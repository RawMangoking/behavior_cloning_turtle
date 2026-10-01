"""Expert: gap-following controller with a short-range safety push.

Each tick it looks at the 24 LiDAR directions, picks the free one closest to
the goal (sticking with its previous choice to avoid dithering), then nudges
away from anything very close. Tested at 92-100% success, 0 collisions.

Records demonstrations to CSV. Labels are the expert's clean commands; the
executed commands get Gaussian noise so the data covers off-path states
(same idea as bc_turtle). Only SUCCESSFUL episodes are saved, so the network
never learns from runs that crashed or got stuck.
"""
import csv
import math
import os
import random

import numpy as np
import rclpy

from bc_tb3.common import (CSV_HEADER, FRONT, N_RAYS, SECTOR_ANGLES, V_MAX, W_MAX,
                           clip_cmd, features, wrap)
from bc_tb3.driver import ArenaDriver

LOOK = 0.8      # a direction is "free" if clear for min(goal dist, LOOK) metres
STICK = 0.3     # preference for keeping the previous direction (anti-dither)
SAFE = 0.4      # obstacles closer than this push the robot away
K_REP = 0.3
K_TURN = 1.5


def expert_cmd(rays, dist, heading_error, prev_target=None):
    """Returns (v, w, chosen_direction)."""
    # a direction is free if it and its neighbours (robot width) are clear
    clear = np.array([min(rays[i - 1], rays[i], rays[(i + 1) % N_RAYS])
                      for i in range(N_RAYS)])
    free = clear > min(dist, LOOK)
    goal_sector = int((heading_error % (2 * math.pi)) / (2 * math.pi / N_RAYS)) % N_RAYS

    if free[goal_sector]:
        target = heading_error                      # straight at the goal
    elif free.any():                                # free direction closest to the goal
        cost = np.abs([wrap(a - heading_error) for a in SECTOR_ANGLES])
        if prev_target is not None:
            cost += STICK * np.abs([wrap(a - prev_target) for a in SECTOR_ANGLES])
        cost[~free] = np.inf
        target = float(SECTOR_ANGLES[int(np.argmin(cost))])
    else:                                           # boxed in: face the most open side
        target = float(SECTOR_ANGLES[int(np.argmax(clear))])

    # short-range safety: push away from anything very close
    fx, fy = math.cos(target), math.sin(target)
    for d, a in zip(rays, SECTOR_ANGLES):
        if d < SAFE:
            m = K_REP * (1.0 / max(d, 0.05) - 1.0 / SAFE)
            fx -= m * math.cos(a)
            fy -= m * math.sin(a)
    heading = math.atan2(fy, fx)

    v = V_MAX * max(0.0, math.cos(heading)) * min(1.0, rays[FRONT].min() / 0.5)
    return (*clip_cmd(v, K_TURN * heading), target)


class Expert(ArenaDriver):
    def __init__(self):
        super().__init__('expert')
        self.declare_parameter('record', False)
        self.declare_parameter('noise', 0.0)
        self.declare_parameter('data_path', '/root/ros2_ws/data/tb3_demos.csv')
        self.declare_parameter('memory', True)   # False = forget previous direction
        self.record = self.get_parameter('record').value
        self.noise = self.get_parameter('noise').value
        self.buffer = []
        self.saved = 0
        self.prev_target = None

        self.csv_file = None
        if self.record:
            path = self.get_parameter('data_path').value
            os.makedirs(os.path.dirname(path), exist_ok=True)
            is_new = not os.path.exists(path)
            self.csv_file = open(path, 'a', newline='')
            self.writer = csv.writer(self.csv_file)
            if is_new:
                self.writer.writerow(CSV_HEADER)

    def act(self, rays, dist, heading_error):
        v, w, target = expert_cmd(rays, dist, heading_error, self.prev_target)
        if self.get_parameter('memory').value:
            self.prev_target = target
        if self.record:   # clean action = label
            self.buffer.append([self.seed + self.episode,
                                *features(rays, dist, heading_error), v, w])
        # noisy action = executed
        return clip_cmd(v + random.gauss(0, self.noise * 0.1 * V_MAX),
                        w + random.gauss(0, self.noise * 0.5 * W_MAX))

    def on_episode_start(self):
        self.buffer = []
        self.prev_target = None

    def on_episode_end(self, outcome):
        if self.csv_file and outcome == 'success':
            self.writer.writerows(self.buffer)
            self.csv_file.flush()
            self.saved += len(self.buffer)
            self.get_logger().info(f'Saved {len(self.buffer)} samples ({self.saved} total)')
        self.buffer = []

    def close(self):
        super().close()
        if self.csv_file:
            self.csv_file.close()


def main():
    rclpy.init()
    node = Expert()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
