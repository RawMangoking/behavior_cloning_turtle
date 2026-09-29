"""Base node: runs episodes in the arena and logs how each one ends.

One episode = new random layout -> drive to the goal -> success / collision / timeout.
Episode i always uses layout seed (seed + i), so the expert and the policy can
be tested on exactly the same layouts. Subclasses only implement act().
"""
import csv
import math
import os

import numpy as np
from geometry_msgs.msg import Twist, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

from bc_tb3 import arena
from bc_tb3.common import (COLLISION_DIST, GOAL_TOL, goal_features, process_scan,
                           se2_compose, se2_relative, yaw_from_quat)


class ArenaDriver(Node):
    def __init__(self, name):
        super().__init__(name)
        self.declare_parameter('n_obstacles', 6)
        self.declare_parameter('episodes', 0)            # 0 = run forever
        self.declare_parameter('seed', 0)
        self.declare_parameter('timeout', 60.0)          # sim seconds per episode
        self.declare_parameter('world', 'rl_arena')
        self.declare_parameter('robot', 'burger')
        self.declare_parameter('cmd_vel_stamped', True)  # check: ros2 topic info /cmd_vel
        self.declare_parameter('results_csv', '')        # optional per-episode log
        p = lambda n: self.get_parameter(n).value   # noqa: E731
        self.n_obstacles, self.episodes, self.seed = p('n_obstacles'), p('episodes'), p('seed')
        self.timeout, self.world, self.robot = p('timeout'), p('world'), p('robot')
        self.stamped = p('cmd_vel_stamped')

        self.results = None
        if p('results_csv'):
            os.makedirs(os.path.dirname(p('results_csv')) or '.', exist_ok=True)
            self.results = open(p('results_csv'), 'w', newline='')
            self.results_writer = csv.writer(self.results)
            self.results_writer.writerow(['episode', 'outcome', 'time_s', 'path_m'])

        msg_type = TwistStamped if self.stamped else Twist
        self.cmd_pub = self.create_publisher(msg_type, '/cmd_vel', 10)
        self.create_subscription(LaserScan, '/scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.on_odom, 10)

        self.odom = None
        self.episode = 0
        self.stats = {'success': 0, 'collision': 0, 'timeout': 0}
        self.success_times, self.success_paths = [], []
        self.state = 'start'
        self.done = False

    # ---------- ROS callbacks ----------
    def on_odom(self, msg):
        p = msg.pose.pose
        self.odom = (p.position.x, p.position.y, yaw_from_quat(p.orientation))

    def on_scan(self, msg):
        # The scan is the control tick (5 Hz). Its stamp is sim time.
        if self.done or self.odom is None:
            return
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.state == 'start':
            self.begin_episode()
        elif self.state == 'settling':           # wait a few scans after teleporting
            self.send(0.0, 0.0)
            self.settle -= 1
            if self.settle == 0:
                self.odom0, self.t0 = self.odom, stamp
                self.prev_xy, self.path = self.start_pose[:2], 0.0
                self.state = 'driving'
        elif self.state == 'driving':
            self.drive(process_scan(msg.ranges), stamp)

    # ---------- episode logic ----------
    def begin_episode(self):
        rng = np.random.default_rng(self.seed + self.episode)
        obstacles, start, goal = arena.sample_layout(rng, self.n_obstacles)
        yaw = float(rng.uniform(-math.pi, math.pi))
        self.send(0.0, 0.0)
        arena.apply_layout(self.world, self.robot, obstacles, start, yaw)
        self.start_pose = (float(start[0]), float(start[1]), yaw)
        self.goal = (float(goal[0]), float(goal[1]))
        self.settle = 3
        self.state = 'settling'
        self.on_episode_start()

    def drive(self, rays, stamp):
        rel = se2_relative(*self.odom0, *self.odom)
        x, y, yaw = se2_compose(*self.start_pose, *rel)
        self.path += math.dist(self.prev_xy, (x, y))
        self.prev_xy = (x, y)
        dist, heading_error = goal_features(x, y, yaw, *self.goal)
        elapsed = stamp - self.t0

        if dist < GOAL_TOL:
            self.end_episode('success', elapsed)
        elif rays.min() < COLLISION_DIST:
            self.end_episode('collision', elapsed)
        elif elapsed > self.timeout:
            self.end_episode('timeout', elapsed)
        else:
            self.send(*self.act(rays, dist, heading_error))

    def end_episode(self, outcome, elapsed):
        self.send(0.0, 0.0)
        self.stats[outcome] += 1
        if outcome == 'success':
            self.success_times.append(elapsed)
            self.success_paths.append(self.path)
        if self.results:
            self.results_writer.writerow([self.seed + self.episode, outcome,
                                          f'{elapsed:.2f}', f'{self.path:.2f}'])
            self.results.flush()
        self.on_episode_end(outcome)
        self.episode += 1
        self.get_logger().info(f'Episode {self.episode}: {outcome} ({elapsed:.1f} s)')

        if self.episodes and self.episode >= self.episodes:
            self.print_summary()
            self.done = True
        else:
            self.state = 'start'

    def print_summary(self):
        n = max(self.episode, 1)
        pct = {k: 100.0 * v / n for k, v in self.stats.items()}
        t = np.mean(self.success_times) if self.success_times else float('nan')
        d = np.mean(self.success_paths) if self.success_paths else float('nan')
        self.get_logger().info(
            f'Finished {self.episode} episodes: success {pct["success"]:.1f}%  '
            f'collision {pct["collision"]:.1f}%  timeout {pct["timeout"]:.1f}%  '
            f'| successful runs: avg {t:.1f} s, {d:.2f} m')

    # ---------- helpers / hooks ----------
    def send(self, v, w):
        if self.stamped:
            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.twist.linear.x, msg.twist.angular.z = float(v), float(w)
        else:
            msg = Twist()
            msg.linear.x, msg.angular.z = float(v), float(w)
        self.cmd_pub.publish(msg)

    def act(self, rays, dist, heading_error):
        raise NotImplementedError

    def on_episode_start(self):
        pass

    def on_episode_end(self, outcome):
        pass

    def close(self):
        self.send(0.0, 0.0)
        if self.results:
            self.results.close()
