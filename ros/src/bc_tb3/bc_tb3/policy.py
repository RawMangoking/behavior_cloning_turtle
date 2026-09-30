"""Policy: drives with the learned network instead of the expert.

Uses the same episode runner as the expert, so running both with the same
seed tests them on identical layouts. Use a seed you did NOT record with,
so the layouts are ones the network has never seen.

DAgger mode (record:=true): the NETWORK drives, while the expert silently
labels every state the network visits with what IT would have done. Those
rows are appended to the dataset, so retraining teaches the network how to
recover from its own mistakes. All episodes are saved, failures included,
because the states just before a failure are the most useful ones.
"""
import csv
import os

import rclpy
import torch

from bc_tb3.common import CSV_HEADER, clip_cmd, features
from bc_tb3.driver import ArenaDriver
from bc_tb3.expert import expert_cmd
from bc_tb3.model import make_model


class Policy(ArenaDriver):
    def __init__(self):
        super().__init__('policy')
        torch.set_num_threads(1)
        self.declare_parameter('model_path', '/root/ros2_ws/data/tb3_policy.pt')
        self.declare_parameter('record', False)
        self.declare_parameter('data_path', '/root/ros2_ws/data/tb3_demos.csv')

        ckpt = torch.load(self.get_parameter('model_path').value)
        self.model = make_model()
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()
        self.x_mean, self.x_std = ckpt['x_mean'], ckpt['x_std']
        self.y_mean, self.y_std = ckpt['y_mean'], ckpt['y_std']

        self.record = self.get_parameter('record').value
        self.buffer, self.saved, self.expert_prev = [], 0, None
        self.csv_file = None
        if self.record:
            path = self.get_parameter('data_path').value
            os.makedirs(os.path.dirname(path), exist_ok=True)
            is_new = not os.path.exists(path)
            self.csv_file = open(path, 'a', newline='')
            self.writer = csv.writer(self.csv_file)
            if is_new:
                self.writer.writerow(CSV_HEADER)

    def net_cmd(self, rays, dist, heading_error):
        # normalize -> network -> undo the normalization
        x = torch.tensor(features(rays, dist, heading_error)).unsqueeze(0)
        with torch.no_grad():
            y = self.model((x - self.x_mean) / self.x_std) * self.y_std + self.y_mean
        return clip_cmd(y[0, 0].item(), y[0, 1].item())

    def act(self, rays, dist, heading_error):
        if self.record:   # expert labels the state, but does NOT drive
            v_exp, w_exp, self.expert_prev = expert_cmd(rays, dist, heading_error,
                                                        self.expert_prev)
            self.buffer.append([self.seed + self.episode,
                                *features(rays, dist, heading_error), v_exp, w_exp])
        return self.net_cmd(rays, dist, heading_error)

    def on_episode_start(self):
        self.buffer, self.expert_prev = [], None

    def on_episode_end(self, outcome):
        if self.csv_file and self.buffer:
            self.writer.writerows(self.buffer)
            self.csv_file.flush()
            self.saved += len(self.buffer)
            self.get_logger().info(f'Saved {len(self.buffer)} DAgger samples '
                                   f'({self.saved} total)')
        self.buffer = []

    def close(self):
        super().close()
        if self.csv_file:
            self.csv_file.close()


def main():
    rclpy.init()
    node = Policy()
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
