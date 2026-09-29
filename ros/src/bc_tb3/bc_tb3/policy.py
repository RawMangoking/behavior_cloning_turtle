"""Policy: drives with the learned network instead of the expert.

Uses the same episode runner as the expert, so running both with the same
seed tests them on identical layouts. Use a seed you did NOT record with,
so the layouts are ones the network has never seen.
"""
import rclpy
import torch

from bc_tb3.common import clip_cmd, features
from bc_tb3.driver import ArenaDriver
from bc_tb3.model import make_model


class Policy(ArenaDriver):
    def __init__(self):
        super().__init__('policy')
        torch.set_num_threads(1)
        self.declare_parameter('model_path', '/root/ros2_ws/data/tb3_policy.pt')

        ckpt = torch.load(self.get_parameter('model_path').value)
        self.model = make_model()
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()
        self.x_mean, self.x_std = ckpt['x_mean'], ckpt['x_std']
        self.y_mean, self.y_std = ckpt['y_mean'], ckpt['y_std']

    def act(self, rays, dist, heading_error):
        # normalize -> network -> undo the normalization
        x = torch.tensor(features(rays, dist, heading_error)).unsqueeze(0)
        with torch.no_grad():
            y = self.model((x - self.x_mean) / self.x_std) * self.y_std + self.y_mean
        return clip_cmd(y[0, 0].item(), y[0, 1].item())


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
