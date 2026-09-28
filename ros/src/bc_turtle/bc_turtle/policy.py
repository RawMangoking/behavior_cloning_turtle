import math
import random

import rclpy
import torch
import torch.nn as nn
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose

MODEL_PATH = '/root/ros2_ws/data/policy.pt'


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class Policy(Node):
    def __init__(self):
        super().__init__('policy')
        torch.set_num_threads(1)

        self.declare_parameter('max_goals', 0)     # 0 = run forever
        self.declare_parameter('timeout', 15.0)    # give up on a goal after this many seconds
        self.max_goals = self.get_parameter('max_goals').value
        self.timeout = self.get_parameter('timeout').value

        # rebuild the same architecture as train.py, then load the learned weights
        ckpt = torch.load(MODEL_PATH)
        self.model = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 2),
        )
        self.model.load_state_dict(ckpt['model'])
        self.model.eval()
        self.x_mean, self.x_std = ckpt['x_mean'], ckpt['x_std']
        self.y_mean, self.y_std = ckpt['y_mean'], ckpt['y_std']

        self.pose = None
        self.goal_count = 0
        self.reached = 0
        self.timeouts = 0
        self.done = False
        self.goal_start = self.get_clock().now()

        self.create_subscription(Pose, '/turtle1/pose', self.on_pose, 10)
        self.cmd_pub = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
        self.create_timer(0.05, self.tick)   # 20 Hz

        self.pick_goal()

    def on_pose(self, msg):
        self.pose = msg

    def pick_goal(self):
        self.gx = random.uniform(1.0, 10.0)
        self.gy = random.uniform(1.0, 10.0)
        self.goal_start = self.get_clock().now()
        self.get_logger().info(f'New goal: ({self.gx:.1f}, {self.gy:.1f})')

    def tick(self):
        if self.done or self.pose is None:
            return

        dx = self.gx - self.pose.x
        dy = self.gy - self.pose.y
        dist = math.hypot(dx, dy)
        heading_error = wrap(math.atan2(dy, dx) - self.pose.theta)
        elapsed = (self.get_clock().now() - self.goal_start).nanoseconds / 1e9

        success = dist < 0.2
        if success or elapsed > self.timeout:
            self.cmd_pub.publish(Twist())
            self.goal_count += 1
            if success:
                self.reached += 1
                self.get_logger().info(f'Goal {self.goal_count} reached in {elapsed:.2f} s')
            else:
                self.timeouts += 1
                self.get_logger().info(f'Goal {self.goal_count} TIMEOUT (dist left {dist:.2f})')
            if self.max_goals and self.goal_count >= self.max_goals:
                self.get_logger().info(
                    f'Finished: {self.reached}/{self.goal_count} reached, {self.timeouts} timeouts')
                self.done = True
                return
            self.pick_goal()
            return

        # the learned policy: normalize -> network -> undo the normalization
        x = torch.tensor([[dist, heading_error]], dtype=torch.float32)
        with torch.no_grad():
            y = self.model((x - self.x_mean) / self.x_std) * self.y_std + self.y_mean

        cmd = Twist()
        cmd.linear.x = float(y[0, 0])
        cmd.angular.z = float(y[0, 1])
        self.cmd_pub.publish(cmd)

    def close(self):
        if rclpy.ok():
            self.cmd_pub.publish(Twist())


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
