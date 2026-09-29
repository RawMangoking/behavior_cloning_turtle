import csv
import math
import os
import random

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from turtlesim.msg import Pose


def wrap(angle):
    # keeps an angle inside -pi..pi
    return math.atan2(math.sin(angle), math.cos(angle))


class Expert(Node):
    def __init__(self):
        super().__init__('expert')

        self.declare_parameter('record', False)
        self.declare_parameter('noise', 0.0)
        self.declare_parameter('max_goals', 0)   # 0 = run forever
        self.record = self.get_parameter('record').value
        self.noise = self.get_parameter('noise').value
        self.max_goals = self.get_parameter('max_goals').value
        self.done = False

        self.csv_file = None
        if self.record:
            os.makedirs('/root/ros2_ws/data', exist_ok=True)
            path = '/root/ros2_ws/data/demos.csv'
            is_new = not os.path.exists(path)
            self.csv_file = open(path, 'a', newline='')
            self.writer = csv.writer(self.csv_file)
            if is_new:
                self.writer.writerow(['dist', 'heading_error', 'linear', 'angular'])

        self.pose = None
        self.goal_count = 0
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

        if dist < 0.2:
            self.cmd_pub.publish(Twist())
            elapsed = (self.get_clock().now() - self.goal_start).nanoseconds / 1e9
            self.goal_count += 1
            self.get_logger().info(f'Goal {self.goal_count} reached in {elapsed:.2f} s')
            if self.csv_file:
                self.csv_file.flush()
            if self.max_goals and self.goal_count >= self.max_goals:
                self.get_logger().info('Done recording')
                self.done = True
                return
            self.pick_goal()
            return

        v = min(1.5 * dist * max(0.0, math.cos(heading_error)), 3.0)
        w = 4.0 * heading_error

        if self.record:
            self.writer.writerow([dist, heading_error, v, w])   # clean action = label

        cmd = Twist()
        cmd.linear.x = v + random.gauss(0, self.noise * 1.0)     # noisy action = executed
        cmd.angular.z = w + random.gauss(0, self.noise * 2.0)
        self.cmd_pub.publish(cmd)

    def close(self):
        if rclpy.ok():
            self.cmd_pub.publish(Twist())
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
