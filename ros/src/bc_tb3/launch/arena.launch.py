"""Gazebo with the arena world + TurtleBot3.

    ros2 launch bc_tb3 arena.launch.py              # with GUI
    ros2 launch bc_tb3 arena.launch.py gui:=false   # headless, faster
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument,
                            IncludeLaunchDescription)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    world = os.path.join(get_package_share_directory('bc_tb3'), 'worlds', 'rl_arena.sdf')
    tb3 = get_package_share_directory('turtlebot3_gazebo')
    gz_sim = os.path.join(get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')
    gui = LaunchConfiguration('gui')

    def include(path, args, **kw):
        return IncludeLaunchDescription(PythonLaunchDescriptionSource(path),
                                        launch_arguments=args.items(), **kw)

    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true'),
        AppendEnvironmentVariable('GZ_SIM_RESOURCE_PATH', os.path.join(tb3, 'models')),
        include(gz_sim, {'gz_args': f'-r -s -v2 {world}', 'on_exit_shutdown': 'true'}),
        include(gz_sim, {'gz_args': '-g -v2', 'on_exit_shutdown': 'true'},
                condition=IfCondition(gui)),
        include(os.path.join(tb3, 'launch', 'robot_state_publisher.launch.py'),
                {'use_sim_time': 'true'}),
        include(os.path.join(tb3, 'launch', 'spawn_turtlebot3.launch.py'),
                {'x_pose': '0.0', 'y_pose': '0.0'}),
    ])
