from glob import glob

from setuptools import find_packages, setup

package_name = 'bc_tb3'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/worlds', glob('worlds/*.sdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='Behavior cloning for LiDAR obstacle avoidance on TurtleBot3',
    license='MIT',
    extras_require={'test': ['pytest']},
    entry_points={
        'console_scripts': [
            'expert = bc_tb3.expert:main',
            'policy = bc_tb3.policy:main',
            'train = bc_tb3.train:main',
            'relabel = bc_tb3.relabel:main',
        ],
    },
)
