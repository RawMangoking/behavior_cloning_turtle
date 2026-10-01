# bc_tb3: behavior cloning with LiDAR obstacle avoidance

Same pipeline as `bc_turtle`, now on a TurtleBot3 in Gazebo with a 2D LiDAR and
random obstacles. Every episode is a new random layout (obstacles, start, goal).

1. `expert` is a gap-following controller (steers to the free direction closest to the goal,
   with a short-range safety push away from obstacles). It records
   demos to CSV: inputs are 24 LiDAR sectors + distance to goal + heading error, and
   labels are the expert's clean commands. Executed commands get noise. Only successful
   episodes are saved.
2. `train` is an MLP with MSE loss and normalized inputs and outputs. The validation split is by episode.
3. `policy` drives with the network. Episode i uses layout seed + i, so the expert
   and the policy can be compared on identical, unseen layouts.

## Run
```bash
colcon build --packages-select bc_tb3 && source install/setup.bash
export TURTLEBOT3_MODEL=burger
ros2 launch bc_tb3 arena.launch.py            # terminal 1 (gui:=false for faster)

# terminal 2
ros2 topic info /cmd_vel    # if it's Twist (not TwistStamped) add: -p cmd_vel_stamped:=false

ros2 run bc_tb3 expert --ros-args -p record:=true -p noise:=0.3 -p episodes:=300 -p seed:=0
ros2 run bc_tb3 train

# compare on 100 layouts neither has seen (seed 100000+)
ros2 run bc_tb3 expert --ros-args -p episodes:=100 -p seed:=100000 -p results_csv:=/root/ros2_ws/data/eval_expert.csv
ros2 run bc_tb3 policy --ros-args -p episodes:=100 -p seed:=100000 -p results_csv:=/root/ros2_ws/data/eval_policy.csv
```
Each run prints a summary: success / collision / timeout % and average time and path length.

## Results
See the [main README](../../../README.md) for results, findings, and the DAgger and
relabeling workflow (`policy` with `record:=true`, `relabel`).
