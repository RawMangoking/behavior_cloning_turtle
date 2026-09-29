Behavior cloning on turtlesim contoled using ROS 2 and trained using PyTorch

Imitation learning pipeline in ROS 2 Jazzy


1. expert.py - proportional controller  Also records demonstrations to a CSV.
2. Recording - inputs are distance to goal and heading error . Labels are the
   expert's clean commands; the executed commands get Gaussian noise so
   the data covers off-path states.
3. train.py - MLP, MSE loss, normalized
   inputs and outputs, 90/10 train/validation split.

## Results so far

- 300 goals recorded, 15,183 samples
- Validation loss 0.0002 (normalized units) after 100 epochs


Deploying the learned policy as a ROS 2 node and comparing it against
the expert (success rate, time to goal) is the next step. Validation
loss only measures agreement on recorded states, so it does not yet say
how well the network actually drives.

## Run

Needs ROS 2 Jazzy with turtlesim and PyTorch (the CPU build is enough).
Paths assume the workspace is mounted at /root/ros2_ws. Start
turtlesim_node, then:

    ros2 run bc_turtle expert --ros-args -p record:=true -p noise:=0.3 -p max_goals:=300
    python3 ros/src/bc_turtle/bc_turtle/train.py
