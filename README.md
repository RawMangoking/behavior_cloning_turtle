# Imitation learning for robot navigation with ROS 2 and PyTorch

A TurtleBot3 learns to drive to goal points through randomly placed obstacles
using only its 2D LiDAR, by imitating a hand-written expert controller. Built
with ROS 2 Jazzy, Gazebo Harmonic and PyTorch.

The project goes from plain behavior cloning (BC), through DAgger, to a
tested diagnosis of what limits the learned policy. Every result below comes
from the same 100 unseen, randomly generated layouts, so all rows are directly
comparable.

[TurtleBot3 driving around obstacles with the DAgger policy (3× speed)] (media/demo.gif)

## Results

Test set: 100 random layouts (6 obstacles, random start and goal), never used
for training. Episode ends on reaching the goal (within 0.25 m), a collision
(LiDAR reading under 0.15 m) or a 30 s timeout.

| Controller | Success | Collision | Timeout | Avg time (s) | Avg path (m) |
|---|---|---|---|---|---|
| Expert (gap following) | 79% | 4% | 17% | 13.8 | 2.50 |
| Expert, no memory | 80% | 6% | 14% | 14.5 | 2.48 |
| Behavior cloning | 64% | 23% | 13% | 12.5 | 2.33 |
| **DAgger, iteration 2** | **73%** | **7%** | 20% | 13.5 | 2.41 |
| DAgger, iteration 3 | 72% | 9% | 19% | 14.4 | 2.49 |
| DAgger, iteration 4 | 72% | 13% | 15% | 13.7 | 2.43 |
| DAgger, iteration 5 | 63% | 15% | 22% | 13.3 | 2.36 |
| DAgger + memory-free relabeling | 73% | 9% | 18% | 13.1 | 2.35 |

Times and path lengths are averaged over successful episodes. With 100
episodes, a success rate is uncertain by roughly ±9 percentage points, so
small differences between rows are not meaningful.

## Findings

**1. DAgger fixes compounding errors.** Plain BC crashed in 23% of episodes,
against the expert's 4%. The network only saw the expert's own trajectories,
so once it drifted off them it had no examples to recover from. DAgger lets
the network drive while the expert labels the states it actually visits.
After two iterations, collisions fell to 7% and success rose from 64% to 73%,
which is 92% of the expert's success rate. On the same layouts, DAgger
succeeded on 19 that BC failed, while BC succeeded on 10 that DAgger failed.

**2. More DAgger iterations stopped helping.** Iterations 3 to 5 plateaued
around 72%, with iteration 5 dropping to 63%.

**3. Validation loss is not driving performance.** Validation loss fell at
every iteration, from 0.53 (BC) to 0.34 (DAgger 5), while closed-loop success
did not improve. Matching the expert's labels more closely did not mean
driving better. Only closed-loop tests in the simulator measure that.

**4. The remaining failures are the network's own hesitation, and the
expert's memory is not the cause.** Timeouts became the main failure mode
(20%). Of DAgger 2's 20 timeouts, the expert also failed on only 5 of those
layouts. Hypothesis tested: the expert remembers its previous steering
direction, which the network cannot see, so identical inputs could carry
conflicting labels. I relabeled the entire dataset offline with a
memory-free expert and retrained. The result was unchanged (73%), and the
memory-free expert drives as well as the original (80% vs 79%), so the
hypothesis was rejected.

The likely remaining cause is that the expert's left/right choice flips
abruptly near decision points, while MSE regression produces a smooth
average there ("go straight, slowly"), which shows up as hesitation.

## How it works

```
expert drives ──> demos CSV ──> train MLP ──> policy drives ──┐
     ^                                                       │
     └───────── DAgger: expert labels the policy's states <──┘
```

- **Arena.** A 4 m × 4 m walled world in Gazebo with a pool of 10
  cylinders. Each episode moves a random subset into the arena and places the
  robot and goal at random free positions, in one Gazebo service call. The
  layout is generated from `seed + episode`, so any controller can be tested
  on exactly the same layouts.
- **Observation (26 values).** The 360-ray LiDAR scan reduced to 24 sectors
  (closest reading per 15° slice), plus distance to the goal and heading
  error. Robot pose comes from odometry, corrected for the teleport at
  each episode start.
- **Expert.** A gap-following controller: it steers toward the free direction
  closest to the goal, with a short-range push away from nearby obstacles.
- **Recording.** Labels are the expert's clean commands, while the commands
  actually executed get Gaussian noise, so the data includes recovery from
  off-path states. Only successful episodes are kept for BC.
- **Model.** An MLP (26 → 128 → 128 → 2) trained with MSE on normalized
  inputs and outputs. Validation is split by episode, not by row, to avoid
  leakage between near-identical consecutive samples. The best epoch is kept.
- **DAgger.** The policy node has a record mode. The network drives, and the
  expert silently labels every state. All episodes are kept, including
  failures, because the states just before a failure are the most useful ones.

Final dataset: 750 episodes, about 54,000 samples (250 expert episodes plus
5 DAgger iterations of 100 episodes).

## Repository layout

```
ros/
├── src/
│   ├── bc_tb3/                TurtleBot3 project (this README)
│   │   ├── bc_tb3/
│   │   │   ├── expert.py      expert controller + demo recording
│   │   │   ├── policy.py      learned policy + DAgger record mode
│   │   │   ├── train.py       MLP training
│   │   │   ├── relabel.py     offline relabeling with the memory-free expert
│   │   │   ├── driver.py      shared episode runner (layouts, outcomes, logging)
│   │   │   ├── arena.py       random layouts + Gazebo teleport
│   │   │   ├── common.py      shared LiDAR/goal features
│   │   │   └── model.py       network definition
│   │   ├── launch/arena.launch.py
│   │   └── worlds/rl_arena.sdf
│   └── bc_turtle/             Part 1: turtlesim warm-up (see below)
├── data/                      trained models (.pt) and evaluation results (.csv)
└── overnight.sh               runs DAgger iterations 3-5 unattended
```

## Running it

Needs ROS 2 Jazzy, Gazebo Harmonic, `ros-jazzy-turtlebot3-gazebo` and
PyTorch. Paths assume the workspace is at `/root/ros2_ws`.

```bash
colcon build && source install/setup.bash
export TURTLEBOT3_MODEL=burger
ros2 launch bc_tb3 arena.launch.py gui:=false      # terminal 1

# terminal 2: record expert demos, train
ros2 run bc_tb3 expert --ros-args -p record:=true -p noise:=0.3 -p episodes:=300 -p seed:=0 -p timeout:=30.0
ros2 run bc_tb3 train

# one DAgger iteration (use a new seed each time)
ros2 run bc_tb3 policy --ros-args -p record:=true -p episodes:=100 -p seed:=200000 -p timeout:=30.0
ros2 run bc_tb3 train

# evaluate on the fixed test layouts
ros2 run bc_tb3 policy --ros-args -p episodes:=100 -p seed:=100000 -p timeout:=30.0 \
    -p results_csv:=/root/ros2_ws/data/eval.csv
```

Seed 100000 is reserved for testing. Training data uses seeds 0 to 299 and
200000 and above, so the test layouts are never trained on. To try the
trained model directly, pass `-p model_path:=/root/ros2_ws/data/tb3_policy_dagger2.pt`.

## Next steps

- **Treat steering as a choice, not an average.** Predict discrete steering
  directions with a classifier, or use a mixture density network, so the
  policy picks one side instead of averaging left and right. This targets
  the hesitation timeouts directly.
- **A better expert.** The expert's own 17% timeout rate caps what imitation
  can reach. An escape behaviour for when progress stalls would raise the ceiling.
- **Go beyond the teacher.** Use the imitation policy as a starting point for
  reinforcement learning, which can improve past the expert.
- **Sim-to-real.** Run the policy node on a physical TurtleBot3.

## Part 1: turtlesim warm-up (`bc_turtle`)

The same pipeline on turtlesim, without obstacles. A proportional controller
drives to random goals and records demonstrations (inputs: distance to goal
and heading error; labels: the expert's clean commands, executed with
Gaussian noise). An MLP trained on 15,183 samples from 300 goals reached a
validation loss of 0.0002.

```bash
ros2 run bc_turtle expert --ros-args -p record:=true -p noise:=0.3 -p max_goals:=300
python3 ros/src/bc_turtle/bc_turtle/train.py
```
