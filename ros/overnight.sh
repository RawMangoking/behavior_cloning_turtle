#!/bin/bash
# Overnight DAgger: iterations 3..5.
# Each iteration: policy drives + expert labels -> retrain -> save copy -> test on fixed layouts.
# Summary lines go to data/overnight.log; full output of every step goes to data/logs/.
#
#   cd /root/ros2_ws && bash overnight.sh
set -e -o pipefail
cd /root/ros2_ws
source install/setup.bash
mkdir -p data/logs
LOG=data/overnight.log

check_sim() {
  if ! gz model --list 2>/dev/null | grep -q burger; then
    echo "$(date '+%H:%M') ERROR: robot not found in Gazebo, stopping." | tee -a $LOG
    exit 1
  fi
}

for i in 3 4 5; do
  seed=$(( (i + 1) * 100000 ))          # 400000, 500000, 600000 (100000 is the test set)
  echo "=== $(date '+%H:%M') DAgger iteration $i (seed $seed)" | tee -a $LOG

  check_sim
  ros2 run bc_tb3 policy --ros-args -p record:=true -p episodes:=100 \
      -p seed:=$seed -p timeout:=30.0 2>&1 \
      | tee data/logs/iter${i}_record.log | tail -1 | tee -a $LOG

  ros2 run bc_tb3 train 2>&1 \
      | tee data/logs/iter${i}_train.log | grep -E "episodes|saved" | tee -a $LOG
  cp data/tb3_policy.pt data/tb3_policy_dagger$i.pt

  check_sim
  echo "--- test on fixed layouts (seed 100000):" | tee -a $LOG
  ros2 run bc_tb3 policy --ros-args -p episodes:=100 -p seed:=100000 -p timeout:=30.0 \
      -p results_csv:=/root/ros2_ws/data/eval_dagger$i.csv 2>&1 \
      | tee data/logs/iter${i}_eval.log | tail -1 | tee -a $LOG
done

echo "=== $(date '+%H:%M') all done" | tee -a $LOG
