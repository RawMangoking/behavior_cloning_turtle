"""Relabel a dataset with the memory-free expert. No simulator needed.

The recording expert remembers its previous direction, which the network
can't see. So the same 26 inputs can carry different labels (left one time,
right another), and MSE training averages them into hesitation. This keeps
every row's inputs and recomputes the label from those inputs alone, so
identical inputs always get identical labels.

    ros2 run bc_tb3 relabel
    ros2 run bc_tb3 train --data /root/ros2_ws/data/tb3_demos_nomem.csv \\
        --out /root/ros2_ws/data/tb3_policy_nomem.pt
"""
import argparse
import csv

import numpy as np

from bc_tb3.common import CSV_HEADER, N_RAYS
from bc_tb3.expert import expert_cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='/root/ros2_ws/data/tb3_demos.csv')
    ap.add_argument('--out', default='/root/ros2_ws/data/tb3_demos_nomem.csv')
    a = ap.parse_args()

    data = np.loadtxt(a.data, delimiter=',', skiprows=1)
    changed, v_diff, w_diff = 0, [], []
    with open(a.out, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for row in data:
            rays = row[1:1 + N_RAYS]
            dist, heading_error = row[1 + N_RAYS], row[2 + N_RAYS]
            v, w, _ = expert_cmd(rays, dist, heading_error, prev_target=None)
            dv, dw = abs(v - row[-2]), abs(w - row[-1])
            if dv > 1e-4 or dw > 1e-4:
                changed += 1
                v_diff.append(dv)
                w_diff.append(dw)
            writer.writerow([int(row[0])] + [f'{x:.9g}' for x in row[1:-2]]
                            + [f'{v:.6f}', f'{w:.6f}'])

    n = len(data)
    print(f'{n} rows relabeled -> {a.out}')
    print(f'{changed} labels changed ({100.0 * changed / n:.1f}%)')
    if changed:
        print(f'  on changed rows: mean |dv| {np.mean(v_diff):.3f} m/s, '
              f'mean |dw| {np.mean(w_diff):.3f} rad/s')


if __name__ == '__main__':
    main()
