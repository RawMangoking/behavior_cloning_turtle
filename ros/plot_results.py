"""Make the README charts from the evaluation CSVs in ros/data/.

    pip install matplotlib
    python3 ros/plot_results.py        # run from the repo root; writes media/*.png
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

DATA = 'ros/data'
OUT = 'media'
COLORS = {'success': '#2e9e5b', 'collision': '#d64545', 'timeout': '#e8a33d'}
OUTCOMES = ['success', 'collision', 'timeout']

CONTROLLERS = [   # (label, file), all tested on the same 100 layouts (seed 100000)
    ('Expert', 'eval_expert.csv'),
    ('Expert, no memory', 'eval_expert_nomem.csv'),
    ('Behavior cloning', 'eval_policy.csv'),
    ('DAgger 2', 'eval_dagger2.csv'),
    ('DAgger 3', 'eval_dagger3.csv'),
    ('DAgger 4', 'eval_dagger4.csv'),
    ('DAgger 5', 'eval_dagger5.csv'),
    ('DAgger + relabel', 'eval_nomem.csv'),
]

# Best validation loss printed by `train` after each stage (not stored in CSVs).
# DAgger 2's value wasn't recorded, so it's left as a gap.
VAL_LOSS = {'BC': 0.533, 'DAgger 1': 0.446, 'DAgger 2': np.nan,
            'DAgger 3': 0.400, 'DAgger 4': 0.354, 'DAgger 5': 0.337}


def load(name):
    with open(os.path.join(DATA, name)) as f:
        return {int(r['episode']): r['outcome'] for r in csv.DictReader(f)}


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(length=0)


def outcomes_chart(results):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    labels = [lab for lab, _ in CONTROLLERS][::-1]
    left = np.zeros(len(labels))
    for k in OUTCOMES:
        vals = np.array([100 * sum(o == k for o in results[lab].values()) / len(results[lab])
                         for lab in labels])
        ax.barh(labels, vals, left=left, color=COLORS[k], label=k.capitalize(), height=0.65)
        for y, (x0, v) in enumerate(zip(left, vals)):
            if v >= 6:
                ax.text(x0 + v / 2, y, f'{v:.0f}%', ha='center', va='center',
                        color='white', fontsize=9, fontweight='bold')
        left += vals
    ax.set_xlim(0, 100)
    ax.set_xlabel('Episodes (%)')
    ax.set_title('Outcomes on 100 unseen random layouts', loc='left', fontweight='bold')
    ax.legend(ncol=3, loc='lower center', bbox_to_anchor=(0.5, -0.32), frameon=False)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'outcomes.png'), dpi=150)


def dagger_chart(results):
    stages = ['BC', 'DAgger 1', 'DAgger 2', 'DAgger 3', 'DAgger 4', 'DAgger 5']
    files = {'BC': 'Behavior cloning', 'DAgger 2': 'DAgger 2', 'DAgger 3': 'DAgger 3',
             'DAgger 4': 'DAgger 4', 'DAgger 5': 'DAgger 5'}   # DAgger 1 wasn't tested
    x = np.arange(len(stages))

    def rate(k):
        return [100 * sum(o == k for o in results[files[s]].values()) / 100
                if s in files else np.nan for s in stages]

    expert = results['Expert']
    exp_success = 100 * sum(o == 'success' for o in expert.values()) / len(expert)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for k in ['success', 'collision']:
        y = np.array(rate(k))
        m = ~np.isnan(y)
        a1.plot(x[m], y[m], 'o-', color=COLORS[k], lw=2.2, label=k.capitalize())
    a1.axhline(exp_success, color='#888', ls='--', lw=1.2)
    a1.text(x[-1], exp_success + 2, f'Expert {exp_success:.0f}%', ha='right', color='#666',
            fontsize=9)
    a1.set_ylim(0, 100)
    a1.set_ylabel('Episodes (%)')
    a1.set_title('Driving: test-set success and collisions', loc='left', fontweight='bold')
    a1.legend(frameon=False, loc='center right')

    v = np.array([VAL_LOSS[s] for s in stages])
    m = ~np.isnan(v)
    a2.plot(x[m], v[m], 'o-', color='#4a6fd1', lw=2.2)
    a2.set_ylim(0, 0.6)
    a2.set_ylabel('Best validation loss')
    a2.set_title('Offline: validation loss', loc='left', fontweight='bold')

    for a in (a1, a2):
        a.set_xticks(x, stages, rotation=30, ha='right', rotation_mode='anchor')
        style(a)
    fig.text(0.5, 0.03, 'Validation loss keeps falling, but driving plateaus after DAgger 2.  '
             '(Not measured: DAgger 1 on the test set, DAgger 2 validation loss.)',
             ha='center', fontsize=9.5, color='#555')
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(os.path.join(OUT, 'dagger.png'), dpi=150)


def layouts_chart(results):
    rows = ['Expert', 'Behavior cloning', 'DAgger 2']
    eps = sorted(results['Expert'])
    code = {'success': 0, 'collision': 1, 'timeout': 2}
    grid = np.array([[code[results[r][e]] for e in eps] for r in rows])
    cmap = plt.matplotlib.colors.ListedColormap([COLORS[k] for k in OUTCOMES])

    fig, ax = plt.subplots(figsize=(10, 2.9))
    fig.subplots_adjust(left=0.15, right=0.98, top=0.83, bottom=0.36)
    ax.imshow(grid, aspect='auto', cmap=cmap, vmin=0, vmax=2, interpolation='nearest')
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xticks([0, 24, 49, 74, 99], ['1', '25', '50', '75', '100'])
    ax.set_xlabel('Test layout')
    ax.set_xticks(np.arange(-0.5, len(eps)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(rows)), minor=True)
    ax.grid(which='minor', color='white', lw=0.6)
    ax.tick_params(which='both', length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    handles = [plt.matplotlib.patches.Patch(color=COLORS[k], label=k.capitalize())
               for k in OUTCOMES]
    fig.legend(handles=handles, ncol=3, loc='lower center', frameon=False)
    ax.set_title('Every test layout, same order for each controller', loc='left',
                 fontweight='bold')
    fig.savefig(os.path.join(OUT, 'layouts.png'), dpi=150)


def main():
    os.makedirs(OUT, exist_ok=True)
    results = {lab: load(f) for lab, f in CONTROLLERS}
    outcomes_chart(results)
    dagger_chart(results)
    layouts_chart(results)
    print(f'wrote {OUT}/outcomes.png, {OUT}/dagger.png, {OUT}/layouts.png')


if __name__ == '__main__':
    main()
