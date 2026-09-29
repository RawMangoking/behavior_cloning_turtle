"""Random arena layouts and moving models in Gazebo.

Must match worlds/rl_arena.sdf: 4 m x 4 m walled arena, obstacle_0..9.
"""
import math
import subprocess

import numpy as np

ARENA_HALF = 2.0        # inner walls at x, y = +-2.0 m
MAX_OBSTACLES = 10
OBSTACLE_HEIGHT = 0.5


def sample_layout(rng, n_obstacles):
    """Random obstacle spots + a free start and goal at least 1.5 m apart."""
    lim = ARENA_HALF - 0.4
    obstacles = []
    for _ in range(5000):
        if len(obstacles) == n_obstacles:
            break
        p = rng.uniform(-lim, lim, size=2)
        if all(np.linalg.norm(p - q) > 0.8 for q in obstacles):   # gaps >= 0.5 m
            obstacles.append(p)

    def clear(p, margin):
        return all(np.linalg.norm(p - q) > margin for q in obstacles)

    lim_r = ARENA_HALF - 0.35
    for _ in range(5000):
        start = rng.uniform(-lim_r, lim_r, size=2)
        goal = rng.uniform(-lim_r, lim_r, size=2)
        if clear(start, 0.55) and clear(goal, 0.5) and np.linalg.norm(start - goal) > 1.5:
            return obstacles, start, goal
    raise RuntimeError('Could not find a free layout, use fewer obstacles')


def move_models(world, poses):
    """Teleport several models in ONE Gazebo call. poses: [(name, x, y, z, yaw)]."""
    parts = []
    for name, x, y, z, yaw in poses:
        parts.append(
            f'pose {{ name: "{name}" position {{ x: {x:.4f} y: {y:.4f} z: {z:.4f} }} '
            f'orientation {{ x: 0 y: 0 z: {math.sin(yaw / 2):.4f} '
            f'w: {math.cos(yaw / 2):.4f} }} }}')
    res = subprocess.run(
        ['gz', 'service', '-s', f'/world/{world}/set_pose_vector',
         '--reqtype', 'gz.msgs.Pose_V', '--reptype', 'gz.msgs.Boolean',
         '--timeout', '5000', '--req', ' '.join(parts)],
        capture_output=True, text=True, timeout=15)
    if 'true' not in res.stdout:
        raise RuntimeError(f'set_pose_vector failed: {res.stdout} {res.stderr}')


def apply_layout(world, robot, obstacles, start, yaw):
    """Robot to start, used obstacles into the arena, the rest parked far outside."""
    poses = [(robot, start[0], start[1], 0.01, yaw)]
    for i in range(MAX_OBSTACLES):
        ox, oy = obstacles[i] if i < len(obstacles) else (20.0 + 2.0 * i, 20.0)
        poses.append((f'obstacle_{i}', ox, oy, OBSTACLE_HEIGHT / 2, 0.0))
    move_models(world, poses)
