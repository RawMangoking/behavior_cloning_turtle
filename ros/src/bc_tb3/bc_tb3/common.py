"""Shared by expert, policy and train, so the features always match."""
import math

import numpy as np

N_RAYS = 24             # LiDAR shrunk to this many sectors
RANGE_MAX = 3.5         # TurtleBot3 LDS max range (m)
COLLISION_DIST = 0.15   # closest ray below this = crash
GOAL_TOL = 0.25         # within this distance = goal reached (m)
V_MAX = 0.22            # burger max linear speed (m/s)
W_MAX = 1.5             # angular limit (rad/s)

N_FEATURES = N_RAYS + 2     # rays + dist + heading_error
CSV_HEADER = (['episode'] + [f'ray{i}' for i in range(N_RAYS)]
              + ['dist', 'heading_error', 'linear', 'angular'])

# Centre angle of each sector. TurtleBot3 scans 0 -> 2*pi, index 0 = straight ahead.
SECTOR_ANGLES = np.array([math.atan2(math.sin(a), math.cos(a))
                          for a in (np.arange(N_RAYS) + 0.5) * 2 * math.pi / N_RAYS])
FRONT = np.abs(SECTOR_ANGLES) < math.pi / 6     # +-30 deg cone


def wrap(angle):
    # keeps an angle inside -pi..pi
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def process_scan(ranges):
    """Clean the raw scan and keep the CLOSEST reading in each sector.

    Min (not every Nth ray) so a thin obstacle between samples isn't missed.
    """
    r = np.asarray(ranges, dtype=np.float32)
    r = np.where(np.isnan(r) | np.isposinf(r), RANGE_MAX, r)   # nothing in range
    r = np.where(np.isneginf(r), 0.0, r)                       # closer than range_min
    r = np.clip(r, 0.0, RANGE_MAX)
    return np.array([s.min() for s in np.array_split(r, N_RAYS)], dtype=np.float32)


def goal_features(x, y, yaw, gx, gy):
    dist = math.hypot(gx - x, gy - y)
    return dist, wrap(math.atan2(gy - y, gx - x) - yaw)


def features(rays, dist, heading_error):
    return np.concatenate([rays, [dist, heading_error]]).astype(np.float32)


def clip_cmd(v, w):
    return float(np.clip(v, 0.0, V_MAX)), float(np.clip(w, -W_MAX, W_MAX))


# Odom doesn't jump when Gazebo teleports the robot, so world pose is
# tracked as: start_pose + (odom motion since the teleport).
def se2_relative(x0, y0, t0, x1, y1, t1):
    """Pose 1 expressed in the frame of pose 0."""
    dx, dy = x1 - x0, y1 - y0
    c, s = math.cos(t0), math.sin(t0)
    return c * dx + s * dy, -s * dx + c * dy, wrap(t1 - t0)


def se2_compose(xa, ya, ta, xb, yb, tb):
    c, s = math.cos(ta), math.sin(ta)
    return xa + c * xb - s * yb, ya + s * xb + c * yb, wrap(ta + tb)
