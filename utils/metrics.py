import json
from pathlib import Path

import numpy as np

from utils.config import CORRECTION_DEG, GRASP_RADIUS_M, LIFT_M


def _angle_deg(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    cos = float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _closed_runs(gripper):
    closed = gripper > 0.0
    runs = []
    start = None
    for i, flag in enumerate(closed):
        if flag and start is None:
            start = i
        if start is not None and (not flag or i == len(closed) - 1):
            end = i if not flag else i + 1
            runs.append((start, end))
            start = None
    return runs


def episode_metrics(action, eef, target, replan):
    """Grasp and correction counts from one episode.

    `target` is (T, N, 3) or None when the task has no grasp object.
    A close is a gripper command crossing from open (<=0) to closed (>0).
    A miss is a close during which every target stays outside GRASP_RADIUS_M.
    A slip is a lifted target that drops while the gripper stays closed.
    A recovery is a miss followed later by a lift.
    """
    zeros = {
        "n_close": 0,
        "n_miss": 0,
        "n_slip": 0,
        "n_recovery": 0,
        "n_correction": 0,
    }
    if target is None or len(action) == 0:
        return zeros

    grip = np.asarray(action)[:, -1]
    eef = np.asarray(eef)
    target = np.asarray(target)
    replan = np.asarray(replan, dtype=bool)
    dist = np.linalg.norm(target - eef[:, None, :], axis=2).min(axis=1)
    height = target[:, :, 2]

    n_close = 0
    n_miss = 0
    n_slip = 0
    saw_miss = False
    recovered = False
    for start, end in _closed_runs(grip):
        n_close += 1
        window = slice(start, end)
        lifted = bool((height[window].max(axis=0) - height[start] > LIFT_M).any())
        missed = bool(dist[window].min() > GRASP_RADIUS_M)
        if missed and not lifted:
            n_miss += 1
            saw_miss = True
        if lifted and saw_miss:
            recovered = True
        held = dist[window] <= GRASP_RADIUS_M
        if held.any():
            first_hold = int(np.argmax(held))
            if (~held[first_hold:]).any():
                n_slip += 1

    n_correction = 0
    for t in np.flatnonzero(replan):
        if t < 4 or t + 1 >= len(eef):
            continue
        if dist[t - 1] <= dist[t - 4]:
            continue
        previous = eef[t - 1, :2] - eef[t - 2, :2]
        nxt = eef[t + 1, :2] - eef[t, :2]
        nearest = int(np.linalg.norm(target[t] - eef[t], axis=1).argmin())
        toward = target[t, nearest, :2] - eef[t, :2]
        turned = _angle_deg(previous, nxt) > CORRECTION_DEG
        closer_heading = float(np.dot(nxt, toward)) > float(np.dot(previous, toward))
        if turned and closer_heading:
            n_correction += 1

    zeros.update(
        {
            "n_close": n_close,
            "n_miss": n_miss,
            "n_slip": n_slip,
            "n_recovery": int(recovered),
            "n_correction": n_correction,
        }
    )
    return zeros


def write_summary(episode_dir: Path, dest: Path):
    rows = [json.loads(path.read_text()) for path in sorted(episode_dir.glob("*.json"))]
    groups = {}
    for row in rows:
        key = (row["block"], row["name"], row["k"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (block, name, k), items in sorted(groups.items()):
        n = len(items)
        summary.append(
            {
                "block": block,
                "name": name,
                "k": k,
                "n": n,
                "successes": sum(int(item["success"]) for item in items),
                "mean_calls": float(np.mean([item["n_calls"] for item in items])),
                "mean_policy_s": float(np.mean([item["policy_s"] for item in items])),
                "mean_wall_s": float(np.mean([item["wall_s"] for item in items])),
                "mean_miss": float(np.mean([item["n_miss"] for item in items])),
                "mean_slip": float(np.mean([item["n_slip"] for item in items])),
                "mean_recovery": float(np.mean([item["n_recovery"] for item in items])),
                "mean_correction": float(np.mean([item["n_correction"] for item in items])),
            }
        )
    dest.write_text(json.dumps(summary, indent=2))
    return summary
