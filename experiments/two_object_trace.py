"""Hand-to-object distance over time for one two-object start at two horizons."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from utils.config import EPISODES, RESULTS

STEM = "E3_soup_and_sauce_init00"
HORIZONS = (1, 25)
LABELS = ("alphabet soup", "tomato sauce")
COLORS = ("#0072B2", "#D55E00")


def main():
    fig, axes = plt.subplots(len(HORIZONS), 1, figsize=(10.2, 5.0), sharex=True)
    for ax, k in zip(axes, HORIZONS):
        meta = json.loads((EPISODES / f"{STEM}_k{k:02d}.json").read_text())
        data = np.load(EPISODES / f"{STEM}_k{k:02d}.npz")
        eef, target, grip = data["eef"], data["target"], data["action"][:, -1]
        dist = np.linalg.norm(target - eef[:, None, :], axis=2) * 100.0
        steps = np.arange(len(eef))
        closed = grip > 0.0
        ax.fill_between(steps, 0, 60, where=closed, color="#999999", alpha=0.25, linewidth=0)
        for j, (label, color) in enumerate(zip(LABELS, COLORS)):
            ax.plot(steps, dist[:, j], color=color, linewidth=1.6, label=label)
        outcome = "success" if meta["success"] else "failure"
        ax.set_title(
            f"K={k}: {outcome} in {meta['steps']} steps, {meta['n_calls']} calls, "
            f"{meta['n_close']} closes, {meta['n_miss']} on nothing",
            fontsize=10,
            loc="left",
        )
        ax.set_ylim(0, 60)
        ax.set_ylabel("hand to object, cm")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].legend(frameon=False, ncol=2, loc="upper right", fontsize=9)
    axes[-1].set_xlabel("control step (grey: gripper commanded closed)")
    fig.tight_layout()
    dest = RESULTS / "figures" / "two_object_trace.pdf"
    fig.savefig(dest)
    fig.savefig(dest.with_suffix(".png"), dpi=110)
    print("wrote", dest)


if __name__ == "__main__":
    main()
