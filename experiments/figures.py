import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from utils.config import EPISODES, HORIZONS, RESULTS
from utils.metrics import write_summary

FIGURES = RESULTS / "figures"
COLORS = {1: "#0072B2", 10: "#E69F00", 25: "#009E73", 50: "#D55E00"}
ORDER = (
    ("E1", "bowl_between"),
    ("E1", "table_center"),
    ("E1", "bowl_stove"),
    ("E2", "alphabet_soup"),
    ("E2", "tomato_sauce"),
    ("E2", "milk"),
    ("E3", "soup_and_sauce"),
    ("E3", "stove_moka"),
    ("E4", "open_drawer"),
)


def _rows():
    return [json.loads(path.read_text()) for path in sorted(EPISODES.glob("E*.json"))]


def _groups(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["block"], row["name"], row["k"])].append(row)
    return groups


def _bars(groups, field, ylabel, dest, ylim=None):
    fig, ax = plt.subplots(figsize=(10.2, 3.6))
    x = np.arange(len(ORDER))
    width = 0.18
    for offset, k in enumerate(HORIZONS):
        vals = []
        for key in ORDER:
            items = groups[(key[0], key[1], k)]
            vals.append(float(np.mean([item[field] for item in items])))
        ax.bar(x + (offset - 1.5) * width, vals, width, label=f"K={k}", color=COLORS[k])
    ax.set_xticks(x, [name.replace("_", "\n") for _, name in ORDER], fontsize=8)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(
        ncol=4,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(dest, bbox_inches="tight")
    fig.savefig(dest.with_suffix(".png"), dpi=140, bbox_inches="tight")
    plt.close(fig)


def _paired(rows):
    by_init = defaultdict(dict)
    for row in rows:
        by_init[(row["block"], row["name"], row["init_index"])][row["k"]] = row["success"]
    lines = []
    for block, name in ORDER:
        keys = [key for key in by_init if key[0] == block and key[1] == name]
        both = lambda a, b: sum(by_init[key][a] and not by_init[key][b] for key in keys)
        lines.append(
            {
                "block": block,
                "name": name,
                "n": len(keys),
                "k1_only": both(1, 50),
                "k50_only": both(50, 1),
                "k10_beats_both_ends": sum(
                    by_init[key][10] and not by_init[key][1] and not by_init[key][50] for key in keys
                ),
            }
        )
    return lines


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = _rows()
    groups = _groups(rows)
    summary = write_summary(EPISODES, RESULTS / "summary.json")
    _bars(groups, "success", "success rate", FIGURES / "success_by_k.pdf", ylim=(0, 1.05))
    _bars(groups, "n_calls", "model calls", FIGURES / "calls_by_k.pdf")
    _bars(groups, "n_miss", "missed closes", FIGURES / "miss_by_k.pdf")
    report = {"n": len(rows), "summary": summary, "paired": _paired(rows)}
    (FIGURES / "aggregates.json").write_text(json.dumps(report, indent=2))
    print(f"episodes {len(rows)}  summary rows {len(summary)}")
    for item in report["paired"]:
        print(
            f"{item['block']} {item['name']}: n={item['n']} "
            f"K1-not-K50={item['k1_only']} K50-not-K1={item['k50_only']} "
            f"only-K10={item['k10_beats_both_ends']}"
        )


if __name__ == "__main__":
    main()
