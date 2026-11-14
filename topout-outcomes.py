"""topout-outcomes: what the game-level outcome distributions actually look like, per arm.

    python topout-outcomes.py --arm NAME=PATH[,PATH...] [--arm ...] [--ref NAME=PATH[,...]] -o outcomes.png

An arm is one or more journals, game directories or directories of games. Each game's outcome is its
journal's final summary record: pieces placed and lines cleared. Games without a summary (still running)
are skipped and counted. One row per arm, plus a pooled row on top (every --arm, not --ref): the
histogram of lines cleared (one bar per value) and of pieces placed (2-piece bins), raw counts.

Prints, per arm, the number of games at each lines-cleared value. No means, no medians: the histogram is
the result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from topout import journals

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = "#2a78d6"


def outcome(j: Path) -> dict | None:
    last = None
    with j.open() as f:
        for line in f:
            if '"kind": "summary"' in line or '"kind":"summary"' in line:
                last = line
    if last is None:
        return None
    d = json.loads(last)
    return {"pieces": int(d.get("pieces") or 0), "lines": int(d.get("lines") or 0),
            "termination": d.get("termination")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, help="NAME=PATH[,PATH...]")
    ap.add_argument("--ref", action="append", default=[], help="NAME=PATH[,...]: plotted, but left out of the pooled row")
    ap.add_argument("-o", "--out", required=True, type=Path)
    a = ap.parse_args()

    arms: list[tuple[str, list[dict]]] = []
    refs = set(journals.arms(a.ref))
    print(f"{'arm':38s} {'games':>5s} {'skip':>4s}   games by lines cleared 0,1,2,...")
    for name, paths in {**journals.arms(a.arm), **journals.arms(a.ref)}.items():
        games, skipped = [], 0
        for j in paths:
            o = outcome(j)
            if o is None:
                skipped += 1
            else:
                games.append(o)
        if not games:
            print(f"{name:38s} {0:5d} {skipped:4d}   (no finished games: skipped)")
            continue
        counts = np.bincount([g["lines"] for g in games])
        print(f"{name:38s} {len(games):5d} {skipped:4d}   " + " ".join(str(int(v)) for v in counts))
        arms.append((name, games))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker

    pool = [g for nm, gs_ in arms if nm not in refs for g in gs_]
    rows = [("all non-reference games", pool)] + arms
    lmax = max(g["lines"] for _, gs_ in rows for g in gs_)
    pmax = max(g["pieces"] for _, gs_ in rows for g in gs_)
    lbins = np.arange(-0.5, lmax + 1.5, 1)
    pbins = np.arange(0, pmax + 3, 2)
    fig, axes = plt.subplots(len(rows), 2, figsize=(14, 1.15 * len(rows) + 0.8), facecolor=SURFACE, squeeze=False,
                             sharex="col", gridspec_kw={"hspace": 0.55, "wspace": 0.12, "width_ratios": [1, 1.4]})
    for r, (name, games) in enumerate(rows):
        for c, (key, bins) in enumerate((("lines", lbins), ("pieces", pbins))):
            ax = axes[r, c]
            ax.set_facecolor(SURFACE)
            counts, edges = np.histogram([g[key] for g in games], bins=bins)
            w = edges[1] - edges[0]
            ax.bar(edges[:-1] + w / 2, counts, width=w * 0.85, color=MUTED if r == 0 else SERIES, zorder=3)
            ax.set_ylim(0, max(1, counts.max()) * 1.15)
            ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True, nbins=3))
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            for sp in ("left", "bottom"):
                ax.spines[sp].set_color(MUTED)
            ax.tick_params(colors=MUTED, labelsize=7)
            if c == 0:
                ax.set_title(f"{name}  (n={len(games)})", loc="left", fontsize=9, color=INK)
        if r == len(rows) - 1:
            axes[r, 0].set_xlabel("lines cleared at game end (games per value)", fontsize=8, color=MUTED)
            axes[r, 1].set_xlabel("pieces placed at game end (games per 2-piece bin)", fontsize=8, color=MUTED)
    fig.savefig(a.out, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
