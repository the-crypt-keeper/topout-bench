"""topout-flows: each game as a path through the triangle's transitions, per arm (a prototype).

    python topout-flows.py --arm NAME=PATH[,PATH...] [--arm ...] -o flows.png

Contract. Reads journals (tick records only, through topout/ledger.py: running games up to their last tick). Writes the
PNG named by -o; prints one line per game with its final totals. The transitions are defined in ledger.flows.

Every row of the 20 is empty, live or dead (threads/topout, the triangle model), and the state alone, plotted over
piece index, says little about play. This plots the transitions' running totals against each other instead, so piece
index leaves the axes and games of any length share a panel without being pooled:

  headroom   x = rows opened (empty -> live), y = rows cleared (live -> empty). The diagonal y = x is every opened row
             cleared; the distance below it is 20 - empty. The dashed line x - y = 20 is the death line: no empty row
             left (a game can top out a row or two before it, at a spawn).
  debt       x = sealings (live -> dead), y = unsealings (dead -> live), both lower bounds (ledger.flows). The diagonal
             is no net debt; the distance below it is dead rows held.

One row of panels per arm, axes shared across arms. Paths are coloured by outcome, read from the journal's summary as
topout-outcomes reads it (unfinished games are skipped); nothing else is judged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from topout import journals, ledger

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SURVIVED, TOPPED = "#2a78d6", "#eb6834"


def plot(arms: dict[str, list[tuple[list[dict], bool]]], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    games = [f for paths in arms.values() for f, _ in paths if f]
    hx = max(f[-1]["opened"] for f in games) + 2
    dx = max(max(f[-1]["sealings"], f[-1]["unsealings"]) for f in games) + 1
    fig, axes = plt.subplots(len(arms), 2, figsize=(11, 3.4 * len(arms)), squeeze=False, facecolor=SURFACE)
    for (name, paths), (hax, dax) in zip(arms.items(), axes):
        for ax, (xk, yk), lim in ((hax, ("opened", "cleared"), hx), (dax, ("sealings", "unsealings"), dx)):
            ax.set_facecolor(SURFACE)
            ax.plot([0, lim], [0, lim], color=MUTED, linewidth=1)
            if ax is hax:
                ax.plot([20, lim], [0, lim - 20], color=MUTED, linewidth=1, linestyle="--")
            for f, survived in paths:
                if not f:
                    continue
                xs, ys = [0] + [r[xk] for r in f], [0] + [r[yk] for r in f]
                col = SURVIVED if survived else TOPPED
                ax.plot(xs, ys, color=col, linewidth=1.4, alpha=0.75, drawstyle="steps-post")
                ax.plot(xs[-1], ys[-1], "o", color=col, markersize=3)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_aspect("equal")
            ax.grid(True, color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            for s in ("left", "bottom"):
                ax.spines[s].set_color(GRID)
            ax.tick_params(colors=MUTED, labelsize=8)
        n = sum(1 for f, _ in paths if f)
        alive = sum(1 for f, s in paths if f and s)
        hax.set_title(f"{name}  headroom  ({alive} / {n} survived)", loc="left", fontsize=10, color=INK)
        dax.set_title(f"{name}  debt (lower bounds)", loc="left", fontsize=10, color=INK)
        hax.set_xlabel("rows opened (empty -> live)", color=MUTED, fontsize=8)
        hax.set_ylabel("rows cleared (live -> empty)", color=MUTED, fontsize=8)
        dax.set_xlabel("sealings (live -> dead)", color=MUTED, fontsize=8)
        dax.set_ylabel("unsealings (dead -> live)", color=MUTED, fontsize=8)
    handles = [Line2D([], [], color=SURVIVED, label="survived"), Line2D([], [], color=TOPPED, label="topped out"),
               Line2D([], [], color=MUTED, label="balance (y = x)"),
               Line2D([], [], color=MUTED, linestyle="--", label="death line (x - y = 20)")]
    axes[0][0].legend(handles=handles, loc="upper left", fontsize=7, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=120, facecolor=SURFACE)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=PATH[,PATH...]")
    ap.add_argument("-o", "--out", type=Path, required=True, help="PNG to write")
    a = ap.parse_args()
    arms = {}
    for name, paths in journals.arms(a.arm).items():
        arms[name] = []
        for p in paths:
            summary = None
            with p.open() as fh:
                for line in fh:
                    if '"kind": "summary"' in line:
                        summary = json.loads(line)
            if summary is None:                     # still running: skipped, as topout-outcomes skips it
                print(f"{name:<18} {journals.game_id(p)}  unfinished, skipped")
                continue
            survived = summary["termination"] == "max_pieces"
            f = ledger.flows(ledger.piece_rows(p))
            arms[name].append((f, survived))
            if f:
                last = f[-1]
                print(f"{name:<18} {journals.game_id(p)}  {'survived' if survived else 'topped  '}  opened "
                      f"{last['opened']:3d} cleared {last['cleared']:3d} (20 - empty = {last['opened'] - last['cleared']:2d})"
                      f"  sealings>={last['sealings']:2d} unsealings>={last['unsealings']:2d}")
    plot(arms, a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
