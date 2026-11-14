"""topout-regions: the height ledger per piece, plotted per game, and exported.

    python topout-regions.py <journal | game dir | dir of games> [...] -o regions.png [--csv out.csv] [--rate K]

Contract. Reads journals (tick records only, through topout/ledger.py: running games up to their last tick). Writes
the PNG named by -o and, with --csv, one row per piece per game: topout/ledger.py's columns, nothing else. Prints one
line per game with its last board's ledger. It judges nothing: no game is classed by outcome, no threshold is applied;
games are coloured by served model. The decomposition and the ledger are defined in topout/ledger.py.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from topout import journals, ledger

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#2f9e6e", "#9b59b6", "#c9a227", "#d6457a"]


def plot(games: list[list[dict]], out: Path, rate: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    panels = [("sealed", "sealed cells"), ("live_rows", "live rows (height - dead rows)"),
              ("dead_rows", "dead rows (a sealed cell)"), ("height", "stack height (rows)")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 3.1 * len(panels)), sharex=True, facecolor=SURFACE)
    models = sorted({rows[0]["model"] for rows in games if rows})
    color = {m: SERIES[i % len(SERIES)] for i, m in enumerate(models)}
    for ax, (key, title) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for rows in games:
            if not rows:
                continue
            xs, ys = [r["piece"] for r in rows], [r[key] for r in rows]
            if rate:                                            # trailing: only what was known at that piece
                xs, ys = xs[rate:], [(ys[i] - ys[i - rate]) / rate for i in range(rate, len(ys))]
            ax.plot(xs, ys, color=color[rows[0]["model"]], linewidth=2, alpha=0.9, solid_capstyle="round")
            last = rows[-1]
            ax.annotate(f"{Path(last['game']).name} {last['lines']}L@{last['piece']}", (xs[-1], ys[-1]),
                        xytext=(4, 0), textcoords="offset points", fontsize=7, color=MUTED, va="center")
        ax.set_title(f"{title}, change per piece over the last {rate}" if rate else title,
                     loc="left", fontsize=11, color=INK)
        if rate:
            ax.axhline(0, color=MUTED, linewidth=1)
        ax.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9)
    axes[-1].set_xlabel("piece", color=MUTED)
    handles = [Line2D([], [], color=color[m], linewidth=2, label=m) for m in models]
    axes[0].legend(handles=handles, loc="upper left", fontsize=8, frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+", help="journals, game directories or directories of games")
    ap.add_argument("-o", "--out", type=Path, required=True, help="PNG to write")
    ap.add_argument("--csv", type=Path, help="per-piece rows to write")
    ap.add_argument("--rate", type=int, default=0, metavar="K",
                    help="plot (v[p] - v[p-K]) / K, the trailing slope over K pieces, instead of v")
    a = ap.parse_args()
    games = [ledger.piece_rows(p) for p in journals.find(a.paths)]
    for rows in games:
        if not rows:
            continue
        last = rows[-1]
        print(f"{last['game']} {last['lines']}L@{last['piece']}  pieces {len(rows):3d}  last board: height "
              f"{last['height']:2d} = packed {last['packed']:4.1f} + open {last['open_slack']:3.1f} + sealed "
              f"{last['sealed_slack']:3.1f}{' (occluded)' if last['occluded'] else ''}   dead {last['dead_rows']:2d}  "
              f"covered {last['covered']:2d} tuckable {last['tuckable']:2d}  occluded boards "
              f"{sum(r['occluded'] for r in rows)}")
    if a.csv:
        with a.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(next(rows for rows in games if rows)[0]))
            w.writeheader()
            for rows in games:
                w.writerows(rows)
    plot(games, a.out, a.rate)
    print(f"wrote {a.out}" + (f" and {a.csv}" if a.csv else ""))


if __name__ == "__main__":
    main()
