"""topout-flows: where each arm's play lives (the line-level triangle) and how each placement moves it (the cell-level map).

    python topout-flows.py --arm NAME=PATH[,PATH...] [--arm ...] -o flows.png [--view rows | moves]

Contract. Reads finished journals (tick records, through topout/ledger.py; games still running are skipped and counted,
as topout-outcomes skips them). Writes the PNG named by -o; prints one line per game: its outcome and its transition
counts (ledger.flows). Judges nothing beyond the outcome the journal's summary records.

Neither view has a clock: piece index is off the axes, so games of any length and either outcome share a panel without
being pooled. Both draw log colour on one range shared by every panel, so darkness compares across arms.

--view rows (default), line level. Every row of the 20 is empty, live (cells, no sealed gap: completable) or dead
(holds a sealed cell), so every board is a point (live, dead) with empty = 20 - live - dead (threads/topout, the
triangle model): where an arm's play spends its pieces. Per arm:

  cells      the share of the arm's spawn boards in each (live, dead) state (the occluded top-out board is not one),
             in log colour on one range shared by every panel, so darkness compares across arms
  bars       those shares summed: live rows across the top, dead rows down the right, one range for every panel
  markers    each game's last clean board, circles topped out, squares survived, sized by games ending there
  dashed     live + dead = 20, the board's limit. Games end when the next piece cannot spawn, often a stack under the
             spawn point, so topped-out games end a row or few inside it.

Reading (operator, 11-17): a dense origin is a good player. How an arm's density smears toward the death line
fingerprints its style, and where its games end says how they failed: along the floor, too much alive (towers that
never clear); up the dead axis, too much dead (the rising tide of sealed rows); or a mix.

The per-game counts are the triangle's transitions, cumulative over the game: rows opened (empty -> live) and cleared
(live -> empty), the headroom balance; sealings (live -> dead) and unsealings (dead -> live), the debt balance, each a
lower bound (ledger.flows).

--view moves, cell level. Each placement (ledger.placements: the change from the board a piece spawned into to the next
one) is a point (lines cleared 0-4, sealed cells added; negative: dug out): how an arm's play changes the board.
  cells      the share of the arm's placements making each move
  bars       the share of ALL the arm's placements clearing 1, 2, 3 and 4 lines (the 0 bar, most placements, is not
             drawn: a piece brings 4 cells and a line takes 10, so at most ~40% of placements can clear): the clear
             profile, comparable across panels
  numbers    per column, the mean change in sealed cells and the column's placements. Debt is the sum of these changes,
             so the mean, catastrophes included, is what a kind of move adds per placement (a median reads 0 nearly
             everywhere). It pools an arm's placements over every board state.
Reading (11-17): the 0-lines column's tail is erosion against catastrophe (single placements sealing 30-40 cells); the
1+ columns carry the dig-outs (digging comes with clearing); a clear profile with doubles, triples and Tetrises on
fewer clearing placements is play that saves its clears.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from topout import journals, ledger

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BAR, TOPPED = "#2a78d6", "#eb6834"
# the reference sequential blue, step 100 -> 700 (dataviz skill, references/palette.md); empty cells stay the surface
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
        "#1c5cab", "#184f95", "#104281", "#0d366b"]
ROWS = ledger.ROWS


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


# Each view: the points a game contributes, its last point (None: no endpoint), axis labels, and whether its axes are
# fixed. rows is a state (every clean spawn board), moves a transition (every placement).
VIEWS = {
    "rows": dict(points=lambda g: [(r["live_rows"], r["dead_rows"]) for r in g["rows"] if not r["occluded"]],
                 last=lambda g: (g["last"]["live_rows"], g["last"]["dead_rows"]),
                 x="live rows", y="dead rows", fixed=(0, ROWS, 0, ROWS)),
    "moves": dict(points=lambda g: [(e["lines"], e["sealed"]) for e in ledger.placements(g["rows"])],
                  last=None, x="lines cleared by the placement", y="sealed cells added (- dug out)", fixed=None),
}
CAPTIONS = {
    "rows": "share of the arm's pieces spawning into each (live, dead) state; dashed: live + dead = 20",
    "moves": "share of the arm's placements making each (lines cleared, change in sealed cells) move",
}


def plot(arms: dict[str, list[dict]], out: Path, view: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import ListedColormap, LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    v = VIEWS[view]
    pts = {name: [p for g in games for p in v["points"](g)] for name, games in arms.items()}
    allp = [p for ps in pts.values() for p in ps]
    x0, x1, y0, y1 = v["fixed"] or (min(0, min(x for x, _ in allp)), max(x for x, _ in allp),
                                    min(0, min(y for _, y in allp)), max(y for _, y in allp))
    shares = {}
    for name, ps in pts.items():
        counts = np.zeros((y1 - y0 + 1, x1 - x0 + 1))
        for x, y in ps:
            counts[y - y0][x - x0] += 1
        share = counts / counts.sum()
        share[counts == 0] = np.nan
        shares[name] = (share, len(ps))
    finite = np.concatenate([s[np.isfinite(s)] for s, _ in shares.values()])
    norm = LogNorm(vmin=finite.min(), vmax=finite.max())
    margins = {name: (np.nansum(s, axis=0), np.nansum(s, axis=1)) for name, (s, _) in shares.items()}
    if view == "moves":
        # most placements clear nothing: a piece brings 4 cells and a line takes 10, so at most ~40% of placements can
        # clear at all. The zero bin is not drawn, but the bars stay shares of ALL the arm's placements, so their
        # heights compare across panels (a panel with 4 clears is not drawn like one with 300).
        def drop_zero(m, zero):
            m = m.copy()
            m[zero] = 0
            return m
        margins = {name: (drop_zero(xm, 0 - x0), ym) for name, (xm, ym) in margins.items()}
    mtop = max((xm.max() if view == "moves" else max(xm.max(), ym.max())) for xm, ym in margins.values()) * 1.05
    cmap = ListedColormap(RAMP)
    cmap.set_bad(SURFACE)
    xs_all, ys_all = range(x0, x1 + 1), range(y0, y1 + 1)

    cols = min(3, len(arms))
    nrows = -(-len(arms) // cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(4.6 * cols, 4.6 * nrows), squeeze=False, facecolor=SURFACE)
    img = None
    for ax, (name, games) in zip(axes.flat, arms.items()):
        share, total = shares[name]
        img = ax.imshow(share, origin="lower", cmap=cmap, norm=norm, interpolation="nearest", aspect="auto",
                        extent=(x0 - 0.5, x1 + 0.5, y0 - 0.5, y1 + 0.5))
        style(ax)
        ax.grid(False)
        if view == "rows":
            ax.plot([0, ROWS], [ROWS, 0], color=MUTED, linewidth=1, linestyle="--")
            ax.set_aspect("equal")
            ticks = range(0, ROWS + 1, 5)                   # whole rows, every 5, the same on both axes
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
        if view == "moves":
            ax.axhline(0, color=MUTED, linewidth=1)
            ax.set_xticks(range(x0, x1 + 1))
            # per column, the mean change in sealed cells: debt accumulates as the sum of these changes, so the mean is
            # what a column contributes per placement, its catastrophe tail included (a median would read 0 nearly
            # everywhere). Written as numbers, centred on each column in one row near the top (a line at this scale
            # sits on 0), with the column's placement count.
            by = {}
            for x, y in pts[name]:
                by.setdefault(x, []).append(y)
            head = 0.14 * (y1 - y0)                 # a strip above the data range, so no stripe runs under a number
            for x, ys in sorted(by.items()):
                ax.text(x, y1 + 0.5 + head / 2, f"{sum(ys) / len(ys):+.2f}\nn={len(ys)}", ha="center", va="center",
                        fontsize=7, color=INK)
            ax.set_ylim(y0 - 0.5, y1 + 0.5 + head)
        if v["last"]:
            for survived, marker, col in ((False, "o", TOPPED), (True, "s", INK)):
                ends = Counter(v["last"](g) for g in games if g["survived"] == survived)
                if ends:
                    exs, eys, ns = zip(*((x, y, k) for (x, y), k in ends.items()))
                    ax.scatter(exs, eys, s=[28 * k for k in ns], marker=marker, facecolors="none", edgecolors=col,
                               linewidths=1.3, zorder=3)
        ax.set_xlabel(v["x"], color=MUTED, fontsize=8)
        ax.set_ylabel(v["y"], color=MUTED, fontsize=8)
        div = make_axes_locatable(ax)
        top, right = div.append_axes("top", 0.6, pad=0.05, sharex=ax), div.append_axes("right", 0.6, pad=0.05, sharey=ax)
        xm, ym = margins[name]
        top.bar(xs_all, xm, width=0.9, color=BAR, linewidth=0)
        if view == "moves":                     # the per-column means on the panel replace the sealed-cells bars
            right.set_visible(False)
        else:
            right.barh(ys_all, ym, height=0.9, color=BAR, linewidth=0)
        top.set_ylim(0, mtop)
        right.set_xlim(0, mtop)
        for mx in (top, right):
            style(mx)
            mx.tick_params(labelsize=6)
        top.tick_params(labelbottom=False)
        right.tick_params(labelleft=False)
        alive = sum(g["survived"] for g in games)
        unit = "placements" if view == "moves" else "pieces"
        top.set_title(f"{name}  ({alive} / {len(games)} survived, {total} {unit})", loc="left", fontsize=9, color=INK)
    for ax in list(axes.flat)[len(arms):]:
        ax.set_visible(False)
    ends = ("\nmarkers: each game's last clean board, circles topped out, squares survived, sized by games ending there"
            if v["last"] else "")
    bars = ("\nbars: share of all the arm's placements clearing 1-4 lines (the 0 bar not drawn); numbers: per column, the mean change "
            "in sealed cells (debt is their sum) and the column's placements" if view == "moves"
            else "bars: the two axes summed, one range")
    fig.suptitle(f"{CAPTIONS[view]}, log colour on one range for all panels; {bars}"
                 f"{ends}", color=MUTED, fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    cb = fig.colorbar(img, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    fig.savefig(out, dpi=110, facecolor=SURFACE, bbox_inches="tight")


def game(path: Path) -> dict | None:
    """A finished game's spawn boards, transitions and outcome; None while it is still running."""
    summary = journals.summary(path)
    if summary is None:
        return None
    rows = ledger.piece_rows(path)
    clean = [r for r in rows if not r["occluded"]]
    if not clean:
        return None
    return {"rows": rows, "flows": ledger.flows(rows), "last": clean[-1],
            "survived": summary["termination"] == "max_pieces"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=PATH[,PATH...]")
    ap.add_argument("-o", "--out", type=Path, required=True, help="PNG to write")
    ap.add_argument("--view", choices=tuple(VIEWS), default="rows",
                    help="rows: the line-level triangle, (live, dead) rows at every spawn board; moves: the "
                         "cell-level map, per placement (lines cleared, sealed cells added)")
    a = ap.parse_args()
    arms = {}
    for name, paths in journals.arms(a.arm).items():
        games, skipped = [], 0
        for p in paths:
            g = game(p)
            if g is None:
                skipped += 1
                continue
            games.append(g)
            f = g["flows"][-1] if g["flows"] else {"opened": 0, "cleared": 0, "sealings": 0, "unsealings": 0}
            print(f"{name:<18} {journals.game_id(p)}  {'survived' if g['survived'] else 'topped  '}  last board: live "
                  f"{g['last']['live_rows']:2d} dead {g['last']['dead_rows']:2d}   opened {f['opened']:3d} cleared "
                  f"{f['cleared']:3d}  sealings>={f['sealings']:2d} unsealings>={f['unsealings']:2d}")
        if skipped:
            print(f"{name:<18} {skipped} unfinished, skipped")
        if games:
            arms[name] = games
    if not arms:
        raise SystemExit("topout-flows: no finished game in any arm: nothing to plot")
    plot(arms, a.out, a.view)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
