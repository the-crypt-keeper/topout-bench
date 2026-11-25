"""topout-triangle-anim: the line-level triangle (topout-flows --view rows), lit up piece by piece.

    python topout-triangle-anim.py --arm NAME=PATH[,PATH...] [--arm ...] -o triangle.mp4 [--pieces 100] [--fps 5]

Contract. Reads finished journals (games still running are skipped and counted, as topout-flows). Writes the animation
named by -o (.mp4 through ffmpeg, .gif through Pillow) and nothing else.

One frame per piece index k = 1..--pieces. Per arm, frame k shows:
  cells      every clean spawn board of piece k or earlier, from every game that reached it, as a share of the arm's
             final total: the grid lights up as play goes on, and the last frame is topout-flows' static triangle
             (log colour, one range for every panel and frame)
  bars       those shares summed along each axis, on the final frame's range
  dots       where each game still alive at piece k is, sized by games in the cell
  markers    each game's last clean board from the piece it ended on: circles topped out, squares survived
The static view has no clock on purpose (threads/topout, 11-17: removing time was the unlock); this puts the clock back
as the frame, never on an axis, to watch how an arm reaches its fingerprint. Kept as its own script (operator, 11-18).
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from topout import journals, ledger

# topout-flows' palette and ramp (the dataviz reference sequential blue), so a frame reads like the static view
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BAR, TOPPED = "#2a78d6", "#eb6834"
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
        "#1c5cab", "#184f95", "#104281", "#0d366b"]
ROWS = ledger.ROWS


def game(path: Path) -> dict | None:
    """A finished game's clean spawn states by piece, its last clean piece, and its outcome; None while running."""
    summary = journals.summary(path)
    if summary is None:
        return None
    clean = [r for r in ledger.piece_rows(path) if not r["occluded"]]
    if not clean:
        return None
    return {"states": {r["piece"]: (r["live_rows"], r["dead_rows"]) for r in clean}, "end": clean[-1]["piece"],
            "last": (clean[-1]["live_rows"], clean[-1]["dead_rows"]), "survived": summary["termination"] == "max_pieces"}


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def animate(arms: dict[str, list[dict]], out: Path, pieces: int, fps: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    from matplotlib.colors import ListedColormap, LogNorm
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # per arm, the cumulative count grid at every frame, as a share of the arm's final total
    frames = {}
    for name, games in arms.items():
        counts = np.zeros((pieces, ROWS + 1, ROWS + 1))
        for g in games:
            for p, (x, y) in g["states"].items():
                if 1 <= p <= pieces:
                    counts[p - 1, y, x] += 1
        cum = counts.cumsum(axis=0)
        frames[name] = cum / cum[-1].sum()
    finals = [f[-1][f[-1] > 0] for f in frames.values()]
    norm = LogNorm(vmin=min(v.min() for v in finals), vmax=max(v.max() for v in finals))
    mtop = max(max(f[-1].sum(axis=0).max(), f[-1].sum(axis=1).max()) for f in frames.values()) * 1.05
    cmap = ListedColormap(RAMP)
    cmap.set_bad(SURFACE)
    span = range(ROWS + 1)

    cols = min(3, len(arms))
    nrows = -(-len(arms) // cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(4.6 * cols, 4.6 * nrows), squeeze=False, facecolor=SURFACE)
    panels = []
    for ax, (name, games) in zip(axes.flat, arms.items()):
        img = ax.imshow(np.full((ROWS + 1, ROWS + 1), np.nan), origin="lower", cmap=cmap, norm=norm,
                        interpolation="nearest", aspect="equal", extent=(-0.5, ROWS + 0.5, -0.5, ROWS + 0.5))
        style(ax)
        ax.plot([0, ROWS], [ROWS, 0], color=MUTED, linewidth=1, linestyle="--")
        ax.set_xticks(range(0, ROWS + 1, 5))
        ax.set_yticks(range(0, ROWS + 1, 5))
        ax.set_xlabel("live rows", color=MUTED, fontsize=8)
        ax.set_ylabel("dead rows", color=MUTED, fontsize=8)
        here = ax.scatter([], [], s=[], color=INK, zorder=4)
        topped = ax.scatter([], [], s=[], marker="o", facecolors="none", edgecolors=TOPPED, linewidths=1.3, zorder=3)
        survived = ax.scatter([], [], s=[], marker="s", facecolors="none", edgecolors=INK, linewidths=1.3, zorder=3)
        div = make_axes_locatable(ax)
        top, right = div.append_axes("top", 0.6, pad=0.05, sharex=ax), div.append_axes("right", 0.6, pad=0.05, sharey=ax)
        tbars = top.bar(span, np.zeros(ROWS + 1), width=0.9, color=BAR, linewidth=0)
        rbars = right.barh(span, np.zeros(ROWS + 1), height=0.9, color=BAR, linewidth=0)
        top.set_ylim(0, mtop)
        right.set_xlim(0, mtop)
        for mx in (top, right):
            style(mx)
            mx.tick_params(labelsize=6)
        top.tick_params(labelbottom=False)
        right.tick_params(labelleft=False)
        title = top.set_title("", loc="left", fontsize=9, color=INK)
        panels.append((name, games, img, here, topped, survived, tbars, rbars, title))
    for ax in list(axes.flat)[len(arms):]:
        ax.set_visible(False)
    head = fig.suptitle("", color=MUTED, fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    cb = fig.colorbar(panels[-1][2], ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)
    cb.ax.tick_params(colors=MUTED, labelsize=7)

    def points(sc, cells: Counter, unit: float) -> None:
        if cells:
            xy, n = zip(*cells.items())
            sc.set_offsets(np.array(xy, dtype=float))
            sc.set_sizes([unit * k for k in n])
        else:
            sc.set_offsets(np.empty((0, 2)))
            sc.set_sizes([])

    def draw(i: int):
        k = i + 1
        changed = [head]
        for name, games, img, here, topped, survived, tbars, rbars, title in panels:
            share = frames[name][i].copy()
            share[share == 0] = np.nan
            img.set_data(share)
            for b, h in zip(tbars, np.nansum(share, axis=0)):
                b.set_height(h)
            for b, w in zip(rbars, np.nansum(share, axis=1)):
                b.set_width(w)
            alive = [g for g in games if k in g["states"]]
            points(here, Counter(g["states"][k] for g in alive), 10)
            points(topped, Counter(g["last"] for g in games if not g["survived"] and g["end"] <= k), 28)
            points(survived, Counter(g["last"] for g in games if g["survived"] and g["end"] <= k), 28)
            ended = sum(1 for g in games if not g["survived"] and g["end"] < k)
            title.set_text(f"{name}  piece {k}: {len(alive)} alive, {ended} topped out")
            changed += [img, here, topped, survived, title, *tbars, *rbars]
        head.set_text(f"piece {k} / {pieces}: spawn boards of pieces 1-{k} as a share of each arm's final total, "
                      f"log colour on one range for all panels and frames\ndots: games alive at this piece; "
                      f"markers: games' last clean boards once ended, circles topped out, squares survived")
        return changed

    anim = FuncAnimation(fig, draw, frames=pieces, blit=False)
    writer = PillowWriter(fps=fps) if out.suffix == ".gif" else FFMpegWriter(fps=fps, codec="libx264",
                                                                              extra_args=["-pix_fmt", "yuv420p"])
    anim.save(out, writer=writer, dpi=100, savefig_kwargs={"facecolor": SURFACE})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=PATH[,PATH...]")
    ap.add_argument("-o", "--out", type=Path, required=True, help="animation to write: .mp4 (ffmpeg) or .gif")
    ap.add_argument("--pieces", type=int, default=100, help="frames: one per piece index, 1..N (default 100)")
    ap.add_argument("--fps", type=int, default=5)
    a = ap.parse_args()
    arms = {}
    for name, paths in journals.arms(a.arm).items():
        games = [g for g in map(game, paths) if g is not None]
        if len(games) < len(paths):
            print(f"{name:<18} {len(paths) - len(games)} unfinished, skipped")
        if games:
            arms[name] = games
            print(f"{name:<18} {len(games)} games, {sum(g['survived'] for g in games)} survived, "
                  f"{sum(len(g['states']) for g in games)} clean spawn boards")
    if not arms:
        raise SystemExit("topout-triangle-anim: no finished game in any arm: nothing to animate")
    animate(arms, a.out, a.pieces, a.fps)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
