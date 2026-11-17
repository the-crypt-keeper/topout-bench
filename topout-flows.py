"""topout-flows: each game through the triangle's transitions, per arm (a prototype).

    python topout-flows.py --arm NAME=PATH[,PATH...] [--arm ...] -o flows.png [--mode paths|fan|triangle]

Contract. Reads journals (tick records only, through topout/ledger.py: running games are skipped, as topout-outcomes
skips them). Writes the PNG named by -o; prints one line per game with its final totals. The transitions are defined in
ledger.flows, the states in ledger.regions.

Every row of the 20 is empty, live or dead (threads/topout, the triangle model). The state over piece index says little
about play; these views take piece index off the axes, so games of any length share a panel without being pooled.

  paths      one path per game. headroom: x = rows opened (empty -> live), y = rows cleared (live -> empty); the
             diagonal is every opened row cleared, the dashed line x - y = 20 the death line (no empty row; a game can
             top out a row or two before it, at a spawn). debt: x = sealings (live -> dead), y = unsealings (dead ->
             live), both lower bounds; the diagonal is no net debt.
  fan        the same two, rotated and as density. headroom: x = rows opened (the game's own row clock), y = rows
             occupied (= opened - cleared = 20 - empty); balance is a band of constant height, the death line the top.
             debt: x = sealings, y = dead rows held. A cell's colour is the share of the games still going at that x
             that pass through it; the strip under each fan is how many games are still going, which is how far each
             column can be trusted.
  triangle   no clock at all: x = live rows, y = dead rows, at every spawn board a game reached (not the occluded one
             at top-out). A cell's colour is its share of the arm's pieces: where the arm's play lives. The ceiling
             live + dead = 20 is drawn.

One row per arm, axes shared across arms. paths colours by outcome, read from the journal's summary as topout-outcomes
reads it; the density views judge nothing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from topout import journals, ledger

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SURVIVED, TOPPED = "#2a78d6", "#eb6834"
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


def cmap():
    from matplotlib.colors import ListedColormap
    m = ListedColormap(RAMP)
    m.set_bad(SURFACE)
    return m


def plot_paths(arms, out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    games = [g["flows"] for paths in arms.values() for g in paths if g["flows"]]
    hx = max(f[-1]["opened"] for f in games) + 2
    dx = max(max(f[-1]["sealings"], f[-1]["unsealings"]) for f in games) + 1
    fig, axes = plt.subplots(len(arms), 2, figsize=(11, 3.4 * len(arms)), squeeze=False, facecolor=SURFACE)
    for (name, paths), (hax, dax) in zip(arms.items(), axes):
        for ax, (xk, yk), lim in ((hax, ("opened", "cleared"), hx), (dax, ("sealings", "unsealings"), dx)):
            style(ax)
            ax.plot([0, lim], [0, lim], color=MUTED, linewidth=1)
            if ax is hax:
                ax.plot([20, lim], [0, lim - 20], color=MUTED, linewidth=1, linestyle="--")
            for g in paths:
                f = g["flows"]
                if not f:
                    continue
                xs, ys = [0] + [r[xk] for r in f], [0] + [r[yk] for r in f]
                col = SURVIVED if g["survived"] else TOPPED
                ax.plot(xs, ys, color=col, linewidth=1.4, alpha=0.75, drawstyle="steps-post")
                ax.plot(xs[-1], ys[-1], "o", color=col, markersize=3)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_aspect("equal")
        n, alive = len(paths), sum(g["survived"] for g in paths)
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


def fan_grid(paths, xk: str, yk: str, width: int):
    """share[y][x] = games passing through (x, y) / games still going at x; going[x]. A placement that opens several
    rows at once moves x by more than one: the columns it skips get the cells between, one row per row opened (the
    rows stack), so a column is never empty for a game that went through it."""
    import numpy as np
    passed = np.zeros((ROWS + 1, width))
    going = np.zeros(width)
    for g in paths:
        f = g["flows"]
        if not f:
            continue
        cells, px, py = {(0, 0)}, 0, 0
        for r in f:
            x, y = r[xk], min(r[yk], ROWS)
            for k in range(px + 1, x):                       # skipped columns, filled along the step
                cells.add((k, min(py + (k - px) * (y - py) // max(x - px, 1), ROWS)))
            cells.add((x, y))
            px, py = x, y
        for x, y in cells:
            passed[y][x] += 1
        going[: f[-1][xk] + 1] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        share = passed / going
    share[passed == 0] = np.nan
    return share, going


def plot_fan(arms, out: Path) -> None:
    import matplotlib.pyplot as plt

    games = [g["flows"] for paths in arms.values() for g in paths if g["flows"]]
    widths = {"opened": max(f[-1]["opened"] for f in games) + 1, "sealings": max(f[-1]["sealings"] for f in games) + 1}
    fig = plt.figure(figsize=(12, 3.9 * len(arms)), facecolor=SURFACE)
    outer = fig.add_gridspec(len(arms), 2, hspace=0.55, wspace=0.18)
    m, img = cmap(), None
    for i, (name, paths) in enumerate(arms.items()):
        n, alive = len(paths), sum(g["survived"] for g in paths)
        for j, (xk, yk, title, xl, yl) in enumerate((
                ("opened", "occupied", f"{name}  headroom fan  ({alive} / {n} survived)",
                 "rows opened (empty -> live)", "rows occupied (20 - empty)"),
                ("sealings", "dead", f"{name}  debt fan", "sealings (live -> dead, lower bound)", "dead rows held"))):
            inner = outer[i, j].subgridspec(2, 1, height_ratios=[4, 1], hspace=0.08)
            ax, strip = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
            share, going = fan_grid(paths, xk, yk, widths[xk])
            img = ax.imshow(share, origin="lower", aspect="auto", cmap=m, vmin=0, vmax=1, interpolation="nearest",
                            extent=(-0.5, widths[xk] - 0.5, -0.5, ROWS + 0.5))
            style(ax)
            ax.grid(False)
            if xk == "opened":
                ax.axhline(ROWS, color=MUTED, linewidth=1, linestyle="--")
            ax.set_title(title, loc="left", fontsize=10, color=INK)
            ax.set_ylabel(yl, color=MUTED, fontsize=8)
            ax.tick_params(labelbottom=False)
            strip.bar(range(widths[xk]), going, width=1.0, color=MUTED, linewidth=0)
            style(strip)
            strip.set_xlim(-0.5, widths[xk] - 0.5)
            strip.set_ylim(0, n)
            strip.set_ylabel("games\ngoing", color=MUTED, fontsize=7)
            strip.set_xlabel(xl, color=MUTED, fontsize=8)
    cb = fig.colorbar(img, ax=fig.axes, fraction=0.015, pad=0.01)
    cb.set_label("share of the games still going at x that pass through the cell", color=MUTED, fontsize=8)
    cb.ax.tick_params(colors=MUTED, labelsize=7)
    fig.savefig(out, dpi=110, facecolor=SURFACE, bbox_inches="tight")


def plot_triangle(arms, out: Path, log: bool) -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LogNorm

    shares = {}
    for name, paths in arms.items():
        counts = np.zeros((ROWS + 1, ROWS + 1))
        for g in paths:
            for r in g["rows"]:
                if not r["occluded"]:
                    counts[r["dead_rows"]][r["live_rows"]] += 1
        total = counts.sum()
        share = counts / total if total else counts
        share[counts == 0] = np.nan
        shares[name] = (share, int(total))
    finite = np.concatenate([s[np.isfinite(s)] for s, _ in shares.values()])
    # log: one range for every panel (smallest nonzero share anywhere to the largest), so darkness compares across
    # arms; linear: each panel its own range, so compare where density sits, not how dark it is
    norm = LogNorm(vmin=finite.min(), vmax=finite.max()) if log and finite.size else None

    # marginals: live summed over dead (top), dead summed over live (right), as shares of the arm's pieces; one
    # range across panels so their heights compare
    margins = {name: (np.nansum(s, axis=0), np.nansum(s, axis=1)) for name, (s, _) in shares.items()}
    mtop = max(max(lv.max(), dd.max()) for lv, dd in margins.values()) * 1.05

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    cols = min(3, len(arms))
    nrows = -(-len(arms) // cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(4.6 * cols, 4.6 * nrows), squeeze=False, facecolor=SURFACE)
    m, img = cmap(), None
    for ax, (name, paths) in zip(axes.flat, arms.items()):
        share, total = shares[name]
        kw = {"norm": norm} if norm else {"vmin": 0, "vmax": np.nanmax(share) if total else 1}
        img = ax.imshow(share, origin="lower", cmap=m, interpolation="nearest",
                        extent=(-0.5, ROWS + 0.5, -0.5, ROWS + 0.5), **kw)
        div = make_axes_locatable(ax)
        top, right = div.append_axes("top", 0.6, pad=0.05, sharex=ax), div.append_axes("right", 0.6, pad=0.05, sharey=ax)
        live, dead = margins[name]
        top.bar(range(ROWS + 1), live, width=0.9, color=SURVIVED, linewidth=0)
        right.barh(range(ROWS + 1), dead, height=0.9, color=SURVIVED, linewidth=0)
        top.set_ylim(0, mtop)
        right.set_xlim(0, mtop)
        for mx in (top, right):
            style(mx)
            mx.tick_params(labelsize=6)
        top.tick_params(labelbottom=False)
        right.tick_params(labelleft=False)
        style(ax)
        ax.grid(False)
        ax.plot([0, ROWS], [ROWS, 0], color=MUTED, linewidth=1, linestyle="--")
        # each game's last state (its last clean spawn board: the occluded top-out board is not one), ringed by
        # outcome and sized by how many games ended in that cell. A topped-out game ends when the next piece cannot
        # spawn, which is often a tower under the spawn point well inside live + dead = 20.
        from collections import Counter
        for survived, marker, col in ((False, "o", TOPPED), (True, "s", INK)):
            ends = Counter((last["live_rows"], last["dead_rows"]) for g in paths if g["survived"] == survived
                           for last in [next((r for r in reversed(g["rows"]) if not r["occluded"]), None)] if last)
            if ends:
                xs, ys, ns = zip(*((x, y, k) for (x, y), k in ends.items()))
                ax.scatter(xs, ys, s=[28 * k for k in ns], marker=marker, facecolors="none", edgecolors=col,
                           linewidths=1.3, zorder=3)
        n, alive = len(paths), sum(g["survived"] for g in paths)
        top.set_title(f"{name}  ({alive} / {n} survived, {total} pieces)", loc="left", fontsize=9, color=INK)
        ax.set_xlabel("live rows", color=MUTED, fontsize=8)
        ax.set_ylabel("dead rows", color=MUTED, fontsize=8)
        if not norm:
            fig.colorbar(img, ax=right, fraction=0.3, pad=0.15).ax.tick_params(colors=MUTED, labelsize=7)
    for ax in list(axes.flat)[len(arms):]:
        ax.set_visible(False)
    scale = "log colour, one range for all panels" if norm else "each panel its own range"
    fig.suptitle(f"share of the arm's pieces spawning into each (live, dead) state ({scale}); dashed: live + dead = 20;"
                 f" bars: live rows (top) and dead rows (right) summed, one range for all panels;\nmarkers: each game's"
                 f" last clean board, circles topped out, squares survived, sized by games ending there",
                 color=MUTED, fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    if norm:
        cb = fig.colorbar(img, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)
        cb.ax.tick_params(colors=MUTED, labelsize=7)
    fig.savefig(out, dpi=110, facecolor=SURFACE, bbox_inches="tight")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=PATH[,PATH...]")
    ap.add_argument("-o", "--out", type=Path, required=True, help="PNG to write")
    ap.add_argument("--mode", choices=("paths", "fan", "triangle"), default="paths")
    ap.add_argument("--log", action="store_true", help="triangle: log colour, one range shared by every panel")
    a = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
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
            rows = ledger.piece_rows(p)
            f = ledger.flows(rows)
            survived = summary["termination"] == "max_pieces"
            arms[name].append({"rows": rows, "flows": f, "survived": survived})
            if f:
                last = f[-1]
                print(f"{name:<18} {journals.game_id(p)}  {'survived' if survived else 'topped  '}  opened "
                      f"{last['opened']:3d} cleared {last['cleared']:3d} occupied {last['occupied']:2d}  "
                      f"sealings>={last['sealings']:2d} unsealings>={last['unsealings']:2d} dead {last['dead']:2d}")
    if a.mode == "triangle":
        plot_triangle(arms, a.out, a.log)
    else:
        {"paths": plot_paths, "fan": plot_fan}[a.mode](arms, a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
