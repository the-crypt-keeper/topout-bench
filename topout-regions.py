"""topout-regions: the board's filled / open / sealed regions, per piece, from topout journals.

    python topout-regions.py <journal | game dir | dir of games> [...] -o regions.png [--csv out.csv]

Each journal carries the board the model was shown on every tick ('#' settled, '@' falling, '.' empty).
The board at a piece's first tick is the settled field that piece spawned into: the previous lock and its
clears, played out. Each such board is split:

  filled      '#'
  sealed      empty, and not reached by a flood from the top that moves down, left and right, never up.
              Fills come from above or the side, never from below, so nothing can ever fill it until
              whatever covers it has been cleared.
  open        the rest of the empty cells.

The flood assumes a 1x1 piece exists: a one-wide pocket under an overhang counts as open though no
tetromino fits it. Piece-aware reachability is the refinement, not built.

Rows: `dead_rows` holds a sealed cell; it cannot complete until dug out. `live_rows` is height minus dead
rows. `height` is the stack height in rows.

`covered` is the classic hole count (an empty cell with a filled cell anywhere above it in its column;
Dellacherie / BCTS). Every sealed cell is covered: a cell open to the sky is reached straight down.
`tuckable` = covered - sealed, the covered cells a slide can still reach (Cold Clear's overhang, in this
game's physics).

The ledger. Each piece adds 4 cells and each line removes 10, so before piece p spawns, with L lines
cleared:

    height = 0.4 (p - 1) - L + open_slack + sealed_slack

`packed` = 0.4 (p - 1) - L is the height a perfectly packed stack would have; `open_slack` is the open
cells below the stack top / 10, `sealed_slack` the sealed cells / 10. It is exact by construction once
filled = 4 (p - 1) - 10 L, which holds on every board except where the falling piece is drawn over settled
cells, at top-out: `occluded` marks those. Each piece's change in height is then 0.4 - (lines it cleared)
+ d_open_slack + d_sealed_slack: packing, clearing, loose slack, dead slack. Survival needs 0.4 lines a
piece; below it the packed term climbs.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path

from topout import journals

ROWS, COLS = 20, 10


def flood(filled) -> set[tuple[int, int]]:
    """Empty cells reached from the top row moving down, left, right: never up."""
    q = deque((0, c) for c in range(COLS) if not filled[0][c])
    seen = set(q)
    while q:
        r, c = q.popleft()
        for s in ((r + 1, c), (r, c - 1), (r, c + 1)):
            rr, cc = s
            if 0 <= rr < ROWS and 0 <= cc < COLS and not filled[rr][cc] and s not in seen:
                seen.add(s)
                q.append(s)
    return seen


def regions(board: str) -> dict:
    filled = [[ch == "#" for ch in row] for row in board.split("\n")]
    assert len(filled) == ROWS and all(len(r) == COLS for r in filled), board
    reach = flood(filled)
    n = dict(filled=0, open=0, sealed=0, covered=0)
    dead = set()
    for r in range(ROWS):
        for c in range(COLS):
            if filled[r][c]:
                n["filled"] += 1
                continue
            if any(filled[k][c] for k in range(r)):
                n["covered"] += 1
            if (r, c) in reach:
                n["open"] += 1
            else:
                n["sealed"] += 1
                dead.add(r)
    top = next((r for r in range(ROWS) if any(filled[r])), ROWS)
    h = ROWS - top
    return n | {"tuckable": n["covered"] - n["sealed"], "dead_rows": len(dead), "height": h,
                "live_rows": h - len(dead), "open_slack": (10 * h - n["filled"] - n["sealed"]) / 10,
                "sealed_slack": n["sealed"] / 10}


def game_rows(path: Path) -> tuple[str, list[dict], dict]:
    recs = [json.loads(line) for line in path.open()]
    summary = next(r for r in recs if r.get("kind") == "summary")
    model = (recs[0].get("model") if recs and recs[0].get("kind") == "run" else None) or "?"
    game = journals.game_id(path)
    label = f"{game} {summary['lines']}L@{summary['pieces']}"
    rows, seen = [], set()
    for t in (r for r in recs if r.get("kind") == "tick"):
        if t["pieces"] in seen:
            continue
        seen.add(t["pieces"])
        p, lines = t["pieces"], t["lines"]
        row = {"game": game, "model": model, "piece": p, "lines": lines} \
            | regions(t["board"]) | {"packed": 0.4 * (p - 1) - lines}
        row["occluded"] = row["filled"] != 4 * (p - 1) - 10 * lines
        prev = rows[-1] if rows else None
        for k in ("lines", "open_slack", "sealed_slack"):
            row[f"d_{k}"] = row[k] - prev[k] if prev else None
        rows.append(row)
    return label, rows, summary


def plot(games: list[tuple[str, list[dict], dict]], out: Path, stable_lines: int, rate: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    ink, muted, grid, surface = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    color = {"stable": "#2a78d6", "collapse": "#eb6834"}
    panels = [("sealed", "sealed cells"), ("live_rows", "live rows (height - dead rows)"),
              ("dead_rows", "dead rows (a sealed cell)"), ("height", "stack height (rows)")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(11, 3.1 * len(panels)), sharex=True, facecolor=surface)
    models = sorted({g[1][0]["model"] for g in games if g[1]})
    styles = ["-", (0, (4, 2)), (0, (1, 2)), (0, (6, 2, 1, 2))]
    dashes = {m: styles[i % len(styles)] for i, m in enumerate(models)}
    for ax, (key, title) in zip(axes, panels):
        ax.set_facecolor(surface)
        for label, rows, summary in games:
            if not rows:
                continue
            mode = "stable" if summary["lines"] >= stable_lines else "collapse"
            xs, ys = [r["piece"] for r in rows], [r[key] for r in rows]
            if rate:                                            # trailing: only what was known at that piece
                xs, ys = xs[rate:], [(ys[i] - ys[i - rate]) / rate for i in range(rate, len(ys))]
            ax.plot(xs, ys, color=color[mode], linewidth=2, linestyle=dashes[rows[0]["model"]], alpha=0.9,
                    solid_capstyle="round")
            short = f"{Path(rows[0]['game']).name} {summary['lines']}L@{summary['pieces']}"
            ax.annotate(short, (xs[-1], ys[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=7, color=muted, va="center")
        ax.set_title(f"{title}, change per piece over the last {rate}" if rate else title,
                     loc="left", fontsize=11, color=ink)
        if rate:
            ax.axhline(0, color=muted, linewidth=1)
        ax.grid(True, color=grid, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(grid)
        ax.tick_params(colors=muted, labelsize=9)
    axes[-1].set_xlabel("piece", color=muted)
    handles = [Line2D([], [], color=color[m], linewidth=2,
                      label=f"{m} (lines {'>=' if m == 'stable' else '<'} {stable_lines})") for m in ("stable", "collapse")]
    handles += [Line2D([], [], color=muted, linewidth=2, linestyle=dashes[m], label=m) for m in models]
    axes[0].legend(handles=handles, loc="upper left", fontsize=8, frameon=False, labelcolor=ink)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=surface)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+", help="journals, game directories or directories of games")
    ap.add_argument("-o", "--out", type=Path, required=True, help="PNG to write")
    ap.add_argument("--csv", type=Path, help="per-piece rows to write")
    ap.add_argument("--rate", type=int, default=0, metavar="K",
                    help="plot (v[p] - v[p-K]) / K, the trailing slope over K pieces, instead of v")
    ap.add_argument("--stable-lines", type=int, default=6, help="lines at the end that count a game as stable")
    a = ap.parse_args()
    games = [game_rows(p) for p in journals.find(a.paths)]
    for label, rows, summary in games:
        last = rows[-1]
        print(f"{label}  pieces {len(rows):3d}  last board: height {last['height']:2d} = packed "
              f"{last['packed']:4.1f} + open {last['open_slack']:3.1f} + sealed {last['sealed_slack']:3.1f}"
              f"{' (occluded)' if last['occluded'] else ''}   dead {last['dead_rows']:2d}  covered "
              f"{last['covered']:2d} tuckable {last['tuckable']:2d}  occluded boards "
              f"{sum(r['occluded'] for r in rows)}")
    if a.csv:
        with a.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(games[0][1][0]))
            w.writeheader()
            for _, rows, _ in games:
                w.writerows(rows)
    plot(games, a.out, a.stable_lines, a.rate)
    print(f"wrote {a.out}" + (f" and {a.csv}" if a.csv else ""))


if __name__ == "__main__":
    main()
