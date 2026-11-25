"""topout-plot: the Explorer's figures as still images or pages, and their numbers as tables.

    python topout-plot.py burden | games | decisions | triangle | moves
        [--dataset DIR] [--arm NAME=PATH[,PATH...] ...] [--only NAME ...] [-o FILE] [--table]

Contract. Reads finished journals (games still running are skipped and counted) through topout/data.py and its cache.
Writes the file named by -o and nothing else; --table prints. At least one of the two. The figures are topout-explore's
own (topout/figures.py).

-o ends in .png, .svg, .pdf or .webp (drawn by a Chrome or Chromium: one on PATH, or named by BROWSER_PATH, or fetched
once with `plotly_get_chrome`) or .html (no browser, and keeps the hover). --only keeps the arms with that name or whose
name starts with it: "Claude Haiku 5.5" is every effort of it.

  burden     burden against tokens per piece; --show full ladder, best performance or cost frontier; --tokens mean or median
  games      every game's burden, by arm
  decisions  planning and execution fingerprints
  triangle   spawn boards by live and dead rows, each game's last board ringed (squares reached the end)
  moves      each placement's lines cleared and sealed cells added
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from topout import decisions, figures
from topout.data import Data, add_arguments, from_arguments

STILL = (".png", ".svg", ".pdf", ".webp")


def choose(data: Data, only: list[str]) -> list[str]:
    for o in only:
        if not any(n == o or n.startswith(o + " ") for n in data.arms):
            raise SystemExit(f"topout-plot: --only {o!r} matches no arm; the arms are: {', '.join(data.arms)}")
    return [n for n in data.arms if not only or any(n == o or n.startswith(o + " ") for o in only)]


def figure(kind: str, data: Data, names: list[str], show: str, tokens: str, columns: int):
    title, note = figures.NOTES[kind]
    games = [data.arms[n]["games"] for n in names]
    if kind == "burden":
        return figures.captioned(figures.burden_figure(data, names, tokens, show), title, f"{note} {figures.NOTES[show]}", 1250)
    if kind == "games":
        return figures.captioned(figures.games_figure(data, names), title, note, 1250)
    if kind == "decisions":
        return figures.captioned(figures.fingerprints(data, names), title, f"{note} {figures.verb_key()}", 1250)
    if kind == "triangle":
        scale = figures.triangle_scale(games)
        panels = [figures.triangle_figure(g, scale) for g in games]
        note += " Rings: each game's last board, squares reached the end, circles topped out. One colour range for all panels."
    else:
        scale = figures.moves_scale(games)
        panels = [figures.moves_figure(g, scale) for g in games]
        note += " One scale for all panels."
    fig = figures.beside(panels, names, columns)
    return figures.captioned(fig, title, note, max(640, 430 * min(columns, len(names))))


def table(kind: str, data: Data, names: list[str]) -> None:
    w = max(len(n) for n in names)
    if kind == "burden":
        print(f"{'arm':{w}s}  games  burden  95% of mean     tokens/piece mean   median  reached end")
        for n in sorted(names, key=lambda n: data.arms[n]["burden"]):
            a = data.arms[n]
            print(f"{n:{w}s}  {len(a['games']):5d}  {a['burden']:6.2f}  [{a['ci'][0]:5.2f}, {a['ci'][1]:5.2f}]  {a['mean']:17,.0f}  {a['median']:7,.0f}  {a['reached']:11d}")
    elif kind == "games":
        print(f"{'arm':{w}s}  game           burden  pieces  lines  ended        sealed cells  truncated  nudged")
        for n in names:
            for g in data.arms[n]["games"]:
                ps = g["pieces"]
                print(f"{n:{w}s}  {g['key']:13s}  {g['burden']:6.2f}  {len(ps):6d}  {g['lines']:5d}  {'reached end' if g['survived'] else 'topped out ':11s}  "
                      f"{ps[-1]['sealed']:12d}  {sum(p['truncated'] for p in ps):9d}  {sum(p['nudged'] for p in ps):6d}")
    elif kind == "decisions":
        print(f"{'arm':{w}s}  phase      replies      tokens p10/p50/p90/max  truncated  nudged")
        for n in names:
            for phase, d in zip(figures.PHASE, data.arms[n]["both"]):
                ts = sorted(d["tokens"])
                q = "/".join(str(ts[min(len(ts) - 1, int(p * len(ts)))]) for p in (.1, .5, .9, 1)) if ts else "-"
                r = decisions.rates(d)
                print(f"{n:{w}s}  {phase:9s}  {d['n']:7d}  {q:>26s}  {r['truncated']:9.1%}  {r['nudged']:6.1%}")
    elif kind == "triangle":
        print(f"{'arm':{w}s}  boards  dead rows p10/p50/p90  live rows p10/p50/p90")
        for n in names:
            ps = [p for g in data.arms[n]["games"] for p in g["pieces"]]
            q = lambda k: "/".join(str(int(v)) for v in np.percentile([p[k] for p in ps], (10, 50, 90), method="nearest"))
            print(f"{n:{w}s}  {len(ps):6d}  {q('dead'):>21s}  {q('live'):>21s}")
    else:
        print(f"{'arm':{w}s}  lines cleared  placements   share  sealed cells, mean")
        for n in names:
            moves = figures.moves_of(data.arms[n]["games"])
            for x in range(5):
                s = [b for a, b in moves if a == x]
                if s:
                    print(f"{n:{w}s}  {x:13d}  {len(s):10d}  {len(s) / len(moves):6.1%}  {np.mean(s):+18.2f}")


def write(fig, out: Path) -> None:
    if out.suffix == ".html":
        fig.write_html(out, include_plotlyjs=True)
        return
    try:
        fig.write_image(out, scale=2)
    except Exception as e:
        if "chrome" not in f"{type(e).__name__} {e}".lower():
            raise
        raise SystemExit(f"topout-plot: {out.suffix} is drawn by Chrome, and none was found: run `uv run plotly_get_chrome` once, or name "
                         "one in BROWSER_PATH, or write a .html") from None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("kind", choices=("burden", "games", "decisions", "triangle", "moves"))
    add_arguments(ap)
    ap.add_argument("--only", action="append", default=[], metavar="NAME", help="an arm, or the start of several arms' names")
    ap.add_argument("-o", "--out", type=Path, help=f"{', '.join(STILL)} or .html")
    ap.add_argument("--table", action="store_true", help="print the numbers")
    ap.add_argument("--show", choices=tuple(figures.SHOW), default="ladders", help="burden: which arms")
    ap.add_argument("--tokens", choices=("mean", "median"), default="mean", help="burden: the mean includes the tail, what it costs")
    ap.add_argument("--columns", type=int, default=3, help="triangle and moves: panels to a row")
    a = ap.parse_args()
    if not a.out and not a.table:
        ap.error("give -o, --table or both")
    if a.out and a.out.suffix not in (*STILL, ".html"):
        ap.error(f"-o {a.out}: not {', '.join(STILL)} or .html")
    if a.columns < 1:
        ap.error("--columns must be >= 1")
    data = from_arguments(ap, a)
    names = choose(data, a.only)
    if data.skipped:
        print(f"{data.skipped} games still running, skipped")
    if a.table:
        table(a.kind, data, names)
    if a.out:
        write(figure(a.kind, data, names, a.show, a.tokens, a.columns), a.out)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
