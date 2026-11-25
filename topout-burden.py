"""topout-burden: how much dead board each arm carries, against what it spends thinking, then every game's own burden.

    python topout-burden.py [--dataset DIR] [--arm NAME=PATH[,PATH...] ...] -o burden.png

Contract. Reads finished journals (games still running are skipped and counted). --dataset adds every
<model>/<sampler> folder of a results-dataset copy as an arm, labelled by its index.json (topout/index.py). Writes the PNG named by -o; prints one row per arm.

Burden (operator, 2026-11-25): a game's dead rows averaged over max_pieces turns, every turn after the game ended counting
as 20 (ledger.burden). The horizon is fixed, so a game that dies early does not look clean for lack of time. Every
arm's games must share one max_pieces.

  top     one dot per arm: burden (mean over its games, up is better) against completion tokens per piece (--tokens
          mean or median over the arm's pieces, each piece's whole conversation, log). A line joins each effort ladder
          of the dataset's index.json, in its order. Bars: 95% interval of the mean, resampling whole games.
  bottom  one column per arm, every game's burden: squares reached max_pieces, circles topped out; the dash is the
          mean. Arms of one model sit together, models ordered by their best arm. This is the comparison; overlap of
          bars says nothing, and bars apart rule equality out.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from topout import index, journals, ledger
from topout.calls import is_reply

INK, MUTED, GRID, SURFACE, BAR, TOPPED = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6", "#eb6834"


def game(path: Path) -> dict | None:
    summary = journals.summary(path)
    if summary is None:
        return None
    run = journals.run_record(path)
    tokens: Counter = Counter()
    for r in journals.records(path):
        if is_reply(r):
            tokens[r["pieces"]] += int((r["usage"] or {}).get("completion_tokens") or 0)
    return {"burden": ledger.burden(ledger.dead_rows(path), run["max_pieces"]), "turns": run["max_pieces"],
            "survived": summary["termination"] == "max_pieces", "tokens": list(tokens.values())}


def interval(xs: list[float], n: int = 10000) -> tuple[float, float]:
    """95% interval of the mean, resampling games (fixed seed)."""
    means = np.random.default_rng(1).choice(xs, (n, len(xs))).mean(axis=1)
    return tuple(float(v) for v in np.percentile(means, (2.5, 97.5)))


def plot(arms: dict[str, dict], out: Path, stat: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax, bx) = plt.subplots(2, 1, figsize=(max(9, 0.55 * len(arms) + 2), 12), facecolor=SURFACE)
    by_model, ladders = defaultdict(list), defaultdict(list)
    for name, a in arms.items():
        by_model[a["arm"].model].append(name)                # in index order, so ladders and their steps in file order
        if a["arm"].ladder is not None:
            ladders[a["arm"].ladder].append(name)
    for names in ladders.values():
        ax.plot([arms[n]["tokens"] for n in names], [arms[n]["mean"] for n in names], color=BAR, alpha=0.4, lw=1.6)
    for name, a in arms.items():
        ax.errorbar(a["tokens"], a["mean"], yerr=[[a["mean"] - a["ci"][0]], [a["ci"][1] - a["mean"]]], fmt="o",
                    color=BAR, ms=6, capsize=3, lw=1.2)
        ax.annotate(name, (a["tokens"], a["mean"]), xytext=(6, -9), textcoords="offset points", fontsize=7, color=INK)
    ax.set_xscale("log")
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel(f"completion tokens per piece, {stat}", color=MUTED)
    ax.set_title("burden against thinking (bars: 95% of the mean, resampling games)", loc="left", fontsize=10, color=INK)

    order = [n for names in sorted(by_model.values(), key=lambda ns: min(arms[n]["mean"] for n in ns)) for n in names]
    rng = np.random.default_rng(0)
    for x, n in enumerate(order):
        a = arms[n]
        for survived, marker, color in ((True, "s", BAR), (False, "o", TOPPED)):
            ys = [g["burden"] for g in a["games"] if g["survived"] == survived]
            bx.scatter(x + rng.uniform(-.22, .22, len(ys)), ys, marker=marker, s=24, facecolors="none", edgecolors=color)
        bx.hlines(a["mean"], x - .32, x + .32, color=INK, lw=2)
        if x and arms[order[x - 1]]["arm"].model != a["arm"].model:
            bx.axvline(x - .5, color=MUTED, lw=0.8)
    bx.set_xticks(range(len(order)), order, rotation=55, ha="right", fontsize=8)
    bx.set_title("every game (squares reached the end, circles topped out; dash: mean)", loc="left", fontsize=10, color=INK)
    for q in (ax, bx):
        q.set_facecolor(SURFACE)
        q.grid(True, color=GRID, lw=0.8)
        q.set_axisbelow(True)
        q.set_ylabel("burden (up is better)", color=MUTED)
        q.set_ylim(20, -0.5)
        q.tick_params(colors=MUTED, labelsize=8)
        for s in q.spines.values():
            s.set_color(GRID)
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor=SURFACE)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", type=Path, help="local copy of a results dataset: every <model>/<sampler> is an arm")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=PATH[,PATH...]")
    ap.add_argument("-o", "--out", type=Path, required=True, help="PNG to write")
    ap.add_argument("--tokens", choices=("mean", "median"), default="mean",
                    help="an arm's tokens per piece; the mean includes the tail (what it costs), the median does not")
    a = ap.parse_args()
    specs = index.arms(a.dataset, a.arm)
    if not specs:
        ap.error("give --dataset, --arm or both")
    arms, turns = {}, set()
    for arm in specs:
        name, paths = arm.name, arm.paths
        games = [g for g in map(game, paths) if g]
        if len(games) < len(paths):
            print(f"{name}: {len(paths) - len(games)} unfinished, skipped")
        if not games:
            continue
        turns |= {g["turns"] for g in games}
        xs = [g["burden"] for g in games]
        arms[name] = {"games": games, "mean": float(np.mean(xs)), "ci": interval(xs), "arm": arm,
                      "tokens": float(getattr(np, a.tokens)([t for g in games for t in g["tokens"]]))}
    if not arms:
        raise SystemExit("topout-burden: no finished game in any arm")
    if len(turns) > 1:
        raise SystemExit(f"topout-burden: arms mix max_pieces {sorted(turns)}; burden needs one horizon")
    print(f"{'arm':32s} {'games':>5s} {'burden':>6s}  {'95% of mean':15s} {'tokens/piece':>12s} {'reached end':>11s}")
    for name, r in sorted(arms.items(), key=lambda kv: kv[1]["mean"]):
        print(f"{name:32s} {len(r['games']):5d} {r['mean']:6.2f}  [{r['ci'][0]:5.2f}, {r['ci'][1]:5.2f}]  "
              f"{r['tokens']:12,.0f} {sum(g['survived'] for g in r['games']):11d}")
    plot(arms, a.out, a.tokens)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
