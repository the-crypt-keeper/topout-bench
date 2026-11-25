"""topout-decisions: each arm's way of deciding, as a fingerprint: planning up, execution down.

    python topout-decisions.py [--dataset DIR] [--arm NAME=PATH[,PATH...] ...] -o decisions.png

Contract. Reads journals (running games up to their last record); --dataset adds every <model>/<sampler> folder of a
results-dataset copy as an arm. Writes the PNG named by -o; prints one row per arm and phase.

A reply is planning when it opens a conversation (conv_turn 0: a piece's first reply, a retry of it, or the first after
a restart) and execution otherwise. The two are different populations (threads/topout: the plan is in turn 0) and are
never pooled: each share is of its own phase (replies per piece excepted). Every attempt the engine returned is a reply, so a truncated one and
its retry both count.

One row per arm:
  tokens            mirrored, planning above the line (blue), execution below (orange): completion tokens, log bins 16
                    to 65k, each half scaled to its own tallest bar (shape only); black: the truncated part
  calls per reply   mirrored the same way: 0, 1 .. 6 or more tool calls, over replies that were not truncated; 0 is a
                    nudge. Each half has one scale for all arms
  buttons           two stacked strips, planning over execution: the share of the phase's tool calls by verb
  replies per piece every reply a piece took, both phases together (a property of the piece, not of a phase)
  truncated, nudged each phase's share of replies cut off, and that made no tool call (wait was pressed instead); one
                    scale per column
Arms of one model sit together, in the order of the dataset's index.json, models by median planning tokens.
"""
from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

from topout import index, journals
from topout.calls import is_reply
from topout.prompt import BUTTONS

INK, MUTED, GRID, SURFACE, PLAN, EXEC = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6", "#eb6834"
VERBS = list(BUTTONS)
VERB_COLORS = ["#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]    # none of the phase colours
PIECE = "#8a8984"                      # replies per piece: both phases together, so neither's colour
BINS, LOW, HIGH = 24, 4, 16            # token bins, log2 range
CALLS, REPLIES = 6, 16                 # the last bar of each is "or more"


def token_bin(t: int) -> int:
    return min(BINS - 1, max(0, int((math.log2(max(t, 1)) - LOW) / (HIGH - LOW) * BINS)))


def phases(paths: list[Path]) -> tuple[list[dict], Counter, set[int]]:
    """Counts for planning (0) and execution (1) over every reply of the arm, replies per piece (both phases), and the
    max_tokens its run records hold."""
    out = [dict(n=0, tokens=[], bins=Counter(), cut=Counter(), calls=Counter(), verbs=Counter(), truncated=0) for _ in range(2)]
    caps, per_piece = set(), Counter()
    for path in paths:
        cap = (journals.run_record(path).get("sampler") or {}).get("max_tokens")
        caps |= {cap} if cap else set()
        pieces: Counter = Counter()
        for r in journals.records(path):
            if not is_reply(r):
                continue
            ph = 0 if r["conv_turn"] == 0 else 1
            d, t = out[ph], int((r["usage"] or {}).get("completion_tokens") or 0)
            d["n"] += 1
            d["tokens"].append(t)
            d["bins"][token_bin(t)] += 1
            pieces[r["pieces"]] += 1
            if r["kind"] == "capped":
                d["truncated"] += 1
                d["cut"][token_bin(t)] += 1
                continue
            calls = r.get("tool_calls") or []
            d["calls"][min(len(calls), CALLS)] += 1
            for c in calls:
                d["verbs"][(c.get("function") or {}).get("name")] += 1
        for n in pieces.values():
            per_piece[min(n, REPLIES)] += 1
    return out, per_piece, caps


def share(counter: Counter, keys) -> list[float]:
    total = sum(counter.values()) or 1
    return [counter[k] / total for k in keys]


COLUMNS = [("tokens", range(BINS), "completion tokens"), ("calls", range(CALLS + 1), "calls per reply"),
           ("verbs", VERBS, "buttons (planning over execution)"), ("per_piece", range(1, REPLIES + 1), "replies per piece"),
           ("truncated", (), "truncated"), ("nudged", (), "nudged")]


def rates(d: dict) -> dict[str, float]:
    """Shares of the phase's replies: cut off by max_tokens or the context, and nudged (no tool call)."""
    return {"truncated": d["truncated"] / (d["n"] or 1), "nudged": d["calls"][0] / (d["n"] or 1)}


def plot(arms: dict[str, list[dict]], per_piece: dict[str, Counter], models: dict[str, str], caps: set[int], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def median(n, ph=0):
        ts = sorted(arms[n][ph]["tokens"])
        return ts[len(ts) // 2] if ts else 0
    first = {}
    for n in arms:
        first[models[n]] = min(first.get(models[n], math.inf), median(n))
    order = sorted(arms, key=lambda n: (first[models[n]], models[n]))      # stable: ladders and steps in index order

    src = {"tokens": "bins"}
    sh = {(n, ph, c): share(arms[n][ph][src.get(c, c)], keys) for n in order for ph in (0, 1) for c, keys, _ in COLUMNS[:3]}
    whole = {n: share(per_piece[n], COLUMNS[3][1]) for n in order}
    top = {(ph, c): max(max(sh[n, ph, c]) for n in order) for ph in (0, 1) for c, _, _ in COLUMNS[:3]}
    bad = {c: max(rates(arms[n][ph])[c] for n in order for ph in (0, 1)) or 1 for c in ("truncated", "nudged")}
    whole_top = max(max(v) for v in whole.values())
    fig, axes = plt.subplots(len(order), 6, figsize=(18, 0.8 * len(order) + 1.7), squeeze=False, facecolor=SURFACE,
                             gridspec_kw=dict(width_ratios=(2.4, 1, 1.6, 1.4, 0.75, 0.75), hspace=0.04, wspace=0.06, top=1 - 1.0 / (0.8 * len(order) + 1.7), bottom=0.6 / (0.8 * len(order) + 1.7)))
    for row, n in enumerate(order):
        new_model = row == 0 or models[order[row - 1]] != models[n]
        for col, (c, keys, title) in enumerate(COLUMNS):
            ax = axes[row][col]
            ax.set_facecolor(SURFACE)
            xs = range(len(list(keys)))
            if c == "per_piece":
                ax.bar(xs, [v / whole_top for v in whole[n]], width=0.88, color=PIECE, linewidth=0)
                ax.axhline(0, color=MUTED, lw=0.8)
                ax.set_ylim(-0.04, 1.1)
                ax.set_xlim(-0.7, len(list(keys)) - 0.3)
            elif c in bad:
                for ph, y in ((0, 0.5), (1, -0.5)):
                    v = rates(arms[n][ph])[c]
                    ax.barh(y, v / bad[c], height=0.6, color=(PLAN, EXEC)[ph], linewidth=0)
                    ax.text(v / bad[c] + 0.04, y, f"{v:.1%}" if v else "0", va="center", fontsize=8,
                            color=INK if v >= 0.0005 else MUTED)
                ax.set_ylim(-1.3, 1.3)
                ax.set_xlim(0, 1.75)
            elif c == "verbs":
                for ph, y in ((0, 0.5), (1, -0.5)):
                    left = 0.0
                    for v, color in zip(sh[n, ph, c], VERB_COLORS):
                        ax.barh(y, v, left=left, height=0.8, color=color, edgecolor=SURFACE, linewidth=0.8)
                        left += v
                    ax.plot([-0.012, -0.012], [y - 0.4, y + 0.4], color=(PLAN, EXEC)[ph], lw=3, solid_capstyle="butt")
                ax.set_ylim(-1.3, 1.3)
                ax.set_xlim(-0.03, 1.0)
            else:
                for ph, sign, color in ((0, 1, PLAN), (1, -1, EXEC)):
                    peak = (max(sh[n, ph, c]) if c == "tokens" else top[ph, c]) or 1     # tokens: each row's own shape
                    ax.bar(xs, [sign * v / peak for v in sh[n, ph, c]], width=0.88, color=color, linewidth=0)
                    if c == "tokens":
                        ax.bar(xs, [sign * arms[n][ph]["cut"][k] / (arms[n][ph]["n"] or 1) / peak for k in keys],
                               width=0.88, color=INK, linewidth=0)
                ax.axhline(0, color=MUTED, lw=0.8)
                ax.set_ylim(-1.1, 1.1)
                ax.set_xlim(-0.7, len(list(keys)) - 0.3)
            ax.set_yticks([])
            ax.set_xticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if new_model:
                ax.spines["top"].set_visible(True)
                ax.spines["top"].set_color(MUTED)
            if row == 0:
                ax.set_title(title, loc="left", fontsize=10, color=INK, pad=10)
            if row == len(order) - 1:
                if c == "tokens":
                    marks = sorted({16, 256, 1024, 4096, 16384} | {c for c in caps if c > 16384})
                    ax.set_xticks([(math.log2(m) - LOW) / (HIGH - LOW) * BINS - 0.5 for m in marks],
                                  [str(m) if m < 1000 else f"{m // 1000}k" for m in marks], fontsize=8)
                elif c in ("calls", "per_piece"):
                    ks = list(keys)
                    shown = [k for k in ks if k in (0, 1, ks[-1]) or (k % 3 == 0 and k < ks[-1] - 1)]
                    ax.set_xticks([ks.index(k) for k in shown], [f"{k}+" if k == ks[-1] else str(k) for k in shown],
                                  fontsize=8)
                ax.tick_params(colors=MUTED, length=2)
        axes[row][0].text(-0.02, 0.5, n, transform=axes[row][0].transAxes, ha="right", va="center", fontsize=9, color=INK)
    fig.suptitle("up, blue: planning replies   down, orange: execution replies   black: truncated   each half is a share "
                 "of its own phase\ntokens: each half scaled to its own tallest bar; calls: each half on one scale for all arms", x=0.5, y=1 - 0.2 / (0.8 * len(order) + 1.7), fontsize=10, color=MUTED)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c, label=v) for v, c in zip(VERBS, VERB_COLORS)], loc="lower center", ncol=6,
               fontsize=9, frameon=False, bbox_to_anchor=(0.6, 0.0), labelcolor=MUTED)
    fig.savefig(out, dpi=100, facecolor=SURFACE, bbox_inches="tight")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", type=Path, help="local copy of a results dataset: every <model>/<sampler> is an arm")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=PATH[,PATH...]")
    ap.add_argument("-o", "--out", type=Path, required=True, help="PNG to write")
    a = ap.parse_args()
    specs = index.arms(a.dataset, a.arm)
    if not specs:
        ap.error("give --dataset, --arm or both")
    arms, pieces, models, all_caps = {}, {}, {}, set()
    print(f"{'arm':28s} {'phase':9s} {'replies':>7s} {'tokens p10/p50/p90':>22s} {'truncated':>9s} {'nudged':>6s}")
    for arm in specs:
        name, paths = arm.name, arm.paths
        both, whole, caps = phases(paths)
        if not any(d["n"] for d in both):
            print(f"{name:28s} no reply yet")
            continue
        arms[name], pieces[name], models[name] = both, whole, arm.model
        all_caps |= caps
        for ph, d in zip(("planning", "execution"), both):
            ts = sorted(d["tokens"])
            q = "/".join(str(ts[min(len(ts) - 1, int(p * len(ts)))]) for p in (.1, .5, .9)) if ts else "-"
            n = d["n"] or 1
            print(f"{name:28s} {ph:9s} {d['n']:7d} {q:>22s} {d['truncated'] / n:9.1%} {d['calls'][0] / n:6.1%}")
    if not arms:
        raise SystemExit("topout-decisions: no reply in any arm")
    plot(arms, pieces, models, all_caps, a.out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
