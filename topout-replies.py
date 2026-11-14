"""topout-replies: how long a model's replies are, per arm, split by where in a piece's conversation they fall.

    python topout-replies.py --arm NAME=PATH[,PATH...] [--arm ...]

Contract. Reads journals (running games up to their last record); an arm is one or more journals, game directories or
directories of games. Writes nothing; prints one row per arm and split. Never pools a piece's first reply with the
rest.

A reply is every generated reply, once (topout/calls.py is_reply): pressing, nudged and cut off alike. Each piece is
one conversation, so its first reply (conv_turn 0) is where the placement is planned, and the rest mostly press the
next button; pooling the two hides the planning under the presses. Per arm and per split: replies, then completion
tokens and reasoning characters at p10 / p50 / p90 / max. No means: one reply in a thousand can carry more tokens than
the rest together.
"""
from __future__ import annotations

import argparse
import json

from topout import journals
from topout.calls import is_reply

PCTS = (0.1, 0.5, 0.9)


def replies(paths) -> dict[str, list[tuple[int, int]]]:
    """(completion tokens, reasoning chars) per reply, under "first" (conv_turn 0) or "later"."""
    out: dict[str, list[tuple[int, int]]] = {"first": [], "later": []}
    for j in paths:
        with j.open() as f:
            for line in f:
                r = json.loads(line)
                if not is_reply(r):
                    continue
                out["first" if r["conv_turn"] == 0 else "later"].append(
                    (int((r["usage"] or {}).get("completion_tokens") or 0), len(r.get("reasoning") or "")))
    return out


def quantiles(xs: list[int]) -> list[int]:
    xs = sorted(xs)
    return [xs[min(len(xs) - 1, int(p * len(xs)))] for p in PCTS] + [xs[-1]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, help="NAME=PATH[,PATH...]")
    a = ap.parse_args()
    print(f"{'arm':24s} {'turn':5s} {'games':>5s} {'replies':>7s}   tokens p10/p50/p90/max      reasoning chars p10/p50/p90/max")
    for name, paths in journals.arms(a.arm).items():
        for split, rows in replies(paths).items():
            if not rows:
                print(f"{name:24s} {split:5s} {len(paths):5d} {0:7d}   (none)")
                continue
            tok = "/".join(map(str, quantiles([t for t, _ in rows])))
            rsn = "/".join(map(str, quantiles([c for _, c in rows])))
            print(f"{name:24s} {split:5s} {len(paths):5d} {len(rows):7d}   {tok:26s} {rsn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
