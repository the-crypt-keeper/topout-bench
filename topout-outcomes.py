"""topout-outcomes: how each game ended, per arm, with games that survived and games that topped out never pooled.

    python topout-outcomes.py --arm NAME=PATH[,PATH...] [--arm ...]

Contract. Reads finished journals (games without a summary, still running, are skipped and counted). Writes nothing;
prints. Reports outcomes only: no trajectory and no per-placement rate pooled across games.

The two outcomes are different measurements and are never put on one axis (operator, 11-16). A game that reached
--max-pieces survived; a game that topped out stopped wherever its stack reached the top. Per arm: survival n / N with
a 95% Wilson interval. Then per set, every game, all read on its last clean board (the last spawn board that is not
occluded, through topout/ledger.py: the journal holds the board before each press, so this is one piece short of the
end for a survivor): pieces placed, lines, lines per turn (lines / placed), `filled` (settled cells, = 4 x placed -
10 x lines) and `sealed` (empty cells no fill can reach). Height is left out: it is where a trajectory stood when the
game stopped, not an outcome.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from topout import journals, ledger


def outcome(path: Path) -> dict | None:
    """The game's outcome and its last clean board; None for a game with no summary."""
    summary = None
    with path.open() as f:
        for line in f:
            if '"kind": "summary"' in line:
                summary = json.loads(line)
    if summary is None:
        return None
    rows = [r for r in ledger.piece_rows(path) if not r["occluded"]]
    last = rows[-1]
    placed = last["piece"] - 1                       # the board piece p spawned into holds p - 1 placed pieces
    return {"game": journals.game_id(path), "survived": summary["termination"] == "max_pieces",
            "lines": last["lines"], "placed": placed, "per_turn": last["lines"] / placed if placed else 0.0,
            "filled": last["filled"], "sealed": last["sealed"]}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, help="NAME=PATH[,PATH...]")
    a = ap.parse_args()
    finished = 0
    for name, paths in journals.arms(a.arm).items():
        games = [g for g in map(outcome, paths) if g is not None]
        skipped = len(paths) - len(games)
        finished += len(games)
        if not games:
            print(f"\n{name}: no finished game ({skipped} skipped)")
            continue
        n_surv = sum(g["survived"] for g in games)
        lo, hi = wilson(n_surv, len(games))
        print(f"\n{name}: survived {n_surv} / {len(games)} [{lo:.2f}, {hi:.2f}]"
              + (f"  ({skipped} unfinished skipped)" if skipped else ""))
        for label, survived in (("survived", True), ("topped out", False)):
            rows = sorted((g for g in games if g["survived"] == survived), key=lambda g: -g["lines"])
            if not rows:
                continue
            print(f"  {label} ({len(rows)}):  {'game':>6s} {'lines':>5s} {'placed':>6s} {'lines/turn':>10s} "
                  f"{'filled':>6s} {'sealed':>6s}")
            for g in rows:
                print(f"  {'':{len(label) + 6}s}{Path(g['game']).name:>6s} {g['lines']:5d} {g['placed']:6d} "
                      f"{g['per_turn']:10.3f} {g['filled']:6d} {g['sealed']:6d}")
    if not finished:
        raise SystemExit("topout-outcomes: no finished game in any arm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
