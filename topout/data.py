"""The arms that topout-explore and topout-plot show: their games digested, grouped and summarised.

Contract. Reads finished journals through topout/store.py (and its cache); writes nothing else. Refuses arms that mix
max_pieces, because burden needs one horizon.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from topout import decisions, index, ledger, store

CACHE = Path.home() / ".cache" / "topout"


class Data:
    """The digests, grouped by arm, and what each arm shows on the first level."""

    def __init__(self, arms: list[index.Arm], cache: Path):
        digests = iter(store.load([p for a in arms for p in a.paths], cache))
        self.arms: dict[str, dict] = {}
        self.skipped = 0
        turns = set()
        for a in arms:
            games = [next(digests) for _ in a.paths]
            self.skipped += sum(g is None for g in games)
            games = sorted((g for g in games if g), key=lambda g: g["burden"])
            if not games:
                continue
            ids = [g["id"].rsplit("/", 1)[-1][:10] for g in games]
            for i, g in enumerate(games):
                g["key"] = ids[i] if ids.count(ids[i]) == 1 else f"{ids[i]}-{i}"
            both, per_piece = decisions.merge(g["phases"] for g in games)
            xs = [g["burden"] for g in games]
            tokens = [sum(p["tokens"]) for g in games for p in g["pieces"]]
            turns |= {g["turns"] for g in games}
            self.arms[a.name] = dict(arm=a, games=games, by_key={g["key"]: g for g in games}, both=both,
                                     per_piece=per_piece, burden=float(np.mean(xs)), ci=ledger.interval(xs),
                                     mean=float(np.mean(tokens)), median=float(np.median(tokens)),
                                     reached=sum(g["survived"] for g in games))
        if not self.arms:
            raise SystemExit("no finished game in any arm")
        if len(turns) > 1:
            raise SystemExit(f"arms mix max_pieces {sorted(turns)}; burden needs one horizon")
        self.turns = turns.pop()


def add_arguments(ap) -> None:
    ap.add_argument("--dataset", type=Path, help="local copy of a results dataset: every <model>/<sampler> is an arm")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=PATH[,PATH...]")
    ap.add_argument("--cache", type=Path, default=CACHE, help=f"digests of journals, rebuilt when missing (default {CACHE})")


def from_arguments(ap, a) -> Data:
    if not a.dataset and not a.arm:
        ap.error("give --dataset, --arm or both")
    return Data(index.arms(a.dataset, a.arm), a.cache)
