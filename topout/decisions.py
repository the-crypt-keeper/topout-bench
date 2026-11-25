"""Replies by phase: planning (conv_turn 0) and execution (the rest), counted, never pooled.

Contract. Counts a journal's replies; draws nothing. topout-decisions and topout-explore both read these counts.

A reply is planning when it opens a conversation: a piece's first reply, a retry of it, or the first after a restart.
Every attempt the engine returned is a reply, so a truncated one and its retry both count. Replies per piece is a
property of the piece, so it is counted over both phases.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from topout import journals
from topout.calls import is_reply
from topout.prompt import BUTTONS

VERBS = list(BUTTONS)
BINS, LOW, HIGH = 24, 4, 16            # token bins, log2 range
CALLS, REPLIES = 6, 16                 # the last bar of each is "or more"


def token_bin(t: int) -> int:
    return min(BINS - 1, max(0, int((math.log2(max(t, 1)) - LOW) / (HIGH - LOW) * BINS)))


def _empty() -> dict:
    return dict(n=0, tokens=[], bins=Counter(), cut=Counter(), calls=Counter(), verbs=Counter(), truncated=0)


def count(records: Iterable[dict]) -> tuple[list[dict], Counter]:
    """Counts for planning (0) and execution (1), and replies per piece, over one game's records."""
    out, pieces = [_empty(), _empty()], Counter()
    for r in records:
        if not is_reply(r):
            continue
        d, t = out[0 if r["conv_turn"] == 0 else 1], int((r["usage"] or {}).get("completion_tokens") or 0)
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
    return out, Counter(min(n, REPLIES) for n in pieces.values())


def merge(games: Iterable[tuple[list[dict], Counter]]) -> tuple[list[dict], Counter]:
    out, per_piece = [_empty(), _empty()], Counter()
    for both, pieces in games:
        per_piece.update(pieces)
        for d, g in zip(out, both):
            for k, v in g.items():
                d[k].update(v) if isinstance(v, Counter) else d.__setitem__(k, d[k] + v)
    return out, per_piece


def phases(paths: list[Path]) -> tuple[list[dict], Counter, set[int]]:
    """An arm's counts, its replies per piece, and the max_tokens its run records hold."""
    caps = {(journals.run_record(p).get("sampler") or {}).get("max_tokens") for p in paths}
    both, per_piece = merge(count(journals.records(p)) for p in paths)
    return both, per_piece, {c for c in caps if c}


def share(counter: Counter, keys) -> list[float]:
    total = sum(counter.values()) or 1
    return [counter[k] / total for k in keys]


def rates(d: dict) -> dict[str, float]:
    """Shares of the phase's replies: cut off by max_tokens or the context, and nudged (no tool call)."""
    return {"truncated": d["truncated"] / (d["n"] or 1), "nudged": d["calls"][0] / (d["n"] or 1)}
