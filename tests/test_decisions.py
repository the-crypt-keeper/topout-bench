"""Decisions: planning and execution never pooled, truncations counted apart, ."""
from __future__ import annotations

import json

from topout import decisions


def _reply(kind, piece, turn, tokens, verbs, **kw):
    calls = [{"function": {"name": v, "arguments": "{}"}} for v in verbs]
    return json.dumps({"kind": kind, "pieces": piece, "conv_turn": turn, "usage": {"completion_tokens": tokens},
                       "tool_calls": calls, **kw})


def test_phases_split_by_conversation_turn_and_a_truncated_reply_has_no_calls():
    records = [json.loads(x) for x in [
        _reply("capped", 1, 0, 49152, []),                       # truncated plan, then its retry
        _reply("reply", 1, 0, 3000, ["start"]),
        _reply("tick", 1, 1, 100, ["left", "drop"]),
        json.dumps({"kind": "tick", "pieces": 1, "conv_turn": 1, "action": "drop"}),       # same reply's second press
        _reply("tick", 1, 2, 50, [], parse_error="no tool call"),
        _reply("tick", 2, 0, 2000, ["start", "drop"])]]
    (plan, execute), per_piece = decisions.merge([decisions.count(records)])
    assert (plan["n"], execute["n"], plan["truncated"], execute["truncated"]) == (3, 2, 1, 0)
    assert plan["calls"] == {1: 1, 2: 1} and execute["calls"] == {2: 1, 0: 1}      # the nudge is the 0
    assert plan["verbs"] == {"start": 2, "drop": 1} and execute["verbs"] == {"left": 1, "drop": 1}
    assert per_piece == {4: 1, 1: 1}                         # both phases, every attempt
