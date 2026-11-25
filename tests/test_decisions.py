"""Decisions: planning and execution never pooled, truncations counted apart, ."""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys

from tests.conftest import ROOT

spec = importlib.util.spec_from_file_location("topout_decisions", ROOT / "topout-decisions.py")
dec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dec)


def _reply(kind, piece, turn, tokens, verbs, **kw):
    calls = [{"function": {"name": v, "arguments": "{}"}} for v in verbs]
    return json.dumps({"kind": kind, "pieces": piece, "conv_turn": turn, "usage": {"completion_tokens": tokens},
                       "tool_calls": calls, **kw})


def test_phases_split_by_conversation_turn_and_a_truncated_reply_has_no_calls(tmp_path):
    path = tmp_path / "journal.jsonl"
    path.write_text("\n".join([
        _reply("capped", 1, 0, 49152, []),                       # truncated plan, then its retry
        _reply("reply", 1, 0, 3000, ["start"]),
        _reply("tick", 1, 1, 100, ["left", "drop"]),
        json.dumps({"kind": "tick", "pieces": 1, "conv_turn": 1, "action": "drop"}),       # same reply's second press
        _reply("tick", 1, 2, 50, [], parse_error="no tool call"),
        _reply("tick", 2, 0, 2000, ["start", "drop"])]) + "\n")
    (plan, execute), per_piece, _ = dec.phases([path])
    assert (plan["n"], execute["n"], plan["truncated"], execute["truncated"]) == (3, 2, 1, 0)
    assert plan["calls"] == {1: 1, 2: 1} and execute["calls"] == {2: 1, 0: 1}      # the nudge is the 0
    assert plan["verbs"] == {"start": 2, "drop": 1} and execute["verbs"] == {"left": 1, "drop": 1}
    assert per_piece == {4: 1, 1: 1}                         # both phases, every attempt


def test_a_just_started_game_is_refused_without_a_traceback(tmp_path):
    (tmp_path / "g").mkdir()
    (tmp_path / "g" / "journal.jsonl").write_text(json.dumps(json.loads((ROOT / "tests" / "fixtures" / "v1-run.json").read_text())) + "\n")
    r = subprocess.run([sys.executable, "topout-decisions.py", "--arm", f"a={tmp_path}", "-o", str(tmp_path / "d.png")],
                       capture_output=True, text=True, cwd=ROOT, timeout=60)
    assert r.returncode and "Traceback" not in r.stderr and not (tmp_path / "d.png").exists(), r.stderr
