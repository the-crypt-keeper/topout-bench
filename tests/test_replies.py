"""topout-replies counts every generated reply (pressing, nudged, cut off) once, under its piece's first or later turn."""
from __future__ import annotations

import subprocess
import sys

from tests.conftest import ROOT, play


def test_every_reply_counted_once_and_split_by_turn(fake_engine, tmp_path):
    r, rows = play(fake_engine(length_rate=0.3, no_call_rate=0.3, seed=2), tmp_path / "g", pieces=4)
    assert r.returncode == 0, r.stdout[-2000:]
    generated = [x for x in rows if "usage" in x and "conv_turn" in x]
    assert {x["kind"] for x in generated} >= {"tick", "capped"}      # cut-off replies are in the sample
    first = sum(1 for x in generated if x["conv_turn"] == 0)
    out = subprocess.run([sys.executable, str(ROOT / "topout-replies.py"), "--arm", f"fake={tmp_path / 'g'}"],
                         capture_output=True, text=True, timeout=60, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    counts = {line.split()[1]: int(line.split()[3]) for line in out.stdout.splitlines() if line.startswith("fake ")}
    assert counts == {"first": first, "later": len(generated) - first}, out.stdout


def test_summary_splits_first_reply_from_later_never_pooled(fake_engine, tmp_path):
    """The summary carries first_reply and later_replies, each counted and summarised from its own replies; no pooled
    completion_tokens_p50 / call_ms_p50."""
    r, rows = play(fake_engine(length_rate=0.3, no_call_rate=0.3, seed=2), tmp_path / "g", pieces=4)
    assert r.returncode == 0, r.stdout[-2000:]
    summary = rows[-1]
    assert "completion_tokens_p50" not in summary and "call_ms_p50" not in summary
    generated = [x for x in rows if "usage" in x and "conv_turn" in x]
    for key, keep in (("first_reply", lambda t: t == 0), ("later_replies", lambda t: t > 0)):
        mine = [x for x in generated if keep(x["conv_turn"])]
        tokens = sorted(x["usage"]["completion_tokens"] for x in mine)
        ms = sorted(x["call_ms"] for x in mine)
        assert summary[key] == {"replies": len(mine), "completion_tokens_p50": tokens[len(tokens) // 2],
                                "call_ms_p50": ms[len(ms) // 2]}, (key, summary[key])
    assert summary["first_reply"]["replies"] + summary["later_replies"]["replies"] == summary["replies"]
