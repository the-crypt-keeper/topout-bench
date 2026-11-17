"""Replies are counted once each, from the journal alone, against what the engine actually served."""
from __future__ import annotations

import subprocess
import sys

from topout.calls import is_reply
from tests.conftest import ROOT, counts, play


def test_replies_counts_every_served_reply_once_split_by_turn(fake_engine, tmp_path):
    """Nudged and cut-off replies included; a lost decision's tick repeats its last capped reply and is not another."""
    r, rows = play(fake_engine(length_rate=0.6, no_call_rate=0.3, seed=0), tmp_path / "g", pieces=4)
    assert r.returncode == 0, r.stdout[-2000:]
    assert any(x.get("parse_error") == "capped" for x in rows)        # a lost decision is in the sample
    first = sum(1 for x in rows if is_reply(x) and x["conv_turn"] == 0)
    out = subprocess.run([sys.executable, str(ROOT / "topout-replies.py"), "--arm", f"fake={tmp_path / 'g'}"],
                         capture_output=True, text=True, timeout=60, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    counts = {line.split()[1]: int(line.split()[3]) for line in out.stdout.splitlines() if line.startswith("fake ")}
    assert counts["first"] + counts["later"] == fake_engine.served(), out.stdout
    assert counts["first"] == first


def test_journal_records_match_the_engine_and_the_live_lines(fake_engine, tmp_path):
    r, rows = play(fake_engine(length_rate=0.6, no_call_rate=0.3, seed=0), tmp_path / "g", pieces=4)
    assert r.returncode == 0, r.stdout[-2000:]
    assert sum(1 for x in rows if is_reply(x)) == fake_engine.served()
    n = counts(rows)
    assert (n["nudges"], n["capped"], n["lost"]) == (r.stdout.count("NUDGE:"), r.stdout.count("CAPPED:"),
                                                     r.stdout.count("LOST:"))
    assert n["nudges"] and n["capped"] and n["lost"]


def test_summary_is_how_the_game_ended_and_nothing_else(fake_engine, tmp_path):
    r, rows = play(fake_engine(), tmp_path / "g", pieces=2)
    assert r.returncode == 0, r.stdout[-2000:]
    assert set(rows[-1]) == {"kind", "termination", "game_over", "pieces", "lines", "score", "frames"}
    assert all("call_ms" in x and "usage" in x for x in rows if is_reply(x))
