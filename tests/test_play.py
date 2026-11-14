"""topout-play end to end against the fake engine."""
from __future__ import annotations

import subprocess
import sys

from tests.conftest import ROOT, play


def test_plays_to_the_piece_cap_and_records_v1(fake_engine, tmp_path):
    r, rows = play(fake_engine(), tmp_path / "g")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    run = rows[0]
    assert (run["kind"], run["mode"], run["decision"], run["prompt"], run["protocol"]) == \
        ("run", "tetris-tool", "verbs", "v1", "tetris-v3")
    assert rows[-1]["kind"] == "summary" and rows[-1]["pieces"] == 3 and rows[-1]["decision"] == "verbs"
    assert "ERROR" not in r.stdout


def test_length_is_retried_never_looped_back_then_rebooted(fake_engine, tmp_path):
    """finish_reason length: not looped back, not nudged; the identical request up to CAPPED_RETRIES more
    times; if all are cut off, wait is pressed (LOST:) and the trajectory reboots (a fresh conversation).
    Neither CAPPED nor LOST is an ERROR."""
    r, rows = play(fake_engine(length_rate=0.6, seed=0), tmp_path / "g", pieces=4)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    summary = rows[-1]
    assert summary["capped_replies"] > 0 and summary["lost_decisions"] > 0, summary
    assert summary["nudges"] == 0                       # a cut-off reply is never treated as no-tool-call
    assert "ERROR" not in r.stdout
    assert r.stdout.count("CAPPED:") == summary["capped_replies"]
    assert r.stdout.count("LOST:") == summary["lost_decisions"]
    capped = [x for x in rows if x.get("kind") == "capped"]
    assert max(x["attempt"] for x in capped) == 4       # 1 + CAPPED_RETRIES
    for a, b in zip(rows, rows[1:]):                   # a retry is the identical request: the conversation does not grow
        if a.get("kind") == "capped" and b.get("kind") == "capped":
            assert b["conv_turn"] == a["conv_turn"]
    for i, x in enumerate(rows):                       # every lost decision reboots the trajectory
        if x.get("parse_error") == "capped":
            nxt = next(y for y in rows[i + 1:] if "conv_turn" in y)
            assert nxt["conv_turn"] == 0 and nxt.get("conv_reset") in ("capped", "new piece"), nxt


def test_no_tool_call_presses_wait_then_a_user_turn(fake_engine, tmp_path):
    """A reply with no tool call: wait is pressed, counted as a nudge, and the next request carries that reply
    and a user turn saying so (the only valid next message). A NUDGE is not an ERROR."""
    r, rows = play(fake_engine(no_call_rate=0.5, seed=1), tmp_path / "g", pieces=4)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    summary = rows[-1]
    nudged = [x for x in rows if x.get("parse_error") == "no tool call"]
    assert summary["nudges"] == len(nudged) > 0
    assert all(x["action"] == "wait" for x in nudged)
    assert r.stdout.count("NUDGE:") == summary["nudges"] and "ERROR" not in r.stdout


def test_parallel_counts_each_game_when_it_exits(fake_engine, tmp_path):
    """--parallel: every game reports its exit, and the last progress line counts all of them."""
    sampler = tmp_path / "s.json"
    sampler.write_text('{"max_tokens": 100}')
    argv = [sys.executable, str(ROOT / "topout-play.py"), "--apibase", fake_engine(), "--model", "fake",
            "--output", str(tmp_path / "g"), "--sampler", str(sampler), "--seed", "1", "--lag-frames", "12",
            "--context-length", "100000", "--max-pieces", "2", "--parallel", "3"]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=300, cwd=ROOT)
    assert r.returncode == 0, r.stdout[-2000:]
    assert sorted(line.split()[0] for line in r.stdout.splitlines() if " exited " in line) == ["[g0]", "[g1]", "[g2]"]
    last = [line for line in r.stdout.splitlines() if line.startswith("[") and "/6]" in line][-1]
    assert last.startswith("[6/6]") and "games running=0" in last, last
    assert all((tmp_path / "g" / str(i) / "journal.jsonl").exists() for i in range(3))


def test_first_turn_takes_its_own_cap_and_retries(fake_engine, tmp_path):
    """--first-turn-max-tokens / --first-turn-retries: a request on a fresh conversation (conv_turn 0) is sent
    with the first-turn cap and given up after 1 + first-turn-retries attempts; every later turn keeps the
    sampler's cap. Without the flags nothing changes."""
    first = ["--first-turn-max-tokens", "200", "--first-turn-retries", "1"]
    r, rows = play(fake_engine(length_rate=0.6, seed=0), tmp_path / "g", first, pieces=4)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert rows[0]["first_turn_max_tokens"] == 200 and rows[0]["first_turn_retries"] == 1
    sent = [x for x in rows if "conv_turn" in x and "max_tokens" in x]
    assert any(x["conv_turn"] == 0 for x in sent) and any(x["conv_turn"] > 0 for x in sent)
    for x in sent:
        assert x["max_tokens"] == (200 if x["conv_turn"] == 0 else 100), x
    capped = [x for x in rows if x.get("kind") == "capped"]
    assert all(x["attempt"] <= 2 for x in capped if x["conv_turn"] == 0)
    r0, rows0 = play(fake_engine(length_rate=0.6, seed=0), tmp_path / "g0", pieces=2)
    assert r0.returncode == 0
    assert rows0[0]["first_turn_max_tokens"] is None
    assert all(x["max_tokens"] == 100 for x in rows0 if "conv_turn" in x and "max_tokens" in x)


def test_engine_overflow_refusal_resets_the_conversation(fake_engine, tmp_path):
    """The engine judges fit: its context-overflow 400 starts the piece's conversation over (RESET: context,
    conv_reset "context", the next request a fresh conversation) instead of crashing. Nothing is predicted."""
    r, rows = play(fake_engine(context=3000), tmp_path / "g", pieces=4)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert "ERROR" not in r.stdout
    resets = [x for x in rows if x.get("conv_reset") == "context"]
    assert resets and r.stdout.count("RESET: context") >= len(resets) > 0
    assert all(x["conv_turn"] == 0 for x in resets)
    r0, rows0 = play(fake_engine(), tmp_path / "g0", pieces=4)
    assert r0.returncode == 0 and "RESET:" not in r0.stdout
    assert not any(x.get("conv_reset") == "context" for x in rows0)
