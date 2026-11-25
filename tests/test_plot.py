"""topout-plot: every figure and table on a real (fake engine) game, refusals without a traceback, and writes only -o."""
from __future__ import annotations

import json
import subprocess
import sys

from tests.conftest import ROOT, play

KINDS = ("burden", "games", "decisions", "triangle", "moves")


def _plot(*argv, cwd=ROOT):
    return subprocess.run([sys.executable, ROOT / "topout-plot.py", *map(str, argv)], capture_output=True, text=True, timeout=120, cwd=cwd)


def test_every_kind_writes_its_page_and_prints_its_table(fake_engine, tmp_path):
    (tmp_path / "games").mkdir()
    assert play(fake_engine(), tmp_path / "games" / "0", pieces=3)[0].returncode == 0
    arms = ["--arm", f"one={tmp_path / 'games'}", "--arm", f"two={tmp_path / 'games'}", "--cache", tmp_path / "c"]
    for kind in KINDS:
        out = tmp_path / f"{kind}.html"
        r = _plot(kind, *arms, "-o", out, "--table")
        assert r.returncode == 0, (kind, r.stderr)
        assert "one" in r.stdout and "two" in r.stdout and out.stat().st_size > 1000
    r = _plot("burden", *arms, "--only", "one", "--table")
    assert "one" in r.stdout and "two" not in r.stdout
    r = _plot("burden", *arms, "--only", "on", "--table")               # the start of a name ends at a word
    assert r.returncode and "matches no arm" in r.stderr
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(["c", "games", *(f"{k}.html" for k in KINDS)])


def test_a_just_started_game_and_bad_arguments_are_refused_without_a_traceback(tmp_path):
    (tmp_path / "g").mkdir()
    (tmp_path / "g" / "journal.jsonl").write_text(json.dumps(json.loads((ROOT / "tests" / "fixtures" / "v1-run.json").read_text())) + "\n")
    arm = ["--arm", f"a={tmp_path / 'g'}", "--cache", tmp_path / "c"]
    for argv in (["burden", *arm, "-o", tmp_path / "b.html"], ["burden", *arm], ["burden", *arm, "-o", tmp_path / "b.gif"], ["burden", "--table"]):
        r = _plot(*argv)
        assert r.returncode and "Traceback" not in r.stderr, (argv, r.stderr)
    assert not list(tmp_path.glob("b.*"))
