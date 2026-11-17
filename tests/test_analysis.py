"""Every analysis reads a game that has only just started (a run record, nothing else) without a traceback: the ones
that report say there is nothing yet; the ones that write a file refuse and write nothing."""
from __future__ import annotations

import json
import subprocess
import sys

from tests.conftest import ROOT

RUN = (ROOT / "tests" / "fixtures" / "v1-run.json").read_text()


def _run(*argv):
    return subprocess.run([sys.executable, *map(str, argv)], capture_output=True, text=True, timeout=120, cwd=ROOT)


def test_a_just_started_game_is_nothing_yet_never_a_traceback(tmp_path):
    game = tmp_path / "games" / "0"
    game.mkdir(parents=True)
    (game / "journal.jsonl").write_text(json.dumps(json.loads(RUN)) + "\n")
    arm = f"a={tmp_path / 'games'}"
    refuse = {"topout-outcomes": ["--arm", arm],
              "topout-flows": ["--arm", arm, "-o", tmp_path / "f.png"]}
    for name, args in refuse.items():
        r = _run(ROOT / f"{name}.py", *args)
        assert r.returncode != 0 and "Traceback" not in r.stderr and f"{name}: " in r.stderr, (name, r.stderr)
    assert not any((tmp_path / f).exists() for f in ("o.png", "f.png"))
    r = _run(ROOT / "topout-replies.py", "--arm", arm)
    assert r.returncode == 0 and "Traceback" not in r.stderr, r.stderr


def test_outcomes_puts_a_survivor_in_the_survived_set_with_its_last_board(fake_engine, tmp_path):
    from tests.conftest import play
    (tmp_path / "games").mkdir()
    r, rows = play(fake_engine(), tmp_path / "games" / "0", pieces=3)
    assert r.returncode == 0 and rows[-1]["termination"] == "max_pieces"
    out = _run(ROOT / "topout-outcomes.py", "--arm", f"a={tmp_path / 'games'}")
    assert out.returncode == 0, out.stderr
    assert "a: survived 1 / 1" in out.stdout and "topped out" not in out.stdout, out.stdout
    game, lines, placed, per_turn, filled, sealed = out.stdout.strip().splitlines()[-1].split()
    assert int(placed) == 2 and int(filled) == 4 * int(placed) - 10 * int(lines)       # piece 3's spawn board
