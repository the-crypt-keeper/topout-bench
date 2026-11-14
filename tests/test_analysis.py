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
    refuse = {"topout-outcomes": ["--arm", arm, "-o", tmp_path / "o.png"],
              "topout-placements": ["--arm", arm],
              "topout-regions": [tmp_path / "games", "-o", tmp_path / "r.png", "--csv", tmp_path / "r.csv"]}
    for name, args in refuse.items():
        r = _run(ROOT / f"{name}.py", *args)
        assert r.returncode != 0 and "Traceback" not in r.stderr and f"{name}: " in r.stderr, (name, r.stderr)
    assert not any((tmp_path / f).exists() for f in ("o.png", "r.png", "r.csv"))
    for name in ("topout-replies", "topout-conduct"):
        r = _run(ROOT / f"{name}.py", "--arm", arm)
        assert r.returncode == 0 and "Traceback" not in r.stderr, (name, r.stderr)
