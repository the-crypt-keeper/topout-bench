"""Burden: the fixed horizon, the kept overlap board, and a just-started game that is nothing yet."""
from __future__ import annotations

import json
import subprocess
import sys

from tests.conftest import ROOT
from topout import ledger


def test_turns_after_the_end_count_as_a_full_board():
    assert ledger.burden([0] * 100, 100) == 0
    assert ledger.burden([0] * 50, 100) == 10          # clean, but gone for half the turns
    assert ledger.burden([], 100) == 20


def _tick(piece, board, lines=0):
    return json.dumps({"kind": "tick", "pieces": piece, "lines": lines, "board": board})


def test_an_overlapping_spawn_is_kept_and_the_falling_piece_is_not_stack(tmp_path):
    rows = ["." * 10 for _ in range(20)]
    clean = rows.copy()
    clean[0] = "...@@@@..."                              # a piece in the air over an empty board: no dead row
    clean[19] = "####......"
    over = rows.copy()
    over[0], over[1], over[2] = "..@@@@....", "##......##", "####..####"
    n = "\n".join
    path = tmp_path / "journal.jsonl"
    path.write_text(_tick(1, n(["...@@@@..."] + rows[1:])) + "\n" + _tick(2, n(clean)) + "\n"
                    + _tick(4, n(over)) + "\n")          # 12 '#' expected: count equals, so nothing is overlapped
    assert ledger.dead_rows(path) == [0, 0, 0]
    lid = rows.copy()
    lid[0] = "@@@@@@@@@@"                                # count short of 4(p-1): the overlap is stack, and seals row 1
    lid[1] = "#########."
    path.write_text(_tick(5, n(lid)) + "\n")
    assert ledger.dead_rows(path) and ledger.dead_rows(path)[0] > 0


def test_a_just_started_game_is_refused_without_a_traceback(tmp_path):
    (tmp_path / "g").mkdir()
    (tmp_path / "g" / "journal.jsonl").write_text(json.dumps(json.loads((ROOT / "tests" / "fixtures" / "v1-run.json").read_text())) + "\n")
    r = subprocess.run([sys.executable, "topout-burden.py", "--arm", f"a={tmp_path}", "-o", str(tmp_path / "b.png")],
                       capture_output=True, text=True, cwd=ROOT, timeout=60)
    assert r.returncode and "Traceback" not in r.stderr and not (tmp_path / "b.png").exists(), r.stderr
