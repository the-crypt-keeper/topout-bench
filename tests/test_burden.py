"""Burden and the boards it is read from: the fixed horizon, and a spawn that overlaps the stack is kept."""
from __future__ import annotations

from topout import ledger

EMPTY = ["." * 10 for _ in range(20)]


def _tick(piece, rows, lines=0):
    return {"kind": "tick", "pieces": piece, "lines": lines, "board": "\n".join(rows)}


def test_turns_after_the_end_count_as_a_full_board():
    assert ledger.burden([0] * 100, 100) == 0
    assert ledger.burden([0] * 50, 100) == 10          # clean, but gone for half the turns
    assert ledger.burden([], 100) == 20


def test_a_falling_piece_in_the_air_is_not_stack_and_an_overlapping_one_is():
    board, overlap = ledger.spawn_board(_tick(2, ["...@@@@...", *EMPTY[2:], "####......"]))
    assert not overlap and "@" not in board and ledger.regions(board)["filled"] == 4
    lid = ["@" * 10, "#########.", *EMPTY[2:]]           # the count is short of 4 (p - 1): the piece is drawn over stack
    board, overlap = ledger.spawn_board(_tick(5, lid))
    assert overlap and ledger.regions(board)["filled"] == 19 and ledger.regions(board)["dead_rows"] == 19


def test_regions_tell_sealed_from_open_and_dead_rows_from_live():
    rows = [*EMPTY[:16], "#####.....", "#...#.....", "#.###.....", "#####.####"]
    r = ledger.regions("\n".join(rows))
    assert r["sealed"] == 4 and r["dead_rows"] == 2 and r["live_rows"] == 2 and r["height"] == 4
    assert r["filled"] + r["open"] + r["sealed"] == 200
