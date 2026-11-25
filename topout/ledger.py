"""The board each piece spawned into: its regions, and burden over a game's boards.

Reads boards and numbers; computes; opens no file and writes nothing. topout/store.py calls it for every piece.

Each journal carries the board the model was shown on every tick ('#' settled, '@' falling, '.' empty). The board at a
piece's first tick is the settled field that piece spawned into: the previous lock and its clears, played out. Each is
split:

  filled      '#'
  sealed      empty, and not reached by a flood from the top that moves down, left and right, never up. Fills come
              from above or the side, never from below, so nothing can ever fill it until whatever covers it has been
              cleared.
  open        the rest of the empty cells.

The flood assumes a 1x1 piece exists: a one-wide pocket under an overhang counts as open though no tetromino fits it.

Rows: a dead row holds a sealed cell and cannot complete until dug out; the rest of the stack's rows are live. Every board
is a point (live, dead) with the 20 rows' remainder empty: the triangle. explore_assets/boards.js repeats the flood to
colour cells.
"""
from __future__ import annotations

from collections import deque

ROWS, COLS = 20, 10


def flood(filled) -> set[tuple[int, int]]:
    """Empty cells reached from the top row moving down, left, right: never up."""
    q = deque((0, c) for c in range(COLS) if not filled[0][c])
    seen = set(q)
    while q:
        r, c = q.popleft()
        for s in ((r + 1, c), (r, c - 1), (r, c + 1)):
            rr, cc = s
            if 0 <= rr < ROWS and 0 <= cc < COLS and not filled[rr][cc] and s not in seen:
                seen.add(s)
                q.append(s)
    return seen


def regions(board: str) -> dict:
    """Filled, open and sealed cells, and the rows: dead (holding a sealed cell), live (the rest of the stack)."""
    filled = [[ch == "#" for ch in row] for row in board.split("\n")]
    assert len(filled) == ROWS and all(len(r) == COLS for r in filled), board
    reach = flood(filled)
    sealed = {(r, c) for r in range(ROWS) for c in range(COLS) if not filled[r][c] and (r, c) not in reach}
    n, dead = sum(map(sum, filled)), len({r for r, _ in sealed})
    height = ROWS - next((r for r in range(ROWS) if any(filled[r])), ROWS)
    return {"filled": n, "open": ROWS * COLS - n - len(sealed), "sealed": len(sealed), "dead_rows": dead,
            "live_rows": height - dead, "height": height}


def spawn_board(tick: dict) -> tuple[str, bool]:
    """The stack on a piece's first tick, and whether the piece overlaps it. An overlapping spawn (a '#' count short of
    4 (p - 1) - 10 L: the piece is drawn over settled cells, or cells were lost above the top) is kept, with the piece's
    cells counted as occupied; on any other board the falling piece is in the air and is not stack."""
    board = tick["board"]
    if board.count("#") == 4 * (tick["pieces"] - 1) - 10 * tick["lines"]:
        return board.replace("@", "."), False
    return board.replace("@", "#"), True


def burden(dead: list[int], turns: int) -> float:
    """Dead rows averaged over `turns` turns; every turn after the game ended counts as all ROWS of them."""
    return (sum(dead) + ROWS * (turns - len(dead))) / turns


def interval(xs: list[float], n: int = 10000) -> tuple[float, float]:
    """95% interval of the mean, resampling games (fixed seed)."""
    import numpy as np
    means = np.random.default_rng(1).choice(xs, (n, len(xs))).mean(axis=1)
    return tuple(float(v) for v in np.percentile(means, (2.5, 97.5)))
