"""The board at each piece's spawn, decomposed: filled / open / sealed regions and the height ledger.

Reads a journal's tick records and nothing else (no summary: a running game reads up to its last tick). Computes;
writes nothing. topout-flows places these rows on the triangle (--view rows) and turns them into placement events (--view moves).

Each journal carries the board the model was shown on every tick ('#' settled, '@' falling, '.' empty). The board at
a piece's first tick is the settled field that piece spawned into: the previous lock and its clears, played out. Each
such board is split:

  filled      '#'
  sealed      empty, and not reached by a flood from the top that moves down, left and right, never up. Fills come
              from above or the side, never from below, so nothing can ever fill it until whatever covers it has been
              cleared.
  open        the rest of the empty cells.

The flood assumes a 1x1 piece exists: a one-wide pocket under an overhang counts as open though no tetromino fits it.
Piece-aware reachability is the refinement, not built.

Rows: `dead_rows` holds a sealed cell; it cannot complete until dug out. `live_rows` is height minus dead rows.
`height` is the stack height in rows. `covered` is the classic hole count (an empty cell with a filled cell anywhere
above it in its column; Dellacherie / BCTS). Every sealed cell is covered: a cell open to the sky is reached straight
down. `tuckable` = covered - sealed, the covered cells a slide can still reach (Cold Clear's overhang, in this game's
physics).

The ledger. Each piece adds 4 cells and each line removes 10, so before piece p spawns, with L lines cleared:

    height = 0.4 (p - 1) - L + open_slack + sealed_slack

`packed` = 0.4 (p - 1) - L is the height a perfectly packed stack would have; `open_slack` is the open cells below the
stack top / 10, `sealed_slack` the sealed cells / 10. It is exact by construction once filled = 4 (p - 1) - 10 L,
which holds on every board except where the falling piece is drawn over settled cells, at top-out: `occluded` marks
those. Each piece's change in height is then 0.4 - (lines it cleared) + d_open_slack + d_sealed_slack: packing,
clearing, loose slack, dead slack. Survival needs 0.4 lines a piece; below it the packed term climbs.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

from topout import journals

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
    filled = [[ch == "#" for ch in row] for row in board.split("\n")]
    assert len(filled) == ROWS and all(len(r) == COLS for r in filled), board
    reach = flood(filled)
    n = dict(filled=0, open=0, sealed=0, covered=0)
    dead = set()
    for r in range(ROWS):
        for c in range(COLS):
            if filled[r][c]:
                n["filled"] += 1
                continue
            if any(filled[k][c] for k in range(r)):
                n["covered"] += 1
            if (r, c) in reach:
                n["open"] += 1
            else:
                n["sealed"] += 1
                dead.add(r)
    top = next((r for r in range(ROWS) if any(filled[r])), ROWS)
    h = ROWS - top
    return n | {"tuckable": n["covered"] - n["sealed"], "dead_rows": len(dead), "height": h,
                "live_rows": h - len(dead), "open_slack": (10 * h - n["filled"] - n["sealed"]) / 10,
                "sealed_slack": n["sealed"] / 10}


def piece_rows(path: Path) -> list[dict]:
    """One row per piece: the board it spawned into, decomposed, and the change since the previous piece."""
    model = journals.run_record(path).get("model") or "?"
    game = journals.game_id(path)
    rows, seen = [], set()
    with path.open() as f:
        for line in f:
            t = json.loads(line)
            if t.get("kind") != "tick" or t["pieces"] in seen:
                continue
            seen.add(t["pieces"])
            p, lines = t["pieces"], t["lines"]
            row = {"game": game, "model": model, "piece": p, "lines": lines} \
                | regions(t["board"]) | {"packed": 0.4 * (p - 1) - lines}
            row["occluded"] = row["filled"] != 4 * (p - 1) - 10 * lines
            prev = rows[-1] if rows else None
            for k in ("lines", "open_slack", "sealed_slack"):
                row[f"d_{k}"] = row[k] - prev[k] if prev else None
            rows.append(row)
    return rows


def flows(rows: list[dict]) -> list[dict]:
    """The triangle's transitions, cumulative from the first piece (threads/topout, the triangle model): each row of
    the 20 is empty, live or dead, and between consecutive spawn boards

      cleared     live -> empty: the lines that piece cleared (a cleared row was full, so live)
      opened      empty -> live: rows newly occupied, d(height) + cleared (only a clear lowers the stack top)
      sealings    live -> dead, and unsealings dead -> live: from d(dead_rows) alone, so each is a LOWER BOUND (a piece
                  that seals one row and unseals another nets to zero); per-row tracking is the refinement, not built

    empty = 20 - (opened - cleared) when no row is skipped. An occluded board (top-out) ends the events, as in
    placements. One dict per placement: the piece, the four running totals, and the state after it: `occupied` (the
    stack height, = opened - cleared, = 20 - empty) and `dead` (dead rows held)."""
    out, o, c, s, u = [], 0, 0, 0, 0
    for a, b in zip(rows, rows[1:]):
        if b["occluded"]:
            break
        cleared = b["lines"] - a["lines"]
        opened = b["height"] - a["height"] + cleared
        if opened < 0:
            raise ValueError(f"{a['game']} piece {a['piece']}: rows opened {opened} < 0 (height {a['height']} -> "
                             f"{b['height']}, cleared {cleared}): the stack top fell by more than its clears")
        d = b["dead_rows"] - a["dead_rows"]
        o, c, s, u = o + opened, c + cleared, s + max(d, 0), u + max(-d, 0)
        out.append({"piece": a["piece"], "opened": o, "cleared": c, "sealings": s, "unsealings": u,
                    "occupied": b["height"], "dead": b["dead_rows"]})
    return out


def placements(rows: list[dict]) -> list[dict]:
    """A placement of piece p is the change from the board p spawned into to the board p+1 spawned into: sealed cells
    added (sealing) or removed (dig-out), and lines cleared. An occluded board (top-out) ends the game's events."""
    out = []
    for a, b in zip(rows, rows[1:]):
        if b["occluded"]:
            break
        out.append({"piece": a["piece"], "sealed": b["sealed"] - a["sealed"], "lines": b["lines"] - a["lines"]})
    return out
