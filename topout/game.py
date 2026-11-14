"""NES Tetris in the emulator, and the timing every press runs under.

Runs on gym-tetris (NES Tetris on nes-py, its bundled ROM). A Game steps the emulator one frame at a time
and reads the board, the falling piece and the counters from RAM.

Timing (tetris-v2, operator 2026-10-04): every button costs at least `lag` frames of game time, the
player's perception lag (12 frames = 200 ms at 60.1 fps). left / right / rotate press for one frame; wait
presses nothing; drop holds Down until the piece is DROP_ROWS lower or locks. Each then runs on until the
play area changes and at least `lag` frames have passed. A decision also ends only on a frame where the
play area did not change since the frame before, so the board the model is shown and the screen are the
same state: the play area has held still for 2 frames, with the piece in play or its lock drawn.

Locks: a decision whose piece locks ends at the lock (the lag is not padded into the lock sequence, which
runs without input either way), settled at playState 3, where a completed row is still on the board in RAM
and on screen. The tick after it advances, without a decision, through the line-clear animation to the next
piece's spawn, settled: every piece runs spawn, decisions, lock.

Every tick records the button held on each of its frames (run-length [[buttons, frames], ...]); from the
seed and those streams the game replays exactly (topout-video.py), so no frames are saved.
"""
from __future__ import annotations

import numpy as np
from gym_tetris.tetris_env import TetrisEnv, _PIECE_ORIENTATION_TABLE

# NES joypad bits as nes-py takes them (one byte per frame).
BUTTON = {"right": 0x80, "left": 0x40, "down": 0x20, "A": 0x01}

# RAM (NES Tetris): falling piece x 0x40, y 0x41, orientation id 0x42 (19 ids, >= 19 means no piece in
# play); board 0x0400-0x04C7 (20x10, 239 = empty); next piece 0xBF; lines BCD 0x50; score BCD 0x53; game
# over 0x58.
RAM_X, RAM_Y, RAM_ORIENT, RAM_GAME_OVER = 0x40, 0x41, 0x42, 0x58
# playState 0x48: 1 while the player controls the falling piece; 2,3,5..8 for ~10 frames of lock / line
# check / spawn, during which 0x40-0x42 still hold the piece that just locked (measured 2026-10-03: 1 for
# whole falls, 2->3->5..8->1 at each lock). Orientation < 19 alone counted a locked piece as in play.
RAM_PLAY_STATE, PLAY_STATE_ACTIVE = 0x48, 1
# Where the screen shows the text board (measured 2026-10-04, every frame outside the clear animation):
# with the piece in play (1) or the lock drawn (3), always once the play area has held still for 2
# frames, and not always before. A decision is observed only at 1 or 3, 2 frames still.
PLAY_STATE_LOCK_DRAWN, SETTLE_FRAMES = 3, 2

# (row, col) offsets of each orientation's four cells from (y, x) at 0x41/0x40, measured (2026-10-02).
PIECE_CELLS = {
    0: ((-1, 0), (0, -1), (0, 0), (0, 1)),    # Tu
    1: ((-1, 0), (0, 0), (0, 1), (1, 0)),     # Tr
    2: ((0, -1), (0, 0), (0, 1), (1, 0)),     # Td
    3: ((-1, 0), (0, -1), (0, 0), (1, 0)),    # Tl
    4: ((-1, 0), (0, 0), (1, -1), (1, 0)),    # Jl
    5: ((-1, -1), (0, -1), (0, 0), (0, 1)),   # Ju
    6: ((-1, 0), (-1, 1), (0, 0), (1, 0)),    # Jr
    7: ((0, -1), (0, 0), (0, 1), (1, 1)),     # Jd
    8: ((0, -1), (0, 0), (1, 0), (1, 1)),     # Zh
    9: ((-1, 1), (0, 0), (0, 1), (1, 0)),     # Zv
    10: ((0, -1), (0, 0), (1, -1), (1, 0)),   # O
    11: ((0, 0), (0, 1), (1, -1), (1, 0)),    # Sh
    12: ((-1, 0), (0, 0), (0, 1), (1, 1)),    # Sv
    13: ((-1, 0), (0, 0), (1, 0), (1, 1)),    # Lr
    14: ((0, -1), (0, 0), (0, 1), (1, -1)),   # Ld
    15: ((-1, -1), (-1, 0), (0, 0), (1, 0)),  # Ll
    16: ((-1, 1), (0, -1), (0, 0), (0, 1)),   # Lu
    17: ((-2, 0), (-1, 0), (0, 0), (1, 0)),   # Iv
    18: ((0, -2), (0, -1), (0, 0), (0, 1)),   # Ih
}

DROP_ROWS = 4


class Game:
    def __init__(self, seed: int):
        """seed seeds gym's np_random, which draws the two NES RNG bytes. NES Tetris advances its RNG
        every frame, so the seed fixes the opening pieces and the deal after them follows the game's
        frame count: games on one seed share the first 2-3 pieces, then diverge."""
        self.env = TetrisEnv(deterministic=False)
        self.env.reset(seed=seed)
        self.ram = self.env.ram
        self.done = False
        self.inputs: list[int] = []        # the button byte held on every frame stepped
        self.last_action: str | None = None
        self.text, self.still = self.board_text(), 0   # frames the play area has held still

    def press(self, buttons: int, frames: int = 1):
        # A top-out can land mid-decision; nes-py refuses to step a done env.
        for _ in range(frames):
            if self.done:
                return
            self.inputs.append(buttons)
            _, _, terminated, truncated, _ = self.env.step(buttons)
            self.done = terminated or truncated
            text = self.board_text()
            self.still = self.still + 1 if text == self.text else 0
            self.text = text

    @property
    def piece_in_play(self) -> bool:
        return int(self.ram[RAM_PLAY_STATE]) == PLAY_STATE_ACTIVE and int(self.ram[RAM_ORIENT]) < 19

    @property
    def game_over(self) -> bool:
        return self.done or bool(self.ram[RAM_GAME_OVER])

    def settled(self) -> np.ndarray:
        return self.ram[0x0400:0x04C8].reshape(20, 10) != 239

    def falling_cells(self) -> list[tuple[int, int]]:
        if not self.piece_in_play:
            return []
        y, x, o = int(self.ram[RAM_Y]), int(self.ram[RAM_X]), int(self.ram[RAM_ORIENT])
        return [(y + dy, x + dx) for dy, dx in PIECE_CELLS[o]]

    def board_text(self) -> str:
        grid = [["#" if c else "." for c in row] for row in self.settled()]
        for r, c in self.falling_cells():
            if 0 <= r < 20 and 0 <= c < 10:
                grid[r][c] = "@"
        return "\n".join("".join(row) for row in grid)

    def info(self) -> dict:
        i = self.env._get_info()
        return {
            "piece": _PIECE_ORIENTATION_TABLE[self.ram[RAM_ORIENT]] if self.piece_in_play else None,
            "next_piece": i["next_piece"],
            "lines": i["number_of_lines"],
            "score": i["score"],
            "pieces": sum(i["statistics"].values()),
            "board_height": int(i["board_height"]),
        }

    def inputs_since(self, start: int) -> list[list[int]]:
        """The buttons held on frames start.. as run-length [[buttons, frames], ...]."""
        runs: list[list[int]] = []
        for b in self.inputs[start:]:
            if runs and runs[-1][0] == b:
                runs[-1][1] += 1
            else:
                runs.append([b, 1])
        return runs


def state_text(board: str, rec: dict) -> str:
    """Piece names are the shape letter, not gym-tetris's orientation codes."""
    return (f"Board (10 columns x 20 rows, top row first):\n{board}\n\n"
            f"Falling piece: {rec['piece'][0]}\nNext piece: {(rec['next_piece'] or '?')[0]}\n"
            f"Lines cleared: {rec['lines']}\nScore: {rec['score']}")


def settle(game: Game, frames: int, max_frames: int) -> int:
    """Run on until the play area has held still for SETTLE_FRAMES frames with the piece in play or its
    lock drawn: only then does the screen show the text board."""
    while (not game.game_over and frames < max_frames
           and not (game.still >= SETTLE_FRAMES
                    and int(game.ram[RAM_PLAY_STATE]) in (PLAY_STATE_ACTIVE, PLAY_STATE_LOCK_DRAWN))):
        game.press(0, 1)
        frames += 1
    return frames


def apply(game: Game, action: str, lag: int, max_frames: int) -> int:
    """Press the button and run the game for max(lag, until the play area changes) frames, or until the
    piece locks, then settle (a lock settles where it is drawn, playState 3); return the frames used. A
    blocked move runs on until gravity changes the board. drop is its own rule (operator): hold Down until
    the piece is DROP_ROWS rows lower or locks, whichever is first; after a lock the next decision is on
    the new piece at the top."""
    if action == "drop":
        frames = drop(game, max_frames)
    else:
        before = game.board_text()
        frames = 0
        if action in ("left", "right", "rotate"):
            game.press(BUTTON["A"] if action == "rotate" else BUTTON[action], 1)
            frames = 1
        while not game.game_over and game.board_text() == before and frames < max_frames:
            game.press(0, 1)
            frames += 1
    while not game.game_over and game.piece_in_play and frames < lag:
        game.press(0, 1)
        frames += 1
    game.last_action = action
    return settle(game, frames, max_frames)


def drop(game: Game, max_frames: int) -> int:
    y0, settled0 = int(game.ram[RAM_Y]), game.settled().copy()
    frames = 0
    if game.last_action == "drop":
        game.press(0, 1)      # NES Tetris soft-drops only on a fresh Down press
        frames = 1
    while (not game.game_over and frames < max_frames and game.piece_in_play
           and (game.settled() == settled0).all() and int(game.ram[RAM_Y]) < y0 + DROP_ROWS):
        game.press(BUTTON["down"], 1)
        frames += 1
    return frames


def advance_to_piece(game: Game, max_frames: int) -> int:
    """Between pieces (lock, line-clear animation, spawn) nothing is decided."""
    frames = 0
    while not game.game_over and not game.piece_in_play and frames < max_frames:
        game.press(0, 1)
        frames += 1
    game.last_action = None
    return settle(game, frames, max_frames)
