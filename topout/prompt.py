"""What the model is shown: the v1 instructions, the board text and the verb tools, byte for byte the text every v1
game was played under.

Contract. Text and tool definitions only; no game logic, no I/O.

v1 is the only text topout plays (operator 2026-11-13): no prompt variant beat it (threads/tetris-v5,
2026-10-11). It is assembled exactly as the reasonlab harness assembled it (the chat text up to its
one-button closing, then the tool loop's closing), so new games pool with the recorded ones;
tests/test_prompt.py holds it to a recorded journal's run record.
"""
from __future__ import annotations

# Operator, 2026-10-02: make the goal unmistakable — horizontal lines, as many at a time as possible.
_CHAT_INSTRUCTIONS = (
    "You are playing Tetris. The board is 10 columns wide and 20 rows tall, "
    "drawn top row first: '#' is a settled block, '@' is the falling piece, "
    "'.' is empty.\n"
    "GOAL: complete HORIZONTAL LINES. A line is one full row of the board: all "
    "10 cells across, from the left wall to the right wall, filled with blocks. "
    "Every completed line disappears and scores. Completing several lines with "
    "one piece scores far more than one at a time; four lines at once is the "
    "best move in the game. The game ends when the blocks stack up to the top, "
    "so keep the stack low and flat with no holes.\n"
    "To fill a row you must steer each piece sideways to where it fits: left "
    "and right move it one column, rotate turns it. The piece falls straight "
    "down on its own; drop only makes it fall faster and wait does nothing, so "
    "neither changes where it lands.\n"
    "Each tick you press one button. Choose the button that moves this piece "
    "toward the place where it completes the most horizontal lines."
)
_CHAT_CLOSING = "Each tick you press one button."
_TOOL_CLOSING = ("The game is paused on this piece: press start to resume it, then press buttons to move "
                 "the piece toward the place where it completes the most horizontal lines. You may press "
                 "several buttons in one reply: they run in order, one after another, each taking game time "
                 "while the piece keeps falling, and each result is the board after that press. Every reply "
                 "must be tool calls.")

INSTRUCTIONS = _CHAT_INSTRUCTIONS[:_CHAT_INSTRUCTIONS.index(_CHAT_CLOSING)] + _TOOL_CLOSING

BUTTONS = {
    "start": "resume the paused game; the buttons do nothing until it is resumed",
    "left": "move the falling piece one column to the left",
    "right": "move the falling piece one column to the right",
    "rotate": "rotate the falling piece a quarter turn clockwise",
    "drop": "drop the falling piece up to 4 rows straight down (it stops early if it lands)",
    "wait": "press nothing this tick; the piece keeps falling",
}

# The decisions are the tools: start() left() right() rotate() drop() wait(), none taking arguments.
TOOLS = [{"type": "function", "function": {
    "name": name, "description": desc[0].upper() + desc[1:] + ".",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False}}}
    for name, desc in BUTTONS.items()]

NUDGE = "Your reply had no tool call, so wait was pressed. Reply with tool calls only."


def state_text(board: str, rec: dict) -> str:
    """The board and counters as the model is shown them. Piece names are the shape letter, not gym-tetris's
    orientation codes."""
    return (f"Board (10 columns x 20 rows, top row first):\n{board}\n\n"
            f"Falling piece: {rec['piece'][0]}\nNext piece: {(rec['next_piece'] or '?')[0]}\n"
            f"Lines cleared: {rec['lines']}\nScore: {rec['score']}")
