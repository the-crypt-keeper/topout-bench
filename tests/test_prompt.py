"""The v1 text, the board text and the verb tools are the bytes every recorded v1 game was played under, so new games
pool with them. The fixtures are a recorded run record and that game's first press (tests/fixtures/v1-run.json,
v1-tick.json, from a 2026-10-13 game)."""
from __future__ import annotations

import json
from pathlib import Path

from topout.prompt import INSTRUCTIONS, NUDGE, TOOLS, state_text

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "v1-run.json").read_text())
TICK = json.loads((Path(__file__).parent / "fixtures" / "v1-tick.json").read_text())


def test_board_text_is_what_the_recorded_game_was_shown():
    """The first press follows start(), whose result is "Resumed." and the board."""
    assert TICK["text"] == "Resumed.\n\n" + state_text(TICK["board"], TICK)


def test_instructions_are_the_recorded_v1_text():
    assert INSTRUCTIONS == FIXTURE["instructions"]


def test_tools_are_the_recorded_verb_tools():
    assert json.dumps(TOOLS, sort_keys=True) == json.dumps(FIXTURE["tools"], sort_keys=True)
    assert [t["function"]["name"] for t in TOOLS] == ["start", "left", "right", "rotate", "drop", "wait"]


def test_nudge_is_the_recorded_text():
    assert FIXTURE["no_tool_call"] == "wait pressed, then a user turn: " + NUDGE
