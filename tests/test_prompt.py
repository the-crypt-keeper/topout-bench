"""The v1 text and the verb tools are the bytes every recorded v1 game was played under, so new games pool with
them. The fixture is a recorded run record (tests/fixtures/v1-run.json, from a 2026-10-13 game)."""
from __future__ import annotations

import json
from pathlib import Path

from topout.prompt import INSTRUCTIONS, NUDGE, TOOLS

FIXTURE = json.loads((Path(__file__).parent / "fixtures" / "v1-run.json").read_text())


def test_instructions_are_the_recorded_v1_text():
    assert INSTRUCTIONS == FIXTURE["instructions"]


def test_tools_are_the_recorded_verb_tools():
    assert json.dumps(TOOLS, sort_keys=True) == json.dumps(FIXTURE["tools"], sort_keys=True)
    assert [t["function"]["name"] for t in TOOLS] == ["start", "left", "right", "rotate", "drop", "wait"]


def test_nudge_is_the_recorded_text():
    assert FIXTURE["no_tool_call"] == "wait pressed, then a user turn: " + NUDGE
