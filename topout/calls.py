"""A reply's tool calls read against the verb tools, and which journal records are replies.

Shared by topout-play (which acts on the calls) and the analysis (which counts them from the journal), so both read a
call the same way. Reads; decides nothing about play.
"""
from __future__ import annotations

import json

from topout.prompt import BUTTONS


def button_of(call: dict) -> tuple[str | None, str | None]:
    """The button a call presses, or None and why (a refusal the model is told about)."""
    name = (call.get("function") or {}).get("name")
    return (name, None) if name in BUTTONS else (None, f"no button named {name!r}")


def extra_args(call: dict) -> dict:
    """Arguments a verb was given (it declares none): ignored in play, counted."""
    raw = (call.get("function") or {}).get("arguments")
    try:
        args = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except json.JSONDecodeError:
        return {"<not JSON>": raw}
    if not isinstance(args, dict):
        return {"<not an object>": args}
    return args


def call_text(call: dict) -> str:
    extra = extra_args(call)
    return f"{(call.get('function') or {}).get('name')}({', '.join(f'{k}={v}' for k, v in extra.items())})"


def is_reply(rec: dict) -> bool:
    """One record per reply the engine returned: a `capped` record (cut off), a nudge's tick, a reply's first pressed
    tick, or a `reply` record (pressed nothing). A lost decision's tick repeats the last capped attempt's reply, which
    its `capped` record already holds, so it is not one."""
    return "usage" in rec and "conv_turn" in rec and rec.get("parse_error") != "capped"
