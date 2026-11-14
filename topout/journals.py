"""Finding journals and naming games, for the analysis scripts.

A path is a journal.jsonl, a game directory holding one, a --parallel output holding <i>/journal.jsonl, or
any directory, searched recursively. An arm is NAME=PATH[,PATH...]: its games pool together.
"""
from __future__ import annotations

import json
from pathlib import Path


def find(paths: list[str | Path]) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_file() and p.name == "journal.jsonl":
            out.append(p)
        elif (p / "journal.jsonl").is_file():
            out.append(p / "journal.jsonl")
        elif p.is_dir():
            hits = sorted(p.rglob("journal.jsonl"), key=_order)
            if not hits:
                raise SystemExit(f"{p}: no journal.jsonl under it")
            out += hits
        else:
            raise SystemExit(f"{p}: not a journal, a game directory or a directory of games")
    return out


def _order(path: Path):
    """--parallel games in index order (2 before 10), everything else by path."""
    name = path.parent.name
    return (str(path.parent.parent), int(name) if name.isdigit() else -1, str(path))


def arms(specs: list[str]) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for spec in specs:
        name, sep, paths = spec.partition("=")
        if not sep or not name or not paths:
            raise SystemExit(f"--arm {spec!r}: expected NAME=PATH[,PATH...]")
        out[name] = find(paths.split(","))
    return out


def run_record(path: Path) -> dict:
    with path.open() as f:
        first = f.readline()
    try:
        rec = json.loads(first)
    except json.JSONDecodeError:
        return {}
    return rec if rec.get("kind") == "run" else {}


def game_id(path: Path) -> str:
    """The game's directory, plus its --parallel index: unique, and readable in a table."""
    return str(path.parent)
