"""Finding, reading and naming journals: the one read path for every script that reads a game.

Contract. Resolves paths and arms to journal files and reads their records; computes nothing over a game.

A journal is a `journal.jsonl` as topout-play writes it, or a gzipped one as the results dataset holds it
(`<id>.jsonl.gz`, topout-publish). The two read the same: every reader here takes either.

A path is a journal, a game directory holding a journal.jsonl, a --parallel output holding <i>/journal.jsonl, or any
directory, searched recursively for both kinds. An arm is NAME=PATH[,PATH...]: its games pool together.
"""
from __future__ import annotations

import gzip
import json
from collections.abc import Iterator
from pathlib import Path

PLAIN = "journal.jsonl"
GZ = ".jsonl.gz"


def is_journal(p: Path) -> bool:
    return p.name == PLAIN or p.name.endswith(GZ)


def find(paths: list[str | Path]) -> list[Path]:
    out: list[Path] = []
    for p in map(Path, paths):
        if p.is_file() and is_journal(p):
            out.append(p)
        elif (p / PLAIN).is_file():
            out.append(p / PLAIN)
        elif p.is_dir():
            hits = sorted([*p.rglob(PLAIN), *p.rglob(f"*{GZ}")], key=_order)
            if not hits:
                raise SystemExit(f"{p}: no journal ({PLAIN} or *{GZ}) under it")
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


def _lines(path: Path) -> Iterator[str]:
    with (gzip.open(path, "rt") if path.name.endswith(GZ) else path.open()) as f:
        for line in f:
            if line.strip():
                yield line


def records(path: Path) -> Iterator[dict]:
    """Every record in order. A running game's last line may be cut off mid-write: a last line that does not parse ends
    the stream; one that does not parse anywhere else raises."""
    pending = None
    for line in _lines(path):
        if pending is not None:
            try:
                yield json.loads(pending)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}: a record that does not parse, before the last line: {e}") from None
        pending = line
    if pending is not None:
        try:
            yield json.loads(pending)
        except json.JSONDecodeError:
            return


def run_record(path: Path) -> dict:
    """The run record, or {} when the journal has none yet."""
    try:
        first = next(_lines(path))
        rec = json.loads(first)
    except (StopIteration, json.JSONDecodeError):
        return {}
    return rec if rec.get("kind") == "run" else {}


def summary(path: Path) -> dict | None:
    """How the game ended (its last record, a summary), or None while it is still running. Parses only the last line."""
    last = None
    for last in _lines(path):
        pass
    try:
        rec = json.loads(last) if last is not None else None
    except json.JSONDecodeError:
        return None
    return rec if rec and rec.get("kind") == "summary" else None


def game_id(path: Path) -> str:
    """The game's directory for a journal.jsonl (with its --parallel index: unique, and readable in a table); a
    published journal's own id."""
    return str(path.parent) if path.name == PLAIN else path.name.removesuffix(GZ)
