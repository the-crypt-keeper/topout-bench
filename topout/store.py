"""Finished games digested for topout-explore, and the cache that makes a second start fast.

Contract. Reads journals; writes only under the cache directory. A digest holds numbers and boards, not the model's
words: a piece's replies are read from the journal when asked for (replies). Deleting the cache loses nothing.

A digest is reused while the journal's path, size and modification time are unchanged and the code that computes it is
(its sources are part of the key). Games still running are not digested.
"""
from __future__ import annotations

import hashlib
import os
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from topout import decisions, journals, ledger
from topout.calls import call_text, is_reply

_HERE = Path(__file__).parent
_CODE = hashlib.sha256(b"".join((_HERE / f).read_bytes() for f in
                                ("store.py", "ledger.py", "decisions.py", "calls.py", "journals.py"))).hexdigest()


def digest(path: Path) -> dict | None:
    summary = journals.summary(path)
    if summary is None:
        return None
    run = journals.run_record(path)
    records = [r for r in journals.records(path) if r.get("kind") in ("tick", "reply", "capped")]
    pieces: dict[int, dict] = {}
    for r in records:
        if r.get("no_piece"):
            continue
        p = pieces.get(r["pieces"])
        if p is None:
            p = pieces[r["pieces"]] = dict(piece=r["pieces"], name=None, board=None, tokens=[0, 0], replies=[0, 0],
                                           presses=[], truncated=0, nudged=0)
        if r["kind"] == "tick":
            if p["board"] is None:
                board, p["overlap"] = ledger.spawn_board(r)
                g = ledger.regions(board)
                p |= dict(board=board.replace("\n", ""), name=(r.get("piece") or "?")[0], lines=r["lines"],
                          live=g["live_rows"], dead=g["dead_rows"], sealed=g["sealed"])
            p["presses"].append(r["action"])
        if is_reply(r):
            ph = 0 if r["conv_turn"] == 0 else 1
            p["tokens"][ph] += int((r["usage"] or {}).get("completion_tokens") or 0)
            p["replies"][ph] += 1
            p["truncated"] += r["kind"] == "capped"
            p["nudged"] += r.get("parse_error") == "no tool call"
    out = [pieces[k] for k in sorted(pieces) if pieces[k]["board"] is not None]
    if not out:
        return None
    for a, b in zip(out, out[1:] + [None]):
        a["cleared"] = (b["lines"] if b else summary["lines"]) - a["lines"]
        # no change is read into an overlapping spawn, or out of one: the falling piece's cells are in that board
        a["sealing"] = b["sealed"] - a["sealed"] if b and not (a["overlap"] or b["overlap"]) else None
    return {"path": str(path), "id": journals.game_id(path), "sampler": run.get("sampler") or {},
            "model": run.get("model"), "survived": summary["termination"] == "max_pieces", "lines": summary["lines"],
            "turns": run["max_pieces"], "burden": ledger.burden([p["dead"] for p in out], run["max_pieces"]),
            "pieces": out, "phases": decisions.count(records)}


def _cached(args: tuple[Path, Path]) -> dict | None:
    path, cache = args
    st = path.stat()
    key = hashlib.sha256(f"{path.resolve()}|{st.st_size}|{st.st_mtime_ns}|{_CODE}".encode()).hexdigest()
    file = cache / f"{key}.pickle"
    try:
        return pickle.loads(file.read_bytes())
    except Exception:                       # absent, or cut off by an interrupted write: computed again
        pass
    d = digest(path)
    if d is not None:
        tmp = file.with_suffix(f".{os.getpid()}.tmp")
        tmp.write_bytes(pickle.dumps(d))
        tmp.replace(file)
    return d


def load(paths: list[Path], cache: Path, workers: int | None = None) -> list[dict | None]:
    """A digest per journal, in order; None for a game still running."""
    cache.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers) as pool:
        return list(pool.map(_cached, [(p, cache) for p in paths], chunksize=4))


def replies(path: Path, piece: int) -> list[dict]:
    """One piece's replies in order, read from the journal: the model's words, its calls, the presses they made and the
    text it was shown before each."""
    out: list[dict] = []
    for r in journals.records(path):
        if r.get("pieces") != piece or r.get("kind") not in ("tick", "reply", "capped") or r.get("no_piece"):
            continue
        if is_reply(r):
            out.append({"phase": 0 if r["conv_turn"] == 0 else 1, "kind": r["kind"], "reset": r.get("conv_reset"),
                        "tokens": int((r["usage"] or {}).get("completion_tokens") or 0), "reason": r.get("reason"),
                        "reasoning": r.get("reasoning") or "", "content": r.get("content") or "",
                        "nudged": r.get("parse_error") == "no tool call",
                        "calls": [call_text(c) for c in r.get("tool_calls") or []], "presses": []})
        if r["kind"] == "tick":
            press = {"action": r["action"], "board": r["board"], "frames": r["frames"],
                     "lost": r.get("parse_error") == "capped"}
            if out:
                out[-1]["presses"].append(press)
    return out
