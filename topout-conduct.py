"""topout-conduct: how a model handled the harness, per game: replies, calls, presses, nudges, caps, refusals.

    python topout-conduct.py --arm NAME=PATH[,PATH...] [--arm ...] [--csv out.csv]

Contract. Reads journals (running games up to their last record) and counts from their records alone, never from a
summary: the counts are the same for a journal of any driver revision that wrote these records. Writes only the CSV
named by --csv. Prints one row per game and a total per arm. Counts; infers nothing.

Per game:
  replies     replies the engine returned (topout/calls.py is_reply)
  decisions   ticks that pressed or waited on a piece (every press, plus a nudge's or a lost decision's wait)
  nudges      replies with no tool call: wait pressed
  capped      replies cut off by max_tokens (retried; a context-full cut is a reset, not counted here)
  lost        decisions lost to max_tokens: every attempt cut off, wait pressed, trajectory rebooted
  resets      fresh conversations because the context was full (a 400 or a cut at the context)
  pressless   replies with tool calls that pressed nothing
  calls       tool calls in replies that had any; per_reply is their count per such reply
  repeated    adjacent calls in a reply naming the same button
  refused     calls refused: no such button, a button before start, start once running
  unpressed   calls left after the piece had locked
  extra_args  calls carrying arguments (the verbs declare none)
  actions     presses per button, the waits of nudges and lost decisions included
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

from topout import journals
from topout.calls import button_of, extra_args, is_reply

COUNTS = ("replies", "decisions", "nudges", "capped", "lost", "resets", "pressless", "calls", "repeated", "refused",
          "unpressed", "extra_args")


def conduct(path: Path) -> dict:
    recs = [json.loads(line) for line in path.open()]
    n = collections.Counter({k: 0 for k in COUNTS})
    actions, per_reply = collections.Counter(), collections.Counter()
    pressed: dict[int, list[int]] = collections.defaultdict(list)     # reply -> index in recs of each pressed tick
    for i, r in enumerate(recs):
        if r.get("kind") == "tick" and not r.get("parse_error") and r.get("action") and "reply" in r:
            pressed[r["reply"]].append(i)
    running = False
    for i, r in enumerate(recs):
        kind = r.get("kind")
        if kind == "tick" and r.get("action"):
            n["decisions"] += 1
            actions[r["action"]] += 1
        if kind == "tick" and r.get("parse_error") == "capped":
            n["lost"] += 1
        if not is_reply(r):
            continue
        n["replies"] += 1
        if r.get("conv_reset"):
            running = False                                  # a fresh paused user turn
            n["resets"] += r["conv_reset"] == "context"
        if kind == "capped":
            n["capped"] += r.get("reason") != "context"
            continue
        if r.get("parse_error") == "no tool call":
            n["nudges"] += 1
            running = True                                   # the player resumes the game on a nudge's wait
            continue
        calls = r.get("tool_calls") or []
        n["pressless"] += kind == "reply"
        per_reply[len(calls)] += 1
        n["calls"] += len(calls)
        n["extra_args"] += sum(1 for c in calls if extra_args(c))
        names = [button_of(c)[0] for c in calls]
        n["repeated"] += sum(1 for x, y in zip(names, names[1:]) if x is not None and x == y)
        ticks = pressed.get(r["reply"], [])
        by_call = {recs[t]["call"]: t for t in ticks}
        locked = False
        for k, (name, why) in enumerate(button_of(c) for c in calls):
            if locked:
                n["unpressed"] += 1
            elif k in by_call:
                # locked if the next tick has no piece in play or a new one (an older driver's tick could run on
                # through the spawn), or the game ended
                nxt = next((x for x in recs[by_call[k] + 1:] if x.get("kind") in ("tick", "summary")), None)
                locked = (nxt is None or nxt.get("kind") == "summary" or bool(nxt.get("no_piece"))
                          or nxt.get("pieces") != recs[by_call[k]]["pieces"])
            elif why:
                n["refused"] += 1
            elif name == "start":
                n["refused"] += running
                running = True
            else:
                n["refused"] += 1                           # a button before start
    return {"game": journals.game_id(path), **n, "per_reply": dict(sorted(per_reply.items())),
            "actions": dict(sorted(actions.items()))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", required=True, help="NAME=PATH[,PATH...]")
    ap.add_argument("--csv", type=Path, help="one row per game")
    a = ap.parse_args()
    rows = []
    head = " ".join(f"{k:>9s}" for k in COUNTS)
    for name, paths in journals.arms(a.arm).items():
        games = [conduct(p) for p in paths]
        print(f"\n{name}: {len(games)} games\n{'game':>6s} {head}   per_reply")
        for g in games:
            print(f"{Path(g['game']).name:>6s} " + " ".join(f"{g[k]:9d}" for k in COUNTS) + f"   {g['per_reply']}")
        print(f"{'total':>6s} " + " ".join(f"{sum(g[k] for g in games):9d}" for k in COUNTS))
        rows += [{"arm": name, **g, "per_reply": json.dumps(g["per_reply"]), "actions": json.dumps(g["actions"])}
                 for g in games]
    if a.csv:
        with a.csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
