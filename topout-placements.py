"""topout-placements: each placement as a ledger event, per game and pooled per arm.

    python topout-placements.py --arm NAME=PATH[,PATH...] [--arm ...] [--window 24]

Contract. Reads journals (an arm is one or more journals, game directories or directories of games, as every topout
analysis takes them), through topout/ledger.py, which defines the decomposition and the placement event. Writes
nothing; prints the tables below. The intervals are its only inference.

A placement of piece p is the change from the board p spawned into to the board p+1 spawned into: sealed cells added
(sealing) or removed (dig-out), and lines cleared. Occluded boards (top-out) end a game's events.

Per game, over the whole game and over placements 1..--window (which every game played, so game length cannot leak
in):
  seal%    placements that added sealed cells
  size     sealed cells per sealing placement
  dig%     placements that removed sealed cells
  clears   lines cleared
  top3     share of all sealing carried by the three biggest sealing placements (low: erosion; high: catastrophes)
  biggest  the largest single sealing placement, and its piece

Per arm, pooled over placements: seal rate, sealed cells per placement, dig-out rate, lines per placement. Intervals
are 95% from a bootstrap that resamples whole games (2000 draws, seed 0): placements of one game are not independent.
"""
from __future__ import annotations

import argparse
import random

from topout import journals, ledger


def game_stats(ev: list[dict]) -> dict:
    n = len(ev)
    seal = [e["sealed"] for e in ev if e["sealed"] > 0]
    big = max(ev, key=lambda e: e["sealed"])
    return {"n": n, "seal": len(seal) / n, "size": sum(seal) / max(1, len(seal)), "cells": sum(seal),
            "dig": sum(e["sealed"] < 0 for e in ev) / n, "clears": sum(e["lines"] for e in ev),
            "top3": sum(sorted(seal, reverse=True)[:3]) / sum(seal) if seal else 0.0,
            "biggest": big["sealed"], "at": big["piece"]}


def pooled(evs: list[list[dict]]) -> tuple[int, float, float, float, float]:
    ev = [e for g in evs for e in g]
    n = len(ev)
    return (n, sum(e["sealed"] > 0 for e in ev) / n, sum(max(e["sealed"], 0) for e in ev) / n,
            sum(e["sealed"] < 0 for e in ev) / n, sum(e["lines"] for e in ev) / n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=PATH[,PATH...]")
    ap.add_argument("--window", type=int, default=24, help="placements 1..N for the shared-window table")
    a = ap.parse_args()

    arms: dict[str, list[str]] = {}
    ev: dict[str, list[dict]] = {}
    lines_of: dict[str, int] = {}
    for name, paths in journals.arms(a.arm).items():
        arms[name] = []
        for p in paths:
            rows = ledger.piece_rows(p)
            if len(rows) < 2:
                continue
            g = journals.game_id(p)
            ev[g], lines_of[g] = ledger.placements(rows), rows[-1]["lines"]
            arms[name].append(g)
    order = sorted(ev, key=lambda g: -lines_of[g])
    width = max(len(g) for g in order)

    head = (f"{'game':{width}s} {'plc':>3s} {'seal%':>5s} {'size':>4s} {'cells':>5s} {'dig%':>4s} {'clears':>6s} "
            f"{'top3':>4s} {'biggest':>11s}")
    for title, sel in (("whole game", lambda e: e),
                       (f"placements 1-{a.window}", lambda e: [x for x in e if x["piece"] <= a.window])):
        print(f"\n{title}\n{head}")
        for g in order:
            chosen = sel(ev[g])
            if not chosen:
                continue
            s = game_stats(chosen)
            print(f"{g:{width}s} {s['n']:3d} {s['seal']:5.0%} {s['size']:4.1f} {s['cells']:5d} {s['dig']:4.0%} "
                  f"{s['clears']:6d} {s['top3']:4.0%} {s['biggest']:4d} ({s['at']:3d})")

    rng = random.Random(0)
    print("\npooled per arm, 95% intervals over whole games (bootstrap, 2000 draws)")
    print(f"{'arm':44s} {'games':>5s} {'plc':>4s} {'seal%':>5s} {'cells/plc':>20s} {'dig%':>4s} {'lines/plc':>22s}")
    for name, gs in arms.items():
        if not gs:
            print(f"{name:44s} no games")
            continue
        n, sr, cells, dig, lines = pooled([ev[g] for g in gs])
        bs = [pooled([ev[rng.choice(gs)] for _ in gs]) for _ in range(2000)]

        def ci(i):
            v = sorted(b[i] for b in bs)
            return f"[{v[50]:.2f}, {v[1949]:.2f}]"
        print(f"{name:44s} {len(gs):5d} {n:4d} {sr:5.0%} {cells:5.2f} {ci(2):>14s} {dig:4.0%} {lines:6.3f} {ci(4):>14s}")


if __name__ == "__main__":
    main()
