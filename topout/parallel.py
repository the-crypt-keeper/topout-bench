"""--games N --parallel P: N games of one seed against one engine, at most P at a time, each its own process and
journal at DIR/<i>/.

A pool: P games start at once, and whenever one ends the next index starts, until N have played. Long and short games
then share the engine's slots instead of the short ones' slots idling until the longest game of a batch ends.

The parent re-runs the calling script once per game without --games / --parallel, prefixes every child line with
"[g<i>]", passes ERROR lines through, and prints the progress contract itself: "[done/total] games running=R
queued=Q" with total = N x max_pieces and done the pieces locked so far (a running game's "[p/max]" is the piece in
play, so p - 1; a finished game counts all its max_pieces; a queued game counts 0). done never decreases and reaches
total only when every game has ended. Its exit code is the worst child's.
"""
from __future__ import annotations

import re
import subprocess
import sys
import threading
from pathlib import Path

_PROGRESS = re.compile(r"^\[(\d+)/(\d+)\]")
_POOL_ARGS = ("--games", "--parallel", "--output")


def fan_out(games: int, parallel: int, output: str, max_pieces: int) -> int:
    args, skip = [], False
    for arg in sys.argv[1:]:                         # this invocation's args minus the pool's own
        if skip:
            skip = False
            continue
        if arg in _POOL_ARGS:
            skip = True
            continue
        args.append(arg)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    pieces, started, ended = [0] * games, [False] * games, [False] * games
    lock, slots = threading.Lock(), threading.Semaphore(min(parallel, games))
    total = games * max_pieces
    shown = [-1]

    def progress():
        done = sum(max_pieces if ended[i] else max(0, min(pieces[i], max_pieces) - 1) for i in range(games))
        if done > shown[0]:
            shown[0] = done
            running = sum(s and not e for s, e in zip(started, ended))
            print(f"[{done}/{total}] games running={running} queued={games - sum(started)}", flush=True)

    codes: list = [None] * games

    def pump(i, proc):
        for line in proc.stdout:
            line = line.rstrip("\n")
            m = _PROGRESS.match(line)
            with lock:
                if m:
                    pieces[i] = max(pieces[i], int(m.group(1)))
                print(f"[g{i}] {line}", flush=True)
                if m:
                    progress()
        # each game is reaped by its own thread, so an exit counts when it happens, not behind a lower index
        code = proc.wait()
        with lock:
            codes[i], ended[i] = code, True
            print(f"[g{i}] exited {code}", flush=True)
            progress()
        slots.release()

    threads = []
    for i in range(games):                           # in index order, each as a slot frees
        slots.acquire()
        with lock:
            started[i] = True
            print(f"[g{i}] started", flush=True)
        proc = subprocess.Popen([sys.executable, sys.argv[0], *args, "--output", str(out / str(i))],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        t = threading.Thread(target=pump, args=(i, proc), daemon=True)
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    bad = [c for c in codes if c != 0]
    if bad:
        print(f"ERROR: {len(bad)} of {games} games exited non-zero: {codes}", flush=True)
    return max(bad, key=abs) if bad else 0
