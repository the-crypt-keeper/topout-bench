"""--parallel N: N games of one seed against one engine, each its own process and journal at DIR/<i>/.

The parent re-runs the calling script N times without --parallel, prefixes every child line with
"[g<i>]", passes ERROR lines through, and prints the progress contract itself: "[done/total] games
running=R" with total = N x max_pieces and done the pieces locked so far (a running game's "[p/max]" is
the piece in play, so p - 1; a finished game counts all its max_pieces). done never decreases and reaches
total only when every game has ended. Its exit code is the worst child's.
"""
from __future__ import annotations

import re
import subprocess
import sys
import threading
from pathlib import Path

_PROGRESS = re.compile(r"^\[(\d+)/(\d+)\]")


def fan_out(n: int, output: str, max_pieces: int) -> int:
    args, skip = [], False
    for arg in sys.argv[1:]:                         # this invocation's args minus --parallel/--output
        if skip:
            skip = False
            continue
        if arg in ("--parallel", "--output"):
            skip = True
            continue
        args.append(arg)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    pieces, ended, lock = [0] * n, [False] * n, threading.Lock()
    total = n * max_pieces
    shown = [-1]

    def progress():
        done = sum(max_pieces if ended[i] else max(0, min(pieces[i], max_pieces) - 1) for i in range(n))
        if done > shown[0]:
            shown[0] = done
            print(f"[{done}/{total}] games running={n - sum(ended)}", flush=True)

    codes: list = [None] * n

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

    procs = [subprocess.Popen([sys.executable, sys.argv[0], *args, "--output", str(out / str(i))],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
             for i in range(n)]
    threads = [threading.Thread(target=pump, args=(i, pr), daemon=True) for i, pr in enumerate(procs)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    bad = [c for c in codes if c != 0]
    if bad:
        print(f"ERROR: {len(bad)} of {n} games exited non-zero: {codes}", flush=True)
    return max(bad, key=abs) if bad else 0
