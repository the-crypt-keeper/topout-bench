"""topout-replies counts every generated reply (pressing, nudged, cut off) once, under its piece's first or later turn."""
from __future__ import annotations

import subprocess
import sys

from tests.conftest import ROOT, play


def test_every_reply_counted_once_and_split_by_turn(fake_engine, tmp_path):
    r, rows = play(fake_engine(length_rate=0.3, no_call_rate=0.3, seed=2), tmp_path / "g", pieces=4)
    assert r.returncode == 0, r.stdout[-2000:]
    generated = [x for x in rows if "usage" in x and "conv_turn" in x]
    assert {x["kind"] for x in generated} >= {"tick", "capped"}      # cut-off replies are in the sample
    first = sum(1 for x in generated if x["conv_turn"] == 0)
    out = subprocess.run([sys.executable, str(ROOT / "topout-replies.py"), "--arm", f"fake={tmp_path / 'g'}"],
                         capture_output=True, text=True, timeout=60, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    counts = {line.split()[1]: int(line.split()[3]) for line in out.stdout.splitlines() if line.startswith("fake ")}
    assert counts == {"first": first, "later": len(generated) - first}, out.stdout
