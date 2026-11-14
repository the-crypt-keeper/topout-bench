"""A game topout-play wrote replays exactly from its seed and inputs (topout-video's replay, without the video)."""
from __future__ import annotations

import importlib.util

from tests.conftest import ROOT, play


def _video_module():
    spec = importlib.util.spec_from_file_location("topout_video", ROOT / "topout-video.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_played_game_replays_without_divergence(fake_engine, tmp_path):
    r, rows = play(fake_engine(no_call_rate=0.2, seed=3), tmp_path / "g", pieces=5)
    assert r.returncode == 0, r.stdout[-2000:]
    arm = _video_module().Arm(tmp_path / "g" / "journal.jsonl")    # sys.exit on the first divergence
    assert arm.total == sum(1 for x in rows if x.get("kind") == "tick" and not x.get("no_piece"))
    assert len(arm.inputs) == rows[-1]["frames"]
