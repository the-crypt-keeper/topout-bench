"""The explorer's digests agree with the functions the scripts use, the cache follows the journal, and the address is
the only state."""
from __future__ import annotations

import importlib.util
import json
import os

import pytest

from tests.conftest import ROOT, play
from topout import decisions, index, journals, ledger, store

spec = importlib.util.spec_from_file_location("topout_explore", ROOT / "topout-explore.py")
ex = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ex)


@pytest.fixture
def game(fake_engine, tmp_path):
    (tmp_path / "games").mkdir()
    r, _ = play(fake_engine(), tmp_path / "games" / "0", pieces=3)
    assert r.returncode == 0
    return tmp_path / "games" / "0" / "journal.jsonl"


def test_a_digest_is_what_the_scripts_compute(game):
    d = store.digest(game)
    run = journals.run_record(game)
    assert [p["dead"] for p in d["pieces"]] == ledger.dead_rows(game)
    assert d["burden"] == ledger.burden(ledger.dead_rows(game), run["max_pieces"])
    assert d["phases"] == decisions.count(journals.records(game))
    assert sum(p["cleared"] for p in d["pieces"]) == d["lines"]
    for p in d["pieces"]:
        rs = store.replies(game, p["piece"])
        assert len(rs) == sum(p["replies"]) and sum(r["tokens"] for r in rs) == sum(p["tokens"])
        assert [q["action"] for r in rs for q in r["presses"]] == p["presses"]


def test_the_cache_is_reused_and_a_changed_journal_is_read_again(game, tmp_path):
    cache = tmp_path / "cache"
    first = store.load([game], cache, 1)
    files = {f: f.stat().st_mtime_ns for f in cache.iterdir()}
    assert store.load([game], cache, 1) == first and {f: f.stat().st_mtime_ns for f in cache.iterdir()} == files
    for f in files:
        f.write_bytes(b"cut off")                              # a damaged entry is computed again
    assert store.load([game], cache, 1) == first
    lines = game.read_text().splitlines()
    game.write_text("\n".join(lines[:-1]) + "\n")              # no summary: running again
    os.utime(game)
    assert store.load([game], cache, 1) == [None]


def test_the_address_round_trips_the_choice_and_keeps_it_on_the_way_down():
    data = type("D", (), {"arms": dict.fromkeys(["a low", "a & b?", "c=d", "e"])})
    for names in ([], ["a low"], ["a & b?", "c=d", "e"], ["a low", "a & b?", "c=d", "e"]):
        nav = ex.Nav(data, {})
        back = ex.Nav(data, ex.query(nav.to(names, stat="median", arm="c=d")[1:]))
        assert back.names == names and back.stat == "median"
    assert ex.link() == "/"


def test_every_level_renders(game, tmp_path):
    data = ex.Data(index.arms(None, [f"a={game.parent}"]), tmp_path / "cache")
    nav = ex.Nav(data, {})
    key = data.arms["a"]["games"][0]["key"]
    assert ex.overview(data, nav) and ex.arm_view(data, nav, "a") and ex.game_view(data, nav, "a", key)
    assert ex.piece_view(data, nav, "a", key, 1)
    ex.build(data)


def test_signed_thinking_blocks_are_not_shown_and_the_words_are():
    assert ex.readable("plan.\n\n\n[anthropic-blocks:" + "Q" * 5000 + "]") == "plan."
    assert ex.readable("board [4, 5] stays") == "board [4, 5] stays"


def test_one_per_model_by_performance_and_by_cost_and_the_rail_does_not_move_the_scale():
    arm = lambda model, tokens, burden: dict(arm=index.Arm("", [], model), mean=tokens, burden=burden)
    data = type("D", (), {"arms": {"m low": arm("m", 1000, 14), "m mid": arm("m", 3000, 3), "m max": arm("m", 30000, 1),
                                   "n": arm("n", 60000, 16)}})
    pick = lambda names, show: ex.one_per_model(data, names, "mean", show)
    assert pick(list(data.arms), "cost") == ["m mid", "n"] and pick(list(data.arms), "performance") == ["m max", "n"]
    assert pick(["m low", "m mid"], "performance") == ["m mid"] and pick(["m low", "m max"], "cost") == ["m max"]
