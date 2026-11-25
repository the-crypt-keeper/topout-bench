"""The index: seeding adds and never changes, an edited file is what the arms are, and neither side's edit is lost."""
from __future__ import annotations

import importlib.util
import json

import pytest

from tests.conftest import ROOT
from topout import index

spec = importlib.util.spec_from_file_location("topout_publish", ROOT / "topout-publish.py")
pub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pub)

FOLDERS = ("ink-2-x/ink-high", "ink-2-x/ink-low", "ink-2-x/ink-high-b40k", "q38-27b/q38-max", "q38-flash/q38-low")


def _dataset(root, folders=FOLDERS):
    for d in folders:
        (root / d).mkdir(parents=True, exist_ok=True)
        (root / d / "a.jsonl.gz").write_bytes(b"")
    return root


def test_seeding_guesses_labels_and_effort_order(tmp_path):
    ix, added = index.seed(_dataset(tmp_path))
    assert sorted(added) == sorted(FOLDERS)
    assert [m["label"] for m in ix["models"].values()] == ["ink", "q38-27b", "q38-flash"]
    assert list(ix["models"]["ink-2-x"]["ladders"][0]["arms"].items()) == [
        ("ink-low", "low"), ("ink-high", "high"), ("ink-high-b40k", "high-b40k")]


def test_an_edited_index_survives_seeding_and_a_new_arm_is_added_to_it(tmp_path):
    _dataset(tmp_path)
    edited = {"models": {"ink-2-x": {"label": "Ink 2", "ladders": [
        {"label": "", "arms": {"ink-high": "HIGH", "ink-low": "low"}},
        {"label": "40k", "arms": {"ink-high-b40k": "high"}}]}}}
    (tmp_path / "index.json").write_text(json.dumps(edited))
    (tmp_path / "ink-2-x" / "ink-max").mkdir()
    (tmp_path / "ink-2-x" / "ink-max" / "a.jsonl.gz").write_bytes(b"")
    ix, added = index.seed(tmp_path)
    assert "ink-2-x/ink-max" in added and "ink-2-x/ink-high" not in added
    m = ix["models"]["ink-2-x"]
    assert m["label"] == "Ink 2" and m["ladders"][1] == edited["models"]["ink-2-x"]["ladders"][1]
    assert list(m["ladders"][0]["arms"].items()) == [("ink-high", "HIGH"), ("ink-low", "low"), ("ink-max", "max")]
    (tmp_path / "index.json").write_text(index.dumps(ix))
    assert index.seed(tmp_path)[1] == []


def test_arms_follow_the_index_and_an_unlisted_one_is_kept_raw_outside_any_ladder(tmp_path):
    _dataset(tmp_path)
    (tmp_path / "index.json").write_text(json.dumps({"models": {
        "ink-2-x": {"label": "Ink 2", "ladders": [{"label": "", "arms": {"ink-low": "low", "ink-high": "high"}},
                                                  {"label": "40k", "arms": {"ink-high-b40k": "high"}}]},
        "absent": {"label": "Gone", "ladders": [{"label": "", "arms": {"x": "x"}}]}}}))
    got = index.arms(tmp_path, [])
    assert [(a.name, a.ladder, a.step) for a in got] == [
        ("Ink 2 low", ("ink-2-x", 0), 0), ("Ink 2 high", ("ink-2-x", 0), 1), ("Ink 2 40k high", ("ink-2-x", 1), 0),
        ("q38-27b/q38-max", None, 0), ("q38-flash/q38-low", None, 0)]


@pytest.mark.parametrize("bad", ["{", '{"models": {"m": {"label": "x"}}}', '{"models": {"m": {"label": "x", "ladders": ['
                                 '{"label": "", "arms": {"a": "1"}}, {"label": "y", "arms": {"a": "2"}}]}}}'])
def test_a_broken_index_is_refused(tmp_path, bad):
    (tmp_path / "index.json").write_text(bad)
    with pytest.raises(SystemExit):
        index.load(tmp_path)


def test_syncing_keeps_an_edit_on_either_side_and_refuses_both(tmp_path):
    path, base = tmp_path / "index.json", tmp_path / pub.BASE
    pub.resolve_index(tmp_path, "v1")                        # no local copy: take the repo's
    assert path.read_text() == base.read_text() == "v1"
    pub.resolve_index(tmp_path, "v2")                        # unedited: take the newer
    assert path.read_text() == "v2"
    path.write_text("mine")
    pub.resolve_index(tmp_path, "v2")                        # edited here only: stays
    assert path.read_text() == "mine"
    with pytest.raises(SystemExit):
        pub.resolve_index(tmp_path, "v3")                    # edited on both sides
    assert path.read_text() == "mine" and base.read_text() == "v2"
