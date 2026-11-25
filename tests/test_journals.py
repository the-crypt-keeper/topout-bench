"""The one read path, attacked: a plain journal and its gzipped copy read the same everywhere; a game cut off mid-write
reads up to its last whole record and is not finished; a broken record anywhere else is an error, never skipped; a
published folder is found and analysed like a game directory."""
from __future__ import annotations

import gzip
import json
import subprocess
import sys

import pytest

from tests.conftest import ROOT
from topout import journals

RECS = [{"kind": "run", "model": "m", "sampler_name": "s", "decision": "verbs"},
        {"kind": "tick", "tick": 0}, {"kind": "tick", "tick": 1},
        {"kind": "summary", "termination": "max_pieces"}]


def _write(path, recs, tail=""):
    text = "".join(json.dumps(r) + "\n" for r in recs) + tail
    if path.name.endswith(".gz"):
        path.write_bytes(gzip.compress(text.encode()))
    else:
        path.write_text(text)
    return path


def test_plain_and_gz_read_the_same(tmp_path):
    plain = _write(tmp_path / "journal.jsonl", RECS)
    gz = _write(tmp_path / "abc.jsonl.gz", RECS)
    for p in (plain, gz):
        assert list(journals.records(p)) == RECS
        assert journals.run_record(p) == RECS[0]
        assert journals.summary(p) == RECS[-1]
    assert journals.game_id(plain) == str(tmp_path) and journals.game_id(gz) == "abc"


def test_a_game_cut_off_mid_write_reads_to_its_last_whole_record_and_is_not_finished(tmp_path):
    for name in ("journal.jsonl", "abc.jsonl.gz"):
        p = _write(tmp_path / name, RECS[:3], tail='{"kind": "tick", "tick": 2, "boa')
        assert list(journals.records(p)) == RECS[:3]
        assert journals.summary(p) is None


def test_a_broken_record_before_the_last_line_is_an_error_not_skipped(tmp_path):
    p = tmp_path / "journal.jsonl"
    p.write_text(json.dumps(RECS[0]) + "\n{not json\n" + json.dumps(RECS[-1]) + "\n")
    with pytest.raises(ValueError, match="does not parse"):
        list(journals.records(p))


def test_empty_and_unstarted_journals_are_nothing_yet(tmp_path):
    for name in ("journal.jsonl", "abc.jsonl.gz"):
        p = _write(tmp_path / name, [])
        assert list(journals.records(p)) == [] and journals.run_record(p) == {} and journals.summary(p) is None


def test_find_takes_gz_files_folders_of_them_and_mixed_trees(tmp_path):
    arm = tmp_path / "model" / "sampler"
    arm.mkdir(parents=True)
    a, b = _write(arm / "aa.jsonl.gz", RECS), _write(arm / "bb.jsonl.gz", RECS)
    (tmp_path / "run" / "0").mkdir(parents=True)
    c = _write(tmp_path / "run" / "0" / "journal.jsonl", RECS)
    assert journals.find([arm]) == [a, b]
    assert journals.find([a]) == [a]
    assert set(journals.find([tmp_path])) == {a, b, c}
    (tmp_path / "none").mkdir()
    with pytest.raises(SystemExit):
        journals.find([tmp_path / "none"])


def test_a_published_folder_is_analysed_like_the_games_it_came_from(fake_engine, tmp_path):
    from tests.conftest import play
    run = tmp_path / "run"
    (run / "game").mkdir(parents=True)
    for i in range(2):
        r, rows = play(fake_engine(seed=i), run / "game" / str(i), pieces=3)
        assert r.returncode == 0 and rows[-1]["kind"] == "summary"
    out = tmp_path / "out"
    p = subprocess.run([sys.executable, ROOT / "topout-publish.py", out, run], capture_output=True, text=True, cwd=ROOT)
    assert p.returncode == 0, p.stderr
    folder = out                                         # conftest.play names each game's sampler after its directory
    assert len(list(folder.rglob("*.jsonl.gz"))) == 2

    def replies(path):
        r = subprocess.run([sys.executable, ROOT / "topout-replies.py", "--arm", f"a={path}"],
                           capture_output=True, text=True, timeout=120, cwd=ROOT)
        assert r.returncode == 0, r.stderr
        return r.stdout

    assert replies(folder) == replies(run / "game")
    o = subprocess.run([sys.executable, ROOT / "topout-outcomes.py", "--arm", f"a={folder}"],
                       capture_output=True, text=True, timeout=120, cwd=ROOT)
    assert o.returncode == 0 and "a: survived 2 / 2" in o.stdout, (o.stdout, o.stderr)
