"""topout-publish's contract, attacked: a published game is the journal's bytes at its one path, nothing unfinished is
published, nothing is published twice, a source journal is never touched, and a refusal writes nothing."""
from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
import sys

from tests.conftest import ROOT

RUN = json.loads((ROOT / "tests" / "fixtures" / "v1-run.json").read_text())


def _publish(out, *run_dirs):
    return subprocess.run([sys.executable, ROOT / "topout-publish.py", *run_dirs, "--local", out],
                          capture_output=True, text=True, timeout=120, cwd=ROOT)


def _journal(run_dir, index, finished=True, **fields):
    game = run_dir / "game" / str(index)
    game.mkdir(parents=True, exist_ok=True)
    rec = {**RUN, "kind": "run", "model": "m-served", "sampler_name": "s-high", "decision": "verbs", **fields}
    lines = [json.dumps(rec), json.dumps({"kind": "tick", "tick": 0, "game": index})]
    if finished:
        lines.append(json.dumps({"kind": "summary", "termination": "max_pieces", "pieces": 3}))
    (game / "journal.jsonl").write_text("\n".join(lines) + "\n")
    return game / "journal.jsonl"


def _target(out, run_dir, index, model="m-served", sampler="s-high"):
    ident = f"{run_dir.name}/game/{index}/journal.jsonl"
    return out / model / sampler / f"{hashlib.sha256(ident.encode()).hexdigest()}.jsonl.gz"


def _files(out):
    return sorted(p for p in out.rglob("*") if p.is_file()) if out.exists() else []


def test_a_finished_game_is_its_bytes_at_its_one_path_and_the_source_is_untouched(tmp_path):
    run = tmp_path / "play-0-20261124T181428Z"
    src = _journal(run, 0)
    before = (src.read_bytes(), src.stat().st_mtime_ns)
    out = tmp_path / "out"
    r = _publish(out, run)
    assert r.returncode == 0, r.stderr
    target = _target(out, run, 0)
    assert _files(out) == [target]
    assert gzip.decompress(target.read_bytes()) == src.read_bytes()
    assert (src.read_bytes(), src.stat().st_mtime_ns) == before
    assert "1 written, 0 unchanged, 0 still running" in r.stdout


def test_a_real_game_from_topout_play_publishes(fake_engine, tmp_path):
    from tests.conftest import play
    run = tmp_path / "run"
    (run / "game").mkdir(parents=True)
    r, rows = play(fake_engine(), run / "game" / "0", pieces=2)
    assert r.returncode == 0 and rows[-1]["kind"] == "summary"
    rec = rows[0]
    out = tmp_path / "out"
    p = _publish(out, run)
    assert p.returncode == 0, p.stderr
    target = _target(out, run, 0, rec["model"], rec["sampler_name"])
    assert gzip.decompress(target.read_bytes()) == (run / "game" / "0" / "journal.jsonl").read_bytes()


def test_unfinished_and_empty_games_are_counted_never_published_and_a_finished_sibling_is(tmp_path):
    run = tmp_path / "run"
    _journal(run, 0)
    _journal(run, 1, finished=False)
    (run / "game" / "2").mkdir(parents=True)
    (run / "game" / "2" / "journal.jsonl").write_text("")
    (run / "game" / "3").mkdir(parents=True)
    (run / "game" / "3" / "journal.jsonl").write_text(json.dumps({**RUN, "kind": "run", "sampler_name": "s"})
                                                      + "\n{\"kind\": \"tic")     # cut mid-record
    out = tmp_path / "out"
    r = _publish(out, run)
    assert r.returncode == 0, r.stderr
    assert _files(out) == [_target(out, run, 0)]
    assert "1 written, 0 unchanged, 3 still running" in r.stdout


def test_publishing_again_changes_nothing(tmp_path):
    run = tmp_path / "run"
    _journal(run, 0)
    _journal(run, 1)
    out = tmp_path / "out"
    assert _publish(out, run).returncode == 0
    first = {p: p.read_bytes() for p in _files(out)}
    r = _publish(out, run)
    assert r.returncode == 0, r.stderr
    assert "0 written, 2 unchanged" in r.stdout
    assert {p: p.read_bytes() for p in _files(out)} == first


def test_a_file_published_by_another_zlib_build_counts_as_unchanged(tmp_path):
    run = tmp_path / "run"
    src = _journal(run, 0)
    out = tmp_path / "out"
    target = _target(out, run, 0)
    target.parent.mkdir(parents=True)
    target.write_bytes(gzip.compress(src.read_bytes(), compresslevel=1, mtime=12345))   # other bytes, same content
    r = _publish(out, run)
    assert r.returncode == 0, r.stderr
    assert "0 written, 1 unchanged" in r.stdout


def test_the_same_game_through_another_directory_name_is_refused_not_published_twice(tmp_path):
    run = tmp_path / "play-0-20261124T181428Z"
    _journal(run, 0)
    out = tmp_path / "out"
    assert _publish(out, run).returncode == 0
    before = _files(out)
    r = _publish(out, run / "game")                     # the subdirectory: another id, the same game
    assert r.returncode != 0 and "already published" in r.stderr and "Traceback" not in r.stderr
    assert _files(out) == before


def test_a_target_holding_other_content_is_refused(tmp_path):
    run = tmp_path / "run"
    _journal(run, 0)
    out = tmp_path / "out"
    target = _target(out, run, 0)
    target.parent.mkdir(parents=True)
    target.write_bytes(gzip.compress(b"something else\n"))
    r = _publish(out, run)
    assert r.returncode != 0 and "other content" in r.stderr
    assert gzip.decompress(target.read_bytes()) == b"something else\n"


def test_bad_names_and_non_verbs_games_are_refused_and_write_nothing(tmp_path):
    cases = [{"sampler_name": None}, {"sampler_name": ""}, {"sampler_name": ".."}, {"model": "a/b"},
             {"model": None}, {"decision": "chat"}]
    for i, fields in enumerate(cases):
        run = tmp_path / f"run{i}"
        _journal(run, 0, **fields)
        out = tmp_path / f"out{i}"
        r = _publish(out, run)
        assert r.returncode != 0 and "refused" in r.stderr and "Traceback" not in r.stderr, (fields, r.stderr)
        assert _files(out) == [], fields


def test_a_stale_tmp_from_an_interrupted_stage_is_refused_before_anything_is_written(tmp_path):
    run = tmp_path / "run"
    _journal(run, 0)
    out = tmp_path / "out"
    (out / "m-served" / "s-high").mkdir(parents=True)
    stale = out / "m-served" / "s-high" / "x.jsonl.gz.tmp"
    stale.write_bytes(b"partial")
    r = _publish(out, run)
    assert r.returncode != 0 and "stale" in r.stderr
    assert _files(out) == [stale]
