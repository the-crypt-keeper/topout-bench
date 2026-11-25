"""topout-publish: stage finished games as the results dataset's files, one gzipped journal per game.

    python topout-publish.py OUT RUN_DIR [RUN_DIR ...]

Contract. Reads every journal.jsonl under each RUN_DIR; writes OUT/<model>/<sampler_name>/<id>.jsonl.gz for each
finished verbs game (last record a summary), the journal's bytes verbatim. Never edits, moves or deletes a journal;
never uploads. Games still running (or that never wrote a line) are skipped and counted.

Layout. <model> and <sampler_name> are the run record's own fields: the served model name and the sampler's name. A
journal without either is refused, as is a name that is empty, '.', '..' or holds a '/'. <id> is the sha256 hex of
"<RUN_DIR's name>/<journal path inside RUN_DIR>", e.g. of "play-0-20261013T063148Z/game/0/journal.jsonl": a game's
index repeats across runs, so the run directory's name is part of its id, and the absolute path is not (it differs
by machine). Pass the run's own directory, the one whose name is unique, not a subdirectory of it.

OUT is a local copy of the dataset repo (hf download), so what is already published is seen. Repeatable: a game
whose file already holds its bytes is left as it is; a file holding other bytes is refused; a game whose bytes are
already published under another id (the same run staged through another directory name) is refused. Compared
decompressed, so the zlib build does not matter. Stale *.tmp files under OUT (an interrupted stage) are refused
before anything is written: hf upload would publish them.

Upload is the Hugging Face CLI, nothing of ours:
    hf upload <user>/topout-bench-results OUT . --repo-type dataset
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import sys
from pathlib import Path

from topout import journals


def _name(rec: dict, field: str, path: Path) -> str:
    value = rec.get(field)
    if not isinstance(value, str) or value in ("", ".", "..") or "/" in value:
        raise SystemExit(f"topout-publish: {path}: run record {field} is {value!r}; refused")
    return value


class Staging:
    def __init__(self, out: Path):
        self.out = out
        stale = sorted(out.rglob("*.tmp")) if out.is_dir() else []
        if stale:
            raise SystemExit(f"topout-publish: {len(stale)} stale *.tmp under {out} (first {stale[0]}); refused")
        self._index: dict[Path, dict[str, Path]] = {}     # folder -> sha256 of decompressed content -> file

    def index(self, folder: Path) -> dict[str, Path]:
        if folder not in self._index:
            self._index[folder] = {hashlib.sha256(gzip.decompress(f.read_bytes())).hexdigest(): f
                                   for f in sorted(folder.glob("*.jsonl.gz"))} if folder.is_dir() else {}
        return self._index[folder]

    def stage(self, run_dir: Path) -> dict[str, int]:
        counts = {"written": 0, "unchanged": 0, "running": 0}
        sources = sorted(run_dir.rglob("journal.jsonl"))
        if not sources:
            raise SystemExit(f"topout-publish: {run_dir}: no journal.jsonl under it")
        for path in sources:
            if journals.summary(path) is None:           # still running, or never wrote a line
                counts["running"] += 1
                continue
            rec = journals.run_record(path)
            if rec.get("decision") != "verbs":
                raise SystemExit(f"topout-publish: {path}: not a verbs game's run record; refused")
            folder = self.out / _name(rec, "model", path) / _name(rec, "sampler_name", path)
            ident = f"{run_dir.name}/{path.relative_to(run_dir).as_posix()}"
            target = folder / f"{hashlib.sha256(ident.encode()).hexdigest()}.jsonl.gz"
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            index = self.index(folder)
            if target.exists():
                if index.get(digest) != target:
                    raise SystemExit(f"topout-publish: {target} exists with other content ({ident}); refused")
                counts["unchanged"] += 1
                continue
            if digest in index:
                raise SystemExit(f"topout-publish: {ident} is already published as {index[digest]}; refused")
            folder.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".tmp")
            tmp.write_bytes(gzip.compress(data, compresslevel=9, mtime=0))
            tmp.replace(target)
            index[digest] = target
            counts["written"] += 1
        return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path, help="local copy of the dataset repo")
    ap.add_argument("run_dirs", type=Path, nargs="+", help="run directories holding games (journal.jsonl under them)")
    a = ap.parse_args()
    for run_dir in a.run_dirs:
        if not run_dir.is_dir():
            raise SystemExit(f"topout-publish: {run_dir}: not a directory")
    staging = Staging(a.out)
    for run_dir in a.run_dirs:
        c = staging.stage(run_dir.resolve())
        print(f"{run_dir.name}: {c['written']} written, {c['unchanged']} unchanged, {c['running']} still running")


if __name__ == "__main__":
    sys.exit(main())
