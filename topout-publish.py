"""topout-publish: put finished games into a results dataset on the Hugging Face Hub, one gzipped journal per game.

    python topout-publish.py RUN_DIR [RUN_DIR ...] [--repo USER/NAME] [--local DIR] [--push]

Contract. Reads every journal.jsonl under each RUN_DIR; writes LOCAL/<model>/<sampler_name>/<id>.jsonl.gz for each
finished verbs game (last record a summary), the journal's bytes verbatim. Never edits, moves or deletes a journal.
Games still running (or that never wrote a line) are skipped and counted. Without --push nothing leaves the machine.

--repo is the dataset repo, by default this project's own (mike-ravkine/topout-bench-results); pass your own to publish
yours. It must exist (hf repos create NAME --repo-type dataset), and pushing needs a write token (hf auth login).
--local is the local copy, by default ./<NAME>. With --push: LOCAL is first brought up to the repo's current revision
(hf download), the games are staged into it, and every game file in LOCAL the repo does not hold yet is committed, in
one commit on top of that revision (a push from elsewhere in between fails the commit rather than racing it). Games
staged earlier without --push go up with the next push. Without --push: staged into LOCAL as it is, offline.

Layout. <model> and <sampler_name> are the run record's own fields: the served model name and the sampler's name. A
journal without either is refused, as is a name that is empty, '.', '..' or holds a '/'. <id> is the sha256 hex of
"<RUN_DIR's name>/<journal path inside RUN_DIR>", e.g. of "play-0-20261013T063148Z/game/0/journal.jsonl": a game's
index repeats across runs, so the run directory's name is part of its id, and the absolute path is not (it differs
by machine). Pass the run's own directory, whose name is unique within the dataset, not a subdirectory of it.

Repeatable: a game whose file already holds its bytes is left as it is; a file holding other bytes is refused; a game
whose bytes are already present under another id (the same run passed through another directory name) is refused.
Compared decompressed, so the zlib build does not matter. Stale *.tmp files under LOCAL (an interrupted stage) are
refused before anything is written.
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


DEFAULT_REPO = "mike-ravkine/topout-bench-results"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dirs", type=Path, nargs="+", help="run directories holding games (journal.jsonl under them)")
    ap.add_argument("--repo", default=DEFAULT_REPO, help=f"dataset repo USER/NAME (default {DEFAULT_REPO})")
    ap.add_argument("--local", type=Path, help="local copy of the dataset (default ./NAME)")
    ap.add_argument("--push", action="store_true", help="sync LOCAL from the repo first, then commit what it lacks")
    a = ap.parse_args()
    local = a.local or Path(a.repo.rsplit("/", 1)[-1])
    for run_dir in a.run_dirs:
        if not run_dir.is_dir():
            raise SystemExit(f"topout-publish: {run_dir}: not a directory")
    if a.push:
        from huggingface_hub import HfApi, snapshot_download
        api = HfApi()
        if not api.repo_exists(a.repo, repo_type="dataset"):
            raise SystemExit(f"topout-publish: dataset {a.repo} not found (or no access): create it first, "
                             f"hf repos create {a.repo} --repo-type dataset")
        base = api.dataset_info(a.repo).sha
        snapshot_download(a.repo, repo_type="dataset", revision=base, local_dir=local)
    staging = Staging(local)
    for run_dir in a.run_dirs:
        c = staging.stage(run_dir.resolve())
        print(f"{run_dir.name}: {c['written']} written, {c['unchanged']} unchanged, {c['running']} still running")
    if not a.push:
        print(f"staged in {local}; nothing uploaded (--push commits to {a.repo})")
        return
    from huggingface_hub import CommitOperationAdd
    remote = set(api.list_repo_files(a.repo, repo_type="dataset", revision=base))
    new = sorted(p.relative_to(local).as_posix() for p in local.glob("*/*/*.jsonl.gz"))
    new = [p for p in new if p not in remote]
    if not new:
        print(f"{a.repo} already holds every game in {local}: no commit")
        return
    info = api.create_commit(a.repo, [CommitOperationAdd(p, local / p) for p in new], repo_type="dataset",
                             commit_message=f"{len(new)} games ({', '.join(d.name for d in a.run_dirs)})",
                             parent_commit=base)
    print(f"{len(new)} games committed to {a.repo}: {info.commit_url}")


if __name__ == "__main__":
    sys.exit(main())
