"""topout-publish: put finished games into a results dataset on the Hugging Face Hub, one gzipped journal per game.

    python topout-publish.py [RUN_DIR ...] [--repo USER/NAME] [--local DIR] [--push]
    python topout-publish.py --sync [--repo USER/NAME] [--local DIR]

Contract. Reads every journal.jsonl under each RUN_DIR; writes LOCAL/<model>/<sampler_name>/<id>.jsonl.gz for each
finished verbs game (last record a summary), the journal's bytes verbatim. Never edits, moves or deletes a journal.
Games still running (or that never wrote a line) are skipped and counted. Without --push nothing leaves the machine.

--sync is the download: LOCAL is brought up to the repo's current revision (every game, and the index as in Index below)
and nothing else happens. It takes no RUN_DIR, stages nothing, commits nothing and needs no token for a public repo. It
is the way to get the dataset, and to bring a local copy up to date before staging into it.

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

Index. After staging, an arm LOCAL/index.json lacks (topout/index.py) is added to it, its label and ladder guessed from its
name and printed; entries already there are never changed, so correct a guess by editing the file. --push commits the
index when it differs from the repo's, and refuses when both were edited since the last sync. With no RUN_DIR only the
index is seeded (and pushed).

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

from topout import index, journals


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


def seed_index(local: Path) -> list[str]:
    """Add arms the index lacks to LOCAL/index.json; returns them. Writes nothing when there are none."""
    ix, added = index.seed(local)
    if added:
        (local / index.NAME).write_text(index.dumps(ix))
    return added


def resolve_index(local: Path, remote: str | None) -> None:
    """Bring LOCAL/index.json up to the repo's without losing an edit on either side. BASE is the repo's index as last
    synced or pushed: an unedited local copy takes the remote, an edited one stays, both changed is refused."""
    path, base = local / index.NAME, local / BASE
    mine = path.read_text() if path.is_file() else None
    was = base.read_text() if base.is_file() else None
    if remote is not None and mine != remote:
        if mine is None or mine == was:
            path.write_text(remote)
        elif was != remote:
            raise SystemExit(f"topout-publish: {path} and the repo's {index.NAME} both changed (or were never synced): "
                             f"merge the repo's version into yours, put a copy of the repo's at {base}, and push again")
    if remote is not None:
        base.parent.mkdir(parents=True, exist_ok=True)
        base.write_text(remote)


BASE = Path(".cache") / "topout" / "index.base.json"
DEFAULT_REPO = "mike-ravkine/topout-bench-results"


def sync(repo: str, local: Path):
    """Bring LOCAL up to REPO's current revision: every file but the index downloaded, the index resolved. Returns the
    API, that revision, the files it holds and its index text (None when it has none)."""
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    api = HfApi()
    if not api.repo_exists(repo, repo_type="dataset"):
        raise SystemExit(f"topout-publish: dataset {repo} not found (or no access): create it first, "
                         f"hf repos create {repo} --repo-type dataset")
    base = api.dataset_info(repo).sha
    snapshot_download(repo, repo_type="dataset", revision=base, local_dir=local, ignore_patterns=[index.NAME])
    remote = set(api.list_repo_files(repo, repo_type="dataset", revision=base))
    remote_index = (Path(hf_hub_download(repo, index.NAME, repo_type="dataset", revision=base)).read_text()
                    if index.NAME in remote else None)
    resolve_index(local, remote_index)
    return api, base, remote, remote_index


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dirs", type=Path, nargs="*", help="run directories holding games (journal.jsonl under them); none: the index alone")
    ap.add_argument("--repo", default=DEFAULT_REPO, help=f"dataset repo USER/NAME (default {DEFAULT_REPO})")
    ap.add_argument("--local", type=Path, help="local copy of the dataset (default ./NAME)")
    ap.add_argument("--push", action="store_true", help="sync LOCAL from the repo first, then commit what it lacks")
    ap.add_argument("--sync", action="store_true", help="only bring LOCAL up to the repo: download, stage nothing, commit nothing")
    a = ap.parse_args()
    local = a.local or Path(a.repo.rsplit("/", 1)[-1])
    for run_dir in a.run_dirs:
        if not run_dir.is_dir():
            raise SystemExit(f"topout-publish: {run_dir}: not a directory")
    if a.sync and (a.push or a.run_dirs):
        raise SystemExit("topout-publish: --sync only downloads: it takes no RUN_DIR and no --push")
    if a.sync or a.push:
        api, base, remote, remote_index = sync(a.repo, local)
    if a.sync:
        games = sum(1 for _ in local.glob("*/*/*.jsonl.gz"))
        print(f"{local}: synced to {a.repo} @ {base[:8]}, {games} games; nothing staged or committed")
        return
    staging = Staging(local)
    for run_dir in a.run_dirs:
        c = staging.stage(run_dir.resolve())
        print(f"{run_dir.name}: {c['written']} written, {c['unchanged']} unchanged, {c['running']} still running")
    for arm in seed_index(local) if local.is_dir() else []:
        print(f"{index.NAME}: added {arm}, guessed from its name; edit its label and ladder if the guess is wrong")
    if not a.push:
        print(f"staged in {local}; nothing uploaded (--push commits to {a.repo})")
        return
    from huggingface_hub import CommitOperationAdd
    new = sorted(p.relative_to(local).as_posix() for p in local.glob("*/*/*.jsonl.gz"))
    new = [p for p in new if p not in remote]
    index_text = (local / index.NAME).read_text() if (local / index.NAME).is_file() else None
    changed = [index.NAME] if index_text is not None and index_text != remote_index else []
    if not new and not changed:
        print(f"{a.repo} already holds every game in {local} and its index: no commit")
        return
    info = api.create_commit(a.repo, [CommitOperationAdd(p, local / p) for p in new + changed], repo_type="dataset",
                             commit_message=(f"{len(new)} games ({', '.join(d.name for d in a.run_dirs)})" if new else "index"),
                             parent_commit=base)
    if changed:
        (local / BASE).parent.mkdir(parents=True, exist_ok=True)
        (local / BASE).write_text(index_text)
    print(f"{len(new)} games{' and the index' if changed else ''} committed to {a.repo}: {info.commit_url}")


if __name__ == "__main__":
    sys.exit(main())
