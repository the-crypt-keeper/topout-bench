"""A results dataset's index.json: what a person says about its arms, the labels and the effort ladders.

Contract. Reads and checks an index, resolves a dataset and --arm arguments to arms, and (for topout-publish) proposes
entries for arms an index lacks. Facts the journals hold (sampler, budget, cap) are not in it. Nothing here knows a
particular model.

    {"models": {"<model folder>": {"label": "Inkling 2 Mini", "ladders": [
        {"label": "",           "arms": {"<sampler folder>": "minimal", ...}},
        {"label": "40k budget", "arms": {"<sampler folder>": "high", ...}}]}}}

A ladder's arms are in effort order: the file's order is the order. An arm in no index is still an arm, under its raw
folder names and in no ladder. An indexed folder that is absent is not an error: a partial download lacks most of them.

seed() proposes by name, and only for arms the index lacks: it never changes or reorders an existing entry. A person
corrects what it guessed, such as splitting one ladder into two.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from topout import journals

NAME = "index.json"


@dataclass
class Arm:
    name: str                          # as shown
    paths: list[Path]
    model: str                         # what groups arms: the model folder, or an --arm's own name
    ladder: tuple[str, int] | None = None     # which ladder of which model, None when in none
    step: int = 0                      # place on the ladder


def load(root: Path) -> dict:
    """The index of a dataset copy ({"models": {}} when it has none), refused when malformed."""
    path = Path(root) / NAME
    if not path.is_file():
        return {"models": {}}
    try:
        index = json.loads(path.read_text())
        for folder, m in index["models"].items():
            seen: set[str] = set()
            if not isinstance(m["label"], str):
                raise ValueError(f"{folder}: label is not text")
            for ladder in m["ladders"]:
                if not isinstance(ladder["label"], str) or not all(isinstance(v, str) for v in ladder["arms"].values()):
                    raise ValueError(f"{folder}: a ladder or arm label is not text")
                if seen & ladder["arms"].keys():
                    raise ValueError(f"{folder}: {sorted(seen & ladder['arms'].keys())} is in two ladders")
                seen |= ladder["arms"].keys()
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError, ValueError) as e:
        raise SystemExit(f"{path}: not a valid index: {type(e).__name__}: {e}") from None
    return index


def folders(root: Path) -> list[Path]:
    return sorted(d for d in Path(root).glob("*/*") if d.is_dir() and any(d.glob("*" + journals.GZ)))


def arms(dataset: Path | None, specs: list[str]) -> list[Arm]:
    """Every arm of a dataset copy (indexed ones first, in file order) and every --arm NAME=PATH."""
    out: list[Arm] = []
    if dataset:
        present = {(d.parent.name, d.name): d for d in folders(dataset)}
        if not present:
            raise SystemExit(f"{dataset}: no <model>/<sampler> folder of games")
        for folder, m in load(dataset)["models"].items():
            for i, ladder in enumerate(m["ladders"]):
                for step, (sampler, label) in enumerate(ladder["arms"].items()):
                    d = present.pop((folder, sampler), None)
                    if d is not None:
                        name = " ".join(x for x in (m["label"], ladder["label"], label) if x)
                        out.append(Arm(name, journals.find([d]), folder, (folder, i), step))
        for (folder, sampler), d in present.items():
            out.append(Arm(f"{folder}/{sampler}", journals.find([d]), folder))
    out += [Arm(name, paths, name) for name, paths in journals.arms(specs).items()]
    names = [a.name for a in out]
    if dupes := sorted({n for n in names if names.count(n) > 1}):
        raise SystemExit(f"two arms are named {dupes}: edit their labels")
    return out


# ---- seeding: guesses from folder names, for a person to correct

_EFFORTS = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]


def _order(sampler: str) -> tuple:
    efforts = [_EFFORTS.index(p) for p in sampler.split("-") if p in _EFFORTS]
    return (not efforts, efforts[0] if efforts else 0, sampler)


def _short(model: str, others: list[str]) -> str:
    parts = model.split("-")
    return next(("-".join(parts[:k]) for k in range(1, len(parts) + 1)
                 if not any(o.split("-")[:k] == parts[:k] for o in others if o != model)), model)


def seed(root: Path) -> tuple[dict, list[str]]:
    """The index with an entry for every arm folder under root, and the "<model>/<sampler>" it added."""
    index, added = load(root), []
    models = index["models"]
    here = folders(root)
    names = sorted({d.parent.name for d in here} | models.keys())
    for d in here:
        folder, sampler = d.parent.name, d.name
        if folder not in models:
            label = _short(folder, names)
            models[folder] = {"label": label, "ladders": []}
        ladders = models[folder]["ladders"]
        if any(sampler in ladder["arms"] for ladder in ladders):
            continue
        parts = sampler.split("-")
        while len(parts) > 1 and parts[0] in folder.split("-"):
            parts = parts[1:]
        if not ladders:
            ladders.append({"label": "", "arms": {}})
        items = list(ladders[0]["arms"].items())
        at = next((i for i, (s, _) in enumerate(items) if _order(sampler) < _order(s)), len(items))
        items.insert(at, (sampler, "-".join(parts)))
        ladders[0]["arms"] = dict(items)
        added.append(f"{folder}/{sampler}")
    return index, added


def dumps(index: dict) -> str:
    return json.dumps(index, indent=2, ensure_ascii=False) + "\n"
