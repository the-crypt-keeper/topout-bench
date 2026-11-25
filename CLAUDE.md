# topout-bench

LLMs play NES Tetris (tetris100: 100 pieces or top-out) through tool calls, one tool per button. Read README.md and
docs/ (run, read, why). This file is for whoever works on the code: where the project stands and the rules it keeps.

Public MIT repo. No ROM here or in the dataset (gym-tetris ships it); keep the no-affiliation line.

## Where it stands

- **Driver (`topout-play.py`, `topout/game.py`, `prompt.py`, `parallel.py`): mature.** Games pool with recorded ones
  only while the v1 text, tools, timing and journal format stay byte-identical (tests/test_prompt.py). Do not change
  any of them in passing.
- **Dataset: public**, `mike-ravkine/topout-bench-results`. `viewer: false`, no Parquet.
- **Analysis: converged** on `topout-explore` (interactive) and `topout-plot` (the same figures as images and tables), plus
  `topout-video`.

## Architecture rules

- The player plays and journals; it computes no statistic. Every count, rate or plot is an analysis script over
  journals. `summary` says only how the game ended.
- Nothing but `topout/journals.py` opens a journal. It reads `journal.jsonl` and published `*.jsonl.gz` alike.
- `topout/`: `ledger.py` (board -> regions, burden), `decisions.py` (reply counts by phase), `store.py` (digests + cache),
  `data.py` (arms), `index.py` (labels, ladders), `figures.py` (every figure, plotly), `calls.py` (what is a reply).
  `topout-explore` and `topout-plot` share all of it: a number or a figure exists once.
- Every script and module opens with a Contract docstring: what it reads, writes and does not do. Keep it true.
- No provenance fields in journals; recipe provenance belongs to whoever ran the game. No derived table is stored or
  published: pieces are recomputed from journals (store.py's cache makes it seconds).
- Scripts that did not earn their place were deleted (regions, placements, conduct, flows, outcomes, replies, triangle-anim).
  Prefer removing a view to keeping one nobody reads; git history keeps them.

## The data

A journal nests game (`run` + `summary`) > piece (one conversation, starts at `conv_reset: "new piece"`) > reply (first
tick carries reasoning / content / tool_calls / usage; `capped` and `reply` records pressed nothing) > tick (one
press: board before it, action, frames, `inputs`). Boards are shown *before* each press.

- A piece's spawn board is the board at its first tick. The last piece has no board after it; only the
  lines it cleared are known (from the summary).
- A spawn board where the piece overlaps the stack is ordinary Tetris near top-out: kept, with the piece's cells counted as
  occupied (`ledger.spawn_board`), so a top-out's last board is 20 rows by construction (docs/why.md). No change in sealed cells is read into or out of one.
- Completion tokens hide a cap: an arm that presses against its sampler's max_tokens is measured under that cap.
- A shared seed fixes only the first 2-3 pieces; the deal then depends on frame count. Games are not paired.

## Reading the data

docs/why.md is the record of what was tried, what failed and the rules that came out of it. Read it before adding or changing
a measure or a view; docs/read.md says what each view means. Add to both when you decide something. The rules most easily
broken: never pool planning with execution or survivors with top-outs; no clock on an axis; filtering never changes what a
number means; overlap of intervals says nothing; no model, label or effort fact in a script.

**No moving figures in README, CLAUDE.md or docs/ (operator).** Nothing that more data would make wrong: no counts of arms,
games or pieces, no dataset size, no arm's burden, share or rank, no "the only" or "every other model". They are always
stale, and anyone who wants them looks at the data. Say what happens and why, name an arm as an example if it helps, and
keep to numbers that cannot move: the rules of the game, arithmetic, design constants, closed experiments. The pictures in
docs/img are snapshots and say so; regenerate one when its view changes, not when the data does.

## The Explorer (`topout-explore.py`)

Dash on waitress, reading a local dataset through `topout/store.py` (a cache that rebuilds; nothing derived is published).
Levels: arms > arm (up to 4 side by side on one scrubber and scale) > game > piece. The rail changes what is drawn, never
what a number means. The address is the only state; `dcc.Location` needs `refresh="callback-nav"` or every choice reloads
the page. Phones get their own figures (`narrow`), with a line steering to a larger screen where that costs something.
The scrubber swaps cells and rings into the Python-built triangle. Board colours and the flood in explore_assets/boards.js mirror topout/ledger.py. Piece search: held off.

## The site (`www/`)

Static, self-contained, served as the root of topoutbench.com (`python3 -m http.server -d www` to look). It follows
reasonscape's `www/index.html` style. Same rule as the docs: no moving figures; `img/` are snapshots. `replay.js` holds two
recorded games' boards.

## Working here

    uv sync; uv run pytest          # minutes, not seconds: tests/test_play.py plays emulated games against a fake engine

Before refactoring a figure, hash `fig.to_json()` for every figure on the dataset before and after: that is how the move to
figures.py was checked. Drive the Explorer in a real browser after touching it; tests do not see layout or JavaScript.
