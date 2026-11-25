# topout-bench

LLMs play NES Tetris (tetris100: 100 pieces or top-out) through tool calls, one tool per button. Read README.md for
what it is and how to run it. This file is what the code and README do not say: where the project stands, and what has
already been settled about reading the data.

Public MIT repo. No ROM here or in the dataset (gym-tetris ships it); keep the no-affiliation line.

## Where it stands (2026-11-25)

- **Driver (`topout-play.py`, `topout/game.py`, `prompt.py`, `parallel.py`): mature.** Games pool with recorded ones
  only while the v1 text, tools, timing and journal format stay byte-identical (tests/test_prompt.py). Do not change
  any of them in passing.
- **Dataset: public**, `mike-ravkine/topout-bench-results`, 400 games = 25 arms x 16. `viewer: false`, no Parquet.
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
  published: pieces are recomputed from journals (8 arms read in about half a minute).
- Scripts that did not earn their place were deleted (regions, placements, conduct, flows, outcomes, replies, triangle-anim).
  Prefer removing a view to keeping one nobody reads; git history keeps them.

## The data

A journal nests game (`run` + `summary`) > piece (one conversation, starts at `conv_reset: "new piece"`) > reply (first
tick carries reasoning / content / tool_calls / usage; `capped` and `reply` records pressed nothing) > tick (one
press: board before it, action, frames, `inputs`). Boards are shown *before* each press.

- A piece's spawn board is the board at its first tick. The last piece has no recorded outcome, on purpose: survivors
  read as 99 placed.
- A spawn board where the piece overlaps the stack is ordinary Tetris near top-out: kept, with the piece's cells counted as
  occupied (`ledger.spawn_board`). No change in sealed cells is read into or out of one.
- Completion tokens hide a cap: arms run with max_tokens 49,152, and strong-effort arms (MiMo, Haiku max, DS4 max)
  press against it. They are measurements under that cap.
- A shared seed fixes only the first 2-3 pieces; the deal then depends on frame count. Games are not paired.

## What is settled about reading it

- **Game-level outcomes are noisy** (CV ~0.74 on lines). 16 games separate survival 0/16 from 13/16, not near rungs.
- **Survived and topped-out games are never pooled on one axis.** Lines per turn is `0.4 - filled / (10 placed)`:
  debris left for a survivor, game length in disguise for a top-out.
- **Burden**: dead rows averaged over max_pieces turns, turns after the end counting as 20. A fixed
  horizon, so dying early does not look clean; it replaces survival as the headline. Sampled per game (n=16), read by the
  per-game dots: bars apart rule equality out, overlap says nothing.
- **Planning (conv_turn 0) and execution replies are never pooled** except where the unit is the piece (replies per
  piece, tokens per piece). Tokens as a cost are a mean; tokens as a description are quantiles.
- **Counters that assume nothing have held up:** survival, placed, lines, filled, sealed. Pooled per-placement rates,
  windows over early pieces and anything against piece index misled, because the trajectory chooses the states.
- **No clock on an axis.** Removing time is what made the triangle readable; the scrubber puts it in the frame. One game's
  own page may use it.
- **Rows are the unit of clearability:** empty / live (completable) / dead (holds a sealed cell). Dead separates arms;
  live sits at 2-4 nearly everywhere. `filled` is not a clearability axis.
- **Levels:** line (triangle), cell (moves), decision (decisions); games carry burden.
- Style and fidelity come apart: effort moved DS4-Flash's fidelity and left its shape alone, while Haiku's ladder
  moves both.

## Open, in the analysis

- The triangle is piece-level: its unit is the spawn board, and a share is of the arm's pieces. Games are not its unit,
  so "never pool survivors and top-outs" (a rule about game outcomes) does not apply to it. Open: its sampling
  uncertainty. Consecutive boards of a game differ by one placement, so boards are not independent draws.
- Untested lead: P(clear | live rows), E[lines | live, dead], i.e. skill in a state apart from which states play
  visits, with each bin trusted by its *game* count. Compare arms only where their states overlap.
- Not built: random-policy reference, per-row seal / unseal tracking, piece-aware reachability (the flood assumes 1x1 cells).
- Labels and effort ladders are the dataset's `index.json` (topout/index.py), seeded by topout-publish from names and edited by
  hand. Never put a model, label or effort fact in a script.
## The Explorer (`topout-explore.py`)

Dash on waitress, reading a local dataset through `topout/store.py` (a cache that rebuilds; nothing derived is published).
Levels: arms > arm (up to 4 side by side on one scrubber and scale) > game > piece. The rail changes what is drawn, never
what a number means. The address is the only state; `dcc.Location` needs `refresh="callback-nav"` or every choice reloads
the page. Phones get their own figures (`narrow`), with a line steering to a larger screen where that costs something.
The scrubber swaps cells and rings into the Python-built triangle. Board colours and the flood in explore_assets/boards.js mirror topout/ledger.py. Piece search: held off.

## Working here

    uv sync; uv run pytest          # minutes, not seconds: tests/test_play.py plays emulated games against a fake engine

Before refactoring a figure, hash `fig.to_json()` for every figure on the dataset before and after: that is how the move to
figures.py was checked. Drive the Explorer in a real browser after touching it; tests do not see layout or JavaScript.
