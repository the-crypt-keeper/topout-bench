# topout-bench

LLMs playing NES Tetris® through tool calls, via [gym-tetris](https://github.com/Kautenja/gym-tetris).
Not affiliated with or endorsed by The Tetris Company. This repository contains no ROM: gym-tetris
provides the emulator and the game.

A model plays **tetris100**: a game of 100 pieces, or until the stack tops out. It sees the board as text
(`#` settled, `@` falling, `.` empty), the current and next piece, lines and score, and plays by calling
tools, one per button: `start() left() right() rotate() drop() wait()`. Every button costs game time (a
12-frame perception lag by default, 200 ms), so thinking is free but moving is not. Each piece is one
conversation, so the model's own thinking for that piece stays in context while it steers it.

Works against any OpenAI-compatible chat completions endpoint that parses tool calls (vLLM, SGLang,
llama-server, hosted APIs).

## Why per piece

A game's final line count is too noisy to separate models at any game count you would run. But every piece
is a fresh conversation, so given the board it was shown, each placement is an independent draw from the
model's policy. topout reads pieces and the decisions inside them, and one graded
number per game, burden, rather than lines or survival.

A board's rows are empty, live (completable) or dead (holding a sealed cell: an empty cell nothing can reach until what
covers it clears). Dead rows are what end a game, so a game's **burden** is its dead rows averaged over the 100 turns, with
every turn after a top-out counting as a full 20. **0.4 lines a piece is break-even**: each piece adds 4 cells and each
line removes 10.

## Install

    git clone https://github.com/the-crypt-keeper/topout-bench && cd topout-bench
    uv sync

`topout-video.py` also needs `ffmpeg` on PATH.

## Play

    uv run topout-play.py --apibase http://localhost:8000/v1 --model NAME --output games/run1 \
        --sampler sampler.json --seed 1 --lag-frames 12 --context-length 32768 --games 32 --parallel 16

`sampler.json` holds the chat-completions sampling fields sent with every request (`temperature`, `top_p`,
`max_tokens`, `chat_template_kwargs`, ...). `--games N` plays N games of the seed, each in `OUTPUT/<i>/`, at
most `--parallel P` at once: as one game ends the next starts (`--games` defaults to `--parallel`). Each game
writes `journal.jsonl`: a `run` record, one `tick` per button press (the board and text shown, the action, its
frames and per-frame inputs, and on a reply's first tick its reasoning, content, tool calls and usage), `capped`
and `reply` records for replies that pressed nothing, and a `summary` of how the game ended. The seed and the
inputs replay the game exactly. The player computes no statistics: everything below reads the journals.

## Replay and analyse

Every analysis takes arms as `--arm NAME=PATH[,PATH...]` (a journal, a game directory, a directory of games or a
folder of the results dataset, below) and/or `--dataset DIR` (every arm of a copy of it). They read only journals, and
keep a cache of digests that rebuilds itself.

    uv run topout-explore.py --dataset results                     # all of it, interactive
    uv run topout-plot.py burden --dataset results -o burden.png   # one of the Explorer's figures as an image (or .html)
    uv run topout-plot.py games --arm run1=games/run1 --table      # or its numbers
    uv run topout-video.py games/run1 -o run1.mp4                  # the games as played, side by side, in game time

`topout-plot.py` draws images with Chrome or Chromium: one on PATH, or named by `BROWSER_PATH`, or fetched once with
`uv run plotly_get_chrome`. `.html` needs none. `--only NAME` keeps some arms.

- **burden**: dead rows averaged over every turn of the game, a turn after the game ended counting as 20, against
  completion tokens per piece. The full effort ladders, or each model's best performance or cost frontier.
- **games**: every game's burden, by arm.
- **decisions**: each arm's fingerprint. Planning replies (a piece's first) against execution replies (the rest): tokens,
  calls per reply, buttons, replies per piece, truncated, nudged.
- **triangle**: every board is a point on the triangle of empty, live (completable) and dead (holding a sealed cell)
  rows; the share of an arm's boards in each (live, dead) state, and where each game ended. A dense origin is a good
  player; how it smears, along the floor (towers) or up the dead axis (sealed rows), fingerprints the style.
- **moves**: each placement as (lines cleared, sealed cells added or dug out).

The Explorer adds the scrubber (an arm's games together, piece by piece), arms side by side, every board of a game, and
every reply of a piece with its thinking. Its address holds the whole view, so a view is a link. There is no login.

## Results dataset

Our games are on the Hugging Face Hub as
[mike-ravkine/topout-bench-results](https://huggingface.co/datasets/mike-ravkine/topout-bench-results): one gzipped
journal per finished game, exactly as topout-play wrote it, at `<model>/<sampler>/<id>.jsonl.gz`. Each
`<model>/<sampler>` folder is an arm, and every analysis reads it as it reads your own games, so ours and yours sit
side by side:

    uv run hf download mike-ravkine/topout-bench-results --repo-type dataset --local-dir results
    uv run hf download mike-ravkine/topout-bench-results --repo-type dataset --local-dir results --include "<model>/*" --include index.json   # one model
    uv run topout-plot.py burden --dataset results --arm mine=games/run1 -o burden.png

To publish your own games to a dataset repo of yours (create it first with `hf repos create`, and log in with
`hf auth login`):

    uv run topout-publish.py games/run1 --repo YOU/NAME --push

Without `--push` it only stages into a local copy (`--local`, default `./NAME`). Labels and effort ladders live in the dataset's `index.json`: publishing adds new arms to it with a guess from their
names, and an edit to it (a label, splitting a ladder) is kept. A run's directory name is part
of each game's id, so keep run directory names unique within a dataset.

## Tests

    uv run pytest

The tests play short games against a fake engine (no GPU) and hold the instruction text, the board text and the
tools to a recorded game, so new games stay comparable with old ones.
