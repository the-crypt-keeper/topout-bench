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
model's policy. topout measures per piece, and reports game outcomes as context.

The **height ledger** splits every placement exactly. Each piece adds 4 cells and each line removes 10, so
before piece p, with L lines cleared:

    height = 0.4 (p - 1) - L + open_slack + sealed_slack

Packing, clearing, loose slack (overhangs, towers) and dead slack (sealed holes). **0.4 lines a piece is
break-even**: below it the stack climbs no matter how well it is packed.

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
folder of the results dataset, below), reads only journals, and writes only what its `-o` / `--csv` names.

    uv run topout-video.py games/run1 -o run1.mp4                  # every game side by side, in game time
    uv run topout-flows.py --arm run1=games/run1 -o flows.png   # the triangle: where play lives, how games ended
    uv run topout-flows.py --arm run1=games/run1 -o moves.png --view moves   # each placement: lines cleared, sealed cells added
    uv run topout-triangle-anim.py --arm run1=games/run1 -o tri.mp4   # the triangle lit up piece by piece (.mp4 or .gif)
    uv run topout-outcomes.py --arm run1=games/run1                # survival, then each set's final boards
    uv run topout-burden.py --dataset results -o burden.png        # every arm: burden against tokens, and every game
    uv run topout-decisions.py --dataset results -o decisions.png  # every arm: planning and execution fingerprints
    uv run topout-replies.py --arm run1=games/run1                 # reply lengths: a piece's first reply vs the rest

- **topout-flows**: every board is a point on the triangle of empty, live (completable) and dead (holding a
  sealed cell) rows; per arm, the share of its pieces in each (live, dead) state with no clock, the live and dead
  marginals, and where each game ended. A dense origin is a good player; how the density smears toward the death
  line, along the floor (towers) or up the dead axis (sealed rows), fingerprints the style. `--view moves` is the
  cell level: each placement as (lines cleared, sealed cells added or dug out), the clear profile, and per kind of
  move the mean change in sealed cells (erosion against catastrophe; digging comes with clearing).
- **topout-outcomes**: survival n / N per arm, then every game's last board, survivors and topped-out games
  never pooled: pieces placed, lines, lines per turn, filled and sealed cells.
- **topout-triangle-anim**: topout-flows' triangle as an animation, one frame per piece: the cells light up as the
  arm's spawn boards accumulate (the last frame is the static view), dots for the games alive at that piece, end
  markers from the piece each game ended. The clock is the frame, never an axis: how an arm reaches its fingerprint.
- **topout-replies**: completion tokens and reasoning per reply, a piece's first reply apart from the rest.

## Results dataset

Our games are on the Hugging Face Hub as
[mike-ravkine/topout-bench-results](https://huggingface.co/datasets/mike-ravkine/topout-bench-results): one gzipped
journal per finished game, exactly as topout-play wrote it, at `<model>/<sampler>/<id>.jsonl.gz`. Each
`<model>/<sampler>` folder is an arm, and every analysis reads it as it reads your own games, so ours and yours sit
side by side:

    uv run hf download mike-ravkine/topout-bench-results --repo-type dataset --local-dir results
    uv run hf download mike-ravkine/topout-bench-results --repo-type dataset --local-dir results --include "<model>/*" --include index.json   # one model
    uv run topout-outcomes.py --arm ours=results/<model>/<sampler> --arm mine=games/run1

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
