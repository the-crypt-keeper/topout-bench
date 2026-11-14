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
most `--parallel P` at once: as one game ends the next starts (`--games` defaults to `--parallel`). Each game writes `journal.jsonl`: a `run` record, one `tick` per button press (the board and
text shown, the action, its frames and per-frame inputs, and on a reply's first tick its reasoning, content,
tool calls and usage), and a `summary`. The seed and the inputs replay the game exactly.

## Replay and analyse

    uv run topout-video.py games/run1 -o run1.mp4                  # every game side by side, in game time
    uv run topout-regions.py games/run1 -o regions.png --csv regions.csv
    uv run topout-placements.py regions.csv                        # per-placement rates, per model
    uv run topout-outcomes.py --arm run1=games/run1 -o outcomes.png

- **topout-regions**: per piece, the board split into filled / open / sealed cells, classic covered holes,
  and the ledger terms.
- **topout-placements**: each placement as a ledger event (sealing, dig-outs, lines); per-model rates with
  95% intervals from a bootstrap over whole games.
- **topout-outcomes**: the distributions of lines cleared and pieces placed per arm.

## Tests

    uv run pytest

The tests play short games against a fake engine (no GPU) and hold the instruction text and tools to a
recorded game, so new games stay comparable with old ones.
