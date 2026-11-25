# Running it

- [Install](#install) · [Play](#play) · [What a journal holds](#what-a-journal-holds) · [Explore](#explore) ·
  [Plot](#plot) · [Video](#video) · [The results dataset](#the-results-dataset) · [Publish](#publish) ·
  [When something goes wrong](#when-something-goes-wrong)

## Install

    uv sync

Two things are needed only by one tool each: `topout-video.py` needs `ffmpeg` on PATH, and `topout-plot.py` needs a
Chrome or Chromium for image files (see [Plot](#plot)).

## Play

    uv run topout-play.py --apibase http://localhost:8000/v1 --model NAME --output games/run1 \
        --sampler sampler.json --seed 1 --lag-frames 12 --context-length 32768 --games 16 --parallel 8

Required:

| option | meaning |
|---|---|
| `--apibase`, `--model` | the endpoint (ending in `/v1`) and the model name it serves |
| `--output` | the run's directory; each game goes in `OUTPUT/<i>/`. Keep run directory names unique: they become part of a published game's id |
| `--sampler` | a JSON file of chat-completions fields sent with every request. The file's name is the arm's sampler name |
| `--seed` | seeds the deal. It fixes only the first 2 or 3 pieces; after that the deal follows the frame count, so games of one seed differ |
| `--lag-frames` | the least game time a button costs. 12 is 200 ms; all our games use it |
| `--context-length` | the engine's context window. It must be the true one: it tells a full context from a token cap |

Useful:

| option | meaning |
|---|---|
| `--games N --parallel P` | N games, at most P at once; as one ends the next starts |
| `--max-pieces` | 100 by default. Every analysis needs one value across all arms |
| `--first-turn-max-tokens N --first-turn-retries R` | a different cap for a piece's first reply |

A sampler file:

    {"temperature": 1.0, "top_p": 0.95, "max_tokens": 49152,
     "chat_template_kwargs": {"enable_thinking": true, "preserve_thinking": true, "reasoning_effort": "high"}}

Two things make or break a run. **The template must keep the model's thinking across tool results**: the player checks
that the prompt grows and prints `ERROR: history not carried` when it does not. **The engine must parse tool calls**
into `tool_calls`: try one game first.

Live lines: `[12/100]` is the piece in play. `NUDGE:`, `CAPPED:`, `LOST:` and `RESET:` are the model's or engine's
behaviour, recorded and measured. `ERROR:` is something broken.

| line | what happened |
|---|---|
| `NUDGE` | a reply with no tool call. `wait` is pressed, and a user turn says so |
| `CAPPED` | a reply cut off by `max_tokens`. The same request is sent again, up to 3 more times |
| `LOST` | every attempt was cut off. `wait` is pressed and the piece's conversation starts over |
| `RESET` | the piece's conversation no longer fits the context. It starts over on the board as it is |

## What a journal holds

`journal.jsonl`, one JSON record per line: a `run` record (the settings, the exact instructions and tools, the sampler),
a `tick` per button press (the board and text shown before it, the button, its frames and per-frame inputs; a reply's
first tick also carries its reasoning, content, tool calls and token usage), `capped` and `reply` records for replies
that pressed nothing, and a `summary` of how the game ended. The seed and the inputs replay the game exactly.

They nest: game > piece (one conversation) > reply (one generation) > tick (one press). The player computes no
statistics. Everything else reads journals.

## Explore

    uv run topout-explore.py --dataset results
    uv run topout-explore.py --dataset results --arm mine=games/run1 --arm other=games/run2,games/run3
    uv run topout-explore.py --arm mine=games/run1 --port 9000 --host 0.0.0.0

An **arm** is a set of games played the same way. `--dataset DIR` makes every `<model>/<sampler>` folder of a dataset copy
an arm; `--arm NAME=PATH[,PATH...]` adds your own (a journal, a game directory, a directory of games). They sit side by
side. Games still running are skipped and counted; restart to pick up new ones.

Finished games are digested into a cache (`--cache`, by default `~/.cache/topout`). A digest is reused until its journal or
the code that computes it changes, and deleting the cache loses nothing. 400 games take a few seconds the first time on a
many-core machine.

The address holds the whole view, so a view is a link, and Back works. There is no login: `--host 0.0.0.0` shows it to
everyone on your network. It works on a phone; a larger screen is better for the fingerprints and for comparing arms.

## Plot

    uv run topout-plot.py burden --dataset results -o burden.png
    uv run topout-plot.py burden --dataset results -o burden.png --show cost --tokens median
    uv run topout-plot.py decisions --dataset results --only "Claude Haiku 5.5" -o haiku.png
    uv run topout-plot.py triangle --dataset results --only "Qwen3.8 27B" --arm mine=games/run1 --only mine -o tri.svg
    uv run topout-plot.py games --dataset results --table

Kinds: `burden`, `games`, `decisions`, `triangle`, `moves`. They are the Explorer's figures, with a title and note so they
stand alone. `--table` prints the numbers instead or as well. `--only` takes an arm's name or the start of several, and
repeats. `--show` is `ladders`, `performance` or `cost`.

`-o` ends in `.png`, `.svg`, `.pdf` or `.webp`, which a Chrome or Chromium draws: one on PATH, or named by
`BROWSER_PATH`, or fetched once with `uv run plotly_get_chrome`. `-o x.html` needs no browser and keeps the hover (about
5 MB, because it carries the plotting library).

## Video

    uv run topout-video.py games/run1 -o run1.mp4
    uv run topout-video.py results/<model>/<sampler>/<id>.jsonl.gz -o game.mp4

Games are played back through the emulator from the seed and inputs, side by side, in game time, with each decision's
button and tokens. A game whose replay does not match its journal refuses to render. Rendering is slow: choose games in
the Explorer first.

## The results dataset

Our games: [mike-ravkine/topout-bench-results](https://huggingface.co/datasets/mike-ravkine/topout-bench-results).

    uv run hf download mike-ravkine/topout-bench-results --repo-type dataset --local-dir results
    uv run hf download mike-ravkine/topout-bench-results --repo-type dataset --local-dir results --include "<model>/*" --include index.json     # one model

    <model>/<sampler>/<id>.jsonl.gz      one finished game's journal, verbatim, gzipped
    index.json                           labels and effort ladders

`<model>` is the served model name and `<sampler>` the sampler's name, both from the run record. `<id>` is a hash of the
run directory's name and the game's path in it. Any tool here reads a `.jsonl.gz` as it reads a `journal.jsonl`.

`index.json` holds only what a person has to say, and what no journal records: a model's label, which arms form an effort
ladder, and in what order.

    {"models": {"inkling-2-mini-nvfp4-...": {"label": "Inkling 2 Mini", "ladders": [
        {"label": "",    "arms": {"inkling-low": "low", "inkling-high": "high", "inkling-max": "max"}},
        {"label": "40k", "arms": {"inkling-high-b40k": "high", "inkling-max-b40k": "max"}}]}}}

A shown name is label + ladder label + arm label; the order in the file is the ladder's order. An arm the index does not
list still appears, under its folder names and in no ladder. Edit the file by hand.

## Publish

To your own dataset repo (create it with `uv run hf repos create NAME --repo-type dataset`, log in with
`uv run hf auth login`):

    uv run topout-publish.py games/run1 --repo YOU/NAME                # stages into ./NAME; nothing leaves the machine
    uv run topout-publish.py games/run1 --repo YOU/NAME --push         # syncs, stages, commits what the repo lacks
    uv run topout-publish.py --repo YOU/NAME --push                    # the index alone, after editing it

Pass the run's own directory, not one inside it. Unfinished games are skipped. An arm the index lacks is added with a label
and ladder guessed from its name, and printed: a variant (a token budget, say) is guessed into the model's first ladder,
so move it if that is wrong. Entries already there are never changed. If the index was edited both here and on the Hub
since the last sync, the push refuses and says how to merge.

Journals hold the model's words and the endpoint's URL as given to `--apibase`. Look before you push.

## When something goes wrong

| you see | it means |
|---|---|
| `arms mix max_pieces` | burden needs one horizon: compare arms played to the same `--max-pieces` |
| `two arms are named ...` | two labels collide: edit `index.json`, or name your `--arm` differently |
| `drawn by Chrome, and none was found` | see [Plot](#plot) |
| `not a valid index` | `index.json` was edited into something unreadable; the message says where |
| raw folder names instead of labels | no `index.json` beside the games: download it too |
| `history not carried` while playing | the chat template drops the model's earlier thinking; fix the sampler's template settings |
