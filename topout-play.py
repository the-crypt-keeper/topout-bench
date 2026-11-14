"""topout-play: an LLM plays NES Tetris (tetris100) through tool calls, one tool per button.

The decisions are the tools: start() left() right() rotate() drop() wait(), against any OpenAI-compatible
chat completions endpoint that parses tool calls. The game, its timing and the v1 instructions are
topout's (topout/game.py, topout/prompt.py):

- Every piece is one conversation. Its one user turn is the board, the instructions and "the game is
  paused: press start to resume"; after that come only tool calls and tool results until the piece locks,
  then it resets. A template that renders thinking only after the last user turn therefore renders all of
  the piece's thinking.
- tool_choice "auto": nothing is forced (every forcing technique is brittle per engine). The instructions
  say every reply is tool calls. parallel_tool_calls is never sent.
- A reply may make several calls. They run in order, one after another, each costing game time while the
  piece falls, each result the board after that press. Calls left once the piece has locked are not
  pressed; a button before start, or start once running, is refused and costs nothing.
- A reply with no tool call is not asked again: wait is pressed, and the reply stays in the conversation
  followed by a user turn saying so (the only valid next message). A last-user-turn template stops
  rendering the thinking before it; that is the model's own exit. The summary counts these as "nudges".
- A reply cut off by max_tokens (finish_reason "length") is not the model's decision: it is never looped
  back and never nudged. The identical request is sent again up to CAPPED_RETRIES times; if every attempt
  is cut off, wait is pressed (a lost decision) and the piece's trajectory is rebooted: a fresh paused
  user turn on the board as it now is.
- When a piece's conversation no longer fits the context, it starts over (RESET: context) on the board as it is,
  nothing pressed. Engines say so two ways: a 400 refusing the request (SGLang, llama-server, vLLM < 0.32, and vLLM
  when the prompt alone fills the context), or, on vLLM >= 0.32, which clamps the output budget to what the context has
  left, a reply cut off with finish_reason "length" at prompt + completion = --context-length. Both are handled alike.
  The engine is the only judge of fit; nothing is predicted.

tetris100: the game is --max-pieces long, counted in pieces. It ends once piece --max-pieces has locked
and its clear has played out, or at a top-out.

Journal (journal.jsonl): a "run" record, one "tick" per executed press (the board and text shown before it,
the action, frames and the per-frame "inputs"), the reply's usage / reasoning / content / tool_calls on
its first tick, "capped" records for cut-off replies, "reply" records for replies that pressed nothing, and
a "summary". The seed and the inputs replay the game exactly (topout-video.py).

Prints one "[pieces/max_pieces]" line per press; errors are lines containing "ERROR:". CAPPED:, LOST:,
NUDGE: and RESET: are the model's or the engine's behaviour, measured, not errors.

Usage:
    python topout-play.py --apibase URL/v1 --model NAME --output DIR --sampler FILE
                          --seed N --lag-frames N --context-length N [--max-pieces 100] [--parallel N]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import requests

from topout import game as tg
from topout.parallel import fan_out
from topout.prompt import BUTTONS, INSTRUCTIONS, NUDGE, TOOLS

CAPPED_RETRIES = 3

# Each engine's context-overflow refusal, read from its source: vLLM (renderers/params.py "This model's
# maximum context length is ..."); SGLang (tokenizer_manager "... longer than the model's context length",
# "Requested token count exceeds the model's maximum context length"); llama-server (type
# exceed_context_size_error). Any other 400 is not this.
CONTEXT_OVERFLOW = ("maximum context length", "longer than the model's context length",
                    "exceeds the available context size", "larger than the max context size")


def context_overflow(r: requests.Response) -> bool:
    """True when the engine refused the request because the conversation does not fit its context."""
    if r.status_code != 400:
        return False
    try:
        err = r.json()
        err = err.get("error", err) if isinstance(err, dict) else {}
    except ValueError:
        err = {}
    if isinstance(err, dict) and err.get("type") == "exceed_context_size_error":
        return True
    text = (err.get("message") if isinstance(err, dict) else None) or r.text
    return any(phrase in text for phrase in CONTEXT_OVERFLOW)


def button_of(call: dict) -> tuple[str | None, str | None]:
    """The button a call presses, or None and why (a refusal the model is told about)."""
    name = (call.get("function") or {}).get("name")
    return (name, None) if name in BUTTONS else (None, f"no button named {name!r}")


def extra_args(call: dict) -> dict:
    """Arguments a verb was given (it declares none): ignored in play, counted."""
    raw = (call.get("function") or {}).get("arguments")
    try:
        args = json.loads(raw) if isinstance(raw, str) else (raw or {})
    except json.JSONDecodeError:
        return {"<not JSON>": raw}
    if not isinstance(args, dict):
        return {"<not an object>": args}
    return args


def call_text(call: dict) -> str:
    extra = extra_args(call)
    return f"{(call.get('function') or {}).get('name')}({', '.join(f'{k}={v}' for k, v in extra.items())})"


def play(a) -> int:
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    sampler = json.loads(Path(a.sampler).read_text())
    url = a.apibase.rstrip("/") + "/chat/completions"
    game, session = tg.Game(a.seed), requests.Session()
    journal = open(out / "journal.jsonl", "w")
    journal.write(json.dumps({"kind": "run", "mode": "tetris-tool", "decision": "verbs", "url": url, "model": a.model,
                              "protocol": "tetris-v3", "prompt": "v1",
                              "instructions": INSTRUCTIONS,
                              "seed": a.seed, "max_pieces": a.max_pieces,
                              "inputs": "per tick, the buttons held on each frame, run-length [[buttons, frames], ...]",
                              "tick": "max(lag, until the play area changes) or the lock, then 2 frames still at playState 1 or 3",
                              "lag_frames": a.lag_frames, "max_frames_per_tick": a.max_frames_per_tick,
                              "board": "text", "drop_rows": tg.DROP_ROWS,
                              "conversation": "piece: one user turn (paused), then tool calls and results, thinking carried",
                              "tool_choice": "auto", "no_tool_call": "wait pressed, then a user turn: " + NUDGE,
                              "on_length": f"not looped back; the request again up to {CAPPED_RETRIES} times, then "
                                           "wait and the trajectory rebooted (a fresh paused user turn)",
                              "first_turn_max_tokens": a.first_turn_max_tokens,
                              "first_turn_retries": a.first_turn_retries,
                              "tools": TOOLS, "context_length": a.context_length,
                              "sampler": sampler}) + "\n")
    counts, latencies, completion = collections.Counter(), [], []
    decisions = nudges = replies = extra = 0
    per_reply, repeats, refused, unpressed, pressless = collections.Counter(), 0, 0, 0, 0
    conv: list = []
    conv_piece, conv_last, running, nudged = None, None, False, False
    capped = lost = capped_streak = 0
    reboot = None                 # "capped" / "context": start the piece's conversation over on the next request
    tick = 0
    while not game.game_over:
        info = game.info()
        if info["pieces"] > a.max_pieces:     # piece max_pieces has locked, its clear played out
            break
        start = len(game.inputs)
        if not game.piece_in_play:
            rec = {"kind": "tick", "tick": tick, **info, "action": None, "no_piece": True,
                   "frames": tg.advance_to_piece(game, a.max_frames_per_tick)}
            rec["inputs"] = game.inputs_since(start)
            journal.write(json.dumps(rec) + "\n")
            journal.flush()
            tick += 1
            continue
        reset = None
        if conv_piece != info["pieces"]:
            reset = "new piece"
        elif reboot:
            reset = reboot
        if reset:   # a fresh paused user turn on the board as it is now
            text = f"{tg.state_text(game.board_text(), info)}\n\n{INSTRUCTIONS}"
            conv = [{"role": "user", "content": [{"type": "text", "text": text}]}]
            conv_piece, conv_last, running, seen = info["pieces"], None, False, text
        reboot = None
        conv_turn = sum(m["role"] == "assistant" for m in conv)
        body = {"model": a.model, "messages": conv, "tools": TOOLS, "tool_choice": "auto", **sampler}
        # First turn: a fresh conversation (new piece or reboot) takes --first-turn-max-tokens with
        # --first-turn-retries in place of CAPPED_RETRIES.
        capped_retries = CAPPED_RETRIES
        if conv_turn == 0 and a.first_turn_max_tokens:
            body["max_tokens"], capped_retries = a.first_turn_max_tokens, a.first_turn_retries
        t0 = time.monotonic()
        r = session.post(url, json=body, timeout=a.timeout)
        ms = round((time.monotonic() - t0) * 1000, 1)
        if context_overflow(r):      # the engine is the judge of fit
            if conv_turn == 0:       # a fresh conversation has nothing left to drop: that is broken
                print(f"ERROR: engine refused a fresh conversation as too long: {r.text[:500]}", flush=True)
                r.raise_for_status()
            print(f"RESET: context at reply {replies + 1}: piece {info['pieces']} conversation ({conv_turn} "
                  f"replies) refused by the engine as too long; a fresh conversation", flush=True)
            reboot = "context"
            continue
        if r.status_code >= 400:
            print(f"ERROR: engine refused the request ({r.status_code}): {r.text[:500]}", flush=True)
        r.raise_for_status()
        resp = r.json()
        msg = resp["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        usage = resp.get("usage") or {}
        completion.append(usage.get("completion_tokens") or 0)
        replies += 1
        latencies.append(ms)
        reasoning = msg.get("reasoning") or msg.get("reasoning_content")
        pt, ct = usage.get("prompt_tokens") or 0, usage.get("completion_tokens") or 0
        # Check: the history reached the prompt. After a nudge the prompt may shrink (the nudge is a user turn
        # and a last-user-turn template drops the thinking before it): the template answering the model's exit.
        carry = None
        if conv_turn > 0 and conv_last:
            carry = {"grew": pt - conv_last[0], "last_completion": conv_last[1], "after_nudge": nudged}
            if carry["grew"] <= 0 and not nudged:
                print(f"ERROR: history not carried at reply {replies}: prompt grew {carry['grew']}", flush=True)
        nudged = False
        reply = {"reply": replies, "conv_turn": conv_turn, "conv_reset": reset, "carry_check": carry,
                 "max_tokens": body.get("max_tokens"),
                 "content": msg.get("content"), "reasoning": reasoning, "usage": usage, "call_ms": ms,
                 "finish_reason": resp["choices"][0].get("finish_reason"), "tool_calls": calls}
        if reply["finish_reason"] == "length" and conv_turn > 0 and pt + ct >= a.context_length:
            # Cut off by the context, not by max_tokens: vLLM >= 0.32 clamps the output budget to what the context has
            # left instead of refusing the request (#57035), so a conversation that no longer fits now ends here rather
            # than in a 400. Handled exactly as that refusal was: the cut reply is journalled, nothing is pressed, and
            # the piece's conversation starts over (not a lost decision; the game is where it was).
            journal.write(json.dumps({"kind": "capped", **reply, "reason": "context", "attempt": 0,
                                      "pieces": info["pieces"]}) + "\n")
            journal.flush()
            print(f"RESET: context at reply {replies}: piece {info['pieces']} conversation ({conv_turn} replies) "
                  f"filled the context ({pt} + {ct} tokens); a fresh conversation", flush=True)
            capped_streak = 0
            reboot = "context"
            continue
        if reply["finish_reason"] == "length":       # cut off by max_tokens: not looped back, not nudged
            reply["reason"] = "max_tokens"
            capped += 1
            capped_streak += 1
            journal.write(json.dumps({"kind": "capped", **reply, "attempt": capped_streak,
                                      "pieces": info["pieces"]}) + "\n")
            journal.flush()
            print(f"CAPPED: reply {replies} cut off at {ct} completion tokens (attempt {capped_streak} of "
                  f"{capped_retries + 1}, piece {info['pieces']})", flush=True)
            if capped_streak <= capped_retries:
                continue                             # the identical request again
            capped_streak, lost = 0, lost + 1
            rec = {"kind": "tick", "tick": tick, **game.info(), **reply, "board": game.board_text(), "text": seen,
                   "action": "wait", "parse_error": "capped", "plan": None, "call": 0, "calls": 0}
            rec["frames"] = tg.apply(game, "wait", a.lag_frames, a.max_frames_per_tick)
            rec["inputs"] = game.inputs_since(start)
            journal.write(json.dumps(rec) + "\n")
            journal.flush()
            decisions += 1
            counts["wait"] += 1
            tick += 1
            reboot = "capped"
            print(f"LOST: decision {decisions}: {capped_retries + 1} attempts cut off; wait pressed, trajectory rebooted",
                  flush=True)
            print(f"[{rec['pieces']}/{a.max_pieces}] wait lines={rec['lines']} decision={decisions}", flush=True)
            continue
        capped_streak = 0
        conv_last = (pt, ct)
        if not calls:              # no tool call: wait is pressed and the model is told so in a user turn
            nudges += 1
            nudged = True
            rec = {"kind": "tick", "tick": tick, **game.info(), **reply, "board": game.board_text(), "text": seen,
                   "action": "wait", "parse_error": "no tool call", "plan": None, "call": 0, "calls": 0}
            rec["frames"] = tg.apply(game, "wait", a.lag_frames, a.max_frames_per_tick)
            rec["inputs"] = game.inputs_since(start)
            journal.write(json.dumps(rec) + "\n")
            journal.flush()
            decisions += 1
            counts["wait"] += 1
            tick += 1
            running = True
            board = (tg.state_text(game.board_text(), game.info()) if game.piece_in_play
                     else "The piece locked.")
            seen = f"{NUDGE}\n\n{board}"
            conv += [{"role": "assistant", "content": msg.get("content") or "", "reasoning_content": reasoning or ""},
                     {"role": "user", "content": [{"type": "text", "text": seen}]}]
            print(f"NUDGE: no tool call at reply {replies} ({reply['finish_reason']}); wait pressed", flush=True)
            print(f"[{rec['pieces']}/{a.max_pieces}] wait lines={rec['lines']} decision={decisions}", flush=True)
            continue
        conv.append({"role": "assistant", "content": msg.get("content") or "",
                     "reasoning_content": reasoning or "", "tool_calls": calls})
        plan = " ".join(call_text(c) for c in calls)
        extra += sum(1 for c in calls if extra_args(c))
        names = [button_of(c)[0] for c in calls]
        repeats += sum(1 for x, y in zip(names, names[1:]) if x is not None and x == y)
        per_reply[len(calls)] += 1
        pressed, piece = 0, info["pieces"]
        for i, c in enumerate(calls):
            button, why = button_of(c)
            if game.game_over or not game.piece_in_play or game.info()["pieces"] != piece:
                result, button = "Not pressed: the piece has already locked.", None
                unpressed += 1
            elif why:
                result, button = f"Not pressed: {why}.", None
                refused += 1
            elif button == "start":
                result = ("The game is already running." if running else
                          f"Resumed.\n\n{tg.state_text(game.board_text(), game.info())}")
                if running:
                    refused += 1
                running = True
            elif not running:
                result, button = "Not pressed: the game is paused; press start first.", None
                refused += 1
            else:
                start, before = len(game.inputs), game.info()
                rec = {"kind": "tick", "tick": tick, **before, **(reply if pressed == 0 else {}),
                       "conv_turn": conv_turn, "conv_reset": reset if pressed == 0 else None,
                       "board": game.board_text(), "text": seen, "action": button, "parse_error": None,
                       "plan": plan, "reply": replies, "call": i, "calls": len(calls)}
                rec["frames"] = tg.apply(game, button, a.lag_frames, a.max_frames_per_tick)
                rec["inputs"] = game.inputs_since(start)
                journal.write(json.dumps(rec) + "\n")
                journal.flush()
                pressed += 1
                decisions += 1
                counts[button] += 1
                tick += 1
                print(f"[{rec['pieces']}/{a.max_pieces}] {button} lines={rec['lines']} decision={decisions}",
                      flush=True)
                if game.piece_in_play:
                    result = f"Pressed {button}.\n\n{tg.state_text(game.board_text(), game.info())}"
                else:
                    result = f"Pressed {button}. The piece locked."
            conv.append({"role": "tool", "tool_call_id": c.get("id"), "content": result})
            seen = result
        if not pressed:
            pressless += 1
            journal.write(json.dumps({"kind": "reply", **reply, "plan": plan, "pieces": piece}) + "\n")
            journal.flush()
    end = game.info()
    latencies.sort()
    completion.sort()
    summary = {"kind": "summary", "ticks": tick, "decisions": decisions, "parse_failures": nudges,
               "nudges": nudges, "seed": a.seed, "lag_frames": a.lag_frames,
               "actions": dict(counts), "lines": end["lines"], "score": end["score"],
               "pieces": end["pieces"] if game.game_over else a.max_pieces,
               "frames": len(game.inputs),
               "game_over": game.game_over, "termination": "game_over" if game.game_over else "max_pieces",
               "call_ms_p50": latencies[len(latencies) // 2] if latencies else None,
               "completion_tokens_p50": completion[len(completion) // 2] if completion else None,
               "decision": "verbs", "replies": replies, "calls_per_reply": dict(sorted(per_reply.items())),
               "repeated_calls": repeats, "refused_calls": refused, "unpressed_after_lock": unpressed,
               "pressless_replies": pressless, "calls_with_extra_arguments": extra,
               "capped_replies": capped, "lost_decisions": lost}
    journal.write(json.dumps(summary) + "\n")
    journal.close()
    print(json.dumps(summary))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--apibase", required=True, help="engine base URL ending in /v1")
    p.add_argument("--model", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--sampler", required=True,
                   help="JSON file of chat-completions sampling fields (temperature, top_p, max_tokens, chat_template_kwargs)")
    p.add_argument("--seed", type=int, required=True, help="deal seed (gym np_random -> NES RNG)")
    p.add_argument("--lag-frames", type=int, required=True,
                   help="minimum game frames per button: the perception lag (12 = 200 ms)")
    p.add_argument("--context-length", type=int, required=True,
                   help="the engine's context window (max_model_len): a reply cut off at prompt + completion = this is "
                        "a full context, not the model's cap; also checks --first-turn-max-tokens")
    p.add_argument("--max-pieces", type=int, default=100,
                   help="tetris100: the game ends once this piece has locked (and cleared), or at a top-out")
    p.add_argument("--max-frames-per-tick", type=int, default=1200,
                   help="safety cap on the frames one press may run")
    p.add_argument("--timeout", type=float, default=7200.0,
                   help="per request; a stall detector, not a budget: 128k tokens at 20 tok/s is ~6400 s")
    p.add_argument("--parallel", type=int, default=1,
                   help="games of this seed at once against the engine, each under OUTPUT/<i>")
    p.add_argument("--first-turn-max-tokens", type=int, default=None,
                   help="max_tokens for a fresh conversation's first reply; with --first-turn-retries")
    p.add_argument("--first-turn-retries", type=int, default=None,
                   help="cut-off retries for that first reply, in place of CAPPED_RETRIES")
    a = p.parse_args()
    if a.lag_frames < 1:
        p.error("--lag-frames must be >= 1: a decision of 0 frames observes a state the game holds for no time")
    if a.max_pieces < 1:
        p.error("--max-pieces must be >= 1")
    if a.context_length <= 0:
        p.error("--context-length must be positive")
    if a.parallel < 1:
        p.error("--parallel must be >= 1")
    if (a.first_turn_max_tokens is None) != (a.first_turn_retries is None):
        p.error("--first-turn-max-tokens and --first-turn-retries go together")
    if a.first_turn_max_tokens is not None and not 0 < a.first_turn_max_tokens < a.context_length:
        p.error("--first-turn-max-tokens must be positive and leave room for the prompt in --context-length")
    if a.first_turn_retries is not None and a.first_turn_retries < 0:
        p.error("--first-turn-retries must be >= 0")
    if a.parallel > 1:
        return fan_out(a.parallel, a.output, a.max_pieces)
    return play(a)


if __name__ == "__main__":
    sys.exit(main())
