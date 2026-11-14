"""An OpenAI-shaped stub for exercising topout-play without a GPU.

POST /v1/chat/completions answers at once with one tool call: start() first in a piece's conversation,
then a random verb. usage counts the prompt by characters / 4 so the player's carry checks have numbers
to read. length_rate cuts a share of replies off at max_tokens (finish_reason "length"); no_call_rate
answers a share with no tool call; context refuses a request whose prompt + max_tokens passes it, in
vLLM's words. thinking picks how the prompt counts the conversation's thinking: "kept" (every message as sent),
"last_user" (a template that renders thinking only after the last user turn, so a nudge's user turn drops the piece's
earlier thinking and the prompt shrinks; replies then think at length so the drop shows), or "forget" (an engine that
renders only the first message: history lost). It plays nothing well and measures nothing about a model: a harness
check only.
"""
from __future__ import annotations

import json
import random

VERBS = ["left", "right", "rotate", "drop", "wait"]
LONG_THINKING = "fake thinking " * 100


def prompt_tokens(body: dict, thinking: str = "kept") -> int:
    msgs = body.get("messages", [])
    if thinking == "forget":
        msgs = msgs[:1]
    elif thinking == "last_user":
        last = max((i for i, m in enumerate(msgs) if m.get("role") == "user"), default=0)
        msgs = [{k: v for k, v in m.items() if k != "reasoning_content"} if i < last and m.get("role") == "assistant"
                else m for i, m in enumerate(msgs)]
    return sum(len(json.dumps(m)) for m in msgs) // 4


def refusal(body: dict, context: int | None) -> dict | None:
    """vLLM's 400 when prompt + max_tokens passes the context, or None."""
    if not context:
        return None
    prompt = sum(len(json.dumps(m)) for m in body.get("messages", [])) // 4
    out = body.get("max_tokens") or 1024
    if prompt + out <= context:
        return None
    return {"error": {"message": f"This model's maximum context length is {context} tokens. However, you "
                                 f"requested {out} output tokens and your prompt contains {prompt} input tokens, "
                                 f"for a total of {prompt + out} tokens. Please reduce the length of the input "
                                 f"prompt or the number of requested output tokens.",
                      "type": "BadRequestError", "param": "input_tokens", "code": 400}}


def clamped(body: dict, context: int) -> tuple[int, dict] | None:
    """vLLM >= 0.32 (#57035): a prompt that alone passes the context is still a 400; a prompt that fits but leaves less
    than max_tokens of room is not refused: the output budget is clamped to the room left, and this fake spends all of
    it, so the reply is cut off with finish_reason "length" at prompt + completion = context. None when it fits."""
    prompt = sum(len(json.dumps(m)) for m in body.get("messages", [])) // 4
    if prompt > context:
        return 400, {"error": {"message": f"This model's maximum context length is {context} tokens. However, your "
                                          f"prompt contains {prompt} input tokens, which exceeds this limit even "
                                          f"without any output tokens. Please reduce the length of the input prompt.",
                               "type": "BadRequestError", "param": "input_tokens", "code": 400}}
    out = body.get("max_tokens") or 1024
    if prompt + out <= context:
        return None
    room = context - prompt
    return 200, {"id": "fake", "object": "chat.completion", "model": body.get("model"),
                 "choices": [{"index": 0, "message": {"role": "assistant", "content": None,
                                                      "reasoning_content": "fake, out of context"},
                              "finish_reason": "length"}],
                 "usage": {"prompt_tokens": prompt, "completion_tokens": room, "total_tokens": context}}


def reply(body: dict, rng: random.Random, length_rate: float = 0.0, no_call_rate: float = 0.0,
          thinking: str = "kept") -> dict:
    prompt = prompt_tokens(body, thinking)
    if rng.random() < length_rate:      # a reply cut off by max_tokens: thinking only, no answer
        cap = body.get("max_tokens") or 1024
        return {"id": "fake", "object": "chat.completion", "model": body.get("model"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": None,
                                                     "reasoning_content": "fake runaway"},
                             "finish_reason": "length"}],
                "usage": {"prompt_tokens": prompt, "completion_tokens": cap, "total_tokens": prompt + cap}}
    msg: dict = {"role": "assistant", "content": None,
                 "reasoning_content": LONG_THINKING if thinking == "last_user" else "fake"}
    finish = "stop"
    started = any(m.get("role") == "tool" for m in body.get("messages", []))
    if started and rng.random() < no_call_rate:   # a reply with no tool call: the player nudges
        msg["content"] = "I will think about it."
    else:
        name = "start" if not started else rng.choice(VERBS)
        msg["tool_calls"] = [{"id": f"call_{rng.randrange(1 << 30)}", "type": "function",
                              "function": {"name": name, "arguments": "{}"}}]
        finish = "tool_calls"
    return {"id": "fake", "object": "chat.completion", "model": body.get("model"),
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": 8, "total_tokens": prompt + 8}}
