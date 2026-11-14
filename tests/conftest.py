"""The fake engine runs in this process, bound to port 0 so the OS assigns a free one, on a thread the fixture
shuts down; topout-play runs as a subprocess with this interpreter and writes under tmp_path."""
from __future__ import annotations

import json
import random
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from tests import fake_engine as fe

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def fake_engine():
    """Yield a function that configures the engine and returns its base URL; shut down at teardown."""
    state = {"rng": random.Random(0), "length_rate": 0.0, "no_call_rate": 0.0, "context": None, "clamp": False}

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if state["clamp"] and state["context"]:          # vLLM >= 0.32
                status, payload = fe.clamped(body, state["context"]) or (200, None)
            else:                                            # vLLM < 0.32, SGLang, llama-server
                refused = fe.refusal(body, state["context"])
                status, payload = (400, refused) if refused else (200, None)
            if payload is None:
                payload = fe.reply(body, state["rng"], state["length_rate"], state["no_call_rate"])
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), H)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def configure(length_rate: float = 0.0, seed: int = 0, context: int | None = None,
                  no_call_rate: float = 0.0, clamp: bool = False) -> str:
        state.update(rng=random.Random(seed), length_rate=length_rate, context=context, no_call_rate=no_call_rate,
                     clamp=clamp)
        return f"http://127.0.0.1:{server.server_address[1]}/v1"

    yield configure
    server.shutdown()
    server.server_close()


def play(apibase: str, out: Path, extra: list[str] = (), pieces: int = 3, max_tokens: int = 100,
         context_length: int = 100000):
    """topout-play.py as a driver would run it (--context-length is the engine's max_model_len, as the Topout driver
    passes seq_max); returns the process and the journal's records."""
    sampler = out.parent / f"{out.name}-sampler.json"
    sampler.write_text(json.dumps({"max_tokens": max_tokens}))
    argv = [sys.executable, str(ROOT / "topout-play.py"), *extra, "--apibase", apibase, "--model", "fake",
            "--output", str(out), "--sampler", str(sampler), "--seed", "1", "--lag-frames", "12",
            "--context-length", str(context_length), "--max-pieces", str(pieces)]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=300, cwd=ROOT)
    path = out / "journal.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return r, rows
