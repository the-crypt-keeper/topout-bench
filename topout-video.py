"""topout-video: topout games replayed through the emulator, in game time, side by side.

    python topout-video.py <game dir | journal | dir of games> [...] [-o out.mp4] [--every 4] [--fps 30]

The games are played back, not stitched from saved frames: every tick's per-frame buttons (the journal's
"inputs") are pressed into a fresh emulator from the journal's seed, and every tick is checked against the
journal (board before the decision, frames used, pieces, lines); the first divergence refuses the render.
Every --every-th frame of the 60.1 fps game is kept and labelled --fps: the defaults (4, 30) are 2x game
time. The model's thinking takes no game time (the emulator waits for it), so this is the game as played.

Screens sit --wrap to a row (default 4), further games on rows below. All screens share one clock; a game
that ends holds its final frame, greyed, marked with how it ended. Under each screen, two ribbons run in
game time (newest at the right, 3 px a frame): the buttons, each placed at the frame it was pressed, and
under each a bar of the tokens that decision burned (grey < 12k, blue 12-24k, purple 24-48k, red >= 48k;
full height at 48k). Tokens, not seconds: a faster GPU thinks faster, the tokens are the model's. Below:
the piece, lines, the current decision and its plan, or for a decision with no plan its parse error and
the last words of its thinking. Labels come from each journal's run record (model, seed, reasoning effort).
Default output: one game, <game>/game.mp4; several games need -o. Needs ffmpeg on PATH.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from topout import journals
from topout.game import Game

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
SCALE = 2                       # NES 256x240 -> 512x480
PW, CAP = 256 * SCALE, 470      # panel width; caption height (two ribbons + 13 plan lines)
GAP, TOP = 12, 50               # between panels; header height
ICON, PXF = 40, 3               # ribbon icon size; ribbon pixels per game frame
BAR_H = 40                      # token bar height at 48k
WRAP, PLAN_LINES = 46, 13
NES_FPS = 60.1
COLORS = {"left": (90, 200, 255), "right": (90, 200, 255), "rotate": (255, 190, 70),
          "drop": (120, 230, 120), "wait": (150, 150, 160), "failed": (255, 90, 90)}
TOKEN_BANDS = [(48_000, (225, 60, 60)), (24_000, (170, 90, 220)), (12_000, (70, 130, 240)), (0, (120, 120, 130))]
ENDED = {"game_over": "GAME OVER", "max_pieces": "PIECE CAP"}


def burned(rec: dict) -> int:
    """Completion tokens of the reply that made this decision, a nudge's or a lost decision's included (the
    reasonlab renderer counted only chat's failed attempts there, so a tool game's nudge showed 0)."""
    return (rec.get("usage") or {}).get("completion_tokens") or 0


class Arm:
    """One game: its journal, its input stream, and its decisions placed in frames."""

    def __init__(self, journal: Path):
        self.game_dir = journal.parent
        recs = [json.loads(line) for line in open(journal)]
        self.run = recs[0] if recs and recs[0].get("kind") == "run" else {}
        if self.run.get("mode") != "tetris-tool" or self.run.get("decision") != "verbs":
            sys.exit(f"topout-video: {journal} is not a tool-verbs journal "
                     f"(mode {self.run.get('mode')!r}, decision {self.run.get('decision')!r})")
        self.ticks = [r for r in recs if r.get("kind") == "tick"]
        self.summary = next((r for r in recs if r.get("kind") == "summary"), None)
        sampler = self.run.get("sampler") or {}
        kwargs = sampler.get("chat_template_kwargs") or {}
        effort = (sampler.get("reasoning_effort") or kwargs.get("reasoning_effort")
                  or kwargs.get("reasoning_strength"))   # gpt-oss top level; Qwen / Muse-Glimmer in the template
        game_no = f"game {self.game_dir.name}" if self.game_dir.name.isdigit() else None
        self.label = " · ".join(str(x) for x in (self.run.get("model"), f"seed {self.run.get('seed')}", effort, game_no) if x)
        self.inputs, self.decisions = self._replay()
        self.total = len(self.decisions)

    def _replay(self) -> tuple[list[int], list[dict]]:
        """Press the journal's recorded stream into a fresh game, checking every tick against the journal;
        refuse at the first divergence."""
        game = Game(self.run["seed"])
        inputs: list[int] = []

        def diverged(tick, what: str):
            sys.exit(f"topout-video: {self.game_dir}: replay diverges from the journal at tick {tick} "
                     f"({what}); the harness replaying it is not the one that played it")

        decisions = []
        for rec in self.ticks:
            info = game.info()
            if (info["pieces"], info["lines"]) != (rec["pieces"], rec["lines"]):
                diverged(rec["tick"], f"pieces/lines {info['pieces']}/{info['lines']} vs {rec['pieces']}/{rec['lines']}")
            start = len(inputs)
            if not rec.get("no_piece") and game.board_text() != rec["board"]:
                diverged(rec["tick"], "board before the decision")
            for buttons, n in rec["inputs"]:
                for _ in range(n):
                    if game.done:
                        break
                    inputs.append(buttons)
                    game.press(buttons, 1)
            if not rec.get("no_piece"):
                decisions.append({**rec, "start": start, "n": len(decisions) + 1, "burned": burned(rec)})
            if len(inputs) - start != rec["frames"]:
                diverged(rec["tick"], f"frames {len(inputs) - start} vs {rec['frames']}")
        return inputs, decisions

    def current(self, f: int) -> dict | None:
        """The decision in force at game frame f: the last one pressed at or before it."""
        lo, hi, best = 0, len(self.decisions) - 1, None
        while lo <= hi:
            mid = (lo + hi) // 2
            if self.decisions[mid]["start"] <= f:
                best, lo = self.decisions[mid], mid + 1
            else:
                hi = mid - 1
        return best


def icon(button: str, failed: bool) -> Image.Image:
    """A button press as a small RGBA glyph, drawn (not a font glyph)."""
    im = Image.new("RGBA", (ICON, ICON), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    c = COLORS["failed" if failed else button] + (255,)
    d.rounded_rectangle((1, 1, ICON - 2, ICON - 2), radius=8, outline=c, width=2, fill=(30, 30, 40, 255))
    m = ICON // 2
    if button in ("left", "right"):
        s = -1 if button == "left" else 1
        d.polygon([(m + s * 12, m), (m - s * 4, m - 11), (m - s * 4, m + 11)], fill=c)
        d.rectangle((m - 12, m - 4, m + 2, m + 4) if s > 0 else (m - 2, m - 4, m + 12, m + 4), fill=c)
    elif button == "rotate":
        d.arc((m - 12, m - 12, m + 12, m + 12), start=-60, end=230, fill=c, width=4)
        d.polygon([(m + 7, m - 15), (m + 15, m - 5), (m + 3, m - 3)], fill=c)
    elif button == "drop":
        for dy in (-8, 2):
            d.polygon([(m - 10, m + dy - 4), (m + 10, m + dy - 4), (m, m + dy + 8)], fill=c)
    else:   # wait
        d.rectangle((m - 8, m - 10, m - 3, m + 10), fill=c)
        d.rectangle((m + 3, m - 10, m + 8, m + 10), fill=c)
    return im


_ICONS: dict = {}


def ribbons(arm: Arm, f: int) -> Image.Image:
    """Buttons above, token bars below, both in game time: a press at frame s sits (f - s) * PXF pixels left
    of the right edge."""
    strip = Image.new("RGB", (PW, ICON + 6 + BAR_H), (12, 12, 18))
    d = ImageDraw.Draw(strip)
    for dec in reversed(arm.decisions):
        if dec["start"] > f:
            continue
        x = PW - ICON - (f - dec["start"]) * PXF
        if x < -ICON:
            break
        failed = bool(dec.get("parse_error"))
        im = _ICONS.setdefault((dec["action"], failed), icon(dec["action"], failed))
        strip.paste(im.convert("RGB"), (int(x), 0), im.getchannel("A"))
        tokens = dec["burned"]
        color = next(c for floor, c in TOKEN_BANDS if tokens >= floor)
        h = max(2, round(BAR_H * min(tokens, 48_000) / 48_000))
        d.rectangle((int(x) + 6, ICON + 6 + BAR_H - h, int(x) + ICON - 7, ICON + 6 + BAR_H), fill=color)
    return strip


def fit(text: str, font, width: int) -> str:
    while text and font.getlength(text) > width:
        text = text[:-2] + "…"
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="game directories, journals or directories of games")
    ap.add_argument("-o", "--output", help="output mp4")
    ap.add_argument("--every", type=int, default=4, help="keep every Nth game frame (60.1 fps game)")
    ap.add_argument("--fps", type=int, default=30, help="the video's frame rate: fps * every / 60.1 is the speed-up")
    ap.add_argument("--wrap", type=int, default=4, help="screens per row; more games wrap onto further rows")
    a = ap.parse_args()
    if not shutil.which("ffmpeg"):
        sys.exit("topout-video: ffmpeg not on PATH")

    games = journals.find(a.paths)
    if a.output:
        out = Path(a.output)
    elif len(games) == 1:
        out = games[0].parent / "game.mp4"
    else:
        sys.exit("topout-video: several games need -o")
    arms = [Arm(g) for g in games]

    font, small = ImageFont.truetype(FONT, 24), ImageFont.truetype(FONT, 17)
    cols = max(1, min(len(arms), a.wrap))
    rows = -(-len(arms) // cols)
    panel_h = 240 * SCALE + CAP
    w, h = cols * PW + (cols - 1) * GAP, TOP + rows * panel_h + (rows - 1) * GAP
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{w}x{h}", "-framerate", str(a.fps), "-i", "-",
                           "-vf", "format=yuv420p", "-c:v", "libx264", "-crf", "18", str(out)],
                          stdin=subprocess.PIPE)
    emus = [Game(arm.run["seed"]) for arm in arms]     # playback: the recorded streams, frame by frame
    last = max(len(arm.inputs) for arm in arms)
    speed = a.fps * a.every / NES_FPS
    written = 0
    for f in range(last + 1):
        for arm, g in zip(arms, emus):
            if 0 < f <= len(arm.inputs):
                g.press(arm.inputs[f - 1], 1)
        if f % a.every and f != last:
            continue
        frame = Image.new("RGB", (w, h), (12, 12, 18))
        d = ImageDraw.Draw(frame)
        secs = f / NES_FPS
        d.text((16, 12), f"game time {int(secs // 60)}:{secs % 60:04.1f}   ({speed:.1f}x)", font=font, fill=(240, 240, 240))
        for n, (arm, g) in enumerate(zip(arms, emus)):
            x0 = (n % cols) * (PW + GAP)
            y0 = TOP + (n // cols) * (panel_h + GAP)
            ended = f >= len(arm.inputs)
            img = Image.fromarray(np.asarray(g.env.screen)).resize((PW, 240 * SCALE), Image.NEAREST)
            if ended:
                img = Image.blend(img.convert("L").convert("RGB"), Image.new("RGB", img.size, (0, 0, 0)), 0.45)
            frame.paste(img, (x0, y0))
            y = y0 + 240 * SCALE + 8
            frame.paste(ribbons(arm, min(f, len(arm.inputs))), (x0, y))
            y += ICON + 6 + BAR_H + 10
            d.text((x0 + 8, y), fit(arm.label, small, PW - 16), font=small, fill=(170, 170, 190))
            y += 24
            info = g.info()
            if ended:
                s = arm.summary or {}
                how = ENDED.get(s.get("termination"), "ENDED")
                d.text((x0 + 8, y), f"{how}   lines {s.get('lines', info['lines'])}   pieces {s.get('pieces', info['pieces'])}",
                       font=font, fill=(255, 120, 120))
                continue
            d.text((x0 + 8, y), f"piece {info['pieces']}  lines {info['lines']}", font=font, fill=(230, 230, 230))
            y += 32
            dec = arm.current(f)
            if dec is None:
                continue
            d.text((x0 + 8, y), f"decision {dec['n']}/{arm.total}: {dec['action']}   {dec['burned']} tok",
                   font=small, fill=(120, 220, 255))
            y += 24
            if dec.get("plan"):
                body, color = dec["plan"], (215, 215, 200)
            else:
                d.text((x0 + 8, y), fit(f"no plan: {dec.get('parse_error') or 'empty reply'}", small, PW - 16),
                       font=small, fill=(255, 120, 120))
                y += 21
                body = "… " + " ".join((dec.get("reasoning") or "").split())[-(WRAP * (PLAN_LINES - 2) - 2):]
                color = (190, 170, 170)
            for line in textwrap.wrap(body, WRAP)[:PLAN_LINES - (not dec.get("plan"))]:
                d.text((x0 + 8, y), line, font=small, fill=color)
                y += 21
        ff.stdin.write(frame.tobytes())
        written += 1
    ff.stdin.close()
    if ff.wait() != 0:
        sys.exit(f"topout-video: ffmpeg exited {ff.returncode}")
    print(f"{out}: {len(arms)} screen(s), {last} game frames ({last / NES_FPS:.0f} s game time), "
          f"{written} video frames at {a.fps} fps ({written / a.fps:.0f} s, {speed:.1f}x)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
