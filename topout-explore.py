"""topout-explore: an interactive look at games, from every arm down to one decision.

    python topout-explore.py [--dataset DIR] [--arm NAME=PATH[,PATH...] ...] [--port 8050] [--host 127.0.0.1]

Contract. Reads finished journals through topout/store.py (games still running are skipped and counted) and serves a web
app until stopped. Writes only its cache (--cache). Every number is computed by the functions the command line scripts
use (topout/ledger.py, topout/decisions.py).

Four levels, each a click from the last: the arms (burden against tokens, every game, decision fingerprints), one arm
(its 16 games scrubbed together beside the triangle, its moves), one game (its boards, tokens and damage), one piece
(every reply: thinking, calls, presses). The rail chooses which arms the first level draws: it changes what is drawn,
never what a number means. The address holds the whole selection, so a view is a link.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path
from urllib.parse import parse_qs, urlencode

import dash
import numpy as np
import plotly.graph_objects as go
from dash import ALL, ClientsideFunction, Input, Output, State, dcc, html
from plotly.subplots import make_subplots

from topout import decisions, index, ledger, store
from topout.decisions import BINS, CALLS, HIGH, LOW, REPLIES, VERBS

INK, MUTED, GRID, SURFACE, PLAN, EXEC, PIECE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6", "#eb6834", "#8a8984"
VERB_COLORS = ["#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
RAMP = [[0, "#cde2fb"], [0.5, "#3987e5"], [1, "#0d366b"]]
PHASE = ("planning", "execution")


class Data:
    """The digests, grouped by arm, and what each arm shows on the first level."""

    def __init__(self, arms: list[index.Arm], cache: Path):
        digests = iter(store.load([p for a in arms for p in a.paths], cache))
        self.arms: dict[str, dict] = {}
        self.skipped = 0
        turns = set()
        for a in arms:
            games = [next(digests) for _ in a.paths]
            self.skipped += sum(g is None for g in games)
            games = sorted((g for g in games if g), key=lambda g: g["burden"])
            if not games:
                continue
            ids = [g["id"].rsplit("/", 1)[-1][:10] for g in games]
            for i, g in enumerate(games):
                g["key"] = ids[i] if ids.count(ids[i]) == 1 else f"{ids[i]}-{i}"
            both, per_piece = decisions.merge(g["phases"] for g in games)
            xs = [g["burden"] for g in games]
            tokens = [sum(p["tokens"]) for g in games for p in g["pieces"]]
            turns |= {g["turns"] for g in games}
            self.arms[a.name] = dict(arm=a, games=games, by_key={g["key"]: g for g in games}, both=both,
                                     per_piece=per_piece, burden=float(np.mean(xs)), ci=ledger.interval(xs),
                                     mean=float(np.mean(tokens)), median=float(np.median(tokens)),
                                     reached=sum(g["survived"] for g in games))
        if not self.arms:
            raise SystemExit("topout-explore: no finished game in any arm")
        if len(turns) > 1:
            raise SystemExit(f"topout-explore: arms mix max_pieces {sorted(turns)}; burden needs one horizon")
        self.turns = turns.pop()


def query(search: str | None) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs((search or "").lstrip("?")).items()}


def link(**q) -> str:
    q = {k: v for k, v in q.items() if v not in (None, "")}
    return "/?" + urlencode(q, safe="~") if q else "/"


SHOW = {"ladders": "full ladder", "performance": "best performance", "cost": "cost frontier"}


class Nav:
    """The first page's choices (which arms, which tokens), read from the address and carried by every link from it."""

    def __init__(self, data: "Data", q: dict[str, str]):
        self.all = list(data.arms)
        named = set(q.get("only", q.get("hide", "")).split("~"))
        self.names = [n for n in self.all if (n in named if "only" in q else n not in named)]
        self.stat = "median" if q.get("stat") == "median" else "mean"
        self.show = q["show"] if q.get("show") in SHOW else "ladders"

    def to(self, names: list[str] | None = None, stat: str | None = None, show: str | None = None, **q) -> str:
        names, stat, show = self.names if names is None else names, stat or self.stat, show or self.show
        chosen = set(names)
        gone = [n for n in self.all if n not in chosen]
        kept = ({"only": "~".join(names) or "~"} if len(names) < len(gone) else {"hide": "~".join(gone)}) if gone else {}
        return link(**q, **kept, stat=stat if stat == "median" else None,
                    show=show if show != "ladders" else None)


def layout(fig: go.Figure, height: int, **kw) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=56, r=12, t=14, b=44), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                      font=dict(color=MUTED, size=12), showlegend=False, hoverlabel=dict(align="left"), **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def one_per_model(data: Data, names: list[str], stat: str, show: str) -> list[str]:
    """One arm of each model. performance: its lowest burden. cost: its arm nearest the top left corner, no burden at the
    fewest tokens of any arm; both axes run 0 to 1 over every arm loaded (tokens in log), not only those shown, so choosing
    arms never moves the corner or the scale."""
    every = data.arms.values()
    lo, hi = min(np.log(a[stat]) for a in every), max(np.log(a[stat]) for a in every)
    top = max(a["burden"] for a in every) or 1

    def far(n):
        a = data.arms[n]
        return a["burden"] if show == "performance" else np.hypot((np.log(a[stat]) - lo) / ((hi - lo) or 1), a["burden"] / top)
    best: dict[str, str] = {}
    for n in names:
        m = data.arms[n]["arm"].model
        if m not in best or far(n) < far(best[m]):
            best[m] = n
    return list(best.values())


def place_labels(points: list[tuple[float, float]], texts: list[str], bars: list[tuple[float, float]]) -> list[str]:
    """A side for each label (plotly textposition) that keeps it off the other labels, points and error bars. Coordinates
    are pixels, y down, for a plot of about the usual width; a much narrower window can still collide. Greedy, in order."""
    sides = {"bottom right": (0, 1), "top right": (0, -1), "middle right": (0, 0), "bottom left": (-1, 1), "top left": (-1, -1),
             "middle left": (-1, 0)}
    fixed = [(x - 6, y - 6, x + 6, y + 6) for x, y in points] + [(x - 3, a, x + 3, b) for (x, _), (a, b) in zip(points, bars)]
    placed: list[tuple] = []
    out = []
    area = lambda a, b: max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    for (x, y), text in zip(points, texts):
        w = 5.9 * len(re.sub("<[^>]*>", "", text))
        def box(side):
            left, down = sides[side]
            x0 = x - 8 - w if left else x + 8
            y0 = {1: y + 4, -1: y - 17, 0: y - 6}[down]
            return (x0, y0, x0 + w, y0 + 13)
        best = min(sides, key=lambda q: (round(sum(3 * area(box(q), b) for b in placed) + sum(area(box(q), b) for b in fixed)), list(sides).index(q)))
        placed.append(box(best))
        out.append(best)
    return out


def burden_figure(data: Data, names: list[str], stat: str, show: str) -> go.Figure:
    fig = go.Figure()
    if show != "ladders":
        names = one_per_model(data, names, stat, show)
    ladders: dict = {}
    for n in names:
        if data.arms[n]["arm"].ladder is not None:
            ladders.setdefault(data.arms[n]["arm"].ladder, []).append(n)
    for ns in ladders.values() if show == "ladders" else ():
        fig.add_scatter(x=[data.arms[n][stat] for n in ns], y=[data.arms[n]["burden"] for n in ns], mode="lines",
                        line=dict(color=PLAN, width=2), opacity=0.35, hoverinfo="skip")
    text = {n: n for n in names}
    for ns in ladders.values() if show == "ladders" else ():
        head, rest = shared(ns)
        if head:
            text |= dict(zip(ns, rest))
            text[ns[-1]] = f"<b>{head}</b> {rest[-1]}"
    A = [data.arms[n] for n in names]
    every = data.arms.values()                       # one range for every choice, so nothing moves when arms or the view change
    lo, hi = (np.log10(f(x[stat] for x in every)) for f in (min, max))
    xr, yr = [lo - .06, hi + .22], [max(x["ci"][1] for x in every) + 1, -.6]
    px = lambda t, b: (1100 * (np.log10(t) - xr[0]) / (xr[1] - xr[0]), 450 * (b - yr[1]) / (yr[0] - yr[1]))
    few = len(names) <= 30
    fig.add_scatter(
        x=[a[stat] for a in A], y=[a["burden"] for a in A], mode="markers+text" if few else "markers", text=[text[n] for n in names],
        textposition=place_labels([px(a[stat], a["burden"]) for a in A], [text[n] for n in names],
                                  [(px(1, a["ci"][0])[1], px(1, a["ci"][1])[1]) for a in A]), textfont=dict(size=10, color=INK), customdata=names,
        marker=dict(size=11, color=PLAN, line=dict(color=SURFACE, width=2)),
        error_y=dict(type="data", symmetric=False, array=[a["ci"][1] - a["burden"] for a in A],
                     arrayminus=[a["burden"] - a["ci"][0] for a in A], color=PLAN, thickness=1.3, width=4),
        hovertext=[f"<b>{n}</b><br>burden {a['burden']:.2f} [{a['ci'][0]:.2f}, {a['ci'][1]:.2f}]<br>tokens per piece: "
                   f"mean {a['mean']:,.0f}, median {a['median']:,.0f}<br>{a['reached']} of {len(a['games'])} reached the end"
                   for n, a in zip(names, A)], hoverinfo="text")
    fig.update_xaxes(type="log", title=f"completion tokens per piece, {stat} (log)", tickvals=[t * 10 ** e for e in range(2, 6) for t in (1, 2, 5)],
                     ticktext=[f"{t * 10 ** e / 1000:g}k" for e in range(2, 6) for t in (1, 2, 5)])
    fig.update_xaxes(range=xr)
    fig.update_yaxes(title="burden (up is better)", range=yr)
    return layout(fig, 520)


def games_figure(data: Data, names: list[str]) -> go.Figure:
    fig, rng = go.Figure(), np.random.default_rng(0)
    for survived, symbol, color, what in ((True, "square-open", PLAN, "reached the end"),
                                          (False, "circle-open", EXEC, "topped out")):
        rows = [(i, n, g) for i, n in enumerate(names) for g in data.arms[n]["games"] if g["survived"] == survived]
        fig.add_scatter(x=[i + rng.uniform(-.22, .22) for i, _, _ in rows], y=[g["burden"] for *_, g in rows],
                        mode="markers", marker=dict(symbol=symbol, color=color, size=8, line=dict(width=1.5)),
                        customdata=[[n, g["key"]] for _, n, g in rows], hoverinfo="text",
                        hovertext=[f"<b>{n}</b><br>game {g['key']}: {what}<br>burden {g['burden']:.2f}, {len(g['pieces'])} "
                                   f"pieces, {g['lines']} lines" for _, n, g in rows])
    fig.add_scatter(x=list(range(len(names))), y=[data.arms[n]["burden"] for n in names], mode="markers", hoverinfo="skip",
                    marker=dict(symbol="line-ew-open", size=22, color=INK, line=dict(width=2.5)))
    fig.update_xaxes(tickvals=list(range(len(names))), ticktext=names, tickangle=-50, automargin=False, range=[-.6, len(names) - .4])
    fig.update_yaxes(title="burden", autorange="reversed")
    layout(fig, 440)
    fig.update_layout(margin=dict(b=150, l=110))
    return fig


def fingerprints(data: Data, names: list[str]) -> go.Figure:
    """One row per arm in six columns. Each column is one set of bars with a per-row base, so a row costs no traces."""
    cols = ["completion tokens", "calls / reply", "buttons", "replies / piece", "truncated", "nudged"]
    fig = make_subplots(1, 6, shared_yaxes=True, horizontal_spacing=0.012, subplot_titles=cols,
                        column_widths=[2.4, 1, 1.7, 1.3, .8, .8])
    rows = {n: -i for i, n in enumerate(names)}
    both = {n: data.arms[n]["both"] for n in names}

    def mirrored(col, keys, values, hover, peak):
        for ph, sign, color in ((0, 1, PLAN), (1, -1, EXEC)):
            xs, ys, bs, hs = [], [], [], []
            for n in names:
                vs = values(n, ph)
                top = peak(n, ph, vs) or 1
                for x, k in enumerate(keys):
                    if vs[x]:
                        xs.append(x); ys.append(sign * .44 * vs[x] / top); bs.append(rows[n]); hs.append(hover(n, ph, k, vs[x]))
            fig.add_bar(x=xs, y=ys, base=bs, marker=dict(color=color, line=dict(width=0)), width=.88, hovertext=hs,
                        hoverinfo="text", row=1, col=col)

    edge = lambda k: 2 ** (LOW + k / BINS * (HIGH - LOW))
    tok = {(n, ph): decisions.share(both[n][ph]["bins"], range(BINS)) for n in names for ph in (0, 1)}
    mirrored(1, range(BINS), lambda n, ph: tok[n, ph],
             lambda n, ph, k, v: f"<b>{n}</b><br>{PHASE[ph]}: {v:.1%} of replies<br>{edge(k):,.0f} to {edge(k + 1):,.0f} tokens",
             lambda n, ph, vs: max(vs))
    for ph, sign in ((0, 1), (1, -1)):              # the truncated part, drawn over its bar
        xs, ys, bs, hs = [], [], [], []
        for n in names:
            d = both[n][ph]
            for k, v in d["cut"].items():
                xs.append(k); ys.append(sign * .44 * v / (d["n"] or 1) / (max(tok[n, ph]) or 1)); bs.append(rows[n])
                hs.append(f"<b>{n}</b><br>{PHASE[ph]}: {v / d['n']:.1%} of replies truncated here")
        fig.add_bar(x=xs, y=ys, base=bs, marker=dict(color=INK), width=.88, hovertext=hs, hoverinfo="text", row=1, col=1)
    calls = {(n, ph): decisions.share(both[n][ph]["calls"], range(CALLS + 1)) for n in names for ph in (0, 1)}
    ctop = [max(max(calls[n, ph]) for n in names) for ph in (0, 1)]
    mirrored(2, range(CALLS + 1), lambda n, ph: calls[n, ph],
             lambda n, ph, k, v: f"<b>{n}</b><br>{PHASE[ph]}: {v:.1%} of replies made {k}{'+' if k == CALLS else ''} calls",
             lambda n, ph, vs: ctop[ph])
    for v, color in zip(VERBS, VERB_COLORS):
        xs, ys, bs, hs = [], [], [], []
        for n in names:
            for ph, off in ((0, .2), (1, -.2)):
                sh = dict(zip(VERBS, decisions.share(both[n][ph]["verbs"], VERBS)))
                xs.append(sh[v]); ys.append(rows[n] + off); bs.append(sum(sh[w] for w in VERBS[:VERBS.index(v)]))
                hs.append(f"<b>{n}</b><br>{PHASE[ph]}: {v} {sh[v]:.1%}")
        fig.add_bar(x=xs, y=ys, base=bs, orientation="h", width=.34, marker=dict(color=color, line=dict(color=SURFACE, width=1)),
                    hovertext=hs, hoverinfo="text", row=1, col=3)
    per = {n: decisions.share(data.arms[n]["per_piece"], range(1, REPLIES + 1)) for n in names}
    ptop = max(max(v) for v in per.values()) or 1
    xs, ys, bs, hs = [], [], [], []
    for n in names:
        for k, v in enumerate(per[n], 1):
            if v:
                xs.append(k); ys.append(.86 * v / ptop); bs.append(rows[n] - .43)
                hs.append(f"<b>{n}</b><br>{v:.1%} of pieces took {k}{'+' if k == REPLIES else ''} replies")
    fig.add_bar(x=xs, y=ys, base=bs, width=.88, marker=dict(color=PIECE), hovertext=hs, hoverinfo="text", row=1, col=4)
    for col, key in ((5, "truncated"), (6, "nudged")):
        top = max(decisions.rates(both[n][ph])[key] for n in names for ph in (0, 1)) or 1
        for ph, off, color in ((0, .2, PLAN), (1, -.2, EXEC)):
            v = [decisions.rates(both[n][ph])[key] for n in names]
            fig.add_bar(x=v, y=[rows[n] + off for n in names], orientation="h", width=.3, marker=dict(color=color),
                        text=[f"{x:.1%}" if x else "0" for x in v], textposition="outside", cliponaxis=False,
                        hovertext=[f"<b>{n}</b><br>{PHASE[ph]}: {x:.1%} {key}" for n, x in zip(names, v)], hoverinfo="text",
                        row=1, col=col)
        fig.update_xaxes(range=[0, top * 1.9], showticklabels=False, showgrid=False, row=1, col=col)
    layout(fig, 90 + 74 * len(names), barmode="overlay")
    fig.update_layout(margin=dict(l=190, t=40))
    fig.update_annotations(font=dict(size=13, color=INK))
    fig.update_yaxes(tickvals=list(rows.values()), ticktext=names, range=[-len(names) + .45, .55], showgrid=False)
    caps = sorted({c for n in names for g in data.arms[n]["games"] if (c := g["sampler"].get("max_tokens"))})
    marks = sorted({16, 256, 1024, 4096, 16384} | {c for c in caps if c > 16384})
    fig.update_xaxes(tickvals=[(np.log2(m) - LOW) / (HIGH - LOW) * BINS - .5 for m in marks],
                     ticktext=[str(m) if m < 1000 else f"{m // 1000}k" for m in marks], range=[-.6, BINS - .4], row=1, col=1)
    fig.update_xaxes(tickvals=list(range(CALLS + 1)), ticktext=[*map(str, range(CALLS)), f"{CALLS}+"], range=[-.6, CALLS + .6], row=1, col=2)
    fig.update_xaxes(range=[0, 1], showticklabels=False, row=1, col=3)
    fig.update_xaxes(tickvals=[1, 4, 8, 12, REPLIES], ticktext=["1", "4", "8", "12", f"{REPLIES}+"], range=[.4, REPLIES + .6], row=1, col=4)
    return fig


def moves_figure(games: list[dict]) -> go.Figure:
    moves = [(p["cleared"], p["sealing"]) for g in games for p in g["pieces"] if p["sealing"] is not None]
    n = Counter(moves)
    ys = [s for _, s in moves] or [0]
    y0, y1 = min(0, *ys), max(0, *ys)
    z = [[np.log10(n[x, y]) if n[x, y] else None for x in range(5)] for y in range(y0, y1 + 1)]
    text = [[f"{x} lines, {y:+d} sealed cells<br>{n[x, y]} of {len(moves)} placements" for x in range(5)] for y in range(y0, y1 + 1)]
    fig = go.Figure(go.Heatmap(z=z, x=list(range(5)), y=list(range(y0, y1 + 1)), colorscale=RAMP, showscale=False, xgap=2,
                               ygap=1, text=text, hovertemplate="%{text}<extra></extra>"))
    head = max(3, .12 * (y1 - y0))
    for x in range(5):
        s = [b for a, b in moves if a == x]
        if s:
            fig.add_annotation(x=x, y=y1 + head / 2, text=f"{np.mean(s):+.2f}<br>n={len(s)}", showarrow=False, font=dict(size=11, color=INK))
    fig.update_xaxes(title="lines cleared", dtick=1, range=[-.6, 4.6])
    fig.update_yaxes(title="sealed cells added (- dug out)", range=[y0 - 1, y1 + head + 1])
    return layout(fig, 400)


def nums(*pairs):
    return html.Div([html.Div([str(v), html.Span(k)]) for k, v in pairs], className="nums")


def segments(id: str, options: dict[str, str], value: str):
    """A segmented control: a radio whose labels are the segments (style.css, .segments)."""
    return dcc.RadioItems([{"label": v, "value": k} for k, v in options.items()], value, id=id, inline=True, className="segments")


def card(title, note, *body):
    return html.Div([html.H2(title), html.P(note), *body], className="card")


def crumbs(*parts):
    """(text, href or None) from the top level down."""
    out = []
    for i, (text, href) in enumerate(parts):
        out += ([" / "] if i else []) + [dcc.Link(text, href=href) if href else html.B(text)]
    return html.Div(out, className="crumbs")


def graph(figure: go.Figure, **kw):
    """A graph whose box is the figure's own height (a flexible box would collapse it)."""
    return dcc.Graph(figure=figure, config=NOBAR, style={"height": figure.layout.height}, **kw)


def readable(text: str) -> str:
    """The model's words, without the encoded blocks an Anthropic proxy appends to carry signed thinking between turns."""
    return re.sub(r"\s*\[anthropic-blocks:[^\]]*\]", "", text)


def shared(names: list[str]) -> tuple[str, list[str]]:
    """The words every name starts with, and what is left of each (never nothing)."""
    words = [n.split() for n in names]
    k = 0
    while len(names) > 1 and all(len(w) > k + 1 and w[:k + 1] == words[0][:k + 1] for w in words):
        k += 1
    return " ".join(words[0][:k]), [" ".join(w[k:]) for w in words]


NOBAR = {"displayModeBar": False, "responsive": True}


def overview(data: Data, nav: Nav):
    names, stat = nav.names, nav.stat
    if not names:
        return html.Div("No arm selected: choose some on the left.", className="card")
    return [
        card("Burden against thinking", "Burden: dead rows averaged over every turn of the game, a turn after the game ended "
             "counting as 20. Bars: 95% interval of the mean, resampling games. Full ladder: every arm, a line "
             "joining each effort ladder. Best performance: each model's lowest burden. Cost frontier: each model's arm nearest the top left "
             "corner, burden and tokens (log) weighing alike. One per model is chosen among the arms ticked on the left. Click an arm.",
             html.Div([segments("show", SHOW, nav.show),
                       segments("stat", {"mean": "mean tokens", "median": "median tokens"}, stat)],
                      className="ctl"),
             graph(burden_figure(data, names, stat, nav.show), id={"kind": "to-arm", "i": 0})),
        card("Every game", "Squares reached the end, circles topped out, the dash is the mean. Click a game.",
             graph(games_figure(data, names), id={"kind": "to-game", "i": 0})),
        card("How each arm decides", "Blue, up: planning replies. Orange, down: execution replies. Black: truncated. Each is a share of "
             "its own phase. Tokens are scaled to each half's tallest bar, calls to one scale for all arms drawn.",
             html.Div(["buttons: "] + [html.Span([html.I(style={"background": c}), v], className="key") for v, c in zip(VERBS, VERB_COLORS)]),
             graph(fingerprints(data, names)))]


def arm_view(data: Data, nav: Nav, name: str):
    a = data.arms[name]
    payload = {"games": [dict(boards=[p["board"] for p in g["pieces"]], live=[p["live"] for p in g["pieces"]],
                              dead=[p["dead"] for p in g["pieces"]], lines=[p["lines"] for p in g["pieces"]],
                              survived=g["survived"]) for g in a["games"]]}
    boards = [dcc.Link([html.Canvas(id=f"b-{i}", width=80, height=160), html.Span(id=f"t-{i}")], id=f"g-{i}",
                       href=nav.to(arm=name, game=g["key"])) for i, g in enumerate(a["games"])]
    return [
        crumbs(("All arms", nav.to()), (name, None)),
        html.Div(nums(("burden", f"{a['burden']:.2f}"), ("95% interval", f"{a['ci'][0]:.2f} to {a['ci'][1]:.2f}"),
                      ("tokens per piece, mean", f"{a['mean']:,.0f}"), ("median", f"{a['median']:,.0f}"),
                      ("reached the end", f"{a['reached']} of {len(a['games'])}")), className="card"),
        html.Div([html.H2("Its games, together"), html.P("Every game at the same piece, best burden first. Orange cells are sealed. "
                  "Faded games have ended. Click one."),
                  html.Div([html.Button("Play", id="play", className="btn"), html.B(id="k-label"),
                            html.Div(dcc.Slider(1, data.turns, 1, value=min(40, data.turns), id="k", marks=None, updatemode="drag"), className="grow")], className="ctl"),
                  html.Div(boards, className="boards"),
                  dcc.Store(id="arm-data", data=payload), dcc.Interval(id="tick", interval=200, disabled=True)], className="card"),
        html.Div([card("The triangle", "Boards spawned so far, by live and dead rows (log colour). Rings: games still going.",
                       dcc.Graph(id="triangle", config=NOBAR, style={"height": 400})),
                  card("Its moves", "Each placement: lines cleared against sealed cells added. Numbers: the column's mean and count.",
                       graph(moves_figure(a["games"])))], className="row"),
        card("How it decides", "Planning up, execution down, as on the first page.", graph(fingerprints(data, [name])))]


def game_figure(g: dict) -> go.Figure:
    ps = g["pieces"]
    fig = make_subplots(2, 1, shared_xaxes=True, vertical_spacing=.06, row_heights=[.6, .4])
    x = [p["piece"] for p in ps]
    hover = [f"<b>piece {p['piece']} ({p['name']})</b><br>planning {p['tokens'][0]:,}, execution {p['tokens'][1]:,} tokens<br>"
             f"{sum(p['replies'])} replies, {len(p['presses'])} presses<br>cleared {p['cleared']}"
             + (f", sealed {p['sealing']:+d}" if p["sealing"] is not None else "")
             + (f"<br>{p['truncated']} truncated" if p["truncated"] else "") + (f"<br>{p['nudged']} nudged" if p["nudged"] else "") for p in ps]
    for ph, color in ((0, PLAN), (1, EXEC)):
        fig.add_bar(x=x, y=[p["tokens"][ph] for p in ps], marker=dict(color=color, line=dict(width=0)), hovertext=hover, hoverinfo="text",
                    customdata=x, row=1, col=1)
    top = max(sum(p["tokens"]) for p in ps)
    for key, symbol, off, color in (("truncated", "x", 1.05, INK), ("nudged", "triangle-down", 1.12, "#c0392b")):
        hit = [p for p in ps if p[key]]
        fig.add_scatter(x=[p["piece"] for p in hit], y=[top * off] * len(hit), mode="markers", marker=dict(symbol=symbol, color=color, size=8),
                        customdata=[p["piece"] for p in hit], hovertext=[f"piece {p['piece']}: {p[key]} {key}" for p in hit], hoverinfo="text", row=1, col=1)
    fig.add_bar(x=x, y=[p["dead"] for p in ps], marker=dict(color=EXEC), hovertext=hover, hoverinfo="text", customdata=x, row=2, col=1)
    fig.add_bar(x=x, y=[p["live"] for p in ps], marker=dict(color="#9ec5f4"), hovertext=hover, hoverinfo="text", customdata=x, row=2, col=1)
    fig.update_yaxes(title="tokens", row=1, col=1)
    fig.update_yaxes(title="dead + live rows", range=[0, 20], row=2, col=1)
    fig.update_xaxes(title="piece", row=2, col=1)
    return layout(fig, 480, barmode="stack")


def game_view(data: Data, nav: Nav, name: str, key: str):
    games = data.arms[name]["games"]
    g = data.arms[name]["by_key"][key]
    i = games.index(g)
    step = [dcc.Link(t, href=nav.to(arm=name, game=games[j]["key"]), className="btn") for t, j in (("Better game", i - 1), ("Worse game", i + 1)) if 0 <= j < len(games)]
    boards = [dcc.Link([html.Canvas(id=f"p-{p['piece']}", width=60, height=120), f"{p['piece']} {p['name']}"],
                       href=nav.to(arm=name, game=key, piece=p["piece"])) for p in g["pieces"]]
    return [
        crumbs(("All arms", nav.to()), (name, nav.to(arm=name)), (f"game {key}", None)), html.Div(step, className="ctl"),
        html.Div(nums(("burden", f"{g['burden']:.2f}"), ("pieces", len(g["pieces"])), ("lines", g["lines"]),
                      ("ended", "reached the end" if g["survived"] else "topped out"),
                      ("truncated replies", sum(p["truncated"] for p in g["pieces"])), ("nudges", sum(p["nudged"] for p in g["pieces"]))), className="card"),
        card("Tokens, and the board", "Blue planning, orange execution; crosses truncated, red triangles nudged. Below: dead (orange) and live (pale) rows "
             "on the board each piece spawned into. Click a piece.", dcc.Graph(id={"kind": "to-piece", "i": f"{name}|{key}"}, figure=game_figure(g), config=NOBAR)),
        card("Every board", "As each piece spawned. Click one.", html.Div(boards, className="boards"),
             dcc.Store(id="boards-data", data=[[f"p-{p['piece']}", p["board"]] for p in g["pieces"]]), html.Span(id="boards-sink"))]


def piece_view(data: Data, nav: Nav, name: str, key: str, number: int):
    g = data.arms[name]["by_key"][key]
    ps = g["pieces"]
    at = next((i for i, p in enumerate(ps) if p["piece"] == number), None)
    if at is None:
        raise KeyError(number)
    p, draw, body = ps[at], [], []
    after = ps[at + 1] if at + 1 < len(ps) else None
    for j, r in enumerate(store.replies(Path(g["path"]), number)):
        r |= {"reasoning": readable(r["reasoning"]), "content": readable(r["content"])}
        tags = [html.Span(t, className="tag bad" if bad else "tag") for t, bad in (
            (f"{PHASE[r['phase']]}", False), (f"{r['tokens']:,} tokens", False),
            (f"restarted: {r['reset']}", False) if r["reset"] and r["reset"] != "new piece" else (None, False),
            (f"truncated ({r['reason'] or 'cut off'})" if r["kind"] == "capped" else None, True),
            ("nudged: no tool call" if r["nudged"] else None, True)) if t]
        strip = []
        for m, q in enumerate(r["presses"]):
            cid = f"r-{j}-{m}"
            draw.append([cid, q["board"].replace("\n", "")])
            strip.append(html.Div([html.Canvas(id=cid, width=60, height=120), f"{q['action']}" + (" (lost decision)" if q["lost"] else "")],
                                  style={"fontSize": 11, "textAlign": "center", "color": MUTED}))
        body.append(html.Div([
            html.Div(tags), html.Div(["calls: ", html.Code(" ".join(r["calls"]) or "none")]),
            html.Div(strip, className="boards") if strip else None,
            html.Details([html.Summary(f"thinking ({len(r['reasoning']):,} characters)"), html.Pre(r["reasoning"])], open=j == 0) if r["reasoning"] else None,
            html.Details([html.Summary("said"), html.Pre(r["content"])]) if r["content"].strip() else None],
            className="reply" + (" x" if r["phase"] else "")))
    draw.append(["before", p["board"]])
    pair = [html.Div([html.Canvas(id="before", width=140, height=280, className="big"), html.Div("as it spawned")])]
    if after:
        draw.append(["after", after["board"]])
        pair.append(html.Div([html.Canvas(id="after", width=140, height=280, className="big"), html.Div("as the next one spawned")]))
    step = [dcc.Link(t, href=nav.to(arm=name, game=key, piece=ps[j]["piece"]), className="btn")
            for t, j in (("Previous piece", at - 1), ("Next piece", at + 1)) if 0 <= j < len(ps)]
    seal = "not read (overlap or last piece)" if p["sealing"] is None else f"{p['sealing']:+d}"
    return [
        crumbs(("All arms", nav.to()), (name, nav.to(arm=name)), (f"game {key}", nav.to(arm=name, game=key)), (f"piece {number} ({p['name']})", None)),
        html.Div(step, className="ctl"),
        html.Div([nums(("lines cleared", p["cleared"]), ("sealed cells", seal), ("planning tokens", f"{p['tokens'][0]:,}"),
                       ("execution tokens", f"{p['tokens'][1]:,}"), ("replies", sum(p["replies"])), ("presses", len(p["presses"]))),
                  html.Div(pair, className="row", style={"marginTop": 14})], className="card"),
        html.Div([html.H2("Its replies"), html.P("Blue: planning. Orange: execution. Boards are what it was shown before each press; grey is the falling piece."),
                  *body, dcc.Store(id="boards-data", data=draw), html.Span(id="boards-sink")], className="card")]


def build(data: Data) -> dash.Dash:
    app = dash.Dash(__name__, title="Topout Explorer", suppress_callback_exceptions=True, update_title=None,
                    assets_folder=str(Path(__file__).parent / "topout" / "explore_assets"))
    groups: dict[str, list[str]] = {}
    for n, a in data.arms.items():
        groups.setdefault(a["arm"].model, []).append(n)
    ids = list(groups)

    rail = [html.H1(dcc.Link("Topout Explorer", href="/")),
            html.Button("All", id="all"), html.Button("None", id="none")]
    for i, m in enumerate(ids):
        head, opts = shared(groups[m])
        if len(groups[m]) > 1:
            rail.append(html.Div(head, className="model"))
        rail.append(dcc.Checklist([{"label": f" {o}", "value": n} for o, n in zip(opts, groups[m])], groups[m],
                                  id={"kind": "pick", "i": i}, inline=True, className="model-one" if len(groups[m]) == 1 else ""))
    if data.skipped:
        rail.append(html.P(f"{data.skipped} games still running, skipped"))
    app.layout = html.Div([dcc.Location(id="url"), html.Div(rail, id="rail"), html.Div(id="main")], id="app")

    # the address is the only state: the rail shows it, and changing the rail changes it
    @app.callback(Output({"kind": "pick", "i": ALL}, "value"), Input("url", "search"), State({"kind": "pick", "i": ALL}, "value"))
    def show_choice(search, now):
        chosen = set(Nav(data, query(search)).names)
        want = [[n for n in groups[m] if n in chosen] for m in ids]
        return [dash.no_update if set(w) == set(v or []) else w for w, v in zip(want, now)]

    @app.callback(Output("url", "search"), Input({"kind": "pick", "i": ALL}, "value"), Input("all", "n_clicks"), Input("none", "n_clicks"),
                  State("url", "search"), prevent_initial_call=True)
    def choose(values, _all, _none, search):
        nav, by = Nav(data, query(search)), dash.ctx.triggered_id
        picked = {n for v in values for n in (v or [])}
        names = nav.all if by == "all" else [] if by == "none" else [n for n in nav.all if n in picked]
        return dash.no_update if names == nav.names else nav.to(names)[1:]

    @app.callback(Output("main", "children"), Input("url", "search"))
    def show(search):
        q = query(search)
        nav = Nav(data, q)
        try:
            if "piece" in q:
                return piece_view(data, nav, q["arm"], q["game"], int(q["piece"]))
            if "game" in q:
                return game_view(data, nav, q["arm"], q["game"])
            if "arm" in q:
                return arm_view(data, nav, q["arm"])
        except (KeyError, ValueError):
            return html.Div(["Nothing at this address: ", dcc.Link("all arms", href="/")], className="card")
        return overview(data, nav)

    @app.callback(Output("url", "search", allow_duplicate=True), Input("stat", "value"), Input("show", "value"), State("url", "search"),
                  prevent_initial_call=True)
    def toggles(stat, show, search):
        nav = Nav(data, query(search))
        return dash.no_update if (nav.stat, nav.show) == (stat, show) else nav.to(stat=stat, show=show)[1:]

    @app.callback(Output("url", "search", allow_duplicate=True), Input({"kind": "to-arm", "i": ALL}, "clickData"),
                  Input({"kind": "to-game", "i": ALL}, "clickData"), Input({"kind": "to-piece", "i": ALL}, "clickData"),
                  State("url", "search"), prevent_initial_call=True)
    def click(_arm, _game, _piece, search):
        t, hit = dash.ctx.triggered_id, dash.ctx.triggered[0]["value"]
        if not t or not hit or (c := hit["points"][0].get("customdata")) is None:
            return dash.no_update
        nav = Nav(data, query(search))
        if t["kind"] == "to-arm":
            return nav.to(arm=c)[1:]
        if t["kind"] == "to-game":
            return nav.to(arm=c[0], game=c[1])[1:]
        name, key = t["i"].rsplit("|", 1)
        return nav.to(arm=name, game=key, piece=c)[1:]

    js = lambda f: ClientsideFunction("topout", f)
    app.clientside_callback(js("arm"), Output("triangle", "figure"), Output("k-label", "children"), Input("k", "value"), Input("arm-data", "data"))
    app.clientside_callback(js("toggle"), Output("tick", "disabled"), Output("play", "children"), Output("k", "value"),
                            Input("play", "n_clicks"), State("tick", "disabled"), State("k", "value"), State("k", "max"))
    app.clientside_callback(js("step"), Output("k", "value", allow_duplicate=True), Output("tick", "disabled", allow_duplicate=True),
                            Output("play", "children", allow_duplicate=True), Input("tick", "n_intervals"), State("k", "value"),
                            State("tick", "disabled"), State("k", "max"), prevent_initial_call=True)
    app.clientside_callback(js("boards"), Output("boards-sink", "children"), Input("boards-data", "data"))
    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dataset", type=Path, help="local copy of a results dataset: every <model>/<sampler> is an arm")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=PATH[,PATH...]")
    ap.add_argument("--cache", type=Path, default=Path.home() / ".cache" / "topout-explore")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to serve other machines; there is no login")
    ap.add_argument("--port", type=int, default=8050)
    a = ap.parse_args()
    if not a.dataset and not a.arm:
        ap.error("give --dataset, --arm or both")
    data = Data(index.arms(a.dataset, a.arm), a.cache)
    games = sum(len(x["games"]) for x in data.arms.values())
    print(f"{len(data.arms)} arms, {games} games" + (f", {data.skipped} still running skipped" if data.skipped else ""), flush=True)
    print(f"http://{a.host}:{a.port}", flush=True)
    from waitress import serve
    serve(build(data).server, host=a.host, port=a.port, threads=16)


if __name__ == "__main__":
    main()
