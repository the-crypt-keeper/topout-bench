"""Every figure, for topout-explore and topout-plot alike.

Contract. Turns arms (topout/data.py) and game digests (topout/store.py) into plotly figures; reads no file, writes
none, and computes nothing a figure alone would know. `narrow` lays a figure out for a phone.
"""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from topout import decisions
from topout.data import Data
from topout.decisions import BINS, CALLS, HIGH, LOW, REPLIES, VERBS
from topout.ledger import ROWS

INK, MUTED, GRID, SURFACE, PLAN, EXEC, PIECE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#2a78d6", "#eb6834", "#8a8984"
VERB_COLORS = ["#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
RAMP = [[0, "#cde2fb"], [0.5, "#3987e5"], [1, "#0d366b"]]
PHASE = ("planning", "execution")
SHOW = {"ladders": "full ladder", "performance": "best performance", "cost": "cost frontier"}

# What each figure shows, for the page around it and for the title of a still image.
NOTES = {
    "burden": ("Burden against thinking", "Burden: dead rows averaged over every turn of the game, a turn after the game ended counting as 20. "
               "Bars: 95% interval of the mean, resampling games."),
    "ladders": "Full ladder: every arm, a line joining each effort ladder.",
    "performance": "Best performance: each model's lowest burden.",
    "cost": "Cost frontier: each model's arm nearest the top left corner, burden and tokens (log) weighing alike.",
    "games": ("Every game", "Squares reached the end, circles topped out, the dash is the mean."),
    "decisions": ("How each arm decides", "Blue, up: planning replies. Orange, down: execution replies. Black: truncated. Each is a share of its "
                  "own phase. Tokens are scaled to each half's tallest bar, calls to one scale for all arms drawn."),
    "triangle": ("The triangle", "Boards spawned, by live and dead rows (log colour, share of the arm's pieces)."),
    "moves": ("Moves", "Each placement: lines cleared against sealed cells added. Numbers: the column's mean and count."),
}


def layout(fig: go.Figure, height: int, narrow: bool = False, **kw) -> go.Figure:
    if narrow:                                       # a finger's drag scrolls the page instead of zooming the plot
        kw["dragmode"] = False
    fig.update_layout(height=height, margin=dict(l=56, r=12, t=14, b=44), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                      font=dict(color=MUTED, size=12), showlegend=False, hoverlabel=dict(align="left"), **kw)
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    if narrow:
        fig.update_layout(margin=dict(l=40, r=6), font=dict(size=11))
        fig.update_xaxes(fixedrange=True)
        fig.update_yaxes(fixedrange=True, title_standoff=2)
    return fig


def place_labels(points: list[tuple[float, float]], texts: list[str], bars: list[tuple[float, float]], right: float = 1e9) -> list[str]:
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
        best = min(sides, key=lambda q: (round(sum(3 * area(box(q), b) for b in placed) + sum(area(box(q), b) for b in fixed)
                                          + 200 * (max(0, box(q)[2] - right) + max(0, -box(q)[0]))), list(sides).index(q)))
        placed.append(box(best))
        out.append(best)
    return out


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


def burden_figure(data: Data, names: list[str], stat: str, show: str, narrow: bool = False) -> go.Figure:
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
            if narrow:                               # room for the ladder's name only
                text |= {n: "" for n in ns[:-1]} | {ns[-1]: f"<b>{head}</b>"}
    A = [data.arms[n] for n in names]
    every = data.arms.values()                       # one range for every choice, so nothing moves when arms or the view change
    lo, hi = (np.log10(f(x[stat] for x in every)) for f in (min, max))
    xr, yr = [lo - .06, hi + (.1 if narrow else .22)], [max(x["ci"][1] for x in every) + 1, -.6]
    wide = 284 if narrow else 1100
    px = lambda t, b: (wide * (np.log10(t) - xr[0]) / (xr[1] - xr[0]), 450 * (b - yr[1]) / (yr[0] - yr[1]))
    few = len(names) <= 30
    fig.add_scatter(
        x=[a[stat] for a in A], y=[a["burden"] for a in A], mode="markers+text" if few else "markers", text=[text[n] for n in names],
        textposition=place_labels([px(a[stat], a["burden"]) for a in A], [text[n] for n in names],
                                  [(px(1, a["ci"][0])[1], px(1, a["ci"][1])[1]) for a in A], wide if narrow else 1e9), textfont=dict(size=10, color=INK), customdata=names,
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
    return layout(fig, 460 if narrow else 520, narrow)


def games_figure(data: Data, names: list[str], narrow: bool = False) -> go.Figure:
    """Every game's burden, one column per arm; on a phone one row per arm, with best on the right."""
    fig, rng, jit = go.Figure(), np.random.default_rng(0), .22 if narrow else .22 * min(1, len(names) / 8)
    place = (lambda at, b: dict(x=b, y=at)) if narrow else (lambda at, b: dict(x=at, y=b))
    for survived, symbol, color, what in ((True, "square-open", PLAN, "reached the end"),
                                          (False, "circle-open", EXEC, "topped out")):
        rows = [(i, n, g) for i, n in enumerate(names) for g in data.arms[n]["games"] if g["survived"] == survived]
        fig.add_scatter(**place([i + rng.uniform(-jit, jit) for i, _, _ in rows], [g["burden"] for *_, g in rows]),
                        mode="markers", marker=dict(symbol=symbol, color=color, size=8, line=dict(width=1.5)),
                        customdata=[[n, g["key"]] for _, n, g in rows], hoverinfo="text",
                        hovertext=[f"<b>{n}</b><br>game {g['key']}: {what}<br>burden {g['burden']:.2f}, {len(g['pieces'])} "
                                   f"pieces, {g['lines']} lines" for _, n, g in rows])
    fig.add_scatter(**place(list(range(len(names))), [data.arms[n]["burden"] for n in names]), mode="markers", hoverinfo="skip",
                    marker=dict(symbol="line-ns-open" if narrow else "line-ew-open", size=14 if narrow else 22, color=INK, line=dict(width=2.5)))
    at = dict(tickvals=list(range(len(names))), ticktext=names)
    if narrow:
        fig.update_yaxes(**at, range=[len(names) - .4, -.6], automargin=False)
        fig.update_xaxes(title="burden (right is better)", autorange="reversed")
        layout(fig, 90 + 30 * len(names), True)
        fig.update_layout(margin=dict(l=158))
        fig.update_yaxes(tickfont=dict(size=10))
        return fig
    slant = len(names) > 6
    fig.update_xaxes(**at, tickangle=-50 if slant else 0, automargin=False, range=[-.6, len(names) - .4])
    fig.update_yaxes(title="burden", autorange="reversed")
    layout(fig, 440 if slant else 320)
    fig.update_layout(margin=dict(b=150, l=110) if slant else dict(b=44))
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
        top = max(.01, *(decisions.rates(both[n][ph])[key] for n in names for ph in (0, 1)))
        for ph, off, color in ((0, .2, PLAN), (1, -.2, EXEC)):
            v = [decisions.rates(both[n][ph])[key] for n in names]
            fig.add_bar(x=v, y=[rows[n] + off for n in names], orientation="h", width=.3, marker=dict(color=color),
                        text=[f"{x:.1%}" if x else "0" for x in v], textposition="outside", cliponaxis=False,
                        hovertext=[f"<b>{n}</b><br>{PHASE[ph]}: {x:.1%} {key}" for n, x in zip(names, v)], hoverinfo="text",
                        row=1, col=col)
        fig.update_xaxes(range=[0, top * 1.9], showticklabels=False, showgrid=False, row=1, col=col)
    layout(fig, 90 + 74 * len(names), barmode="overlay")
    fig.update_layout(margin=dict(l=190, t=40, r=12))
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


def moves_of(games: list[dict]) -> list[tuple[int, int]]:
    return [(p["cleared"], p["sealing"]) for g in games for p in g["pieces"] if p["sealing"] is not None]


def moves_scale(groups: list[list[dict]]) -> tuple[int, int, float, float]:
    """The sealed-cells range and the log share range that every one of these arms' moves are drawn on."""
    every = [moves_of(g) for g in groups]
    ys = [y for m in every for _, y in m] or [0]
    shares = [np.log10(c / len(m)) for m in every if m for c in Counter(m).values()] or [0]
    return min(0, *ys), max(0, *ys), min(shares), max(shares)


def moves_figure(games: list[dict], scale: tuple, narrow: bool = False) -> go.Figure:
    moves, (y0, y1, z0, z1) = moves_of(games), scale
    n = Counter(moves)
    z = [[np.log10(n[x, y] / len(moves)) if n[x, y] else None for x in range(5)] for y in range(y0, y1 + 1)]
    text = [[f"{x} lines, {y:+d} sealed cells<br>{n[x, y]} of {len(moves)} placements" for x in range(5)] for y in range(y0, y1 + 1)]
    fig = go.Figure(go.Heatmap(z=z, x=list(range(5)), y=list(range(y0, y1 + 1)), zmin=z0, zmax=z1, colorscale=RAMP, showscale=False,
                               xgap=2, ygap=1, text=text, hovertemplate="%{text}<extra></extra>"))
    head = max(3, .12 * (y1 - y0))
    for x in range(5):
        s = [b for a, b in moves if a == x]
        if s:
            fig.add_annotation(x=x, y=y1 + head / 2, text=f"{np.mean(s):+.2f}<br>n={len(s)}", showarrow=False, font=dict(size=11, color=INK))
    fig.update_xaxes(title="lines cleared", dtick=1, range=[-.6, 4.6])
    fig.update_yaxes(title="sealed cells added (- dug out)", range=[y0 - 1, y1 + head + 1])
    return layout(fig, 400, narrow)


def shared(names: list[str]) -> tuple[str, list[str]]:
    """The words every name starts with, and what is left of each (never nothing)."""
    words = [n.split() for n in names]
    k = 0
    while len(names) > 1 and all(len(w) > k + 1 and w[:k + 1] == words[0][:k + 1] for w in words):
        k += 1
    return " ".join(words[0][:k]), [" ".join(w[k:]) for w in words]


def game_figure(g: dict, narrow: bool = False) -> go.Figure:
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
    return layout(fig, 420 if narrow else 480, narrow, barmode="stack")


def triangle_grid(games: list[dict]) -> tuple[list[list[int]], int]:
    """Spawn boards by [dead rows][live rows], and how many there are."""
    z = [[0] * (ROWS + 1) for _ in range(ROWS + 1)]
    for g in games:
        for p in g["pieces"]:
            z[p["dead"]][p["live"]] += 1
    return z, sum(map(sum, z))


def triangle_scale(groups: list[list[dict]]) -> tuple[float, float]:
    """The log share range that every one of these arms' triangles is coloured on, from their whole games."""
    grids = [triangle_grid(g) for g in groups]
    return min(np.log10(1 / t) for _, t in grids), max(np.log10(max(map(max, z)) / t) for z, t in grids)


def triangle_figure(games: list[dict], scale: tuple[float, float], ends: bool = True, narrow: bool = False) -> go.Figure:
    """Trace 0 the cells, 1 the board's limit, 2 the rings (each game's last board, or none). The explorer's scrubber
    swaps the cells and the rings (explore_assets/boards.js) and leaves the rest as built here."""
    z, total = triangle_grid(games)
    fig = go.Figure(go.Heatmap(z=[[np.log10(c / total) if c else None for c in r] for r in z], customdata=z, zmin=scale[0], zmax=scale[1],
                               colorscale=RAMP, showscale=False, xgap=1, ygap=1,
                               hovertemplate="%{x} live, %{y} dead rows<br>%{customdata} pieces<extra></extra>"))
    fig.add_scatter(x=[0, ROWS], y=[ROWS, 0], mode="lines", line=dict(dash="dash", color=MUTED, width=1), hoverinfo="skip")
    last = Counter((g["pieces"][-1]["live"], g["pieces"][-1]["dead"], g["survived"]) for g in games) if ends else {}
    fig.add_scatter(x=[k[0] for k in last], y=[k[1] for k in last], mode="markers", text=[f"{n} games" for n in last.values()],
                    hovertemplate="%{text}<extra></extra>",
                    marker=dict(symbol=["square-open" if k[2] else "circle-open" for k in last], color=[INK if k[2] else EXEC for k in last],
                                size=[8 + 3 * n ** .5 for n in last.values()], line=dict(width=2)))
    axis = dict(range=[-.5, ROWS + .5], constrain="domain", dtick=5, fixedrange=True)
    layout(fig, 400, narrow, dragmode=False)
    fig.update_layout(margin=dict(l=44, r=8, t=8, b=40))
    fig.update_xaxes(title="live rows", constraintoward="left", **axis)
    fig.update_yaxes(title="dead rows", scaleanchor="x", **axis)
    return fig


def beside(figs: list[go.Figure], titles: list[str], columns: int = 3) -> go.Figure:
    """Single-panel figures of one kind as panels of one still figure."""
    columns = min(columns, len(figs))
    rows = -(-len(figs) // columns)
    out = make_subplots(rows, columns, subplot_titles=titles, horizontal_spacing=.06, vertical_spacing=.22 / max(rows, 2))
    out.update_annotations(font=dict(size=13, color=INK))
    for i, f in enumerate(figs):
        row, col = i // columns + 1, i % columns + 1
        n = i + 1
        for t in f.data:
            out.add_trace(t, row=row, col=col)
        for a in f.layout.annotations:
            out.add_annotation(a.to_plotly_json() | {"xref": f"x{n if n > 1 else ''}", "yref": f"y{n if n > 1 else ''}"})
        x, y = f.layout.xaxis.to_plotly_json(), f.layout.yaxis.to_plotly_json()
        if y.get("scaleanchor"):
            y["scaleanchor"] = f"x{n if n > 1 else ''}"
        out.update_xaxes(x, row=row, col=col)
        out.update_yaxes(y, row=row, col=col)
    return layout(out, 60 + 400 * rows)


def captioned(fig: go.Figure, title: str, note: str, width: int) -> go.Figure:
    """A figure that explains itself away from a page: a title and its note, wrapped, above it."""
    lines, seen = [""], lambda s: len(re.sub("<[^>]*>", "", s))
    for w in re.findall(r"(?:<[^>]*>|[^\s<])+", note):
        if seen(lines[-1]) + seen(w) > width / 6.6:
            lines.append("")
        lines[-1] += (" " if lines[-1] else "") + w
    top = 56 + 17 * len(lines)
    old = fig.layout.margin.t or 14
    fig.update_layout(width=width, height=fig.layout.height + top, margin=dict(t=old + top),
                      title=dict(text=f"<b>{title}</b><br><span style='font-size:12px;color:{MUTED}'>{'<br>'.join(lines)}</span>",
                                 x=0, xref="container", xanchor="left", pad=dict(l=16, t=18), y=1, yref="container", yanchor="top",
                                 font=dict(size=17, color=INK)))
    return fig


def verb_key() -> str:
    return "Buttons: " + ", ".join(f"<span style='color:{c}'>\u25a0</span>\u00a0{v}" for v, c in zip(VERBS, VERB_COLORS)) + "."
