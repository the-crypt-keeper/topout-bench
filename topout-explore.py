"""topout-explore: an interactive look at games, from every arm down to one decision.

    python topout-explore.py [--dataset DIR] [--arm NAME=PATH[,PATH...] ...] [--port 8050] [--host 127.0.0.1]

Contract. Reads finished journals through topout/store.py (games still running are skipped and counted) and serves a web
app until stopped. Writes only its cache (--cache). Every number is computed by the functions the command line scripts
use (topout/ledger.py, topout/decisions.py), and every figure is topout-plot's (topout/figures.py).

Four levels, each a click from the last: the arms (burden against tokens, every game, decision fingerprints), one arm
(its 16 games scrubbed together beside the triangle, its moves), one game (its boards, tokens and damage), one piece
(every reply: thinking, calls, presses). The rail chooses which arms the first level draws: it changes what is drawn,
never what a number means. The address holds the whole selection, so a view is a link.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import parse_qs, urlencode

import dash
import plotly.graph_objects as go
from dash import ALL, ClientsideFunction, Input, Output, State, dcc, html

from topout import store
from topout.data import Data, add_arguments, from_arguments
from topout.decisions import VERBS
from topout.figures import (MUTED, NOTES, PHASE, SHOW, VERB_COLORS, burden_figure, fingerprints, game_figure, games_figure,
                            moves_figure, moves_scale, shared, triangle_figure, triangle_scale)


def query(search: str | None) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs((search or "").lstrip("?")).items()}


def link(**q) -> str:
    q = {k: v for k, v in q.items() if v not in (None, "")}
    return "/?" + urlencode(q, safe="~") if q else "/"


MOST = 4                               # arms side by side


class Nav:
    """The first page's choices (which arms, which tokens), read from the address and carried by every link from it."""

    def __init__(self, data: "Data", q: dict[str, str], narrow: bool = False):
        self.narrow = narrow                          # a phone-sized window: the figures are laid out for it
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


def wide_graph(figure: go.Figure, nav: "Nav"):
    """A figure that needs its width: on a phone it scrolls sideways inside its card."""
    if not nav.narrow:
        return graph(figure)
    figure.update_layout(width=960, dragmode=False)
    figure.update_xaxes(fixedrange=True)
    figure.update_yaxes(fixedrange=True)
    return html.Div([html.P("Swipe sideways. These read best on a larger screen, where the columns line up in one view.", className="steer"),
                     html.Div(graph(figure), className="sideways")])


def readable(text: str) -> str:
    """The model's words, without the encoded blocks an Anthropic proxy appends to carry signed thinking between turns."""
    return re.sub(r"\s*\[anthropic-blocks:[^\]]*\]", "", text)


NOBAR = {"displayModeBar": False, "responsive": True}


def overview(data: Data, nav: Nav):
    names, stat = nav.names, nav.stat
    if not names:
        return html.Div("No arm selected: choose some on the left.", className="card")
    return [
        dcc.Link(f"Compare these {len(names)} side by side", href=nav.to(arm=names[0], vs="~".join(names[1:])), className="btn go")
        if 2 <= len(names) <= MOST else None,
        card(NOTES["burden"][0], " ".join([NOTES["burden"][1], *(NOTES[k] for k in SHOW), "One per model is chosen among the arms "
             "ticked on the left. Click an arm."]),
             html.Div([segments("show", SHOW, nav.show),
                       segments("stat", {"mean": "mean tokens", "median": "median tokens"}, stat)],
                      className="ctl"),
             graph(burden_figure(data, names, stat, nav.show, nav.narrow), id={"kind": "to-arm", "i": 0})),
        card(NOTES["games"][0], NOTES["games"][1] + " Click a game.",
             graph(games_figure(data, names, nav.narrow), id={"kind": "to-game", "i": 0})),
        card(*NOTES["decisions"],
             html.Div(["buttons: "] + [html.Span([html.I(style={"background": c}), v], className="key") for v, c in zip(VERBS, VERB_COLORS)]),
             wide_graph(fingerprints(data, names), nav))]


def arm_view(data: Data, nav: Nav, name: str, vs: list[str]):
    """One arm, or up to MOST of them side by side on one scrubber and one scale."""
    names = list(dict.fromkeys([name, *vs]))[:MOST]
    arms = [data.arms[n] for n in names]
    several = len(names) > 1
    payload = {"arms": [{"games": [dict(boards=[p["board"] for p in g["pieces"]], live=[p["live"] for p in g["pieces"]],
                                        dead=[p["dead"] for p in g["pieces"]], lines=[p["lines"] for p in g["pieces"]],
                                        survived=g["survived"]) for g in a["games"]]} for a in arms]}
    size = 50 if several or nav.narrow else 80
    scale, tri_scale = moves_scale([a["games"] for a in arms]), triangle_scale([a["games"] for a in arms])

    def column(j, n, a):
        boards = [dcc.Link([html.Canvas(id=f"b-{j}-{i}", width=size, height=2 * size), html.Span(id=f"t-{j}-{i}")], id=f"g-{j}-{i}",
                           href=nav.to(arm=n, game=g["key"])) for i, g in enumerate(a["games"])]
        return boards, graph(triangle_figure(a["games"], tri_scale, ends=False, narrow=nav.narrow), id={"kind": "triangle", "i": j}), \
            graph(moves_figure(a["games"], scale, nav.narrow)), \
            nums(("burden", f"{a['burden']:.2f}"), ("95% interval", f"{a['ci'][0]:.2f} to {a['ci'][1]:.2f}"),
                 ("tokens per piece, mean", f"{a['mean']:,.0f}"), ("median", f"{a['median']:,.0f}"),
                 ("reached the end", f"{a['reached']} of {len(a['games'])}"))
    cols = [column(j, n, a) for j, (n, a) in enumerate(zip(names, arms))]
    others = [n for n in data.arms if n != name]
    head = [crumbs(("All arms", nav.to()), (" vs ".join(names), None)),
            html.Div([html.Span("compare with"), dcc.Dropdown(others, names[1:], id="vs", multi=True, placeholder="another arm")], className="versus")]
    ctl = html.Div([html.Button("Play", id="play", className="btn"), html.B(id="k-label"),
                    html.Div(dcc.Slider(1, data.turns, 1, value=min(40, data.turns), id="k", marks=None, updatemode="drag"), className="grow"),
                    dcc.Store(id="arm-data", data=payload), dcc.Interval(id="tick", interval=200, disabled=True)], className="ctl scrub")
    about = ("Every game at the same piece, best burden first. Orange cells are sealed. Faded games have ended. Click one.")
    triangle = NOTES["triangle"][1] + " The cells fill in up to the piece above. Rings: games still going."
    moves = NOTES["moves"][1]
    how = card("How it decides" if not several else "How they decide", "Planning up, execution down, as on the first page.", wide_graph(fingerprints(data, names), nav))
    if not several:
        boards, tri, mv, numbers = cols[0]
        return head + [html.Div(numbers, className="card"),
                       html.Div([html.H2("Its games, together"), html.P(about), ctl, html.Div(boards, className="boards" + (" small" if nav.narrow else ""))], className="card"),
                       html.Div([card(NOTES["triangle"][0], triangle, tri), card(NOTES["moves"][0], moves, mv)], className="row"), how]
    title = lambda n, j: html.H2(dcc.Link(n, href=nav.to(arm=n)))
    side = lambda cells: html.Div([html.Div([title(n, j), *([c] if not isinstance(c, list) else c)], className="card")
                                   for j, (n, c) in enumerate(zip(names, cells))], className=f"side n{len(names)}")
    return head + [
        html.P("Arms are stacked here; a larger screen puts them beside each other.", className="steer") if nav.narrow else None,
        side([c[3] for c in cols]),
        card(NOTES["games"][0], NOTES["games"][1] + " Click a game.",
             graph(games_figure(data, names, nav.narrow), id={"kind": "to-game", "i": 0})),
        html.Div([html.P(about), ctl], className="card stick"),
        side([html.Div(c[0], className="boards small") for c in cols]),
        html.P(triangle, className="note"), side([c[1] for c in cols]),
        html.P(moves + " One scale for all.", className="note"), side([c[2] for c in cols]), how]


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
             "on the board each piece spawned into. Click a piece.", dcc.Graph(id={"kind": "to-piece", "i": f"{name}|{key}"}, figure=game_figure(g, nav.narrow), config=NOBAR)),
        card("Every board", "As each piece spawned. Click one.", html.Div(boards, className="boards" + (" small" if nav.narrow else "")),
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
                  html.Div(pair, className="pair")], className="card"),
        html.Div([html.H2("Its replies"), html.P("Blue: planning. Orange: execution. Boards are what it was shown before each press; grey is the falling piece."),
                  *body, dcc.Store(id="boards-data", data=draw), html.Span(id="boards-sink")], className="card")]


def build(data: Data) -> dash.Dash:
    app = dash.Dash(__name__, title="Topout Explorer", suppress_callback_exceptions=True, update_title=None,
                    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
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
    app.layout = html.Div([dcc.Location(id="url", refresh="callback-nav"), dcc.Store(id="narrow"),
                           html.Div([dcc.Link("Topout Explorer", href="/"), html.Button("Choose arms", id="menu", className="btn")], id="bar"),
                           html.Div(rail, id="rail"), html.Div(id="main")], id="app")

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

    @app.callback(Output("main", "children"), Input("url", "search"), Input("narrow", "data"))
    def show(search, narrow):
        if narrow is None:                                       # the window's size is not known yet
            return dash.no_update
        q = query(search)
        nav = Nav(data, q, narrow)
        try:
            if "piece" in q:
                return piece_view(data, nav, q["arm"], q["game"], int(q["piece"]))
            if "game" in q:
                return game_view(data, nav, q["arm"], q["game"])
            if "arm" in q:
                return arm_view(data, nav, q["arm"], [n for n in q.get("vs", "").split("~") if n])
        except (KeyError, ValueError):
            return html.Div(["Nothing at this address: ", dcc.Link("all arms", href="/")], className="card")
        return overview(data, nav)

    @app.callback(Output("url", "search", allow_duplicate=True), Input("vs", "value"), State("url", "search"), prevent_initial_call=True)
    def versus(value, search):
        q = query(search)
        if "arm" not in q or "~".join(value or []) == q.get("vs", ""):
            return dash.no_update
        return Nav(data, q).to(arm=q["arm"], vs="~".join((value or [])[:MOST - 1]))[1:]

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
    app.clientside_callback(js("arm"), Output({"kind": "triangle", "i": ALL}, "figure"), Output("k-label", "children"), Input("k", "value"), Input("arm-data", "data"),
                            State({"kind": "triangle", "i": ALL}, "figure"))
    app.clientside_callback(js("toggle"), Output("tick", "disabled"), Output("play", "children"), Output("k", "value"),
                            Input("play", "n_clicks"), State("tick", "disabled"), State("k", "value"), State("k", "max"))
    app.clientside_callback(js("step"), Output("k", "value", allow_duplicate=True), Output("tick", "disabled", allow_duplicate=True),
                            Output("play", "children", allow_duplicate=True), Input("tick", "n_intervals"), State("k", "value"),
                            State("tick", "disabled"), State("k", "max"), prevent_initial_call=True)
    app.clientside_callback(js("narrow"), Output("narrow", "data"), Input("url", "pathname"))
    app.clientside_callback(js("top"), Output("bar", "title"), Input("url", "search"))
    app.clientside_callback(js("menu"), Output("menu", "title"), Input("menu", "n_clicks"), prevent_initial_call=True)
    app.clientside_callback(js("boards"), Output("boards-sink", "children"), Input("boards-data", "data"))
    return app


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_arguments(ap)
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 to serve other machines; there is no login")
    ap.add_argument("--port", type=int, default=8050)
    a = ap.parse_args()
    data = from_arguments(ap, a)
    games = sum(len(x["games"]) for x in data.arms.values())
    print(f"{len(data.arms)} arms, {games} games" + (f", {data.skipped} still running skipped" if data.skipped else ""), flush=True)
    print(f"http://{a.host}:{a.port}", flush=True)
    from waitress import serve
    serve(build(data).server, host=a.host, port=a.port, threads=16)


if __name__ == "__main__":
    main()
