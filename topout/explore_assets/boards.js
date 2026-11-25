// Boards drawn on canvas. A board is 200 characters, '#' filled, '.' empty; '@' is drawn as the falling piece.
window.dash_clientside = window.dash_clientside || {};
(function () {
  const INK = {"#": "#2a78d6", "@": "#8a8984"}, SEALED = "#eb6834", EMPTY = "#fcfcfb";
  function sealed(b) {                       // the ledger's flood: from the top row, down and sideways, never up
    const seen = new Set(), q = [];
    for (let c = 0; c < 10; c++) if (b[c] !== "#") { seen.add(c); q.push(c); }
    while (q.length) {
      const i = q.pop(), r = Math.floor(i / 10), c = i % 10;
      for (const [a, d] of [[r + 1, c], [r, c - 1], [r, c + 1]]) {
        const j = a * 10 + d;
        if (a < 20 && d >= 0 && d < 10 && b[j] !== "#" && !seen.has(j)) { seen.add(j); q.push(j); }
      }
    }
    return i => b[i] !== "#" && !seen.has(i);
  }
  function draw(cv, b) {
    const n = cv.width / 10, x = cv.getContext("2d"), s = sealed(b);
    x.fillStyle = EMPTY; x.fillRect(0, 0, cv.width, cv.height);
    for (let i = 0; i < 200; i++) {
      const fill = INK[b[i]] || (s(i) ? SEALED : null);
      if (fill) { x.fillStyle = fill; x.fillRect((i % 10) * n, Math.floor(i / 10) * n, n - 1, n - 1); }
    }
  }
  function when(id, f, tries = 40) {         // the canvas may land just after its data does
    const e = document.getElementById(id);
    if (e) f(e); else if (tries) setTimeout(() => when(id, f, tries - 1), 25);
  }
  window.dash_clientside.topout = {
    arm: function (k, data) {                // every game's board at piece k, and each arm's triangle up to k
      if (!data) return [window.dash_clientside.no_update, ""];
      const grids = data.arms.map(arm => {
        const full = Array.from({length: 21}, () => Array(21).fill(0)), z = full.map(r => r.slice());
        let total = 0;
        arm.games.forEach(g => g.live.forEach((l, i) => { total++; full[g.dead[i]][l]++; if (i < k) z[g.dead[i]][l]++; }));
        return {z, total, top: Math.max(...full.flat()) / total};
      });
      const top = Math.log10(Math.max(...grids.map(g => g.top)));       // one colour range for the arms shown, from their whole games
      const low = Math.log10(Math.min(...grids.map(g => 1 / g.total)));
      const figs = data.arms.map((arm, j) => {
        const {z, total} = grids[j], dots = {};
        arm.games.forEach((g, i) => {
          const last = g.boards.length, at = Math.min(k, last) - 1;
          when(`b-${j}-${i}`, cv => draw(cv, g.boards[at]));
          when(`g-${j}-${i}`, a => { a.className = k > last ? "over" : ""; });
          when(`t-${j}-${i}`, t => { t.textContent = k > last ? (g.survived ? "finished" : "out at " + last) : "lines " + g.lines[at]; });
          if (k <= last) { const q = g.live[k - 1] + "," + g.dead[k - 1]; dots[q] = (dots[q] || 0) + 1; }
        });
        const pts = Object.entries(dots).map(([q, n]) => [...q.split(",").map(Number), n]);
        const axis = {range: [-.5, 20.5], constrain: "domain", dtick: 5, gridcolor: "#e4e3df", zeroline: false, fixedrange: true};
        return {data: [
          {type: "heatmap", z: z.map(r => r.map(c => c ? Math.log10(c / total) : null)), customdata: z, zmin: low, zmax: top, showscale: false,
           xgap: 1, ygap: 1, colorscale: [[0, "#cde2fb"], [.5, "#3987e5"], [1, "#0d366b"]],
           hovertemplate: "%{x} live, %{y} dead rows<br>%{customdata} pieces<extra></extra>"},
          {type: "scatter", mode: "lines", x: [0, 20], y: [20, 0], line: {dash: "dash", color: "#52514e", width: 1}, hoverinfo: "skip"},
          {type: "scatter", mode: "markers", x: pts.map(p => p[0]), y: pts.map(p => p[1]), text: pts.map(p => p[2] + " games here now"),
           hovertemplate: "%{text}<extra></extra>", marker: {symbol: "circle-open", color: "#0b0b0b", size: pts.map(p => 8 + 3 * Math.sqrt(p[2])), line: {width: 2}}}],
          layout: {height: 400, margin: {l: 44, r: 8, t: 8, b: 40}, showlegend: false, dragmode: false, paper_bgcolor: "#fcfcfb", plot_bgcolor: "#fcfcfb",
            xaxis: {...axis, title: {text: "live rows"}, constraintoward: "left"},
            yaxis: {...axis, title: {text: "dead rows"}, scaleanchor: "x"}}};
      });
      return [figs, "piece " + k];
    },
    narrow: function () {                    // a phone-sized window, and again whenever it crosses the line
      const m = window.matchMedia("(max-width: 800px)");
      if (!window.topoutNarrow) {
        window.topoutNarrow = true;
        m.addEventListener("change", e => window.dash_clientside.set_props("narrow", {data: e.matches}));
      }
      return m.matches;
    },
    top: function (search) {                 // a different arm, game or piece starts at its top; a changed choice stays put
      const q = new URLSearchParams(search), at = ["arm", "game", "piece"].map(k => q.get(k)).join("|");
      if (window.topoutAt !== undefined && window.topoutAt !== at) window.scrollTo(0, 0);
      window.topoutAt = at;
      return "";
    },
    menu: function () { document.body.classList.toggle("menu"); return ""; },   // on the body: Dash redraws the rail
    step: function (n, k, off, max) {         // play: advance the slider until the last piece
      const nu = window.dash_clientside.no_update;
      return !off ? (k < max ? [k + 1, nu, nu] : [nu, true, "Play"]) : [nu, true, "Play"];
    },
    toggle: function (clicks, disabled, k, max) {
      const nu = window.dash_clientside.no_update;
      if (!clicks) return [nu, nu, nu];
      return disabled ? [false, "Pause", k >= max ? 1 : nu] : [true, "Play", nu];
    },
    boards: function (data) {                // [[canvas id, board], ...]
      (data || []).forEach(([id, b]) => when(id, cv => draw(cv, b)));
      return "";
    }
  };
})();
