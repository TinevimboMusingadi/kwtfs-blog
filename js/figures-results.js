// Results figures: RL dynamics, accuracy–token frontier, seed stability, genres,
// token-price sweep, latency, trace explorer. All data from data/*.json.

const E2_RUNS = ["e2", "e2s1", "e2-s4"];
const E3B_RUNS = ["e3b", "e3bs1", "e3b-s2", "e3b-s4"];

/* ---------------- RL training dynamics ---------------- */
KWT.register("fig-dynamics", async (el) => {
  const C = KWT.color();
  const dyn = await KWT.data("dynamics");
  const metrics = { mode_cot: "share of rollouts that reason (CoT)", mode_jev: "share of one-step (JEV) rollouts",
    correct: "rollout accuracy", chars: "mean completion length (characters)" };
  let metric = "mode_cot";
  const ctr = d3.select(el).append("div").attr("class", "fig-controls");
  ctr.append("span").text("Show:");
  const btns = ctr.selectAll("button").data(Object.keys(metrics)).join("button")
    .text((k) => ({ mode_cot: "CoT share", mode_jev: "JEV share", correct: "accuracy", chars: "length" }[k]));
  const box = d3.select(el).append("div");
  KWT.legend(el, [["RL + regret reward (4 seeds)", C.regret], ["RL, outcome-only (3 seeds)", C.outcome]]);
  const smooth = (arr, k = 15) => arr.map((_, i) => d3.mean(arr.slice(Math.max(0, i - k + 1), i + 1)));
  function render() {
    btns.classed("on", (k) => k === metric);
    box.selectAll("*").remove();
    const { g, w, h } = KWT.svg(box.node(), { width: 1000, height: 340, margin: { t: 14, r: 20, b: 40, l: 56 } });
    const series = [];
    E2_RUNS.concat(E3B_RUNS).forEach((name) => {
      if (!dyn[name]) return;
      const vals = smooth(dyn[name].map((r) => r[metric]));
      series.push({ name, fam: E3B_RUNS.includes(name) ? "e3b" : "e2", vals });
    });
    const x = d3.scaleLinear().domain([0, 149]).range([0, w]);
    const all = series.flatMap((s) => s.vals);
    const y = d3.scaleLinear().domain(metric === "chars" ? [0, d3.max(all) * 1.05] : [Math.max(0, d3.min(all) - 0.05), Math.min(1, d3.max(all) + 0.05)]).range([h, 0]);
    KWT.axes(g, x, y, w, h, { xLabel: "GRPO step (15-step moving average)", yLabel: metrics[metric],
      yFmt: metric === "chars" ? d3.format(".0f") : d3.format(".0%") });
    const line = d3.line().x((_, i) => x(i)).y((v) => y(v));
    g.selectAll(".s").data(series).join("path").attr("fill", "none")
      .attr("stroke", (s) => (s.fam === "e3b" ? C.regret : C.outcome)).attr("stroke-width", 2).attr("opacity", 0.85)
      .attr("d", (s) => line(s.vals))
      .on("mousemove", (e, s) => KWT.showTip(`<b>${s.fam === "e3b" ? "RL + regret" : "outcome-only"}</b> · run ${s.name}<br>final ${metric === "chars" ? s.vals[149].toFixed(0) : KWT.pct(s.vals[149])}`, e))
      .on("mouseleave", KWT.hideTip);
  }
  btns.on("click", (e, k) => { metric = k; render(); });
  render();
});

/* ---------------- accuracy vs tokens ---------------- */
KWT.register("fig-frontier", async (el) => {
  const C = KWT.color();
  const fr = await KWT.data("frontier");
  let split = "test";
  const ctr = d3.select(el).append("div").attr("class", "fig-controls");
  const sb = ctr.selectAll("button").data(["test", "ood"]).join("button").text((s) => (s === "test" ? "sealed test (799)" : "out-of-distribution (460)"));
  const box = d3.select(el).append("div");
  KWT.legend(el, [["RL + regret reward, model chooses (5 seeds)", C.regret, "circle"], ["RL outcome-only, model chooses (5 seeds)", C.outcome],
    ["same weights forced to always reason", C.ink3, "diamond"], ["SFT only", C.base]]);
  function render() {
    sb.classed("on", (s) => s === split);
    box.selectAll("*").remove();
    const { g, w, h } = KWT.svg(box.node(), { width: 1000, height: 400, margin: { t: 14, r: 30, b: 44, l: 56 } });
    const pts = fr.filter((d) => d.split === split && (d.family === "e2" || d.family === "e3b" || d.model === "sft")
      && (d.control === "auto" || d.control === "cot"));
    const tx = d3.extent(pts, (d) => d.tokens);
    const x = d3.scaleLinear().domain([Math.floor(tx[0] / 10) * 10 - 5, Math.ceil(tx[1] / 10) * 10 + 5]).range([0, w]);
    const y = d3.scaleLinear().domain(d3.extent(pts, (d) => d.acc).map((v, i) => v + (i ? 0.01 : -0.02))).range([h, 0]);
    KWT.axes(g, x, y, w, h, { xLabel: "mean generated tokens per question", yLabel: "accuracy", yFmt: d3.format(".0%") });
    const col = (d) => (d.control === "cot" ? C.ink3 : d.family === "e3b" ? C.regret : d.family === "e2" ? C.outcome : C.base);
    const sym = (d) => (d.control === "cot" ? d3.symbolDiamond : d.family === "e3b" ? d3.symbolCircle : d3.symbolSquare);
    g.selectAll(".p").data(pts).join("path")
      .attr("transform", (d) => `translate(${x(d.tokens)},${y(d.acc)})`)
      .attr("d", (d) => d3.symbol().type(sym(d)).size(d.model === "sft" ? 110 : 80)())
      .attr("fill", col).attr("stroke", C.surface).attr("stroke-width", 1.5)
      .on("mousemove", (e, d) => KWT.showTip(`<b>${d.model}</b> · ${d.control === "auto" ? "model chooses" : "forced CoT"}<br>accuracy ${KWT.pct(d.acc)} · ${d.tokens.toFixed(1)} tokens<br>JEV ${KWT.pct(d.share.jev, 0)} · direct ${KWT.pct(d.share.direct, 0)} · CoT ${KWT.pct(d.share.cot, 0)}`, e))
      .on("mouseleave", KWT.hideTip);
  }
  sb.on("click", (e, s) => { split = s; render(); });
  render();
});

/* ---------------- seed stability ---------------- */
KWT.register("fig-seeds", async (el) => {
  const C = KWT.color();
  const sd = await KWT.data("seeds");
  const W = 1000, H = 300;
  const root = d3.select(el).append("svg").attr("viewBox", `0 0 ${W} ${H}`);
  [["test", "sealed test (799)"], ["ood", "out-of-distribution (460)"]].forEach(([split, title], i) => {
    const g = root.append("g").attr("transform", `translate(${70 + i * 490},30)`);
    const pw = 380, ph = 220;
    const vals = [...sd[`${split}_e2`].acc, ...sd[`${split}_e3b`].acc];
    const y = d3.scaleLinear().domain([d3.min(vals) - 0.01, d3.max(vals) + 0.01]).range([ph, 0]);
    const x = d3.scalePoint().domain(["e2", "e3b"]).range([0, pw]).padding(0.6);
    g.append("g").attr("class", "grid").call(d3.axisLeft(y).ticks(5).tickSize(-pw).tickFormat(""));
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".0%")));
    g.append("text").attr("class", "label ink").attr("x", 0).attr("y", -12).attr("font-weight", 650).text(title);
    ["e2", "e3b"].forEach((fam) => {
      const s = sd[`${split}_${fam}`];
      const colr = fam === "e3b" ? C.regret : C.outcome;
      s.acc.forEach((v, k) => {
        g.append("path").attr("transform", `translate(${x(fam) + (k - 2) * 12},${y(v)})`)
          .attr("d", d3.symbol().type(fam === "e3b" ? d3.symbolCircle : d3.symbolSquare).size(70)())
          .attr("fill", colr).attr("stroke", C.surface).attr("stroke-width", 1.5)
          .on("mousemove", (e) => KWT.showTip(`<b>${fam === "e3b" ? "RL + regret" : "outcome-only"}</b>, seed ${k}<br>accuracy ${KWT.pct(v)} · ${s.tokens[k].toFixed(1)} tokens`, e))
          .on("mouseleave", KWT.hideTip);
      });
      g.append("line").attr("x1", x(fam) - 40).attr("x2", x(fam) + 40).attr("y1", y(s.mean)).attr("y2", y(s.mean)).attr("stroke", colr).attr("stroke-width", 2.5);
      g.append("text").attr("class", "label").attr("x", x(fam)).attr("y", ph + 22).attr("text-anchor", "middle")
        .text(fam === "e3b" ? "+ regret reward" : "outcome-only");
      g.append("text").attr("class", "label").attr("x", x(fam)).attr("y", ph + 38).attr("text-anchor", "middle")
        .text(`mean ${KWT.pct(s.mean)} · sd ${(100 * s.sd).toFixed(1)} pts`);
    });
  });
});

/* ---------------- reasoning share by genre ---------------- */
KWT.register("fig-genres", async (el) => {
  const C = KWT.color();
  const gs = await KWT.data("genres");
  let metric = "cot_share";
  const ctr = d3.select(el).append("div").attr("class", "fig-controls");
  const bs = ctr.selectAll("button").data(["cot_share", "acc"]).join("button").text((k) => (k === "acc" ? "accuracy" : "how often it reasons"));
  const box = d3.select(el).append("div");
  KWT.legend(el, [["SFT only", C.base], ["RL outcome-only (mean of 5)", C.outcome], ["RL + regret (mean of 5)", C.regret]]);
  function render() {
    bs.classed("on", (k) => k === metric);
    box.selectAll("*").remove();
    const { g, w, h } = KWT.svg(box.node(), { width: 1000, height: 320, margin: { t: 14, r: 10, b: 40, l: 56 } });
    const x0 = d3.scaleBand().domain(gs.map((d) => d.genre)).range([0, w]).padding(0.22);
    const keys = ["sft", "e2", "e3b"];
    const x1 = d3.scaleBand().domain(keys).range([0, x0.bandwidth()]).padding(0.08);
    const y = d3.scaleLinear().domain([0, 1]).range([h, 0]);
    KWT.axes(g, d3.scaleLinear().range([0, 0]), y, w, h, { yLabel: metric === "acc" ? "accuracy (test)" : "% of questions answered with CoT (test)", yFmt: d3.format(".0%") });
    g.selectAll(".axis").filter((d, i) => i === 0).remove();
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${h})`).call(d3.axisBottom(x0).tickSizeOuter(0));
    const colr = { sft: C.base, e2: C.outcome, e3b: C.regret };
    gs.forEach((d) => {
      keys.forEach((k) => {
        const v = d[`${k}_${metric}`];
        g.append("rect").attr("x", x0(d.genre) + x1(k)).attr("width", x1.bandwidth()).attr("y", y(v)).attr("height", h - y(v))
          .attr("rx", 3).attr("fill", colr[k])
          .on("mousemove", (e) => KWT.showTip(`<b>${d.genre}</b> · ${{ sft: "SFT only", e2: "outcome-only", e3b: "+ regret" }[k]}<br>${metric === "acc" ? "accuracy" : "reasons on"} ${KWT.pct(v)} (n=${d.n})`, e))
          .on("mouseleave", KWT.hideTip);
      });
    });
  }
  bs.on("click", (e, k) => { metric = k; render(); });
  render();
});

/* ---------------- token price sweep ---------------- */
KWT.register("fig-sweep", async (el) => {
  const C = KWT.color();
  const sw = await KWT.data("sweep");
  const { g, w, h } = KWT.svg(el, { width: 1000, height: 320, margin: { t: 14, r: 20, b: 44, l: 64 } });
  const x = d3.scaleLinear().domain([0, 0.004]).range([0, w]);
  const y = d3.scaleLinear().domain([d3.min(sw, (d) => d.lo) - 0.005, d3.max(sw, (d) => d.hi) + 0.005]).range([h, 0]);
  KWT.axes(g, x, y, w, h, { xLabel: "token price λ (accuracy given up per generated token)", yLabel: "utility: model chooses − always reason (95% CI)",
    xFmt: d3.format(".4f"), yFmt: d3.format("+.2f") });
  g.append("path").datum(sw).attr("fill", C.regret).attr("opacity", 0.15)
    .attr("d", d3.area().x((d) => x(d.lam)).y0((d) => y(d.lo)).y1((d) => y(d.hi)));
  g.append("path").datum(sw).attr("fill", "none").attr("stroke", C.regret).attr("stroke-width", 2.2)
    .attr("d", d3.line().x((d) => x(d.lam)).y((d) => y(d.diff)));
  g.append("line").attr("x1", 0).attr("x2", w).attr("y1", y(0)).attr("y2", y(0)).attr("stroke", C.ink2);
  g.append("line").attr("x1", x(0.0005)).attr("x2", x(0.0005)).attr("y1", 0).attr("y2", h).attr("stroke", C.ink3).attr("stroke-dasharray", "4 3");
  g.append("text").attr("class", "label").attr("x", x(0.0005) + 6).attr("y", 12).text("price used in training");
  g.selectAll("circle").data(sw).join("circle").attr("cx", (d) => x(d.lam)).attr("cy", (d) => y(d.diff)).attr("r", 9).attr("fill", "transparent")
    .on("mousemove", (e, d) => KWT.showTip(`λ = ${d.lam}<br>difference ${d.diff >= 0 ? "+" : ""}${d.diff.toFixed(3)} [${d.lo.toFixed(3)}, ${d.hi.toFixed(3)}]`, e))
    .on("mouseleave", KWT.hideTip);
});

/* ---------------- latency ---------------- */
KWT.register("fig-latency", async (el) => {
  const C = KWT.color();
  const lat = await KWT.data("latency");
  const rows = [["model chooses", "auto", C.regret], ["always reason", "cot", C.ink3]];
  const stats = [["mean_s", "mean"], ["median_s", "median"], ["p90_s", "90th percentile"]];
  const { g, w, h } = KWT.svg(el, { width: 900, height: 230, margin: { t: 10, r: 60, b: 40, l: 140 } });
  const y0 = d3.scaleBand().domain(stats.map((s) => s[1])).range([0, h]).padding(0.25);
  const y1 = d3.scaleBand().domain(["auto", "cot"]).range([0, y0.bandwidth()]).padding(0.1);
  const x = d3.scaleLinear().domain([0, 3.2]).range([0, w]);
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${h})`).call(d3.axisBottom(x).ticks(6).tickFormat((d) => `${d}s`));
  stats.forEach(([k, label]) => {
    g.append("text").attr("class", "label ink").attr("x", -10).attr("text-anchor", "end").attr("y", y0(label) + y0.bandwidth() / 2 + 4).text(label);
    rows.forEach(([name, key, col]) => {
      const v = lat[key][k];
      g.append("rect").attr("x", 0).attr("y", y0(label) + y1(key)).attr("height", y1.bandwidth()).attr("width", x(v)).attr("rx", 3).attr("fill", col);
      g.append("text").attr("class", "label").attr("x", x(v) + 6).attr("y", y0(label) + y1(key) + y1.bandwidth() / 2 + 4).text(`${v.toFixed(2)}s · ${name}`);
    });
  });
});

/* ---------------- trace explorer ---------------- */
KWT.register("fig-traces", async (el) => {
  const traces = await KWT.data("traces");
  const genres = ["all", ...new Set(traces.map((t) => t.genre))];
  const st = { genre: "all", mode: "all", outcome: "all" };
  const ctr = d3.select(el).append("div").attr("class", "fig-controls");
  const mk = (label, key, opts) => {
    const l = ctr.append("label");
    l.append("span").text(label);
    l.append("select").on("change", function () { st[key] = this.value; render(); })
      .selectAll("option").data(opts).join("option").attr("value", (d) => d).text((d) => d);
  };
  mk("genre", "genre", genres);
  mk("chosen mode", "mode", ["all", "jev", "direct", "cot"]);
  mk("outcome", "outcome", ["all", "correct", "wrong"]);
  const count = ctr.append("span").attr("class", "muted");
  const list = d3.select(el).append("div").attr("class", "trace-list");
  const esc = (s) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  function render() {
    const rows = traces.filter((t) => (st.genre === "all" || t.genre === st.genre) && (st.mode === "all" || t.mode === st.mode)
      && (st.outcome === "all" || (st.outcome === "correct") === t.correct)).slice(0, 8);
    count.text(`showing ${rows.length} of ${traces.length} sampled test questions`);
    list.html(rows.map((t) => `
      <div class="trace">
        <div class="meta"><span>${t.genre}</span><span>chose <span class="mode ${t.mode}">${t.mode}</span></span>
          <span>${t.tokens} tokens</span><span class="${t.correct ? "ok" : "no"}">${t.correct ? "correct" : "wrong"}</span>
          <span>measured best mode: ${t.suggested || "–"}</span></div>
        <div class="q">${esc(t.question)}</div>
        ${t.options.length ? `<div class="opts">${t.options.map((o, i) => `${"ABCDEF"[i]}) ${esc(o)}`).join(" · ")}</div>` : ""}
        <div class="small">gold answer: <b>${esc(t.gold)}</b></div>
        <pre>${esc(t.completion)}</pre>
        <details><summary class="small">Same question, forced to reason (${t.cot_tokens} tokens, ${t.cot_correct ? "correct" : "wrong"})</summary><pre>${esc(t.cot_completion)}</pre></details>
      </div>`).join(""));
  }
  render();
});
