// Method figures: hero (three modes), landscape, RLVR toy, label explorer,
// router headroom, probe calibration, SFT mix, reward explorer.

/* ---------------- Figure 1: one model, three modes (animated) ---------------- */
KWT.register("fig-hero", async (el) => {
  const C = KWT.color();
  const traces = await KWT.data("traces");
  const pick = (m) => traces.find((t) => t.mode === m && t.correct && (m !== "cot" || t.completion.includes("<mode:jev>")))
    || traces.find((t) => t.mode === m);
  const examples = { jev: pick("jev"), direct: pick("direct"), cot: pick("cot") };
  const ctr = d3.select(el).append("div").attr("class", "fig-controls");
  ctr.append("span").text("Play a real test question answered in mode:");
  const btns = ctr.selectAll("button").data(["jev", "direct", "cot"]).join("button")
    .text((m) => KWT.MODE_LABEL[m]);
  const box = d3.select(el).append("div");
  const W = 1000, H = 300;
  const { svg } = KWT.svg(box.node(), { width: W, height: H, margin: { t: 0, r: 0, b: 0, l: 0 } });

  function draw(mode) {
    btns.classed("on", (m) => m === mode);
    svg.selectAll("*").remove();
    const ex = examples[mode];
    const g = svg.append("g");
    // prompt
    g.append("rect").attr("x", 10).attr("y", 30).attr("width", 250).attr("height", 220).attr("rx", 10)
      .attr("fill", C.surface).attr("stroke", C.rule);
    g.append("text").attr("class", "label").attr("x", 22).attr("y", 52).text("question (" + ex.genre + ")");
    const fo = g.append("foreignObject").attr("x", 20).attr("y", 60).attr("width", 230).attr("height", 184);
    fo.append("xhtml:div").style("font-size", "11.5px").style("line-height", "1.35").style("color", C.ink2)
      .style("overflow", "hidden").style("height", "184px")
      .text(ex.question.slice(0, 260) + (ex.options.length ? "  " + ex.options.map((o, i) => "ABCDEF"[i] + ") " + o).join("  ") : ""));
    // model
    g.append("rect").attr("x", 300).attr("y", 90).attr("width", 150).attr("height", 100).attr("rx", 12)
      .attr("fill", C.surface).attr("stroke", C.ink3);
    g.append("text").attr("class", "label ink").attr("x", 375).attr("y", 135).attr("text-anchor", "middle").text("one network");
    g.append("text").attr("class", "label").attr("x", 375).attr("y", 155).attr("text-anchor", "middle").text("Qwen3-4B + LoRA");
    g.append("path").attr("d", "M262,140 L296,140").attr("stroke", C.ink3).attr("marker-end", "url(#arr)");
    svg.append("defs").append("marker").attr("id", "arr").attr("viewBox", "0 0 10 10").attr("refX", 9).attr("refY", 5)
      .attr("markerWidth", 7).attr("markerHeight", 7).attr("orient", "auto")
      .append("path").attr("d", "M0,0 L10,5 L0,10 z").attr("fill", C.ink3);
    // lanes
    const lanes = [["jev", 60, "one forward pass → read the option letter"], ["direct", 140, "a few tokens"],
      ["cot", 220, "reason in tokens, then hand off to a one-step pick"]];
    lanes.forEach(([m, y, text]) => {
      const active = m === mode;
      g.append("path").attr("d", `M452,140 C 500,140 500,${y} 540,${y}`).attr("fill", "none")
        .attr("stroke", active ? C[m] : C.rule).attr("stroke-width", active ? 2.5 : 1.2);
      g.append("rect").attr("x", 540).attr("y", y - 15).attr("width", 92).attr("height", 30).attr("rx", 6)
        .attr("fill", active ? C[m] : "none").attr("stroke", C[m]).attr("opacity", active ? 1 : 0.5);
      g.append("text").attr("x", 586).attr("y", y + 4).attr("text-anchor", "middle").attr("font-family", "var(--mono)")
        .attr("font-size", 11.5).attr("fill", active ? C.surface : C[m]).text(`<mode:${m}>`);
      g.append("text").attr("class", "label").attr("x", 642).attr("y", y - 20).attr("opacity", active ? 1 : 0.55).text(text);
    });
    // animated output for the active lane
    const y = lanes.find((l) => l[0] === mode)[1];
    const body = ex.completion.split("\n").slice(1).join(" ").replace(/<mode:jev>/, " ⟶ ");
    const tokens = body.split(/\s+/).filter(Boolean).slice(0, 26);
    const out = g.append("g").attr("transform", `translate(642,${y})`);
    let x = 0;
    tokens.forEach((tok, i) => {
      const w = Math.min(10 + tok.length * 6.4, 120);
      if (x + w > 340) return;
      const isPick = mode !== "direct" && i === tokens.length - 1;
      const t = out.append("g").attr("transform", `translate(${x},-10)`).attr("opacity", 0);
      t.append("rect").attr("width", w).attr("height", 20).attr("rx", 4).attr("fill", isPick ? C[mode] : C.surface)
        .attr("stroke", isPick ? C[mode] : C.rule);
      t.append("text").attr("x", w / 2).attr("y", 14).attr("text-anchor", "middle").attr("font-size", 10.5)
        .attr("font-family", "var(--mono)").attr("fill", isPick ? C.surface : C.ink2).text(tok.length > 17 ? tok.slice(0, 16) + "…" : tok);
      t.transition().delay(mode === "jev" ? 250 : 120 * i).duration(200).attr("opacity", 1);
      x += w + 4;
    });
    const verdict = ex.correct ? "✓ correct" : "✗ wrong";
    g.append("text").attr("class", "label").attr("x", 642).attr("y", 290)
      .text(`${verdict} · ${ex.tokens} tokens generated · gold: ${ex.gold}`).attr("fill", ex.correct ? C.good : C.bad);
  }
  btns.on("click", (e, m) => draw(m));
  draw("cot");
});

/* ---------------- Figure 2: the landscape of fast/slow approaches ---------------- */
KWT.register("fig-landscape", (el) => {
  const C = KWT.color();
  const rows = [
    { name: "Always-reasoning model (RLVR)", who: "nobody: it always thinks", steps: [["LM", "box"], ["long reasoning tokens", "chain"], ["answer", "out"]] },
    { name: "User/API think switch", who: "the user, per request", steps: [["flag", "flag"], ["LM", "box"], ["think tokens | answer", "chain"]] },
    { name: "Learned think switch", who: "the model, first token", steps: [["LM", "box"], ["<think>/<short>", "tag"], ["tokens", "chain"]] },
    { name: "Latent / continuous CoT", who: "fixed budget of latent steps", steps: [["LM", "box"], ["hidden vectors ↻", "latent"], ["answer", "out"]] },
    { name: "Router + two models", who: "an external router", steps: [["router", "flag"], ["small LM | big reasoning LM", "box2"], ["answer", "out"]] },
    { name: "Decision head on an LM", who: "always one-step", steps: [["LM", "box"], ["option scores", "head"], ["pick", "out"]] },
    { name: "This work", who: "the model, per question", steps: [["LM", "box"], ["<mode:x>", "tag"], ["pick | short | CoT→pick", "ours"]], ours: true },
  ];
  const W = 1000, rowH = 52, H = rows.length * rowH + 30;
  const { svg } = KWT.svg(el, { width: W, height: H, margin: { t: 0, r: 0, b: 0, l: 0 } });
  svg.append("text").attr("class", "label").attr("x", 0).attr("y", 16).text("approach");
  svg.append("text").attr("class", "label").attr("x", 250).attr("y", 16).text("who decides how much to think");
  svg.append("text").attr("class", "label").attr("x", 480).attr("y", 16).text("pipeline at test time");
  rows.forEach((r, i) => {
    const y = 30 + i * rowH;
    const g = svg.append("g").attr("transform", `translate(0,${y})`);
    if (r.ours) g.append("rect").attr("x", -6).attr("y", 2).attr("width", W + 6).attr("height", rowH - 6).attr("rx", 8)
      .attr("fill", C.regret).attr("opacity", 0.08);
    g.append("text").attr("class", r.ours ? "label ink" : "label").attr("x", 0).attr("y", 30)
      .attr("font-weight", r.ours ? 650 : 400).text(r.name);
    g.append("text").attr("class", "label").attr("x", 250).attr("y", 30).text(r.who);
    let x = 480;
    r.steps.forEach(([txt, kind], j) => {
      const w = Math.max(54, txt.length * 6.6 + 18);
      const col = kind === "ours" ? C.regret : kind === "tag" ? C.cot : kind === "latent" ? C.jev : kind === "head" ? C.jev : C.ink3;
      g.append("rect").attr("x", x).attr("y", 12).attr("width", w).attr("height", 26).attr("rx", kind === "box" || kind === "box2" ? 7 : 13)
        .attr("fill", kind === "chain" ? C.grid : C.surface).attr("stroke", col).attr("stroke-dasharray", kind === "latent" ? "3 2" : null);
      g.append("text").attr("x", x + w / 2).attr("y", 29).attr("text-anchor", "middle").attr("font-size", 11.5)
        .attr("font-family", kind === "tag" || kind === "ours" ? "var(--mono)" : "var(--sans)").attr("fill", C.ink2).text(txt);
      if (j < r.steps.length - 1) g.append("path").attr("d", `M${x + w + 3},25 L${x + w + 17},25`).attr("stroke", C.ink3);
      x += w + 20;
    });
  });
});

/* ---------------- Figure 3: RLVR from first principles (toy policy gradient) ---------------- */
KWT.register("fig-rlvr", (el) => {
  const C = KWT.color();
  // three ways to answer one kind of question, as a softmax policy over logits
  const acts = [
    { k: "guess", label: "answer immediately", p: 0.35, len: 3 },
    { k: "short", label: "short reasoning", p: 0.80, len: 30 },
    { k: "long", label: "long reasoning", p: 0.90, len: 120 },
  ];
  let logits = [1.2, 0.2, -0.6];
  let step = 0;
  const ctr = d3.select(el).append("div").attr("class", "fig-controls");
  const stepBtn = ctr.append("button").text("Take 10 RL steps");
  const resetBtn = ctr.append("button").text("Reset");
  const lamLab = ctr.append("label");
  lamLab.append("span").text("token price λ");
  const lam = lamLab.append("input").attr("type", "range").attr("min", 0).attr("max", 0.004).attr("step", 0.0005).attr("value", 0);
  const lamOut = lamLab.append("span").attr("class", "num").text("0");
  const W = 760, H = 250;
  const { g, w, h } = KWT.svg(el, { width: W, height: H, margin: { t: 26, r: 20, b: 40, l: 170 } });
  const y = d3.scaleBand().domain(acts.map((a) => a.k)).range([0, h]).padding(0.35);
  const x = d3.scaleLinear().domain([0, 1]).range([0, w]);
  g.append("g").attr("class", "grid").call(d3.axisBottom(x).ticks(5).tickSize(h).tickFormat("")).attr("transform", "translate(0,0)");
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${h})`).call(d3.axisBottom(x).ticks(5).tickFormat(d3.format(".0%")));
  g.selectAll(".alab").data(acts).join("text").attr("class", "label ink").attr("x", -12).attr("text-anchor", "end")
    .attr("y", (a) => y(a.k) + y.bandwidth() / 2 + 4).text((a) => a.label);
  g.selectAll(".alab2").data(acts).join("text").attr("class", "label").attr("x", -12).attr("text-anchor", "end")
    .attr("y", (a) => y(a.k) + y.bandwidth() / 2 + 18).attr("font-size", 10.5).text((a) => `right ${Math.round(a.p * 100)}% · ${a.len} tokens`);
  const bars = g.selectAll(".bar").data(acts).join("rect").attr("class", "bar").attr("x", 0).attr("y", (a) => y(a.k))
    .attr("height", y.bandwidth()).attr("rx", 3).attr("fill", (a, i) => [C.direct, C.jev, C.cot][i]);
  const vals = g.selectAll(".val").data(acts).join("text").attr("class", "label").attr("y", (a) => y(a.k) + y.bandwidth() / 2 + 4);
  const info = g.append("text").attr("class", "label").attr("x", 0).attr("y", -10);
  const softmax = (z) => { const m = Math.max(...z); const e = z.map((v) => Math.exp(v - m)); const s = d3.sum(e); return e.map((v) => v / s); };
  function render() {
    const pi = softmax(logits);
    bars.transition().duration(350).attr("width", (a, i) => x(pi[i]));
    vals.transition().duration(350).attr("x", (a, i) => x(pi[i]) + 6).text((a, i) => d3.format(".0%")(pi[i]));
    const L = +lam.property("value");
    const exp = d3.sum(acts.map((a, i) => pi[i] * (a.p - L * a.len * a.p)));  // cost only on correct answers
    info.text(`after ${step} steps · expected reward ${exp.toFixed(3)} · mean length ${d3.sum(acts.map((a, i) => pi[i] * a.len)).toFixed(0)} tokens`);
  }
  // exact expected GRPO-style update for a categorical policy:
  // d logit_a = lr * pi_a * (R_a - E[R]),   R_a = p_a - λ * len_a * p_a  (cost only on correct answers)
  function update(n) {
    const L = +lam.property("value");
    for (let s = 0; s < n; s++) {
      const pi = softmax(logits);
      const R = acts.map((a) => a.p - L * a.len * a.p);
      const base = d3.sum(R.map((r, i) => r * pi[i]));
      logits = logits.map((z, i) => z + 3.0 * pi[i] * (R[i] - base));
      step++;
    }
    render();
  }
  stepBtn.on("click", () => update(10));
  resetBtn.on("click", () => { logits = [1.2, 0.2, -0.6]; step = 0; render(); });
  lam.on("input", () => { lamOut.text((+lam.property("value")).toFixed(4)); });
  render();
});

/* ---------------- Figure 4: measured labels, with a λ slider ---------------- */
KWT.register("fig-labels", async (el) => {
  const C = KWT.color();
  const rows = await KWT.data("labels_sample");
  const genres = ["compare", "knowledge", "decision", "rubric", "arith", "wordproblem"];
  const ctr = d3.select(el).append("div").attr("class", "fig-controls");
  const lab = ctr.append("label");
  lab.append("span").text("token price λ");
  const slider = lab.append("input").attr("type", "range").attr("min", 0).attr("max", 0.004).attr("step", 0.00025).attr("value", 0.0005);
  const out = lab.append("span").attr("class", "num");
  const W = 1000, H = 320;
  const { g, w, h } = KWT.svg(el, { width: W, height: H, margin: { t: 10, r: 120, b: 36, l: 110 } });
  const y = d3.scaleBand().domain(genres).range([0, h]).padding(0.28);
  const x = d3.scaleLinear().domain([0, 1]).range([0, w]);
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${h})`).call(d3.axisBottom(x).ticks(5).tickFormat(d3.format(".0%")));
  g.selectAll(".gl").data(genres).join("text").attr("class", "label ink").attr("x", -10).attr("text-anchor", "end")
    .attr("y", (d) => y(d) + y.bandwidth() / 2 + 4).text((d) => d);
  const segs = g.append("g");
  const util = g.append("g");
  KWT.legend(el, [["JEV (one step)", C.jev], ["direct", C.direct], ["CoT", C.cot]]);
  function render() {
    const L = +slider.property("value");
    out.text(L.toFixed(5));
    const data = [];
    genres.forEach((gname) => {
      const rs = rows.filter((r) => r.g === gname);
      const cnt = { jev: 0, direct: 0, cot: 0 };
      rs.forEach((r) => {
        let best = null, bu = -1e9;
        for (const m of ["jev", "direct", "cot"]) {
          if (r.p[m] == null) continue;
          const u = r.p[m] - L * r.t[m];
          if (u > bu + 1e-9) { bu = u; best = m; }
        }
        cnt[best]++;
      });
      let acc = 0;
      ["jev", "direct", "cot"].forEach((m) => {
        const v = cnt[m] / rs.length;
        data.push({ g: gname, m, x0: acc, v, n: cnt[m], tot: rs.length });
        acc += v;
      });
    });
    segs.selectAll("rect").data(data, (d) => d.g + d.m).join("rect")
      .attr("y", (d) => y(d.g)).attr("height", y.bandwidth()).attr("fill", (d) => C[d.m])
      .attr("stroke", C.surface).attr("stroke-width", 2)
      .on("mousemove", (e, d) => KWT.showTip(`<b>${d.g}</b> · best mode <b>${KWT.MODE_LABEL[d.m]}</b><br>${d.n} of ${d.tot} questions (${KWT.pct(d.v)})`, e))
      .on("mouseleave", KWT.hideTip)
      .transition().duration(250).attr("x", (d) => x(d.x0)).attr("width", (d) => Math.max(0, x(d.v)));
    segs.selectAll("text").data(data.filter((d) => d.v > 0.07), (d) => d.g + d.m).join("text")
      .attr("class", "label").style("fill", (d) => KWT.onColor(C[d.m])).attr("font-size", 11)
      .attr("y", (d) => y(d.g) + y.bandwidth() / 2 + 4).transition().duration(250)
      .attr("x", (d) => x(d.x0) + 6).text((d) => KWT.pct(d.v, 0));
  }
  slider.on("input", render);
  render();
});

/* ---------------- Figure 5: router headroom (label report) ---------------- */
KWT.register("fig-headroom", async (el) => {
  const C = KWT.color();
  const rep = await KWT.data("label_report");
  const r = rep.routers_utility.ALL;
  const rows = [
    ["always one-step (JEV)", r["always-jev"], C.base], ["always direct", r["always-direct"], C.base],
    ["always reason (CoT)", r["always-cot"], C.base], ["best mode per question type", r.lookup_xfit, C.ink3],
    ["perfect per-question router (cross-fitted)", r.oracle_xfit, C.regret],
  ];
  const W = 900, H = 230;
  const { g, w, h } = KWT.svg(el, { width: W, height: H, margin: { t: 6, r: 60, b: 36, l: 300 } });
  const y = d3.scaleBand().domain(rows.map((d) => d[0])).range([0, h]).padding(0.3);
  const x = d3.scaleLinear().domain([0.5, 0.86]).range([0, w]);
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${h})`).call(d3.axisBottom(x).ticks(5));
  g.append("text").attr("class", "label").attr("x", w).attr("y", h + 32).attr("text-anchor", "end").text("utility = accuracy − λ·tokens (λ = 0.0005)");
  g.selectAll(".b").data(rows).join("rect").attr("y", (d) => y(d[0])).attr("height", y.bandwidth()).attr("x", 0)
    .attr("width", (d) => x(d[1])).attr("rx", 3).attr("fill", (d) => d[2]);
  g.selectAll(".l").data(rows).join("text").attr("class", "label ink").attr("x", -10).attr("text-anchor", "end")
    .attr("y", (d) => y(d[0]) + y.bandwidth() / 2 + 4).text((d) => d[0]);
  g.selectAll(".v").data(rows).join("text").attr("class", "label").attr("x", (d) => x(d[1]) + 6)
    .attr("y", (d) => y(d[0]) + y.bandwidth() / 2 + 4).text((d) => d[1].toFixed(3));
});

/* ---------------- Figure 6: probe calibration ---------------- */
KWT.register("fig-probe", async (el) => {
  const C = KWT.color();
  const pr = await KWT.data("probe");
  const W = 1000, H = 330;
  const panels = ["jev", "direct", "cot"];
  const root = d3.select(el).append("svg").attr("viewBox", `0 0 ${W} ${H}`);
  const pw = 280, ph = 250;
  panels.forEach((m, i) => {
    const g = root.append("g").attr("transform", `translate(${50 + i * 330},20)`);
    const x = d3.scaleLinear().domain([0, 1]).range([0, pw]);
    const y = d3.scaleLinear().domain([0, 1]).range([ph, 0]);
    KWT.axes(g, x, y, pw, ph, { xLabel: "predicted P(correct)", yLabel: i === 0 ? "observed accuracy" : "", xFmt: d3.format(".1f"), yFmt: d3.format(".1f"), xTicks: 5, yTicks: 5 });
    g.append("line").attr("x1", 0).attr("y1", ph).attr("x2", pw).attr("y2", 0).attr("stroke", C.rule).attr("stroke-dasharray", "4 3");
    const d = pr[m].bins;
    g.append("path").datum(d).attr("fill", "none").attr("stroke", C[m]).attr("stroke-width", 2)
      .attr("d", d3.line().x((b) => x(b.pred)).y((b) => y(b.true)));
    g.selectAll("circle").data(d).join("circle").attr("cx", (b) => x(b.pred)).attr("cy", (b) => y(b.true))
      .attr("r", (b) => 3 + Math.sqrt(b.n) / 4).attr("fill", C[m]).attr("stroke", C.surface).attr("stroke-width", 1.5)
      .on("mousemove", (e, b) => KWT.showTip(`<b>${KWT.MODE_LABEL[m]}</b><br>predicted ${KWT.pct(b.pred)} · observed ${KWT.pct(b.true)}<br>${b.n} questions`, e))
      .on("mouseleave", KWT.hideTip);
    g.append("text").attr("class", "label ink").attr("x", 4).attr("y", 14).attr("font-weight", 650).text(KWT.MODE_LABEL[m]);
    g.append("text").attr("class", "label").attr("x", 4).attr("y", 30).text(`AUROC ${pr[m].auroc.toFixed(2)} · ECE ${pr[m].ece.toFixed(3)}`);
  });
});

/* ---------------- Figure 7: the SFT mix ---------------- */
KWT.register("fig-sft", async (el) => {
  const C = KWT.color();
  const n = await KWT.data("numbers");
  const mix = n.sft_mix;
  const rows = [["routing (Mode: auto)", "route"], ["obedience (Mode: forced)", "obey"]];
  const W = 900, H = 150;
  const { g, w, h } = KWT.svg(el, { width: W, height: H, margin: { t: 6, r: 20, b: 30, l: 200 } });
  const total = d3.max(rows, ([, k]) => d3.sum(["jev", "direct", "cot"], (m) => mix[`${k}_${m}`] || 0));
  const x = d3.scaleLinear().domain([0, total]).range([0, w]);
  const y = d3.scaleBand().domain(rows.map((r) => r[0])).range([0, h]).padding(0.3);
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${h})`).call(d3.axisBottom(x).ticks(6));
  rows.forEach(([label, k]) => {
    let acc = 0;
    g.append("text").attr("class", "label ink").attr("x", -10).attr("text-anchor", "end").attr("y", y(label) + y.bandwidth() / 2 + 4).text(label);
    ["jev", "direct", "cot"].forEach((m) => {
      const v = mix[`${k}_${m}`] || 0;
      g.append("rect").attr("x", x(acc)).attr("y", y(label)).attr("width", x(v)).attr("height", y.bandwidth())
        .attr("fill", C[m]).attr("stroke", C.surface).attr("stroke-width", 2)
        .on("mousemove", (e) => KWT.showTip(`<b>${label}</b><br>${KWT.MODE_LABEL[m]}: ${v.toLocaleString()} examples`, e))
        .on("mouseleave", KWT.hideTip);
      if (v > total * 0.08) g.append("text").attr("class", "label").style("fill", KWT.onColor(C[m])).attr("font-size", 11)
        .attr("x", x(acc) + 6).attr("y", y(label) + y.bandwidth() / 2 + 4).text(v.toLocaleString());
      acc += v;
    });
  });
  KWT.legend(el, [["JEV (one step)", C.jev], ["direct", C.direct], ["CoT", C.cot]]);
});

/* ---------------- Figure 8: reward explorer (mirrors kwt/rewards_v2.py) ---------------- */
const REWARD = {
  TOKEN_COST: 0.0005, MAX_COST: 0.3, FALSE_CLAIM: 0.05, MAX_FALSE: 0.2, GENRE_BONUS: 0.05, BETA: 0.5, ANNEAL: 2880,
  // returns each term exactly as rewards_v2 computes it
  terms({ correct, obeyed, valid, chars, falseClaims, genreOk, control, mode, utilities, seen }) {
    const correctness = obeyed && valid && correct ? 1 : 0;
    const wellFormed = !obeyed ? -1 : valid ? 0.1 : -0.2;
    const cost = correctness ? -Math.min(this.MAX_COST, this.TOKEN_COST * chars / 4) : 0;
    const faithful = -Math.min(this.MAX_FALSE, this.FALSE_CLAIM * falseClaims);
    const genre = genreOk ? this.GENRE_BONUS : 0;
    const beta = this.BETA * Math.max(0, 1 - seen / this.ANNEAL);
    let regret = 0;
    if (control === "auto" && utilities && mode) {
      const best = Math.max(...Object.values(utilities));
      regret = utilities[mode] == null ? -beta : -beta * (best - utilities[mode]);
    }
    return { correctness, well_formed: wellFormed, compute_cost: cost, faithful, genre, route_regret: regret };
  },
};

KWT.register("fig-reward", (el) => {
  const C = KWT.color();
  const presets = {
    "word problem (reasoning pays)": { jev: 0.25, direct: 0.30, cot: 0.88 - 0.0005 * 110 },
    "comparison (one step is enough)": { jev: 0.99, direct: 0.93, cot: 0.97 - 0.0005 * 70 },
    "knowledge MCQ (near tie)": { jev: 0.86, direct: 0.85, cot: 0.88 - 0.0005 * 90 },
  };
  const state = { preset: Object.keys(presets)[0], mode: "jev", correct: true, chars: 30, falseClaims: 0, progress: 0 };
  const ctr = d3.select(el).append("div").attr("class", "fig-controls");
  const sel = ctr.append("select");
  sel.selectAll("option").data(Object.keys(presets)).join("option").text((d) => d);
  const mbtn = ctr.selectAll("button.mode").data(["jev", "direct", "cot"]).join("button").attr("class", "mode-btn").text((m) => `chose ${KWT.MODE_LABEL[m]}`);
  const cbtn = ctr.append("button");
  const ctr2 = d3.select(el).append("div").attr("class", "fig-controls");
  const pl = ctr2.append("label"); pl.append("span").text("training progress");
  const prog = pl.append("input").attr("type", "range").attr("min", 0).attr("max", 100).attr("value", 0);
  const progOut = pl.append("span").attr("class", "num");
  const fl = ctr2.append("label"); fl.append("span").text("false equations in the trace");
  const fc = fl.append("input").attr("type", "range").attr("min", 0).attr("max", 4).attr("value", 0);
  const W = 900, H = 260;
  const { g, w, h } = KWT.svg(el, { width: W, height: H, margin: { t: 10, r: 60, b: 30, l: 170 } });
  const keys = ["correctness", "well_formed", "compute_cost", "faithful", "genre", "route_regret", "total"];
  const y = d3.scaleBand().domain(keys).range([0, h]).padding(0.25);
  const x = d3.scaleLinear().domain([-0.6, 1.2]).range([0, w]);
  g.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", 0).attr("y2", h).attr("stroke", C.ink3);
  g.append("g").attr("class", "axis").attr("transform", `translate(0,${h})`).call(d3.axisBottom(x).ticks(7));
  g.selectAll(".kl").data(keys).join("text").attr("class", "label ink").attr("x", -10).attr("text-anchor", "end")
    .attr("y", (k) => y(k) + y.bandwidth() / 2 + 4).attr("font-weight", (k) => (k === "total" ? 650 : 400)).text((k) => k.replace("_", " "));
  const bars = g.selectAll(".rb").data(keys).join("rect").attr("y", (k) => y(k)).attr("height", y.bandwidth()).attr("rx", 3);
  const vals = g.selectAll(".rv").data(keys).join("text").attr("class", "label").attr("y", (k) => y(k) + y.bandwidth() / 2 + 4);
  function render() {
    const u = presets[state.preset];
    const chars = { jev: 26, direct: 45, cot: 380 }[state.mode];
    const t = REWARD.terms({ correct: state.correct, obeyed: true, valid: true, chars, falseClaims: state.mode === "cot" ? state.falseClaims : 0,
      genreOk: true, control: "auto", mode: state.mode, utilities: u, seen: state.progress / 100 * 4800 });
    t.total = d3.sum(Object.values(t));
    mbtn.classed("on", (m) => m === state.mode);
    cbtn.text(state.correct ? "answer was right ✓ (click to flip)" : "answer was wrong ✗ (click to flip)");
    progOut.text(`${state.progress}%`);
    bars.transition().duration(200).attr("x", (k) => x(Math.min(0, t[k]))).attr("width", (k) => Math.abs(x(t[k]) - x(0)))
      .attr("fill", (k) => (k === "total" ? C.ink : k === "route_regret" ? C.regret : t[k] >= 0 ? C.good : C.bad));
    vals.transition().duration(200).attr("x", (k) => (t[k] >= 0 ? x(t[k]) + 6 : x(t[k]) - 6)).attr("text-anchor", (k) => (t[k] >= 0 ? "start" : "end"))
      .text((k) => (t[k] === 0 ? "0" : t[k].toFixed(3)));
  }
  sel.on("change", function () { state.preset = this.value; render(); });
  mbtn.on("click", (e, m) => { state.mode = m; render(); });
  cbtn.on("click", () => { state.correct = !state.correct; render(); });
  prog.on("input", function () { state.progress = +this.value; render(); });
  fc.on("input", function () { state.falseClaims = +this.value; render(); });
  render();
});
