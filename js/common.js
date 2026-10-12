// Shared helpers for every figure: theme-aware colors, data loading, tooltip,
// responsive SVG, number formatting, footnotes, and the sticky contents.

const KWT = (() => {
  const cache = {};
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const color = () => ({
    jev: css("--jev"), direct: css("--direct"), cot: css("--cot"),
    regret: css("--regret"), outcome: css("--outcome"), base: css("--base"),
    ink: css("--ink"), ink2: css("--ink-2"), ink3: css("--ink-3"), rule: css("--rule"),
    grid: css("--grid"), surface: css("--surface"), good: css("--good"), bad: css("--bad"),
    accent: css("--accent"),
  });
  const MODE_LABEL = { jev: "JEV (one step)", direct: "direct", cot: "CoT" };
  const FAMILY_LABEL = { e3b: "RL + regret reward", e2: "RL, outcome-only" };

  async function data(name) {
    if (!cache[name]) cache[name] = fetch(`data/${name}.json`).then((r) => r.json());
    return cache[name];
  }

  // One tooltip for the page.
  const tip = document.createElement("div");
  tip.className = "tip";
  document.addEventListener("DOMContentLoaded", () => document.body.appendChild(tip));
  function showTip(html, event) {
    tip.innerHTML = html;
    tip.style.opacity = 1;
    const pad = 14, w = tip.offsetWidth, h = tip.offsetHeight;
    let x = event.clientX + pad, y = event.clientY + pad;
    if (x + w > window.innerWidth - 8) x = event.clientX - w - pad;
    if (y + h > window.innerHeight - 8) y = event.clientY - h - pad;
    tip.style.left = `${x}px`;
    tip.style.top = `${y}px`;
  }
  const hideTip = () => { tip.style.opacity = 0; };

  // SVG sized by viewBox so it scales with the column; returns {svg, g, w, h}.
  function svg(el, { width = 680, height = 360, margin = { t: 16, r: 16, b: 40, l: 48 } } = {}) {
    const root = d3.select(el).append("svg").attr("viewBox", `0 0 ${width} ${height}`)
      .attr("role", "img").style("min-width", `${Math.round(width * 0.6)}px`);  // phones scroll the figure, not shrink it
    const g = root.append("g").attr("transform", `translate(${margin.l},${margin.t})`);
    return { svg: root, g, w: width - margin.l - margin.r, h: height - margin.t - margin.b, margin };
  }

  function axes(g, x, y, w, h, { xLabel = "", yLabel = "", xFmt = null, yFmt = null, xTicks = 6, yTicks = 5 } = {}) {
    g.append("g").attr("class", "grid").call(d3.axisLeft(y).ticks(yTicks).tickSize(-w).tickFormat(""));
    g.append("g").attr("class", "axis").attr("transform", `translate(0,${h})`)
      .call(d3.axisBottom(x).ticks(xTicks).tickFormat(xFmt).tickSizeOuter(0));
    g.append("g").attr("class", "axis").call(d3.axisLeft(y).ticks(yTicks).tickFormat(yFmt).tickSizeOuter(0));
    if (xLabel) g.append("text").attr("class", "label").attr("x", w).attr("y", h + 34).attr("text-anchor", "end").text(xLabel);
    if (yLabel) g.append("text").attr("class", "label").attr("x", -2).attr("y", -6).attr("text-anchor", "start").text(yLabel);
  }

  const pct = (v, d = 1) => (v == null ? "–" : `${(100 * v).toFixed(d)}%`);
  const fix = (v, d = 1) => (v == null ? "–" : (+v).toFixed(d));

  function legend(el, items) {
    const box = d3.select(el).append("div").attr("class", "legend");
    items.forEach(([label, col, shape]) => {
      const s = box.append("span");
      s.append("span").attr("class", "sw").style("background", col)
        .style("border-radius", shape === "circle" ? "50%" : shape === "diamond" ? "1px" : "3px")
        .style("transform", shape === "diamond" ? "rotate(45deg) scale(.8)" : null);
      s.append("span").text(label);
    });
  }

  function tableView(el, columns, rows) {
    const det = d3.select(el).append("details").attr("class", "table-view");
    det.append("summary").text("Show the data as a table");
    const t = det.append("table");
    t.append("thead").append("tr").selectAll("th").data(columns).join("th").text((c) => c.label)
      .attr("class", (c) => (c.num ? "n" : null));
    t.append("tbody").selectAll("tr").data(rows).join("tr").selectAll("td")
      .data((r) => columns.map((c) => ({ c, v: c.fmt ? c.fmt(r[c.key], r) : r[c.key] })))
      .join("td").attr("class", (d) => (d.c.num ? "n" : null)).text((d) => d.v);
  }

  // Re-render figures when the theme changes so colors follow the theme.
  const renderers = [];
  // Ink for text drawn on top of a filled mark: dark on light fills, white on dark ones.
  function onColor(fill) {
    return d3.lab(fill).l > 58 ? "#16161a" : "#ffffff";
  }

  function register(id, fn) {
    renderers.push([id, fn]);
    const el = document.getElementById(id);
    if (el) fn(el);
  }
  function rerender() {
    renderers.forEach(([id, fn]) => {
      const el = document.getElementById(id);
      if (el) { el.innerHTML = ""; fn(el); }
    });
  }

  function initTheme() {
    const btn = document.querySelector(".theme-toggle");
    const saved = (() => { try { return localStorage.getItem("kwt-theme"); } catch (e) { return null; } })();
    if (saved) document.documentElement.dataset.theme = saved;
    if (btn) btn.addEventListener("click", () => {
      const dark = getComputedStyle(document.documentElement).getPropertyValue("--bg").trim().startsWith("#16");
      const next = dark ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem("kwt-theme", next); } catch (e) { /* storage blocked */ }
      rerender();
    });
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", rerender);
  }

  function initFootnotes() {
    let box = null;
    document.querySelectorAll("sup.fn").forEach((s, i) => {
      s.textContent = i + 1;
      s.tabIndex = 0;
      const open = () => {
        if (box) box.remove();
        box = document.createElement("div");
        box.className = "fn-box";
        box.innerHTML = s.dataset.note;
        document.body.appendChild(box);
        const r = s.getBoundingClientRect();
        const left = Math.min(window.scrollX + r.left, window.scrollX + window.innerWidth - box.offsetWidth - 12);
        box.style.left = `${Math.max(8, left)}px`;
        box.style.top = `${window.scrollY + r.bottom + 6}px`;
      };
      s.addEventListener("mouseenter", open);
      s.addEventListener("focus", open);
      s.addEventListener("mouseleave", () => box && box.remove());
      s.addEventListener("blur", () => box && box.remove());
    });
  }

  function initToc() {
    const links = [...document.querySelectorAll("nav.toc a")];
    const targets = links.map((a) => document.querySelector(a.getAttribute("href"))).filter(Boolean);
    const obs = new IntersectionObserver((entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          links.forEach((a) => a.classList.toggle("active", a.getAttribute("href") === `#${e.target.id}`));
        }
      });
    }, { rootMargin: "0px 0px -75% 0px" });
    targets.forEach((t) => obs.observe(t));
  }

  document.addEventListener("DOMContentLoaded", () => { initTheme(); initFootnotes(); initToc(); });

  return { data, color, onColor, svg, axes, pct, fix, legend, tableView, register, showTip, hideTip, MODE_LABEL, FAMILY_LABEL };
})();
