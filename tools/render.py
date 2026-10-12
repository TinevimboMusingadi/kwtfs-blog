"""Render the post: src/post.html + data/numbers.json + tools/refs.json -> index.html.

    python tools/render.py            # writes index.html and medium.md

Placeholders in src/post.html:
  {{n:path|fmt}}   a number from numbers.json (dot path; list indices allowed)
  {{cite:a,b}}     numbered citation links, numbered in order of first use
Every placeholder must resolve, or rendering fails. That is the fact check:
no number in the post is typed by hand.
"""

from __future__ import annotations

import html
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SITE_URL = "https://tinevimbomusingadi.github.io/kwtfs-blog/"


def lookup(numbers, path):
  cur = numbers
  for part in path.split("."):
    if isinstance(cur, list):
      cur = cur[int(part)]
    elif isinstance(cur, dict) and part in cur:
      cur = cur[part]
    else:
      raise KeyError(path)
  return cur


def fmt(value, kind):
  if kind == "pct0":
    return f"{100 * value:.0f}%"
  if kind == "pct1":
    return f"{100 * value:.1f}%"
  if kind == "pp1":  # signed percentage points
    return f"{100 * value:+.1f}"
  if kind == "pts1":  # unsigned points
    return f"{100 * value:.1f}"
  if kind == "ci_pp":  # [mean, lo, hi] in percentage points
    m, lo, hi = value
    return f"{100 * m:+.1f} points [{100 * lo:+.1f}, {100 * hi:+.1f}]"
  if kind == "ci_u":  # [mean, lo, hi] utility
    m, lo, hi = value
    return f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]"
  if kind == "ci_tok":
    m, lo, hi = value
    return f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}]"
  if kind in ("f0", "f1", "f2", "f3", "f4"):
    return f"{float(value):.{kind[1]}f}"
  if kind == "int":
    return f"{int(value):,}"
  if kind == "p3":
    return f"{value:.3f}"
  if kind == "raw":
    return str(value)
  raise ValueError(kind)


def render_body(src: str, numbers: dict, refs: dict):
  order: list[str] = []

  def num(m):
    path, kind = m.group(1), m.group(2) or "raw"
    try:
      return fmt(lookup(numbers, path), kind)
    except (KeyError, IndexError, TypeError) as exc:
      sys.exit(f"render: number {path!r} not found in numbers.json ({exc})")

  def cite(m):
    keys = [k.strip() for k in m.group(1).split(",")]
    links = []
    for k in keys:
      if k not in refs:
        sys.exit(f"render: citation {k!r} not in refs.json")
      if k not in order:
        order.append(k)
      links.append(f'<a class="cite" href="#ref-{k}">{order.index(k) + 1}</a>')
    return "[" + ", ".join(links) + "]"

  body = re.sub(r"\{\{n:([\w.\-]+)(?:\|(\w+))?\}\}", num, src)
  body = re.sub(r"\{\{cite:([\w,\s\-]+)\}\}", cite, body)
  leftover = re.findall(r"\{\{[^}]*\}\}", body)
  if leftover:
    sys.exit(f"render: unresolved placeholders {leftover[:5]}")
  ref_list = "\n".join(
      f'<li id="ref-{k}">{html.escape(refs[k]["text"])} <a href="{refs[k]["url"]}">{html.escape(refs[k]["url"])}</a></li>'
      for k in order)
  body = body.replace("<!--REFERENCES-->", f'<ol class="refs">{ref_list}</ol>')
  return body, order


def toc(body: str):
  items = re.findall(r'<h2 id="([\w-]+)"><span class="sec">([^<]*)</span>([^<]*)</h2>', body)
  return "\n".join(f'<a href="#{i}"><span class="sec">{s.strip()}</span>{html.escape(t.strip())}</a>' for i, s, t in items)


SHELL = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Knowing When to Think Fast or Slow at Test Time</title>
<meta name="description" content="One small language model learns, per question, whether to decide in a single forward pass, answer directly, or reason first, using routing labels measured on the model itself and a regret reward.">
<meta property="og:title" content="Knowing When to Think Fast or Slow at Test Time">
<meta property="og:description" content="Teaching one 4B model when to think: measured mode labels, native one-step decisions, and a regret reward that makes RL stable.">
<meta property="og:image" content="{site}img/og.png">
<link rel="stylesheet" href="style.css">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js"></script>
</head>
<body>
<header class="masthead"><div class="inner"><span>Independent research · test-time compute</span>
<button class="theme-toggle" type="button">Light / dark</button></div></header>
<div class="title-block">
<h1>Knowing When to Think Fast or Slow at Test Time</h1>
<p class="lede">One small language model learns, question by question, whether to decide in a single forward pass, answer in a few words, or reason first. Its routing labels are measured on the model itself, and a regret reward keeps reinforcement learning stable.</p>
</div>
<div class="byline">
<div><b>Author</b>Tinevimbo Musingadi</div>
<div><b>Published</b>October 2026</div>
<div><b>Code</b><a href="https://github.com/TinevimboMusingadi/kwtfs-public">kwtfs-public</a></div>
<div><b>Model</b>Qwen3-4B + LoRA, trained with Tunix on spot TPUs</div>
</div>
<div class="layout">
<nav class="toc"><b>Contents</b>
{toc}
</nav>
<article>
{body}
</article>
</div>
<footer>Knowing When to Think Fast or Slow at Test Time · Tinevimbo Musingadi · 2026. Every number on this page is generated from the experiment outputs by <code>tools/build_data.py</code> and filled in by <code>tools/render.py</code>.</footer>
<script src="js/common.js"></script>
<script src="js/figures-method.js"></script>
<script src="js/figures-results.js"></script>
<script>
document.addEventListener("DOMContentLoaded", () => {{
  if (window.renderMathInElement) renderMathInElement(document.body, {{delimiters: [{{left: "$$", right: "$$", display: true}}, {{left: "\\\\(", right: "\\\\)", display: false}}]}});
}});
</script>
</body>
</html>
"""


def main():
  numbers = json.loads((ROOT / "data" / "numbers.json").read_text(encoding="utf-8"))
  refs = json.loads((ROOT / "tools" / "refs.json").read_text(encoding="utf-8"))
  src = (ROOT / "src" / "post.html").read_text(encoding="utf-8")
  body, order = render_body(src, numbers, refs)
  page = SHELL.format(site=SITE_URL, toc=toc(body), body=body)
  (ROOT / "index.html").write_text(page, encoding="utf-8")
  print(f"index.html: {len(page):,} chars, {len(order)} references cited")
  sys.path.insert(0, str(ROOT / "tools"))
  import export_md  # pylint: disable=g-import-not-at-top
  export_md.export(body, ROOT / "medium.md", SITE_URL)


if __name__ == "__main__":
  main()
