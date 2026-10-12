"""Convert the rendered post body (our restricted HTML) to Markdown for Medium / GitHub.

Figures become a static PNG (img/<fig-id>.png, made by tools/snapshot.py) plus the
caption and a link to the interactive version. Footnotes become numbered notes.
"""

from __future__ import annotations

import html
import pathlib
import re


def _inline(s: str) -> str:
  s = re.sub(r'<a class="cite" href="#ref-[\w-]+">(\d+)</a>', r"\1", s)
  s = re.sub(r'<a href="([^"]+)">(.*?)</a>', r"[\2](\1)", s, flags=re.S)
  s = re.sub(r"<(strong|b)>(.*?)</\1>", r"**\2**", s, flags=re.S)
  s = re.sub(r"<(em|i)>(.*?)</\1>", r"*\2*", s, flags=re.S)
  s = re.sub(r'<span class="mode \w+">(.*?)</span>', r"`\1`", s, flags=re.S)
  s = re.sub(r"<code>(.*?)</code>", r"`\1`", s, flags=re.S)
  s = re.sub(r"\\\((.*?)\\\)", r"$\1$", s)
  s = re.sub(r"<[^>]+>", "", s)
  return html.unescape(re.sub(r"\s+", " ", s)).strip()


def export(body: str, out: pathlib.Path, site: str) -> None:
  notes: list[str] = []

  def fn(m):
    notes.append(_inline(html.unescape(m.group(1))))
    return f"[^{len(notes)}]"

  body = re.sub(r'<sup class="fn" data-note="(.*?)"></sup>', fn, body, flags=re.S)
  lines: list[str] = []
  tokens = re.split(r"(<h2.*?</h2>|<h3.*?</h3>|<figure.*?</figure>|<pre>.*?</pre>|<table>.*?</table>|"
                    r"<ul.*?</ul>|<ol.*?</ol>|<div class=\"diary\">.*?</div>\s*</div>|<div class=\"math\">.*?</div>|"
                    r"<p.*?</p>)", body, flags=re.S)
  for tok in tokens:
    t = tok.strip()
    if not t:
      continue
    if t.startswith("<h2"):
      lines.append("## " + _inline(t.replace("</span>", "</span> ")))
    elif t.startswith("<h3"):
      lines.append("### " + _inline(t.replace("</span>", "</span> ")))
    elif t.startswith("<figure"):
      fid = re.search(r'id="(fig-[\w-]+)"', t).group(1)
      cap = _inline(re.search(r"<figcaption>(.*?)</figcaption>", t, flags=re.S).group(1))
      lines.append(f"![{cap[:120]}]({site}img/{fid}.png)\n*{cap}* [Interactive version]({site}#{fid})")
    elif t.startswith("<pre"):
      code = html.unescape(re.sub(r"<[^>]+>", "", t))
      lines.append("```\n" + code.strip() + "\n```")
    elif t.startswith("<table"):
      rows = re.findall(r"<tr>(.*?)</tr>", t, flags=re.S)
      md = []
      for i, row in enumerate(rows):
        cells = [_inline(c) for c in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, flags=re.S)]
        md.append("| " + " | ".join(cells) + " |")
        if i == 0:
          md.append("|" + "---|" * len(cells))
      lines.append("\n".join(md))
    elif t.startswith("<ul") or t.startswith("<ol"):
      items = re.findall(r"<li[^>]*>(.*?)</li>", t, flags=re.S)
      num = t.startswith("<ol")
      lines.append("\n".join(f"{i + 1}. {_inline(x)}" if num else f"- {_inline(x)}" for i, x in enumerate(items)))
    elif t.startswith('<div class="diary"'):
      rows = re.findall(r'<span class="tag">(.*?)</span><span>(.*?)</span>', t, flags=re.S)
      lines.append("\n".join(f"> **{_inline(a)}.** {_inline(b)}" + "\n>" for a, b in rows).rstrip(">\n"))
    elif t.startswith('<div class="math"'):
      lines.append(html.unescape(re.sub(r"<[^>]+>", "", t)).strip())
    elif t.startswith('<p class="limit"'):
      lines.append("> **What this doesn't show.** " + _inline(t))
    elif t.startswith('<p class="callout"'):
      lines.append("> " + _inline(t))
    elif t.startswith("<p"):
      lines.append(_inline(t))
    elif "<!--" in t or t.startswith("<ol class=\"refs\""):
      continue
  refs = re.findall(r'<li id="ref-[\w-]+">(.*?)</li>', body, flags=re.S)
  text = ("# Knowing When to Think Fast or Slow at Test Time\n\n*Tinevimbo Musingadi · October 2026 · "
          f"[interactive version with all figures]({site})*\n\n" + "\n\n".join(lines))
  text = text.replace("## § References", "## References")
  text += "\n\n" + "\n".join(f"{i + 1}. {_inline(r)}" for i, r in enumerate(refs))
  if notes:
    text += "\n\n" + "\n".join(f"[^{i + 1}]: {n}" for i, n in enumerate(notes))
  out.write_text(text + "\n", encoding="utf-8")
  print(f"{out.name}: {len(text):,} chars, {len(notes)} footnotes")
