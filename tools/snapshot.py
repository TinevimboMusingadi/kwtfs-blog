"""Static PNGs of every figure (for medium.md) and an og image, via headless Chrome.

    python -m http.server 8765 &         # serve the site
    python tools/snapshot.py             # writes img/fig-*.png and img/og.png
    python tools/snapshot.py --dark --pages out/   # optional: dark mode, plus full-page slices for QA

Drives Chrome over the DevTools protocol (needs `websocket-client`), so each
figure is clipped exactly from the live page, with its caption.
"""

from __future__ import annotations

import argparse
import base64
import json
import pathlib
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

import websocket

ROOT = pathlib.Path(__file__).resolve().parents[1]
CHROME = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "google-chrome", "chromium", "chromium-browser",
]


class Tab:

  def __init__(self, ws_url):
    self.ws = websocket.create_connection(ws_url, timeout=60, suppress_origin=True)
    self.n = 0

  def call(self, method, **params):
    self.n += 1
    self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
    while True:
      msg = json.loads(self.ws.recv())
      if msg.get("id") == self.n:
        if "error" in msg:
          raise RuntimeError(f"{method}: {msg['error']}")
        return msg.get("result", {})

  def js(self, expr):
    r = self.call("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
    return r.get("result", {}).get("value")

  def shot(self, path, x, y, w, h, scale=2):
    r = self.call("Page.captureScreenshot", format="png", captureBeyondViewport=True,
                  clip={"x": x, "y": y, "width": w, "height": h, "scale": scale})
    path.write_bytes(base64.b64decode(r["data"]))


def find_chrome():
  for c in CHROME:
    p = shutil.which(c) or (c if pathlib.Path(c).exists() else None)
    if p:
      return p
  raise SystemExit("snapshot: Chrome not found")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--url", default="http://localhost:8765/")
  ap.add_argument("--out", default=str(ROOT / "img"))
  ap.add_argument("--dark", action="store_true")
  ap.add_argument("--width", type=int, default=1400)
  ap.add_argument("--pages", default=None, help="also save full-page slices here (QA)")
  args = ap.parse_args()
  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)

  with socket.socket() as sk:  # a free port, so a stray Chrome cannot answer
    sk.bind(("127.0.0.1", 0))
    port = sk.getsockname()[1]
  prof = tempfile.mkdtemp()
  proc = subprocess.Popen([find_chrome(), "--headless=new", f"--remote-debugging-port={port}",
                           f"--user-data-dir={prof}", "--remote-debugging-address=127.0.0.1", "--hide-scrollbars", "--no-first-run",
                           f"--window-size={args.width},1000", "about:blank"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  try:
    tabs = None
    for _ in range(150):
      try:
        tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
        break
      except OSError:
        time.sleep(0.2)
    if tabs is None:
      raise SystemExit("snapshot: Chrome did not open its debugging port")
    tab = Tab(next(t for t in tabs if t["type"] == "page")["webSocketDebuggerUrl"])
    tab.call("Page.enable")
    tab.call("Page.addScriptToEvaluateOnNewDocument", source=(
        "window.__kwtErrors=[];addEventListener('error',e=>__kwtErrors.push(e.message));"
        "addEventListener('unhandledrejection',e=>__kwtErrors.push(String(e.reason)));"))
    tab.call("Emulation.setEmulatedMedia", features=[
        {"name": "prefers-color-scheme", "value": "dark" if args.dark else "light"}])
    tab.call("Emulation.setDeviceMetricsOverride", width=args.width, height=1000,
             deviceScaleFactor=1, mobile=args.width < 768)
    tab.call("Page.navigate", url=args.url)
    for _ in range(100):
      if tab.js("document.readyState") == "complete":
        break
      time.sleep(0.2)
    # Scroll through so lazy observers fire, then let figures settle.
    h = tab.js("document.documentElement.scrollHeight")
    for y in range(0, h, 800):
      tab.js(f"window.scrollTo(0,{y})")
      time.sleep(0.05)
    tab.js("window.scrollTo(0,0)")
    time.sleep(2.0)
    errors = tab.js("(window.__kwtErrors||[]).join('\\n')")
    if errors:
      print("page errors:\n" + errors)

    # Medium gets the first three trace cards, not all eight.
    tab.js("document.querySelectorAll('#fig-traces .trace-list > *').forEach((c, i) => { if (i >= 3) c.style.display = 'none'; })")
    figs = tab.js("""[...document.querySelectorAll('figure')].map(f => {
        const d = f.querySelector('.fig[id]'); const r = f.getBoundingClientRect();
        return d && {id: d.id, x: r.left + scrollX, y: r.top + scrollY, w: r.width, h: r.height};
      }).filter(Boolean)""")
    suffix = "-dark" if args.dark else ""
    for f in figs:
      pad = 16
      tab.shot(out / f"{f['id']}{suffix}.png", max(0, f["x"] - pad), max(0, f["y"] - pad),
               f["w"] + 2 * pad, f["h"] + 2 * pad)
      print(f"{f['id']}{suffix}.png  {f['w']:.0f}x{f['h']:.0f}")
    if not args.dark:
      # og image: the title over the hero figure, composed at 1200x630.
      t = tab.js("(() => {const r = document.querySelector('.title-block h1').getBoundingClientRect();"
                 "return {x: r.left + scrollX, y: r.top + scrollY, w: r.width, h: r.height};})()")
      tab.shot(out / "_title.png", t["x"], t["y"], t["w"], t["h"] + 8, scale=1)
      from PIL import Image  # pylint: disable=g-import-not-at-top
      bg = Image.open(out / "fig-hero.png").getpixel((4, 4))
      og = Image.new("RGB", (1200, 630), bg)
      title = Image.open(out / "_title.png")
      title.thumbnail((1100, 200))
      og.paste(title, (50, 36))
      hero = Image.open(out / "fig-hero.png")
      hero.thumbnail((1100, 630 - 60 - title.height - 30))
      og.paste(hero, ((1200 - hero.width) // 2, 36 + title.height + 20))
      og.save(out / "og.png")
      (out / "_title.png").unlink()
      print("og.png")
    if args.pages:
      pd = pathlib.Path(args.pages)
      pd.mkdir(parents=True, exist_ok=True)
      h = tab.js("document.documentElement.scrollHeight")
      for i, y in enumerate(range(0, h, 1800)):
        tab.shot(pd / f"page{suffix}{i:02d}.png", 0, y, args.width, min(1800, h - y), scale=0.5)
      print(f"{len(range(0, h, 1800))} page slices, height {h}")
  finally:
    if sys.platform == "win32":  # Chrome's helper processes outlive terminate() on Windows
      subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
      proc.terminate()
    shutil.rmtree(prof, ignore_errors=True)


if __name__ == "__main__":
  main()
