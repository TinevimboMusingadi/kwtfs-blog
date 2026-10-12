"""Check every external link in index.html and medium.md returns HTTP 200.

    python tools/linkcheck.py
"""

import concurrent.futures as cf
import pathlib
import re
import sys
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]


def check(url):
  try:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 kwtfs-linkcheck"})
    return urllib.request.urlopen(req, timeout=25).status, url
  except Exception as exc:  # pylint: disable=broad-except
    return getattr(exc, "code", str(exc)[:60]), url


def main():
  html = (ROOT / "index.html").read_text(encoding="utf-8")
  md = (ROOT / "medium.md").read_text(encoding="utf-8")
  urls = sorted(set(re.findall(r'href="(https?://[^"]+)"', html)) | set(re.findall(r"\((https?://[^)\s]+)\)", md)))
  with cf.ThreadPoolExecutor(8) as ex:
    bad = [r for r in ex.map(check, urls) if r[0] != 200]
  print(f"{len(urls)} links, {len(bad)} failing")
  for code, url in bad:
    print(f"  {code}  {url}")
  sys.exit(1 if bad else 0)


if __name__ == "__main__":
  main()
