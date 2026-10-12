"""Check the reward explorer (js/figures-method.js REWARD.terms) against kwt/rewards_v2.py.

    python tests/reward_parity.py --src "D:/knowing-when-to -think"

Python scores real-format completions; the same parsed features go through the
JS function under Node; every term must agree to 1e-9.
"""

import argparse
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
args = ap.parse_args()
sys.path.insert(0, args.src)

from kwt import checks, rewards_v2 as R, format as fmt  # noqa: E402

U = {"jev": 0.55, "direct": 0.60, "cot": 0.86}
OPTS = json.dumps(["12", "15", "18", "21"])
CASES = [  # (completion, control, answer, genre, seen)
    ("<genre:wordproblem><mode:jev>\nB", "auto", "15", "wordproblem", 0),
    ("<genre:wordproblem><mode:jev>\nA", "auto", "15", "wordproblem", 0),
    ("<genre:arith><mode:cot>\n3 * 5 = 15.\n<mode:jev>\nB", "auto", "15", "wordproblem", 1000),
    ("<genre:wordproblem><mode:cot>\n3 * 5 = 16. So 15.\n<answer>15</answer>", "auto", "15", "wordproblem", 2000),
    ("<genre:wordproblem><mode:direct>\n<answer>15</answer>", "auto", "15", "wordproblem", 1500),
    ("<genre:wordproblem><mode:direct>\n<answer>15</answer>", "auto", "15", "wordproblem", 3000),
    ("<genre:wordproblem><mode:direct>\n<answer>15</answer>", "cot", "15", "wordproblem", 0),
    ("<genre:wordproblem><mode:cot>\n" + "4 + 4 = 9.\n" * 6 + "<answer>15</answer>", "cot", "15", "wordproblem", 500),
    ("<genre:wordproblem><mode:cot>\n" + "x" * 3000 + "\n<answer>15</answer>", "auto", "15", "wordproblem", 100),
    ("garbage", "auto", "15", "wordproblem", 0),
]
TERMS = ["correctness", "well_formed", "compute_cost", "faithful", "genre", "route_regret"]

py, feats = [], []
for comp, ctrl, ans, g, seen in CASES:
  kw = dict(prompts=[""], completions=[comp], answer=[ans], options=[OPTS], control=[ctrl],
            genre=[g], utilities=[json.dumps(U)])
  row = {}
  for t in TERMS:
    R._seen = seen
    row[t] = getattr(R, t)(**kw)[0]
  py.append(row)
  p = fmt.parse_v2(comp, 4, ctrl)
  feats.append(dict(correct=fmt.is_correct_v2(p.trace, ans, json.loads(OPTS)), obeyed=p.obeyed,
                    valid=p.trace.valid, chars=len(comp), falseClaims=len(checks.false_claims(comp)),
                    genreOk=p.genre == g, control=ctrl, mode=p.trace.first_mode, utilities=U, seen=seen))

js_src = (ROOT / "js" / "figures-method.js").read_text(encoding="utf-8")
start = js_src.index("const REWARD = {")
end = js_src.index("\n};", start) + 3
node = js_src[start:end] + f"\nconsole.log(JSON.stringify({json.dumps(feats)}.map(f => REWARD.terms(f))));"
js = json.loads(subprocess.run(["node", "-e", node], capture_output=True, text=True, check=True).stdout)

for r in py: print({k: round(v, 4) for k, v in r.items()})
bad = 0
for i, (a, b) in enumerate(zip(py, js)):
  for t in TERMS:
    if abs(a[t] - b[t]) > 1e-9:
      bad += 1
      print(f"case {i} {t}: python {a[t]} js {b[t]}")
print(f"{len(CASES)} cases x {len(TERMS)} terms, {bad} mismatches")
sys.exit(1 if bad else 0)
