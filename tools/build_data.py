"""Build every number and figure dataset for the research post from saved outputs.

    python tools/build_data.py --src "D:/knowing-when-to -think"

Writes data/numbers.json (every number the post quotes, keyed) plus one JSON per
figure. Nothing is computed by hand: results are re-scored from the raw model
completions with the same code the experiments used (kwt.report_v2, kwt.data.labels,
kwt.mode_value). No TPU, no network.
"""

from __future__ import annotations

import argparse
import collections
import glob
import itertools
import json
import pathlib
import random
import sys

import numpy as np

OUT = pathlib.Path(__file__).resolve().parents[1] / "data"
E2 = ["e2", "e2s1", "e2-s2", "e2-s3", "e2-s4"]
E3B = ["e3b", "e3bs1", "e3b-s2", "e3b-s3", "e3b-s4"]
NO_TEXT = ("bev-", "openbookqa-")  # sources released as IDs only (see DATASHEET)
GENRES = ["compare", "knowledge", "decision", "rubric", "arith", "wordproblem"]
LAM = 0.0005


def r(x, n=4):
  return None if x is None else round(float(x), n)


class Builder:

  def __init__(self, src: pathlib.Path):
    self.src = src
    sys.path[:0] = [str(src), str(src / "docs")]
    from kwt import report_v2  # pylint: disable=g-import-not-at-top
    from kwt.data import labels  # pylint: disable=g-import-not-at-top
    self.report_v2, self.labels = report_v2, labels
    self.items = {}
    with open(src / "data" / "kwt-v2" / "labelled.jsonl", encoding="utf-8") as handle:
      for line in handle:
        rec = json.loads(line)
        self.items[rec["id"]] = rec
    runs = sorted(glob.glob(str(src / "runs" / "kwt" / "final-*")))
    self.data = {s: report_v2.load(runs, s, self.items) for s in ("test", "ood")}
    self.raw = self._raw_completions(runs)
    self.numbers: dict[str, object] = {}

  def _raw_completions(self, runs):
    out = collections.defaultdict(dict)
    for d in runs:
      for sub in pathlib.Path(d).iterdir():
        for f in sub.glob("completions_*_*.jsonl"):
          _, split, control = f.stem.split("_")
          rows = {}
          for line in f.open(encoding="utf-8"):
            rec = json.loads(line)
            rows[rec["id"]] = rec
          out[(sub.name, split)][control] = rows
    return out

  def stats(self, rows):
    ids = sorted(rows)
    c, t, u = self.report_v2.utility(rows, ids)
    modes = [rows[i][2] for i in ids]
    return {"n": len(ids), "acc": r(c.mean()), "tokens": r(t.mean(), 2), "utility": r(u.mean()),
            "share": {m: r(modes.count(m) / len(modes), 4) for m in ("jev", "direct", "cot")}}

  # -- results -----------------------------------------------------------------

  def frontier(self):
    rows = []
    for split, data in self.data.items():
      for name, ctl in sorted(data.items()):
        fam = ("e2" if name in E2 else "e3b" if name in E3B else name)
        seed = (E2.index(name) if name in E2 else E3B.index(name) if name in E3B else None)
        for control, crow in ctl.items():
          rows.append({"split": split, "model": name, "family": fam, "seed": seed,
                       "control": control, **self.stats(crow)})
    self.write("frontier.json", rows)
    return rows

  def seeds(self, frontier):
    out = {}
    for split in ("test", "ood"):
      fam_acc = {}
      for fam, names in (("e2", E2), ("e3b", E3B)):
        accs = [next(x["acc"] for x in frontier if x["model"] == n and x["split"] == split
                     and x["control"] == "auto") for n in names]
        toks = [next(x["tokens"] for x in frontier if x["model"] == n and x["split"] == split
                     and x["control"] == "auto") for n in names]
        cot_acc = [next(x["acc"] for x in frontier if x["model"] == n and x["split"] == split
                        and x["control"] == "cot") for n in names]
        cot_tok = [next(x["tokens"] for x in frontier if x["model"] == n and x["split"] == split
                        and x["control"] == "cot") for n in names]
        fam_acc[fam] = np.array(accs)
        out[f"{split}_{fam}"] = {"acc": accs, "tokens": toks, "cot_acc": cot_acc,
                                 "cot_tokens": cot_tok, "mean": r(np.mean(accs)),
                                 "sd": r(np.std(accs, ddof=1)), "min": r(min(accs)),
                                 "max": r(max(accs)), "mean_tokens": r(np.mean(toks), 2),
                                 "mean_cot_acc": r(np.mean(cot_acc)),
                                 "mean_cot_tokens": r(np.mean(cot_tok), 2)}
      a, b = fam_acc["e3b"], fam_acc["e2"]
      pooled, n = np.concatenate([a, b]), len(a)
      diff = a.mean() - b.mean()
      perms = [pooled[list(c)].mean() - np.delete(pooled, list(c)).mean()
               for c in itertools.combinations(range(len(pooled)), n)]
      out[f"{split}_diff"] = {"diff": r(diff), "p_one_sided": r(np.mean([d >= diff - 1e-12 for d in perms]), 4),
                              "sd_ratio": r(np.std(b, ddof=1) / np.std(a, ddof=1), 2)}
    self.write("seeds.json", out)
    return out

  def paired(self, a_name, a_ctl, b_name, b_ctl, split="test", lam=LAM):
    ra, rb = self.data[split][a_name][a_ctl], self.data[split][b_name][b_ctl]
    ids = sorted(set(ra) & set(rb))
    ca, ta, ua = self.report_v2.utility(ra, ids, lam)
    cb, tb, ub = self.report_v2.utility(rb, ids, lam)
    m, lo, hi = self.report_v2.boot_diff(ua, ub)
    ma, la, ha = self.report_v2.boot_diff(ca, cb)
    mt, lt, ht = self.report_v2.boot_diff(ta, tb)
    return {"utility": [r(m), r(lo), r(hi)], "acc": [r(ma), r(la), r(ha)],
            "tokens": [r(mt, 2), r(lt, 2), r(ht, 2)], "n": len(ids)}

  def sweep(self):
    """Seed-averaged utility(auto) - utility(always-CoT, same weights), per λ."""
    lams = [round(x, 5) for x in np.linspace(0, 0.004, 17)]  # 0.00025 steps: includes 0.0005 and 0.001
    rows = []
    rng = np.random.default_rng(0)
    ids = sorted(set.intersection(*[set(self.data["test"][n]["auto"]) for n in E3B]))
    idx = rng.integers(0, len(ids), (3000, len(ids)))
    for lam in lams:
      d = np.mean([self.report_v2.utility(self.data["test"][n]["auto"], ids, lam)[2]
                   - self.report_v2.utility(self.data["test"][n]["cot"], ids, lam)[2]
                   for n in E3B], axis=0)
      means = d[idx].mean(axis=1)
      rows.append({"lam": lam, "diff": r(d.mean()), "lo": r(np.percentile(means, 2.5)),
                   "hi": r(np.percentile(means, 97.5))})
    self.write("sweep.json", rows)
    first_sig = next((x["lam"] for x in rows if x["lo"] > 0), None)
    return rows, first_sig

  def genres(self):
    out = []
    for g in GENRES:
      row = {"genre": g}
      for label, names in (("sft", ["sft"]), ("e2", E2), ("e3b", E3B)):
        shares, accs = [], []
        for n in names:
          rows = self.data["test"][n]["auto"]
          ids = [i for i in rows if self.items[i]["genre"] == g]
          shares.append(sum(rows[i][2] == "cot" for i in ids) / len(ids))
          accs.append(sum(rows[i][0] for i in ids) / len(ids))
        row[f"{label}_cot_share"] = r(np.mean(shares))
        row[f"{label}_acc"] = r(np.mean(accs))
      row["n"] = sum(1 for i in self.data["test"]["sft"]["auto"] if self.items[i]["genre"] == g)
      out.append(row)
    self.write("genres.json", out)
    return out

  def dynamics(self):
    out = {}
    names = {"rl-e2": "e2", "rl-e2s1": "e2s1", "rl-e2-s4": "e2-s4", "rl-e3b": "e3b",
             "rl-e3bs1": "e3bs1", "rl-e3b-s2": "e3b-s2", "rl-e3b-s4": "e3b-s4",
             "rl-e3": "e3", "rl-e6": "e6", "rl-e3bfix": "e3bfix"}
    for run, name in names.items():
      path = self.src / "runs" / "kwt" / "steplogs" / f"{run}.jsonl"
      rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
      if len(rows) < 100:
        continue
      out[name] = [{k: r(v, 4) for k, v in row.items()} for row in rows]
    self.write("dynamics.json", out)
    return out

  def traces(self, k=72, seed=0):
    """Real test outputs from E3b seed 3 (best seed), auto + the same question forced to CoT."""
    rng = random.Random(seed)
    auto = self.raw[("e3b-s3", "test")]["auto"]
    cot = self.raw[("e3b-s3", "test")]["cot"]
    scored = self.data["test"]["e3b-s3"]["auto"]
    buckets = collections.defaultdict(list)
    for i, rec in auto.items():
      it = self.items[i]
      if str(i).startswith(NO_TEXT):  # licence does not clearly allow republishing the text
        continue
      ok, _, mode = scored[i]
      handoff = "<mode:cot>" in rec["completion"] and "<mode:jev>" in rec["completion"]
      buckets[(it["genre"], mode, ok, handoff)].append(i)
    keys = sorted(buckets)
    picked = []
    while len(picked) < k and any(buckets[x] for x in keys):
      for key in keys:
        if buckets[key] and len(picked) < k:
          picked.append(buckets[key].pop(rng.randrange(len(buckets[key]))))
    out = []
    for i in picked:
      it = self.items[i]
      ok, tok, mode = scored[i]
      out.append({
          "id": i, "genre": it["genre"], "question": it["question"][:700], "options": it["options"],
          "gold": str(it["gold"]), "completion": auto[i]["completion"][:900], "tokens": tok,
          "mode": mode, "correct": bool(ok), "jev_probs": auto[i].get("jev_probs"),
          "cot_completion": cot.get(i, {}).get("completion", "")[:900],
          "cot_tokens": cot.get(i, {}).get("tokens"),
          "cot_correct": bool(self.data["test"]["e3b-s3"]["cot"].get(i, (False,))[0]),
          "suggested": (it.get("label") or {}).get("suggested"),
          "label_p": (it.get("label") or {}).get("p"), "label_t": (it.get("label") or {}).get("t")})
    self.write("traces.json", out)
    return out

  # -- data and labels -----------------------------------------------------------

  def labels_sample(self, per_genre=500, seed=0):
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for rec in self.items.values():
      if rec.get("label"):
        by[rec["genre"]].append(rec)
    out = []
    for g in GENRES:
      rows = by[g]
      rng.shuffle(rows)
      for rec in rows[:per_genre]:
        lab = rec["label"]
        it = self.labels_item(rec)
        out.append({"g": g, "b": self.labels.difficulty_band(it),
                    "p": {m: r(v, 3) for m, v in lab["p"].items()},
                    "t": {m: r(v, 1) for m, v in lab["t"].items()}})
    self.write("labels_sample.json", out)
    return out

  def labels_item(self, rec):
    from kwt.data import schema  # pylint: disable=g-import-not-at-top
    fields = {k: rec[k] for k in ("id", "source", "licence", "genre", "domain", "question",
                                  "options", "gold", "answer_type", "difficulty", "split", "meta")}
    return schema.Item(**fields)

  def label_report(self):
    path = self.src / "runs" / "kwt" / "profile-v2" / "label_report.json"
    rep = json.loads(path.read_text(encoding="utf-8"))
    keep = {k: rep[k] for k in ("n", "ambiguous", "mean_p", "mean_tokens", "gates",
                                "share_by_genre", "routers_utility")}
    self.write("label_report.json", keep)
    return keep

  def probe(self):
    """Retrain the mode-value probe (CPU) on SFT+RL items; calibration on the test split."""
    from kwt import mode_value  # pylint: disable=g-import-not-at-top
    from sklearn.metrics import roc_auc_score  # pylint: disable=g-import-not-at-top
    d = np.load(self.src / "runs" / "kwt" / "prep-v2" / "hidden.npz")
    H, ids = d["H"], [str(x) for x in d["ids"]]
    profiles = self.labels.load_profiles(self.src / "runs" / "kwt" / "profile-v2" / "profile.jsonl")
    split = [self.items[x]["split"] if x in self.items else None for x in ids]
    train = {i for i, s in enumerate(split) if s in ("sft", "rl")}
    test = [i for i, s in enumerate(split) if s == "test"]
    model = mode_value.fit(H, ids, profiles, train)
    pred = mode_value.predict(model, H[test])
    out = {}
    for m, (p, _) in pred.items():
      rows = [(k, profiles[ids[i]][m]) for k, i in enumerate(test) if m in profiles.get(ids[i], {})]
      true_p = np.array([sum(x["correct"]) / len(x["correct"]) for _, x in rows])
      pp = p[[k for k, _ in rows]]
      bins = []
      for b in range(10):
        mask = (pp >= b / 10) & (pp < (b + 1) / 10 if b < 9 else pp <= 1)
        if mask.sum():
          bins.append({"pred": r(pp[mask].mean()), "true": r(true_p[mask].mean()), "n": int(mask.sum())})
      y = (true_p >= 0.5).astype(int)
      ece = sum(abs(x["pred"] - x["true"]) * x["n"] for x in bins) / len(pp)
      out[m] = {"n": len(pp), "auroc": r(roc_auc_score(y, pp)), "ece": r(ece), "bins": bins}
    self.write("probe.json", out)
    return out

  # -- assemble ----------------------------------------------------------------------

  def write(self, name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(obj, separators=(",", ":"), ensure_ascii=False),
                            encoding="utf-8")

  def run(self):
    n = self.numbers
    fr = self.frontier()
    sd = self.seeds(fr)
    sw, first_sig = self.sweep()
    gen = self.genres()
    dyn = self.dynamics()
    self.traces()
    self.labels_sample()
    lr = self.label_report()
    pr = self.probe()

    def get(model, split="test", control="auto"):
      return next(x for x in fr if x["model"] == model and x["split"] == split and x["control"] == control)

    sft_auto, sft_cot = get("sft"), get("sft", control="cot")
    n["test_n"] = sft_auto["n"]
    n["ood_n"] = get("sft", "ood")["n"]
    n["sft_auto_acc"], n["sft_auto_tokens"] = sft_auto["acc"], sft_auto["tokens"]
    n["sft_auto_jev_share"] = sft_auto["share"]["jev"]
    n["sft_cot_acc"], n["sft_cot_tokens"] = sft_cot["acc"], sft_cot["tokens"]
    n["sft_ood_acc"] = get("sft", "ood")["acc"]
    for split in ("test", "ood"):
      for fam in ("e2", "e3b"):
        s = sd[f"{split}_{fam}"]
        for k in ("mean", "sd", "min", "max", "mean_tokens", "mean_cot_acc", "mean_cot_tokens"):
          n[f"{split}_{fam}_{k}"] = s[k]
      n[f"{split}_seed_diff"] = sd[f"{split}_diff"]["diff"]
      n[f"{split}_seed_p"] = sd[f"{split}_diff"]["p_one_sided"]
      n[f"{split}_seed_sd_ratio"] = sd[f"{split}_diff"]["sd_ratio"]
    n["test_token_saving_vs_cot"] = r(1 - sd["test_e3b"]["mean_tokens"] / sd["test_e3b"]["mean_cot_tokens"], 3)
    n["sweep_first_significant_lambda"] = first_sig
    n["sweep_at_train_lambda"] = next(x for x in sw if abs(x["lam"] - LAM) < 1e-9)
    n["sweep_at_0001"] = next(x for x in sw if abs(x["lam"] - 0.001) < 1e-9)
    n["sweep_at_0"] = sw[0]
    for key, args in {
        "e3b_vs_e2": ("e3b", "auto", "e2", "auto"),
        "e3b_vs_e3": ("e3b", "auto", "e3", "auto"),
        "e3b_vs_e6": ("e3b", "auto", "e6", "auto"),
        "e3bfix_vs_e3b": ("e3bfix", "auto", "e3b", "auto"),
        "e3b_vs_sft": ("e3b", "auto", "sft", "auto"),
        "e3b_auto_vs_cot": ("e3b", "auto", "e3b", "cot"),
    }.items():
      n[f"paired_{key}"] = self.paired(*args)
    for m in ("e3", "e6", "e3bfix"):
      n[f"{m}_test_acc"] = get(m)["acc"]
    for g in gen:
      n[f"genre_{g['genre']}"] = g
    n["probe"] = {m: {"auroc": v["auroc"], "ece": v["ece"]} for m, v in pr.items()}
    n["labels"] = {"n": lr["n"], "ambiguous": lr["ambiguous"], "mean_p": lr["mean_p"],
                   "mean_tokens": lr["mean_tokens"], "stability": lr["gates"]["stability"],
                   "headroom": lr["gates"]["headroom"],
                   "router_all": lr["routers_utility"]["ALL"]}
    lat = json.loads((self.src / "runs" / "kwt" / "latency" / "e3bs1" / "latency.json").read_text())
    n["latency"] = lat
    self.write("latency.json", lat)
    n["dynamics_runs"] = sorted(dyn)
    # v1 (Qwen3-1.7B) sealed-test results, from the v1 result files.
    v1 = {}
    for key, f in (("base", "test_base"), ("rl_from_base", "test_run1_grpo_from_base"),
                   ("sft", "test_sft"), ("rl_lr3e6", "test_run2_grpo_lr3e-6"), ("rl_lr3e5", "test_run3_grpo_lr3e-5")):
      res = json.loads((self.src / "docs" / "results" / f"{f}.json").read_text())
      v1[key] = {"acc": r(res["learned"]["all"]["accuracy"]), "tokens": r(res["learned"]["all"]["mean_tokens"], 1),
                 "cot_acc": r(res["cot"]["all"]["accuracy"]), "cot_tokens": r(res["cot"]["all"]["mean_tokens"], 1)}
    run3 = json.loads((self.src / "docs" / "results" / "test_run3_grpo_lr3e-5.json").read_text())
    v1["run3_direct_behavior_acc"] = r(run3["learned"]["direct"]["accuracy"])
    n["v1"] = v1
    # dataset and audit counts
    br = json.loads((self.src / "data" / "kwt-v2" / "build_report.json").read_text())
    n["data"] = {"built": sum(sum(v.values()) for v in br["splits"].values()),
                 "dropped_long": len(br["dropped_long_prompt"]["ids"]), "final": len(self.items),
                 "flagged": sum(1 for _ in open(self.src / "runs" / "kwt" / "profile-v2" / "audit_queue.jsonl", encoding="utf-8")),
                 "sources": 6, "genres": len(GENRES)}
    shares = lr["share_by_genre"]
    n["label_share"] = {g: {m: r(c.get(m, 0) / sum(c.values()), 3) for m in ("jev", "direct", "cot")}
                        for g, c in shares.items()}
    # probe used as a router (validation), from the mode-value report
    mv = json.loads((self.src / "runs" / "kwt" / "mode_value" / "report.json").read_text())
    val_items = [self.labels_item(x) for x in self.items.values() if x["split"] == "val"]
    profiles = self.labels.load_profiles(self.src / "runs" / "kwt" / "profile-v2" / "profile.jsonl")
    val_rep = self.labels.report(val_items, profiles)
    n["probe_router_val"] = {"probe": mv["val"]["router_utility"]["probe"],
                             "lookup": r(val_rep["routers_utility"]["ALL"]["lookup"]),
                             "oracle": r(val_rep["routers_utility"]["ALL"]["oracle"])}
    # self-distilled CoT trace statistics (SFT split)
    from kwt import checks  # pylint: disable=g-import-not-at-top
    from kwt.data import sft_data  # pylint: disable=g-import-not-at-top
    st = collections.Counter()
    usable = total_items = 0
    for line in open(self.src / "runs" / "kwt" / "prep-v2" / "traces.jsonl", encoding="utf-8"):
      rec = json.loads(line)
      if rec["id"] not in self.items:
        continue
      total_items += 1
      for s in rec["samples"]:
        st["samples"] += 1
        st["correct"] += s["correct"]
        cl = checks.claims(s["text"])
        bad = [c for c in cl if c.true is False]
        st["with_claims"] += bool(cl)
        st["with_false"] += bool(bad)
        st["correct_but_false"] += bool(bad) and s["correct"]
        st["truncated"] += "<answer>" not in s["text"]
      usable += sft_data.best_cot_body(self.labels_item(self.items[rec["id"]]), rec["samples"]) is not None
    n["traces"] = {"samples": st["samples"], "correct": r(st["correct"] / st["samples"], 3),
                   "truncated": r(st["truncated"] / st["samples"], 3),
                   "false_among_checkable": r(st["with_false"] / st["with_claims"], 3),
                   "false_but_correct": r(st["correct_but_false"] / st["with_false"], 3),
                   "usable_items": r(usable / total_items, 3)}
    sft = collections.Counter()
    with open(self.src / "data" / "kwt-v2" / "sft.jsonl", encoding="utf-8") as handle:
      for line in handle:
        ex = json.loads(line)
        sft[f"{ex['kind']}_{ex['mode']}"] += 1
    n["sft_mix"] = dict(sft)
    n["sft_examples"] = sum(sft.values())
    audit = json.loads((self.src / "docs" / "data-v2" / "audit_decisions.json").read_text(encoding="utf-8"))
    n["audit"] = {"reviewed": audit["reviewed_counts"], "fixed": len(audit["fix_gold"]),
                  "dropped": len(audit["drop"])}
    # derived comparisons quoted in the text
    n["v1_gap"] = r(v1["sft"]["cot_acc"] - v1["sft"]["acc"])
    n["sft_gap"] = r(sft_cot["acc"] - sft_auto["acc"])
    n["e3b_mean_minus_sft"] = r(sd["test_e3b"]["mean"] - sft_auto["acc"])
    n["latency_saving"] = {"mean": r(1 - lat["auto"]["mean_s"] / lat["cot"]["mean_s"], 3),
                           "median": r(1 - lat["auto"]["median_s"] / lat["cot"]["median_s"], 3)}
    rates = {}
    for src_name, c in audit["reviewed_counts"].items():
      rates[src_name] = {"wrong": r(c["gold_wrong"] / c["audited"], 3), "ambiguous": r(c["ambiguous"] / c["audited"], 3)}
    n["audit_rates"] = rates
    n["audit_wrong_range"] = [min(v["wrong"] for k, v in rates.items() if k != "openbookqa"),
                              max(v["wrong"] for v in rates.values())]
    # forced-mode obedience over every forced-control evaluation we ran
    from kwt import checks, format as fmt  # pylint: disable=g-import-not-at-top
    obey = []
    for (model, split), ctls in self.raw.items():
      for control, rows in ctls.items():
        if control == "auto":
          continue
        ok = sum(fmt.parse_v2(x["completion"], len(self.items[i]["options"]), control).obeyed for i, x in rows.items())
        obey.append(ok / len(rows))
    n["obedience_min"] = r(min(obey), 4)
    n["obedience_evals"] = len(obey)
    # true-equation rate in reasoning traces (test, auto), regret vs regret+faithful
    def faithful_rate(model):
      rows = self.raw[(model, "test")]["auto"]
      cot = [x["completion"] for x in rows.values() if "<mode:cot>" in x["completion"]]
      return r(sum(checks.faithful(c) for c in cot) / len(cot), 3)
    n["faithful_rate"] = {"e3b": faithful_rate("e3b"), "e3": faithful_rate("e3"), "e2": faithful_rate("e2")}
    self.write("numbers.json", n)
    print(json.dumps({k: n[k] for k in ("test_e3b_mean", "test_e3b_sd", "test_e2_mean", "test_e2_sd",
                                        "test_seed_p", "test_token_saving_vs_cot",
                                        "sweep_first_significant_lambda")}, indent=1))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--src", required=True)
  Builder(pathlib.Path(ap.parse_args().src)).run()


if __name__ == "__main__":
  main()
