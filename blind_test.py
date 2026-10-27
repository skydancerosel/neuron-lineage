#!/usr/bin/env python
"""
blind_test.py

The blind two-choice test of lineage. For each dated new job in one or more
neuron_lineage reports, a reader sees the new job's contexts and two candidate
earlier jobs: its real parent (the unit's own nearest job at the checkpoint just
before birth) and a decoy (another unit's job at that checkpoint, same Pile
source, nearest the new job, picked the way the null picks them). The reader
names the one that comes from the same unit. Items from every report are mixed
and shuffled, and the key stays in a separate file, so readers can't tell real
neurons from random directions. Accuracy above 50% means the lineage is visible
to a reader, whatever the embeddings say.

  python blind_test.py make  REPORT [REPORT ...] --out blind --per-report 60 --files 4
  python blind_test.py score --out blind      # reads blind/answers_*.json
"""
import argparse
import glob
import json
import math
import os

import numpy as np

PROMPT = """Each item shows a NEW pattern: up to five text snippets where one feature of a
language model fires on the token marked [[like this]]. Then two CANDIDATE patterns, A and B,
from earlier in training. Exactly one candidate is an earlier pattern of the same feature; the
other is a pattern of a different feature, taken from the same kind of document. For every item,
pick the candidate you think comes from the same feature. Use any cue you like."""


def show(t):
    return t.replace("\n", " ⏎ ").strip()


def make(a):
    rng = np.random.default_rng(a.seed)
    items, key = [], {}
    for path in a.reports:
        r = json.load(open(path))
        dated = [e for e in r["events"] if e["kind"] == "dated" and "decoy_examples" in e]
        if a.adjacent is None:
            pick = [dated[i] for i in rng.permutation(len(dated))[:a.per_report]]
        else:  # stratified by the role-space label
            pick = []
            for lab, n in (("adjacent", a.adjacent), ("no_lineage", a.other)):
                pool = [e for e in dated if e["role"]["label"] == lab]
                pick += [pool[i] for i in rng.permutation(len(pool))[:n]]
        for e in pick:
            iid = "".join(rng.choice(list("abcdefghjkmnpqrstuvwxyz23456789"), 6))
            parent_is_a = bool(rng.random() < 0.5)
            cand = (e["parent_examples"], e["decoy_examples"]) if parent_is_a else (e["decoy_examples"], e["parent_examples"])
            items.append((iid, e["examples"], cand))
            key[iid] = {"report": path, "basis": r["basis"], "layer": r["layer"], "neuron": e["neuron"],
                        "step": e["step"], "birth": e["birth"], "answer": "A" if parent_is_a else "B",
                        "genre": e["genre"]["label"], "role": e["role"]["label"]}
    order = rng.permutation(len(items))
    os.makedirs(a.out, exist_ok=True)
    for f in range(a.files):
        part = [items[j] for j in order[f::a.files]]
        lines = [PROMPT, ""]
        for iid, new, (A, B) in part:
            lines += [f"### Item {iid}", "", "NEW:"] + [f"{n + 1}. {show(t)}" for n, t in enumerate(new)]
            lines += ["", "Candidate A:"] + [f"{n + 1}. {show(t)}" for n, t in enumerate(A)]
            lines += ["", "Candidate B:"] + [f"{n + 1}. {show(t)}" for n, t in enumerate(B)] + [""]
        with open(os.path.join(a.out, f"items_{f + 1}.md"), "w") as fh:
            fh.write("\n".join(lines))
    with open(os.path.join(a.out, "key.json"), "w") as fh:
        json.dump(key, fh, indent=1)
    print(f"{len(items)} items in {a.files} files under {a.out}/ (key in key.json)")


def wilson(k, n):
    if not n:
        return None
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 3), round(c + h, 3)]


def binom_p(k, n):
    """One-sided P(X >= k) for X ~ Binomial(n, 1/2)."""
    return sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n


def score(a):
    key = json.load(open(os.path.join(a.out, "key.json")))
    answers = {}
    for path in sorted(glob.glob(os.path.join(a.out, "answers_*.json"))):
        for x in json.load(open(path)):
            answers[x["id"]] = x
    rows = [(v, answers[i]) for i, v in key.items() if i in answers]
    out = {"answered": len(rows), "of": len(key)}

    def acc(sel):
        k = sum(ans["choice"] == v["answer"] for v, ans in sel)
        return {"correct": k, "n": len(sel), "accuracy": round(k / len(sel), 3) if sel else None,
                "ci95": wilson(k, len(sel)), "p_vs_half": round(binom_p(k, len(sel)), 4) if sel else None}

    for basis in sorted({v["basis"] for v, _ in rows}):
        sel = [(v, x) for v, x in rows if v["basis"] == basis]
        out[basis] = {"all": acc(sel),
                      "confident (3)": acc([(v, x) for v, x in sel if x.get("confidence") == 3]),
                      **{f"{sp} {lab}": acc([(v, x) for v, x in sel if v[sp] == lab])
                         for sp in ("genre", "role") for lab in ("adjacent", "no_lineage")}}
    with open(os.path.join(a.out, "score.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["make", "score"])
    p.add_argument("reports", nargs="*")
    p.add_argument("--out", default="blind")
    p.add_argument("--per-report", type=int, default=60)
    p.add_argument("--adjacent", type=int, default=None,
                   help="stratify: this many role-adjacent jobs per report (with --other)")
    p.add_argument("--other", type=int, default=0, help="and this many role no-lineage jobs")
    p.add_argument("--files", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    make(a) if a.stage == "make" else score(a)
