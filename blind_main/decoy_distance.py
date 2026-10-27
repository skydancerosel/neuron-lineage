"""For the main-run blind items: distance from the new job to its real parent and to its decoy,
in each space, and reader accuracy by which one was closer. Decoys are rebuilt exactly as
analyze picked them and checked against the stored ones."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the repo root; run from there
import neuron_lineage as nl  # noqa: E402

sys.argv = ["x", "analyze", "--layers", "12", "--early-steps", "128", "256", "512",
            "--steps", "1000", "2000", "4000", "8000", "16000", "32000", "64000", "143000",
            "--n-neurons", "512", "--neuron-seed", "2", "--sem-weight", "0.5",
            "--cluster-thresh", "0.65", "--new-frac", "0.75", "--tag", "main"]
a = nl.parse()
tok, toks, lengths, sources = nl.load_corpus(a)
key = json.load(open("blind_main/key.json"))
ans = {}
for i in range(1, 7):
    for x in json.load(open(f"blind_main/answers_{i}.json")):
        ans[x["id"]] = x
out = []
for L in (6, 12, 18):
    for basis in ("neuron", "random"):
        d_out = nl.out_dir(a, L, basis)
        path = f"{d_out}/report_main.json"
        items = {i: v for i, v in key.items() if v["report"] == path}
        r = json.load(open(path))
        S, neurons, data = nl.load_units(a, d_out, toks, lengths)
        cs = nl.context_space(a, tok, toks, lengths, sources, d_out, S, data)
        D, slices = nl.spaces(a, cs)
        jobs = [[nl.unit_jobs(a, D, x, cs["src"]) for x in ru] for ru in nl.unit_rows(S, data, cs["pos"])]
        col = {int(n): i for i, n in enumerate(neurons)}
        rd = np.random.default_rng(a.seed + 7)
        ok = 0
        dated = [e for e in r["events"] if e["kind"] == "dated"]
        dist = {}
        for e in dated:
            rows = np.searchsorted(cs["pos"], e["new_pos"])
            v, drows = nl.pick_decoy(D, jobs, len(neurons), col[e["neuron"]], S.index(e["parent_step"]), rows, rd)
            if v is None:
                continue
            text = nl.window_texts(a, tok, toks, cs["pos"][drows[:1]])[0][0]
            ok += text == e["decoy_examples"][0]
            prow = np.searchsorted(cs["pos"], e["parent_pos"])
            dd = {}
            for k, sl in {**slices, "joined": slice(None)}.items():
                c = nl.unit_norm(D[rows, sl].mean(0))
                dd[k] = (float(1 - nl.unit_norm(D[prow, sl].mean(0)) @ c), float(1 - nl.unit_norm(D[drows, sl].mean(0)) @ c))
            dist[(e["neuron"], e["step"], e["birth"], e["examples"][0])] = dd
        print(f"L{L}/{basis}: rebuilt decoys match stored ones for {ok} of {len(dated)} dated jobs", flush=True)
        for iid, v in items.items():
            m = [dd for (n, s, b, ex), dd in dist.items() if n == v["neuron"] and s == v["step"] and b == v["birth"]]
            if len(m) == 1 and iid in ans:
                out.append({**v, "correct": ans[iid]["choice"] == v["answer"], "dist": m[0]})
json.dump(out, open("blind_main/decoy_distance.json", "w"))
print(f"matched {len(out)} of {len(key)} blind items")
for strat in ("adjacent", "no_lineage"):
    sel = [x for x in out if x["role"] == strat]
    for k in ("role", "genre", "joined"):
        closer = [x for x in sel if x["dist"][k][1] < x["dist"][k][0]]
        far = [x for x in sel if x["dist"][k][1] >= x["dist"][k][0]]
        acc = lambda xs: f"{sum(x['correct'] for x in xs)}/{len(xs)}"
        print(f"role {strat:10s} {k:6s}: decoy closer than parent in {len(closer)}/{len(sel)}; "
              f"readers right {acc(closer)} when decoy closer, {acc(far)} when parent closer; "
              f"median parent {np.median([x['dist'][k][0] for x in sel]):.3f} decoy {np.median([x['dist'][k][1] for x in sel]):.3f}")
