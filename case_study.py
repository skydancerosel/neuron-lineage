#!/usr/bin/env python
"""
case_study.py

One neuron's jobs traced checkpoint by checkpoint. The jobs (sets of corpus
positions) come from a neuron_lineage report: every new job of the neuron and
every parent. At each checkpoint the whole corpus is rerun to get every unit's
top-k cutoff (at most --per-chunk hits per chunk, as collect does), then for each
job: the neuron's mean activation over the job's contexts as a share of its
cutoff, and the other unit firing most on them (the taker) with its share.

  python case_study.py --report lineage_out_pilot_v2/L12/report_pilot.json \
      --tokens lineage_out_pilot_v2/tokens.npy --neuron 1038 \
      --steps 4000 8000 9000 10000 11000 12000 13000 14000 15000 16000
"""
import argparse
import json
import os

import numpy as np
import torch

from neuron_lineage import _Stop, device, free


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", required=True)
    p.add_argument("--tokens", required=True, help="the corpus the report's positions index")
    p.add_argument("--neuron", type=int, required=True)
    p.add_argument("--steps", type=int, nargs="+", required=True)
    p.add_argument("--model", default="EleutherAI/pythia-410m-deduped")
    p.add_argument("--layer", type=int, default=12)
    p.add_argument("--k", type=int, default=200)
    p.add_argument("--per-chunk", type=int, default=2)
    p.add_argument("--window", type=int, default=24)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="case_study")
    a = p.parse_args()
    from transformers import GPTNeoXForCausalLM

    r = json.load(open(a.report))
    jobs = {}
    for e in r["events"]:
        if e["neuron"] != a.neuron:
            continue
        jobs.setdefault(tuple(e["new_pos"]), f"new at {e['step']}: {e['examples'][0][-50:]}")
        jobs.setdefault(tuple(e["parent_pos"]), f"parent from {e['parent_step']}: {e['parent_examples'][0][-50:]}")
    names = [n.replace("\n", " ⏎ ").replace("|", "/") for n in jobs.values()]
    sets = [np.array(k) for k in jobs]
    pos = np.unique(np.concatenate(sets))
    toks = torch.from_numpy(np.load(a.tokens)).long()
    ctx = toks.shape[1]
    dev = device(a)
    share, taker = np.zeros((len(sets), len(a.steps))), []
    for si, s in enumerate(a.steps):
        model = GPTNeoXForCausalLM.from_pretrained(a.model, revision=f"step{s}",
                                                   dtype=torch.float32).to(dev).eval()
        mlp = model.gpt_neox.layers[a.layer].mlp
        d = mlp.dense_4h_to_h.in_features
        buf = {}

        def hook(m, inp):
            buf["a"] = inp[0]
            raise _Stop

        h = mlp.dense_4h_to_h.register_forward_pre_hook(hook)
        top = torch.full((d, a.k), -float("inf"), device=dev)
        at = np.zeros((len(pos), d), dtype=np.float32)
        with torch.no_grad():
            for b0 in range(0, len(toks), a.batch):
                try:
                    model.gpt_neox(toks[b0:b0 + a.batch].to(dev))
                except _Stop:
                    pass
                act = buf["a"].float()  # (B, ctx, d)
                sel = np.flatnonzero((pos // ctx >= b0) & (pos // ctx < b0 + len(act)))
                if len(sel):
                    at[sel] = act[torch.as_tensor(pos[sel] // ctx - b0, device=dev),
                                  torch.as_tensor(pos[sel] % ctx, device=dev)].cpu().numpy()
                act[:, :a.window] = -float("inf")
                cv = act.topk(a.per_chunk, dim=1).values.permute(2, 0, 1).reshape(d, -1)
                top = torch.cat([top, cv], 1).topk(a.k, dim=1).values
        h.remove()
        cut = top[:, -1].cpu().numpy()
        del model
        free(dev)
        row = []
        for j, ps in enumerate(sets):
            m = at[np.searchsorted(pos, ps)].mean(0) / cut
            share[j, si] = m[a.neuron]
            m[a.neuron] = -np.inf
            row.append((int(m.argmax()), round(float(m.max()), 3)))
        taker.append(row)
        print(f"step {s}: " + ", ".join(f"{x:.2f}" for x in share[:, si]))

    os.makedirs(a.out, exist_ok=True)
    stem = os.path.join(a.out, f"neuron{a.neuron}")
    with open(stem + ".json", "w") as fh:
        json.dump({"neuron": a.neuron, "steps": a.steps, "jobs": names, "share": share.round(3).tolist(),
                   "taker": [[taker[si][j] for si in range(len(a.steps))] for j in range(len(sets))]},
                  fh, indent=1, ensure_ascii=False)
    lines = [f"# Neuron {a.neuron}: share of its top-{a.k} cutoff on each job, by checkpoint", "",
             "| job | " + " | ".join(map(str, a.steps)) + " |", "|---|" + "---|" * len(a.steps)]
    lines += [f"| {n} | " + " | ".join(f"{x:.2f}" for x in share[j]) + " |"
              for j, n in enumerate(names)]
    lines += ["", "Other neuron firing most on each job (neuron: share of its own cutoff):", "",
              "| job | " + " | ".join(map(str, a.steps)) + " |", "|---|" + "---|" * len(a.steps)]
    lines += [f"| {n} | " + " | ".join(f"{t[0]}: {t[1]:.2f}" for t in
                                                         (taker[si][j] for si in range(len(a.steps)))) + " |"
              for j, n in enumerate(names)]
    with open(stem + ".md", "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {stem}.md")


if __name__ == "__main__":
    main()
