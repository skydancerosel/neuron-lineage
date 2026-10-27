#!/usr/bin/env python
"""
fade_probe.py

Where do faded jobs go? Parents of dated new jobs in neuron_lineage reports are split
into those that faded (below --collapse-share of their unit's top-k cutoff at the last
checkpoint) and those that stayed. Each is measured at its own step, at the first
checkpoint after it where it is below that share (faded only), and at the last one:

  probe     a regularized linear probe on the whole layer's activations, separating
            the job's contexts from other occurrences of the same tokens (so token
            identity alone can't do it); cross-validated AUC
  best      the single unit (a neuron, or a random direction for that basis) that best
            separates them, chosen on the training folds, AUC on the held-out folds
  own       the parent's own unit, AUC with no selection
  loss      next-token loss on the job's contexts minus on the token-matched contexts:
            the model's skill on exactly that job, relative to the same tokens elsewhere

Random context sets, with the same token matching, give the AUC a job needs to count
as readable. A faded job that the layer still reads while no unit carries it moved into
a distributed code; one the layer no longer reads, with relative loss up, was dropped;
no longer read with loss flat, it moved elsewhere or stopped mattering.

  python fade_probe.py --layers 6 12 18      # the main run's settings are the defaults
"""
import argparse
import json
import os
import types

import numpy as np
import torch
from scipy.stats import rankdata

import neuron_lineage as nl


def parse(extra=None):
    p = argparse.ArgumentParser()
    if extra:
        extra(p)
    p.add_argument("--model", default="EleutherAI/pythia-410m-deduped")
    p.add_argument("--layers", type=int, nargs="+", default=[6, 12, 18])
    p.add_argument("--bases", nargs="+", choices=["neuron", "random"], default=["neuron", "random"])
    p.add_argument("--early-steps", type=int, nargs="+", default=[128, 256, 512])
    p.add_argument("--steps", type=int, nargs="+",
                   default=[1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000])
    p.add_argument("--dataset", default="NeelNanda/pile-10k")
    p.add_argument("--split", default="train")
    p.add_argument("--n-tokens", type=int, default=2_000_000)
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--window", type=int, default=24)
    p.add_argument("--k", type=int, default=200)
    p.add_argument("--collapse-share", type=float, default=0.25)
    p.add_argument("--n-faded", type=int, default=150, help="faded parents per layer and basis")
    p.add_argument("--n-stayed", type=int, default=75, help="parents that stayed, as the control")
    p.add_argument("--n-random", type=int, default=40, help="random context sets, the readable bar")
    p.add_argument("--random-size", type=int, default=20)
    p.add_argument("--neg-per-pos", type=int, default=5, help="token-matched contexts per job context")
    p.add_argument("--C", type=float, default=0.05, help="inverse L2 strength of the probe")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--device", default=None)
    p.add_argument("--dtype", default=None)
    p.add_argument("--tag", default="main")
    p.add_argument("--out", default="lineage_out")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def auc(scores, y):
    """AUC of each column of scores (n, m) for labels y; ties count half."""
    r = rankdata(scores, axis=0)
    n1 = y.sum()
    return (r[y == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def measure(a, X, y, groups, U, own, rng):
    """Cross-validated probe AUC and best-unit AUC, and the own unit's AUC, for layer
    activations X (n, d) and labels y. Each job context and its own token-matched
    contexts share a group and a fold: split apart, the probe would learn "this token
    means negative" from the matches of a held-out context and score it backwards.
    U holds the candidate units' activations on the same rows: X itself for neurons, X
    rotated for random directions, dictionary codes for an SAE."""
    from sklearn.linear_model import LogisticRegression

    npos = int(y.sum())
    f = (rng.permutation(npos) % min(a.folds, npos))[groups]
    probe, best, w = [], [], []
    for k in np.unique(f):  # AUC within each held-out fold: pooled scores mix offsets
        tr, te = f != k, f == k
        if y[te].min() == y[te].max() or y[tr].min() == y[tr].max():
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        clf = LogisticRegression(C=a.C, max_iter=2000).fit((X[tr] - mu) / sd, y[tr])
        probe.append(auc(clf.decision_function((X[te] - mu) / sd)[:, None], y[te])[0])
        best.append(auc(U[te, int(np.argmax(auc(U[tr], y[tr])))][:, None], y[te])[0])
        w.append(te.sum())
    if not w:  # no fold with both labels on each side
        return {"probe": None, "best": None, "own": None}
    return {"probe": round(float(np.average(probe, weights=w)), 3),
            "best": round(float(np.average(best, weights=w)), 3),
            "own": None if own is None else round(float(auc(U[:, own:own + 1], y)[0]), 3)}


def setup(a):
    """Corpus, checkpoints and a same-token index of valid positions."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    toks, lengths, _ = nl.corpus(a, tok)
    flat = np.flatnonzero(nl.valid(a, lengths).ravel())
    ids = toks.ravel()
    order = np.argsort(ids[flat], kind="stable")
    return types.SimpleNamespace(tok=tok, toks=toks, lengths=lengths, S=nl.all_steps(a), flat=flat, ids=ids,
                                 by_tok=flat[order], starts=np.searchsorted(ids[flat][order], np.arange(ids.max() + 2)))


def matched(a, c, rng, pos):
    """Token-matched negatives: other valid positions with the same firing token, each
    tagged with the job context it was drawn for (its group)."""
    neg, grp, seen = [], [], set(pos.tolist())
    for gi, p in enumerate(pos):
        pool = c.by_tok[c.starts[c.ids[p]]:c.starts[c.ids[p] + 1]]
        draw = rng.choice(pool, min(len(pool), 4 * a.neg_per_pos), replace=False).tolist()
        pick = [q for q in draw if q not in seen][:a.neg_per_pos]
        seen.update(pick)
        neg += pick
        grp += [gi] * len(pick)
    return np.array(neg, dtype=np.int64), np.array(grp, dtype=int)


def build_jobs(a, c, rng):
    """The sampled parents, faded and stayed, and random context sets as the readable bar.
    Deterministic in rng, so later stages rebuild exactly these jobs."""
    S, jobs = c.S, []
    for L in a.layers:
        for basis in a.bases:
            path = os.path.join(nl.out_dir(a, L, basis), nl.tagged(a, "report"))
            r = json.load(open(path))
            dated = [e for e in r["events"] if e["kind"] == "dated"]
            faded = [e for e in dated if e["parent_history"][-1] < a.collapse_share]
            stayed = [e for e in dated if e["parent_history"][-1] >= a.collapse_share]
            pick = [("faded", faded[i]) for i in rng.permutation(len(faded))[:a.n_faded]]
            pick += [("stayed", stayed[i]) for i in rng.permutation(len(stayed))[:a.n_stayed]]
            fades = []
            for kind, e in pick:
                ps = S.index(e["parent_step"])
                fade = None
                if kind == "faded":
                    fade = next(t for t in range(ps + 1, len(S)) if e["parent_history"][t] < a.collapse_share)
                    fades.append(ps)
                pos = np.array(sorted(e["parent_pos"]), dtype=np.int64)
                neg, grp = matched(a, c, rng, pos)
                if len(neg) < 10:
                    continue
                unit = e["neuron"]
                jobs.append({"layer": L, "basis": basis, "kind": kind, "neuron": unit, "pos": pos, "neg": neg, "grp": grp,
                             "times": sorted({ps, len(S) - 1} | ({fade} if fade is not None else set())),
                             "before": ps, "fade": fade, "end": len(S) - 1, "own": unit})
            mid = int(np.median(fades)) if fades else 0
            for _ in range(a.n_random):
                pos = np.sort(rng.choice(c.flat, a.random_size, replace=False))
                neg, grp = matched(a, c, rng, pos)
                jobs.append({"layer": L, "basis": basis, "kind": "random", "neuron": None, "pos": pos, "neg": neg, "grp": grp,
                             "times": sorted({mid, len(S) - 1}), "before": mid, "fade": None,
                             "end": len(S) - 1, "own": None})
    return jobs


def rotation(a, d):
    """The fixed rotation collect used for the random-direction basis."""
    g = torch.Generator().manual_seed(a.seed)
    return torch.linalg.qr(torch.randn(d, d, generator=g))[0].numpy().astype(np.float32)


def load_model(a, s, dev):
    from transformers import GPTNeoXForCausalLM

    dt = getattr(torch, a.dtype or ("float16" if dev == "cuda" else "float32"))
    return GPTNeoXForCausalLM.from_pretrained(a.model, revision=f"step{s}", dtype=dt).to(dev).eval()


def acts_and_loss(a, c, model, dev, P):
    """Post-GELU activations (fp16) at the positions P[L] of each layer, and next-token loss
    at every position in any P[L] (NaN where the document has no next token)."""
    allp = np.unique(np.concatenate(list(P.values())))
    has_next = (allp % a.ctx) + 1 < c.lengths[allp // a.ctx]
    loss = np.full(len(allp), np.nan, dtype=np.float32)
    d = model.gpt_neox.layers[a.layers[0]].mlp.dense_4h_to_h.in_features
    H = {L: np.zeros((len(P[L]), d), dtype=np.float16) for L in P}
    buf = {}
    hooks = [model.gpt_neox.layers[L].mlp.dense_4h_to_h.register_forward_pre_hook(
        lambda m, inp, L=L: buf.__setitem__(L, inp[0])) for L in P]
    tt = torch.from_numpy(c.toks).long()
    chunks = np.unique(allp // a.ctx)
    with torch.no_grad():
        for b0 in range(0, len(chunks), a.batch):
            rb = chunks[b0:b0 + a.batch]
            hid = model.gpt_neox(tt[rb].to(dev)).last_hidden_state
            for L in P:
                sel = np.flatnonzero(np.isin(P[L] // a.ctx, rb))
                bi = torch.as_tensor(np.searchsorted(rb, P[L][sel] // a.ctx), device=dev)
                ti = torch.as_tensor(P[L][sel] % a.ctx, device=dev)
                H[L][sel] = buf[L][bi, ti].half().cpu().numpy()
            sel = np.flatnonzero(np.isin(allp // a.ctx, rb) & has_next)
            bi = torch.as_tensor(np.searchsorted(rb, allp[sel] // a.ctx), device=dev)
            ti = torch.as_tensor(allp[sel] % a.ctx, device=dev)
            logp = torch.log_softmax(model.embed_out(hid[bi, ti]).float(), -1)
            nxt = torch.as_tensor(c.ids[allp[sel] + 1].astype(np.int64), device=dev)
            loss[sel] = (-logp.gather(1, nxt[:, None])[:, 0]).cpu().numpy()
    for h in hooks:
        h.remove()
    return H, allp, loss


def rows(H, P, j):
    """A job's activation matrix (contexts, then their matches), labels and CV groups."""
    L = j["layer"]
    X = np.r_[H[L][np.searchsorted(P[L], j["pos"])], H[L][np.searchsorted(P[L], j["neg"])]].astype(np.float32)
    y = np.r_[np.ones(len(j["pos"])), np.zeros(len(j["neg"]))].astype(int)
    return X, y, np.r_[np.arange(len(j["pos"])), j["grp"]]


def positions(a, jobs):
    return {L: np.unique(np.concatenate([np.r_[j["pos"], j["neg"]] for j in jobs if j["layer"] == L]
                                        or [np.zeros(0, dtype=np.int64)])) for L in a.layers}


def main():
    a = parse()
    c = setup(a)
    S = c.S
    rng = np.random.default_rng(a.seed)
    jobs = build_jobs(a, c, rng)
    print(f"{len(jobs)} jobs: " + ", ".join(f"{k} {sum(j['kind'] == k for j in jobs)}"
                                           for k in ("faded", "stayed", "random")), flush=True)

    dev = nl.device(a)
    Q = None
    results = [{"res": {}} for _ in jobs]
    for si, s in enumerate(S):
        here = [i for i, j in enumerate(jobs) if si in j["times"]]
        if not here:
            continue
        P = positions(a, [jobs[i] for i in here])
        model = load_model(a, s, dev)
        H, allp, loss = acts_and_loss(a, c, model, dev, P)
        if "random" in a.bases and Q is None:
            Q = rotation(a, H[a.layers[0]].shape[1])
        del model
        nl.free(dev)
        for i in here:
            j = jobs[i]
            X, y, groups = rows(H, P, j)
            m = measure(a, X, y, groups, X @ Q if j["basis"] == "random" else X, j["own"], rng)
            lp, ln = loss[np.searchsorted(allp, j["pos"])], loss[np.searchsorted(allp, j["neg"])]
            m["rel_loss"] = round(float(np.nanmean(lp) - np.nanmean(ln)), 3)
            results[i]["res"][str(s)] = m
        print(f"step {s}: {len(here)} jobs measured", flush=True)

    out = []
    for j, rr in zip(jobs, results):
        out.append({"layer": j["layer"], "basis": j["basis"], "kind": j["kind"], "neuron": j["neuron"],
                    "n_pos": len(j["pos"]), "n_neg": len(j["neg"]), "before": S[j["before"]],
                    "fade": None if j["fade"] is None else S[j["fade"]], "end": S[j["end"]], "at": rr["res"]})
    path = os.path.join(a.out, "fade_probe.json")
    with open(path, "w") as fh:
        json.dump({"args": vars(a), "steps": S, "jobs": out}, fh, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
