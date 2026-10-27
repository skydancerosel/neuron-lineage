#!/usr/bin/env python
"""
fade_followup.py

Three tests of the claim that a faded job is retained by the layer as a whole, on exactly the
jobs fade_probe.py measured (rebuilt from the same seed and checked against fade_probe.json)
and the coherent non-job sets of fade_control.py:

  step0    The same probe on the untrained network (step 0). If a random network already
           separates a job's contexts from the token-matched ones, the layer reading them
           at the end shows input statistics, not a retained job.
  ablate   At the last checkpoint, remove the job's direction at the job's contexts and
           measure what the model's next-token prediction loses (KL from the clean
           prediction, and the change in loss). The direction is the difference of means
           between job contexts and matched contexts, fitted on the other half of the
           contexts (2-fold); removal sets the activation's component along it to the
           matched contexts' mean. Controls at every position: a kick of the same size in
           a random direction, and the job's own neuron set to its matched mean.
           --scope all removes the direction (or kicks, or resets the neuron) at every
           position of the context, and also measures loss over the next --after tokens.
  sae      Whether a dictionary feature picks out the job: the best of an SAE's latents
           (train_sae.py), chosen on training folds, against the best neuron and the best
           of as many random directions as the SAE has latents, all on the same folds.

  python fade_followup.py step0
  python fade_followup.py ablate
  python fade_followup.py sae
"""
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

import fade_control as fc
import fade_probe as fpm
import neuron_lineage as nl


def extra(p):
    p.add_argument("stage", choices=["step0", "ablate", "sae"])
    p.add_argument("--ablate-pos", type=int, default=16, help="job contexts ablated per job, at most")
    p.add_argument("--rows", type=int, default=64, help="sequences per ablation batch")
    p.add_argument("--scope", choices=["pos", "all"], default="pos",
                   help="ablate at the job position only, or at every position of the context")
    p.add_argument("--after", type=int, default=16, help="scope all: loss measured over this many next tokens")
    p.add_argument("--sae-dir", default="lineage_out/sae")


def rebuild(a, c):
    """fade_probe's jobs, checked against fade_probe.json, and fade_control's sets."""
    jobs = fpm.build_jobs(a, c, np.random.default_rng(a.seed))
    ref = json.load(open(os.path.join(a.out, "fade_probe.json")))["jobs"]
    key = lambda j: (j["layer"], j["basis"], j["kind"], j["neuron"], j["n_pos"], j["n_neg"])
    got = [key({**j, "n_pos": len(j["pos"]), "n_neg": len(j["neg"])}) for j in jobs]
    if got != [key(j) for j in ref]:
        raise SystemExit("rebuilt jobs differ from fade_probe.json")
    print(f"rebuilt {len(jobs)} jobs, identical to fade_probe.json", flush=True)
    return jobs, fc.coherent_sets(a, c)


def summary_row(j):
    return {"layer": j["layer"], "basis": j["basis"], "kind": j["kind"], "neuron": j["neuron"], "n_pos": len(j["pos"])}


def step0(a, c, jobs, sets, dev):
    model = fpm.load_model(a, 0, dev)
    P = fpm.positions(a, jobs + sets)
    H, allp, loss = fpm.acts_and_loss(a, c, model, dev, P)
    del model
    nl.free(dev)
    Q = fpm.rotation(a, H[a.layers[0]].shape[1])
    rng = np.random.default_rng(a.seed + 21)
    out = {"jobs": [], "sets": []}
    for j in jobs + sets:
        X, y, g = fpm.rows(H, P, j)
        m = fpm.measure(a, X, y, g, X @ Q if j["basis"] == "random" else X, j["own"], rng)
        lp, ln = loss[np.searchsorted(allp, j["pos"])], loss[np.searchsorted(allp, j["neg"])]
        m["rel_loss"] = round(float(np.nanmean(lp) - np.nanmean(ln)), 3)
        out["sets" if j["kind"] == "coherent" else "jobs"].append({**summary_row(j), "step0": m})
    return out


def ablate(a, c, jobs, sets, dev):
    items = [j for j in jobs if j["basis"] == "neuron" and j["kind"] in ("faded", "stayed")] + sets
    s_end = c.S[-1]
    model = fpm.load_model(a, s_end, dev)
    P = fpm.positions(a, items)
    H, _, _ = fpm.acts_and_loss(a, c, model, dev, P)
    rng = np.random.default_rng(a.seed + 31)
    d = H[a.layers[0]].shape[1]

    # per item and fold: the direction, the matched mean along it, and the test rows
    bank_u, bank_c0, bank_own, rows = [], [], [], []
    for ii, j in enumerate(items):
        X, y, g = fpm.rows(H, P, j)
        npos = len(j["pos"])
        fold = rng.permutation(npos) % 2
        keep = set(rng.permutation(npos)[:a.ablate_pos].tolist())
        for k in (0, 1):
            tr = fold[g] != k
            mp, mn = X[tr & (y == 1)].mean(0), X[tr & (y == 0)].mean(0)
            u = mp - mn
            u /= np.linalg.norm(u) + 1e-9
            bi = len(bank_u)
            bank_u.append(u)
            bank_c0.append(float(mn @ u))
            bank_own.append(float(mn[j["own"]]) if j["own"] is not None else 0.0)
            for gi in np.flatnonzero(fold == k):
                if gi not in keep:
                    continue
                p = int(j["pos"][gi])
                rows.append((ii, bi, p, 1))
                negs = j["neg"][j["grp"] == gi]
                if len(negs):
                    rows.append((ii, bi, int(negs[0]), 0))
    U = torch.as_tensor(np.array(bank_u), device=dev)
    C0 = torch.as_tensor(np.array(bank_c0, dtype=np.float32), device=dev)
    OWNV = torch.as_tensor(np.array(bank_own, dtype=np.float32), device=dev)
    own_idx = np.array([items[r[0]]["own"] if items[r[0]]["own"] is not None else -1 for r in rows])
    has_next = np.array([(r[2] % a.ctx) + 1 < c.lengths[r[2] // a.ctx] for r in rows])
    print(f"ablation: {len(items)} items, {len(rows)} positions, {d}-dim directions", flush=True)

    modes = ("clean", "real", "kick", "own")
    res = np.full((len(rows), len(modes), 2), np.nan, dtype=np.float32)  # (KL, change in loss) per mode
    shift = np.full(len(rows), np.nan, dtype=np.float32)
    res_span = np.full((len(rows), len(modes)), np.nan, dtype=np.float32)  # scope all: loss change over the span
    tt = torch.from_numpy(c.toks).long()
    gen = torch.Generator(device="cpu").manual_seed(a.seed + 41)
    per = a.rows // len(modes)
    for L in a.layers:
        idx = [i for i, r in enumerate(rows) if items[r[0]]["layer"] == L]
        idx.sort(key=lambda i: rows[i][2] % a.ctx)
        for b0 in range(0, len(idx), per):
            sel = idx[b0:b0 + per]
            n = len(sel)
            pos = np.array([rows[i][2] for i in sel])
            t_last = pos % a.ctx
            span = a.after if a.scope == "all" else 0
            T = min(a.ctx, -(-(int(t_last.max()) + 1 + span) // 64) * 64)  # a few fixed lengths: the MPS cache keeps every size it sees
            x = tt[pos // a.ctx, :T].repeat_interleave(len(modes), 0).to(dev)
            bi = torch.as_tensor([rows[i][1] for i in sel], device=dev).repeat_interleave(len(modes))
            mode = torch.arange(len(modes), device=dev).repeat(n)
            ti = torch.as_tensor(t_last, device=dev).repeat_interleave(len(modes))
            own = torch.as_tensor(own_idx[sel], device=dev).repeat_interleave(len(modes))
            r = torch.randn(n * len(modes), d, generator=gen).to(dev)
            r /= r.norm(dim=1, keepdim=True)
            ar = torch.arange(n * len(modes), device=dev)
            got = {}

            def hook(m, inp):
                h = inp[0].clone()
                if a.scope == "all":  # every position of the context
                    s = (h * U[bi][:, None]).sum(-1) - C0[bi][:, None]
                    delta = torch.zeros_like(h)
                    delta = torch.where((mode == 1)[:, None, None], s[..., None] * U[bi][:, None], delta)
                    delta = torch.where((mode == 2)[:, None, None], s.abs()[..., None] * r[:, None], delta)
                    o = own.clamp(min=0)
                    own_delta = torch.zeros_like(h)
                    own_delta[ar, :, o] = h[ar, :, o] - OWNV[bi][:, None]
                    delta = torch.where(((mode == 3) & (own >= 0))[:, None, None], own_delta, delta)
                    got["s"] = s[ar, ti]
                    return (h - delta,)
                hv = h[ar, ti]
                s = (hv * U[bi]).sum(1) - C0[bi]
                delta = torch.zeros_like(hv)
                delta = torch.where((mode == 1)[:, None], s[:, None] * U[bi], delta)
                delta = torch.where((mode == 2)[:, None], s.abs()[:, None] * r, delta)
                o = own.clamp(min=0)
                own_delta = torch.zeros_like(hv)
                own_delta[ar, o] = hv[ar, o] - OWNV[bi]
                delta = torch.where(((mode == 3) & (own >= 0))[:, None], own_delta, delta)
                h[ar, ti] = hv - delta
                got["s"] = s
                return (h,)

            hk = model.gpt_neox.layers[L].mlp.dense_4h_to_h.register_forward_pre_hook(hook)
            with torch.no_grad():
                full = model.gpt_neox(x).last_hidden_state
                hid = full[ar, ti]
                logp = torch.log_softmax(model.embed_out(hid).float(), -1).view(n, len(modes), -1)
                if span:  # loss over the job token and the next span-1 tokens, inside the document
                    tt_idx = (ti[:, None] + torch.arange(span, device=dev)[None]).clamp(max=T - 2)
                    lp_span = torch.log_softmax(model.embed_out(full[ar[:, None], tt_idx]).float(), -1)
                    nx = x[ar[:, None], tt_idx + 1]
                    ls = -lp_span.gather(-1, nx[..., None])[..., 0].view(n, len(modes), span)
                    lim = torch.as_tensor(c.lengths[pos // a.ctx] - 1 - t_last, device=dev).clamp(min=0, max=span)
                    valid = (torch.arange(span, device=dev)[None] < lim[:, None]).float()
                    span_d = ((ls - ls[:, :1]) * valid[:, None]).sum(-1) / valid.sum(-1, keepdim=True).clamp(min=1)
                    span_d[lim == 0] = float("nan")
            hk.remove()
            kl = (logp[:, :1].exp() * (logp[:, :1] - logp)).sum(-1)  # KL(clean || mode)
            nxt = torch.as_tensor(c.ids[np.minimum(pos + 1, c.ids.size - 1)].astype(np.int64), device=dev)
            lo = -logp.gather(2, nxt[:, None, None].expand(n, len(modes), 1))[..., 0]
            res[sel, :, 0] = kl.cpu().numpy()
            res[sel, :, 1] = (lo - lo[:, :1]).cpu().numpy()
            if span:
                res_span[sel] = span_d.cpu().numpy()
            shift[sel] = got["s"].view(n, len(modes))[:, 1].abs().cpu().numpy()
            if (b0 // per) % 25 == 0:
                nl.free(dev)
        print(f"L{L}: {len(idx)} positions ablated", flush=True)
    del model
    nl.free(dev)
    res[~has_next, :, 1] = np.nan

    out = []
    for ii, j in enumerate(items):
        mine = np.array([i for i, r in enumerate(rows) if r[0] == ii])
        if not len(mine):
            continue
        posm = np.array([rows[i][3] == 1 for i in mine])
        rec = summary_row(j)
        used = [m for m in modes[1:] if m != "own" or j["own"] is not None]
        for lab, msk in (("pos", posm), ("neg", ~posm)):
            if not msk.any():
                continue
            rr = res[mine[msk]]
            rec[lab] = {"n": int(msk.sum()), "shift": round(float(np.mean(shift[mine[msk]])), 3),
                        **{f"kl_{m}": round(float(np.nanmean(rr[:, modes.index(m), 0])), 5) for m in used},
                        **{f"dloss_{m}": round(float(np.nanmean(rr[:, modes.index(m), 1])), 5) for m in used},
                        **({f"dspan_{m}": round(float(np.nanmean(res_span[mine[msk], modes.index(m)])), 5) for m in used}
                           if a.scope == "all" else {})}
        out.append(rec)
    return {"step": s_end, "items": out}


def sae(a, c, jobs, sets, dev):
    import train_sae

    s_end = c.S[-1]
    model = fpm.load_model(a, s_end, dev)
    P = fpm.positions(a, jobs + sets)
    H, _, _ = fpm.acts_and_loss(a, c, model, dev, P)
    del model
    nl.free(dev)
    d = H[a.layers[0]].shape[1]
    Q = torch.as_tensor(fpm.rotation(a, d), device=dev)
    dicts = {}
    for L in a.layers:
        sd = torch.load(os.path.join(a.sae_dir, f"L{L}.pt"))
        m = sd["W_enc"].shape[1]
        s = train_sae.TopK(d, m, sd["k"]).to(dev)
        s.load_state_dict({k: sd[k].float() for k in ("W_enc", "W_dec", "b_enc", "b_dec")})
        dicts[L] = (s, sd["scale"])
    m = dicts[a.layers[0]][0].W_enc.shape[1]
    R = torch.randn(d, m, generator=torch.Generator().manual_seed(a.seed + 51))
    R = (R / R.norm(dim=0, keepdim=True)).to(dev)
    rng = np.random.default_rng(a.seed + 61)
    out = []
    for n_done, j in enumerate(jobs + sets):
        X, y, g = fpm.rows(H, P, j)
        N = next(p for p in PADS if p >= len(y))
        s, scale = dicts[j["layer"]]
        with torch.no_grad():
            xt = torch.zeros(N, d, device=dev)
            xt[:len(y)] = torch.as_tensor(X, device=dev)
            Us = {"unit": xt @ Q if j["basis"] == "random" else xt, "sae": s.encode(xt * scale), "random_m": xt @ R}
            res = held_out_best(a, X, y, g, Us, rng, dev)
            pos, neg = masks(np.flatnonzero(y == 1), N, dev), masks(np.flatnonzero(y == 0), N, dev)
            Z = Us["sae"]
            top = int(torch.argmax(auc_masked(Z, pos, neg)))
            res["sae_fires_pos"] = round(float(((Z[:, top] > 0) & pos).sum() / pos.sum()), 3)
            res["sae_fires_neg"] = round(float(((Z[:, top] > 0) & neg).sum() / neg.sum()), 3)
        out.append({**summary_row(j), **res})
        if n_done % 50 == 0:
            nl.free(dev)
            print(f"{n_done}/{len(jobs) + len(sets)} items", flush=True)
    return {"step": s_end, "latents": m, "items": out}


PADS = (256, 512, 1280)  # rows per item, padded: the MPS backend keeps memory for every new shape


def masks(idx, N, dev):
    msk = torch.zeros(N, dtype=torch.bool, device=dev)
    msk[torch.as_tensor(idx, device=dev)] = True
    return msk


def auc_masked(U, pos, neg):
    """AUC of every column of U (N, m) for rows marked pos against rows marked neg, ties
    counting half, in fixed shapes: rank sum over one sort per column, with the other rows
    pushed to +inf (last) and tied values given their average rank. (torch.searchsorted is
    slow on MPS.)"""
    inf = torch.tensor(float("inf"), device=U.device)
    v = torch.where((pos | neg)[:, None], U, inf).T.contiguous()
    sv, perm = torch.sort(v, dim=1)
    N = v.shape[1]
    ar = torch.arange(N, device=U.device).expand_as(sv)
    new = torch.ones_like(sv, dtype=torch.bool)
    new[:, 1:] = sv[:, 1:] != sv[:, :-1]
    last = torch.ones_like(new)
    last[:, :-1] = new[:, 1:]
    start = torch.cummax(torch.where(new, ar, 0), dim=1).values
    end = torch.flip(torch.cummin(torch.flip(torch.where(last, ar, N - 1), [1]), dim=1).values, [1])
    rank = (start + end).float() / 2 + 1
    n1, n0 = pos.sum(), neg.sum()
    return ((rank * pos[perm]).sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)


def held_out_best(a, X, y, groups, Us, rng, dev):
    """Probe AUC and, for each candidate set in Us (padded device tensors), the held-out AUC
    of its best member chosen on the training folds, all on the same folds."""
    from sklearn.linear_model import LogisticRegression

    npos = int(y.sum())
    f = (rng.permutation(npos) % min(a.folds, npos))[groups]
    N = next(iter(Us.values())).shape[0]
    acc = {k: [] for k in ["probe", *Us]}
    w = []
    for k in np.unique(f):
        tr, te = f != k, f == k
        if y[te].min() == y[te].max() or y[tr].min() == y[tr].max():
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        clf = LogisticRegression(C=a.C, max_iter=2000).fit((X[tr] - mu) / sd, y[tr])
        acc["probe"].append(fpm.auc(clf.decision_function((X[te] - mu) / sd)[:, None], y[te])[0])
        ptr, ntr = masks(np.flatnonzero(tr & (y == 1)), N, dev), masks(np.flatnonzero(tr & (y == 0)), N, dev)
        pte, nte = masks(np.flatnonzero(te & (y == 1)), N, dev), masks(np.flatnonzero(te & (y == 0)), N, dev)
        for name, U in Us.items():
            best = int(torch.argmax(auc_masked(U, ptr, ntr)))
            acc[name].append(float(auc_masked(U[:, best:best + 1], pte, nte)[0]))
        w.append(te.sum())
    if not w:
        return {k: None for k in acc}
    return {k: round(float(np.average(v, weights=w)), 3) for k, v in acc.items()}


def main():
    a = fpm.parse(extra)
    c = fpm.setup(a)
    jobs, sets = rebuild(a, c)
    dev = nl.device(a)
    res = {"step0": step0, "ablate": ablate, "sae": sae}[a.stage](a, c, jobs, sets, dev)
    path = os.path.join(a.out, f"fade_{a.stage}{'_all' if a.stage == 'ablate' and a.scope == 'all' else ''}.json")
    with open(path, "w") as fh:
        json.dump({"args": vars(a), **res}, fh, indent=1)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
