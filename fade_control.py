#!/usr/bin/env python
"""
fade_control.py

Control for fade_probe.py: is any coherent set of contexts readable from a layer, job or
not? Clusters of random corpus contexts (formed the way jobs are, in the joined
genre-and-role space, at least 10 contexts) are probed at the last checkpoint exactly as
faded jobs are: whole-layer probe and best single neuron against token-matched contexts.

  python fade_control.py --layers 6 12 18
"""
import json
import os
import types

import numpy as np

import fade_probe as fpm
import neuron_lineage as nl


def coherent_sets(a, c):
    """Up to n_random coherent non-job sets per layer, each with token-matched negatives.
    Deterministic, so later stages rebuild exactly these sets."""
    rng = np.random.default_rng(a.seed + 11)
    rand = np.random.default_rng(a.seed + 1).choice(c.flat, size=20000, replace=False)  # context_space's sample
    sets = []
    for L in a.layers:
        with np.load(os.path.join(nl.out_dir(a, L, "neuron"), "contexts.npz")) as f:
            cs = {"pos": f["pos"], "sem": f["sem"], "lex": f["lex"], "role": f["role"]}
        D, _ = nl.spaces(types.SimpleNamespace(sem_weight=0.5, role_weight=0.5), cs)
        sub = np.sort(rng.choice(rand, 4000, replace=False))
        rows = np.searchsorted(cs["pos"], sub)
        _, members = nl.clusters(types.SimpleNamespace(cluster_thresh=0.65, min_cluster=10), D[rows])
        for m in [members[i] for i in rng.permutation(len(members))[:a.n_random]]:
            pos = np.sort(sub[m][:30])
            neg, grp, seen = [], [], set(pos.tolist())
            for gi, p in enumerate(pos):
                pool = c.by_tok[c.starts[c.ids[p]]:c.starts[c.ids[p] + 1]]
                pick = [q for q in rng.choice(pool, min(len(pool), 4 * a.neg_per_pos), replace=False).tolist()
                        if q not in seen][:a.neg_per_pos]
                seen.update(pick)
                neg += pick
                grp += [gi] * len(pick)
            if len(neg) >= 10:
                sets.append({"layer": L, "basis": "neuron", "kind": "coherent", "neuron": None, "own": None,
                             "pos": pos, "neg": np.array(neg, dtype=np.int64), "grp": np.array(grp, dtype=int)})
        print(f"L{L}: {len(members)} coherent clusters among 4,000 random contexts", flush=True)
    return sets


def main():
    a = fpm.parse()
    c = fpm.setup(a)
    sets = coherent_sets(a, c)
    dev = nl.device(a)
    s = c.S[-1]
    model = fpm.load_model(a, s, dev)
    P = fpm.positions(a, sets)
    H, _, _ = fpm.acts_and_loss(a, c, model, dev, P)
    rng = np.random.default_rng(a.seed + 11)
    out = []
    for x in sets:
        X, y, groups = fpm.rows(H, P, x)
        out.append({"layer": x["layer"], "n_pos": len(x["pos"]), **fpm.measure(a, X, y, groups, X, None, rng)})
    path = os.path.join(a.out, "fade_control.json")
    with open(path, "w") as fh:
        json.dump({"step": s, "sets": out}, fh, indent=1)
    for L in a.layers:
        v = [o for o in out if o["layer"] == L]
        print(f"L{L}: {len(v)} coherent non-job sets at step {s}: probe median {np.median([o['probe'] for o in v]):.2f}, "
              f"best neuron median {np.median([o['best'] for o in v]):.2f}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
