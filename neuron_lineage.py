#!/usr/bin/env python
"""
neuron_lineage.py

Do MLP neurons grow by recursive generalization (new activation contexts
sprout next to old ones) or by superposition (unrelated contexts arrive with
no lineage)?  Tracked across Pythia training checkpoints.

Corpus: one document per ctx-token chunk (truncated, padded), each tagged with
its Pile source. For each sampled neuron at each checkpoint:
  1. top-k activating positions, at most --per-chunk per chunk (so per
     document), with word-for-word repeated windows dropped
  2. each position -> two FROZEN context embeddings
       genre: sentence embedding of the window + char n-grams around the token
       role:  a different model's (GPT-2 medium) hidden state at the firing
              token + its embedding of the next token
  3. participation ratio (PR) of the k genre embeddings = effective dimensionality
  4. jobs = clusters of the k contexts in the two spaces joined
  5. NEW jobs at an analysis step t: cluster all of the neuron's contexts so far
     together; a cluster made mostly of contexts first seen at t is a new job.
  6. birth = the first checkpoint (early ones included) from which the job's
     contexts average at least --birth-share of the neuron's top-k cutoff until
     it is found, from rerunning those contexts through every checkpoint. Born at
     the first checkpoint: "old".
  7. lineage, in each space separately: distance from the new job to the
     neuron's nearest job at the checkpoint just before birth, against each
     OTHER neuron's nearest job there from the same Pile source.
       p < 0.05 -> sprouted next to an earlier job (recursive generalization)
       else     -> no closer than another neuron's job: no lineage
     Adjacent in role but not genre = same role, different genre.

Control: the random basis. Activations are rotated by a fixed random
orthogonal matrix, so "neurons" become random directions. If the lineage
results look the same there, the effect isn't about the neuron basis. Both
bases come out of the same forward pass.

Planted answers (the controls stage), scored over a grid of clustering and
new-job parameters without looking at any real neuron's lineage labels:
  no-change      odd vs even ranks of one checkpoint's top-2k: should give no new jobs
  superposition  a neuron's last-step top-k gets a coherent block of contexts around
                 a random seed: should be labelled adjacent ~5% of the time
  lineage        same, seeded the r-th nearest free context to one of its jobs at the
                 step before: the adjacent rate is the power, by distance rank r
It prints the setting with the most power among those that pass, to freeze for analyze.

Usage (CUDA or Apple MPS; each 410m checkpoint is a ~1 GB download):
  pip install torch transformers datasets sentence-transformers scikit-learn
  python neuron_lineage.py collect --layers 12    # early steps and analysis steps
  python neuron_lineage.py controls --layers 12
  python neuron_lineage.py analyze --layers 12    # with the parameters controls picked

Outputs in lineage_out/L12[_rand]/: step*.npz (top-2k per unit), contexts.npz
(embedding cache), controls.json (grid of planted-answer scores), report.json
(every new job with its birth, activation history, source, and per-space
lineage test with its own and its parent's examples; summaries per space and
genre x role).
"""
import argparse
import json
import os

import numpy as np
import torch

EARLY_DEFAULT = [128, 256, 512]
STEPS_DEFAULT = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 143000]
TOKEN_RANKS = (50, 100, 200, 400, 800)  # corpus frequency ranks of the single-token pseudo-neurons


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["collect", "controls", "analyze", "redecoy"])
    p.add_argument("--model", default="EleutherAI/pythia-410m-deduped")
    p.add_argument("--layers", type=int, nargs="+", default=[12])
    p.add_argument("--bases", nargs="+", choices=["neuron", "random"], default=["neuron", "random"])
    p.add_argument("--steps", type=int, nargs="+", default=STEPS_DEFAULT,
                   help="analysis steps: new jobs are found here")
    p.add_argument("--early-steps", type=int, nargs="+", default=EARLY_DEFAULT,
                   help="earlier checkpoints, collected too, for dating births and finding parents")
    p.add_argument("--device", default=None, help="cuda | mps | cpu (default: best available)")
    p.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default=None,
                   help="model dtype (default: float16 on cuda, float32 elsewhere; MPS float16 "
                        "overflows to NaN in Pythia's attention output)")
    p.add_argument("--dataset", default="NeelNanda/pile-10k")
    p.add_argument("--split", default="train")
    p.add_argument("--n-tokens", type=int, default=2_000_000, help="corpus size (chunks x ctx)")
    p.add_argument("--ctx", type=int, default=512, help="chunk length; one document per chunk")
    p.add_argument("--k", type=int, default=200,
                   help="top-k contexts per neuron (collect keeps 2k for the no-change control)")
    p.add_argument("--per-chunk", type=int, default=2, help="most top-k hits from one chunk (document)")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--n-neurons", type=int, default=512, help="neurons sampled for analysis")
    p.add_argument("--neuron-seed", type=int, default=None, help="seed for the neuron sample (default --seed)")
    p.add_argument("--window", type=int, default=24, help="tokens before the firing token")
    p.add_argument("--after", type=int, default=4, help="tokens after it (next-token neurons)")
    p.add_argument("--lex-span", type=int, default=3, help="tokens each side for char n-grams")
    p.add_argument("--sem-weight", type=float, default=0.5, help="genre: semantic vs lexical weight")
    p.add_argument("--role-model", default="gpt2-medium",
                   help="a different, fixed GPT-2-family model for the role space")
    p.add_argument("--role-layer", type=int, default=12, help="its layer whose output is the role state")
    p.add_argument("--next-weight", type=float, default=0.25, help="role: weight of the next token")
    p.add_argument("--role-dims", type=int, default=256, help="role space dimensions after PCA")
    p.add_argument("--role-weight", type=float, default=0.5,
                   help="weight of the role space when the two are joined to find jobs")
    p.add_argument("--min-cluster", type=int, default=10)
    p.add_argument("--cluster-thresh", type=float, default=0.55, help="cosine distance cut")
    p.add_argument("--stable-corr", type=float, default=0.5,
                   help="probe-activation corr needed to call a unit the same across checkpoints")
    p.add_argument("--stable-jacc", type=float, default=0.1,
                   help="top-k Jaccard that also counts as the same unit (rescues sparse units "
                        "the probe tokens miss)")
    p.add_argument("--new-frac", type=float, default=0.9,
                   help="a pooled cluster is a new job when at least this share of its contexts "
                        "first appear at the step being tested")
    p.add_argument("--birth-share", type=float, default=0.2,
                   help="a job is born at the first checkpoint from which its contexts average at "
                        "least this share of the unit's top-k cutoff through the step it is found")
    p.add_argument("--collapse-share", type=float, default=0.25,
                   help="a parent collapsed if it ends below this share of the unit's cutoff")
    p.add_argument("--controls-span", type=int, default=0,
                   help="controls: only the last this-many checkpoints (0 = every analysis step), "
                        "to keep pooled clustering affordable on long runs")
    p.add_argument("--min-new-dist", type=float, default=0.0,
                   help="optional: a new job's centre must be farther than this from every job at "
                        "its parent checkpoint (cluster-thresh/2 is the distinct-job rule; 0 = off)")
    p.add_argument("--min-null", type=int, default=20, help="fewest other neurons for a lineage test")
    p.add_argument("--boot", type=int, default=1000, help="bootstrap reps over neurons")
    p.add_argument("--n-random-ctx", type=int, default=20000,
                   help="random corpus contexts embedded for the controls (plants, PR ceiling)")
    p.add_argument("--plant-size", type=int, default=50, help="contexts per planted job")
    p.add_argument("--plant-ranks", type=int, nargs="+", default=[25, 100, 400, 1600],
                   help="lineage plants: seed rank among free contexts nearest the parent job")
    p.add_argument("--plant-reps", type=int, default=2, help="plants per neuron per kind and rank")
    p.add_argument("--grid-sem-weight", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    p.add_argument("--grid-cluster-thresh", type=float, nargs="+", default=[0.45, 0.55, 0.65])
    p.add_argument("--grid-new-frac", type=float, nargs="+", default=[0.75, 0.9, 1.0])
    p.add_argument("--max-fpr", type=float, default=0.08,
                   help="controls: most superposition plants labelled adjacent, in every space")
    p.add_argument("--max-false-new", type=float, default=0.05,
                   help="controls: most new jobs per unit in the no-change control")
    p.add_argument("--tag", default="", help="suffix for report/controls file names")
    p.add_argument("--out", default="lineage_out")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def device(a):
    if a.device:
        return a.device
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def free(dev):
    if dev == "cuda":
        torch.cuda.empty_cache()
    elif dev == "mps":
        torch.mps.empty_cache()


def out_dir(a, layer, basis):
    d = os.path.join(a.out, f"L{layer}" + ("_rand" if basis == "random" else ""))
    os.makedirs(d, exist_ok=True)
    return d


def tagged(a, name):
    return f"{name}_{a.tag}.json" if a.tag else f"{name}.json"


def all_steps(a):
    """Every checkpoint collected and dated over: the early ones, then the analysis steps."""
    return sorted(set(a.early_steps) | set(a.steps))


def corpus(a, tok):
    """Fixed corpus shared by every checkpoint: one document per ctx-token chunk, truncated and
    padded with EOS, so the per-chunk cap on top-k hits is a per-document cap. Returns the
    tokens (n_chunks, ctx), each chunk's document length, and its Pile source."""
    path = os.path.join(a.out, "corpus.npz")
    meta_path = os.path.join(a.out, "corpus.json")
    meta = {"model": a.model, "dataset": a.dataset, "split": a.split, "n_tokens": a.n_tokens,
            "ctx": a.ctx, "packing": "one document per chunk"}
    if os.path.exists(path):
        with open(meta_path) as fh:
            if json.load(fh) != meta:
                raise SystemExit(f"{path} was built with other corpus settings; pass another --out")
        with np.load(path) as f:
            return f["toks"], f["lengths"], f["sources"]
    from datasets import load_dataset
    ds = load_dataset(a.dataset, split=a.split, streaming=True)
    rows, lengths, sources = [], [], []
    for ex in ds:
        ids = tok(ex["text"][:10 * a.ctx])["input_ids"][:a.ctx]
        if len(ids) <= a.window:
            continue  # no position with a full left window
        rows.append(ids + [tok.eos_token_id] * (a.ctx - len(ids)))
        lengths.append(len(ids))
        sources.append((ex.get("meta") or {}).get("pile_set_name", "unknown"))
        if len(rows) == a.n_tokens // a.ctx:
            break
    toks = np.array(rows, dtype=np.int32)
    lengths, sources = np.array(lengths, dtype=np.int32), np.array(sources)
    os.makedirs(a.out, exist_ok=True)
    np.savez(path, toks=toks, lengths=lengths, sources=sources)
    with open(meta_path, "w") as fh:
        json.dump(meta, fh)
    return toks, lengths, sources


def load_corpus(a):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    return (tok, *corpus(a, tok))


def valid(a, lengths):
    """(n_chunks, ctx) mask of positions with a full left window inside their document."""
    c = np.arange(a.ctx)[None]
    return (c >= a.window) & (c < lengths[:, None])


# ----------------------------------------------------------------- stage 1
class _Stop(Exception):
    pass


def collect(a):
    from transformers import AutoTokenizer, GPTNeoXForCausalLM

    tok = AutoTokenizer.from_pretrained(a.model)
    toks_np, lengths, _ = corpus(a, tok)
    toks = torch.from_numpy(toks_np).long()
    usable = torch.from_numpy(valid(a, lengths))
    dev = device(a)
    dt = getattr(torch, a.dtype or ("float16" if dev == "cuda" else "float32"))
    Q = None

    for step in all_steps(a):
        todo = [(L, b) for L in a.layers for b in a.bases
                if not os.path.exists(os.path.join(out_dir(a, L, b), f"step{step}.npz"))]
        if not todo:
            continue
        model = GPTNeoXForCausalLM.from_pretrained(
            a.model, revision=f"step{step}", dtype=dt).to(dev).eval()
        layers = sorted({L for L, _ in todo})
        d = model.gpt_neox.layers[layers[0]].mlp.dense_4h_to_h.in_features  # post-GELU neuron count

        if "random" in a.bases and Q is None:  # same rotation for every checkpoint and layer
            g = torch.Generator().manual_seed(a.seed)
            Q = torch.linalg.qr(torch.randn(d, d, generator=g))[0].to(dev, dt)

        buf = {}

        def grab(L):
            def hook(m, inp):
                buf[L] = inp[0]
                if L == layers[-1]:
                    raise _Stop  # nothing past the deepest layer we need
            return hook

        hooks = [model.gpt_neox.layers[L].mlp.dense_4h_to_h.register_forward_pre_hook(grab(L))
                 for L in layers]
        K = 2 * a.k
        top = {c: (torch.full((d, K), -float("inf"), device=dev),
                   torch.zeros((d, K), dtype=torch.long, device=dev)) for c in todo}
        probe = {}

        with torch.no_grad():
            for b0 in range(0, len(toks), a.batch):
                x = toks[b0:b0 + a.batch].to(dev)
                try:
                    model.gpt_neox(x)  # skip the LM head
                except _Stop:
                    pass
                for L in layers:  # topk ranks NaN above everything, so it would fill the top-k
                    if not torch.isfinite(buf[L]).all():
                        raise SystemExit(f"non-finite activations at layer {L}, step {step}; "
                                         "try --dtype float32")
                B = x.shape[0]
                keep = usable[b0:b0 + B].to(dev)
                row0 = torch.arange(b0, b0 + B, device=dev).view(B, 1, 1) * a.ctx
                for c in todo:
                    L, basis = c
                    act = buf[L]  # (B, ctx, d)
                    if basis == "random":
                        act = act @ Q
                    act = act.to(torch.float32, copy=True)  # buf[L] is shared by both bases
                    if c not in probe:  # fixed probe set for cross-checkpoint identity
                        probe[c] = act[keep].cpu().numpy().astype(np.float16)
                    act[~keep] = -float("inf")  # no full left window, or padding past the document
                    # best --per-chunk positions of each chunk, merged into the running top-k.
                    # A chunk lies inside one batch, so this is exactly the capped top-k.
                    cv, ct = act.topk(a.per_chunk, dim=1)  # (B, per_chunk, d)
                    tv, ti = top[c]
                    v = torch.cat([tv, cv.permute(2, 0, 1).reshape(d, -1)], 1)
                    i = torch.cat([ti, (row0 + ct).permute(2, 0, 1).reshape(d, -1)], 1)
                    nv, ni = v.topk(K, dim=1)
                    top[c] = (nv, i.gather(1, ni))

        for h in hooks:
            h.remove()
        for c in todo:
            L, basis = c
            np.savez(os.path.join(out_dir(a, L, basis), f"step{step}.npz"),
                     vals=top[c][0].cpu().numpy(), idx=top[c][1].cpu().numpy(), probe=probe[c])
        del model
        free(dev)
        print(f"collected step {step}: " + ", ".join(f"L{L}/{b}" for L, b in todo))


# ----------------------------------------------------------------- stage 2
def window_texts(a, tok, toks, positions):
    """Context strings per position; windows are cut at end-of-document tokens."""
    eos = tok.eos_token_id
    full, short = [], []
    for p in positions:
        r, c = divmod(int(p), a.ctx)
        left = toks[r, max(0, c - a.window):c]
        e = np.flatnonzero(left == eos)
        left = left[e[-1] + 1:] if len(e) else left
        right = toks[r, c + 1:min(a.ctx, c + 1 + a.after)]
        e = np.flatnonzero(right == eos)
        right = right[:e[0]] if len(e) else right
        mid = toks[r, c:c + 1]
        full.append(f"{tok.decode(left)}[[{tok.decode(mid)}]]{tok.decode(right)}")
        n = min(len(left), a.lex_span)
        short.append(tok.decode(np.concatenate([left[len(left) - n:], mid, right[:a.lex_span]])))
    return full, short


def embed(a, dev, full, short):
    from sentence_transformers import SentenceTransformer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    sem = SentenceTransformer("all-MiniLM-L6-v2", device=dev).encode(
        [t.replace("[[", "").replace("]]", "") for t in full],
        batch_size=256, normalize_embeddings=True, show_progress_bar=True)
    tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2).fit_transform(short)
    lex = normalize(TruncatedSVD(min(256, tf.shape[1] - 1), random_state=a.seed).fit_transform(tf))
    return sem, lex


def unit_norm(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-9)


def role_embed(a, dev, tok, toks, lengths, positions):
    """The firing token's role: a different, fixed model's hidden state at that token (so the
    space isn't built from the model under study), and that model's input embedding of the
    next token. Each part is centred and scaled per dimension (GPT-2's states are dominated
    by a few huge dimensions), then they are joined and reduced to role_dims by PCA."""
    from sklearn.decomposition import PCA
    from transformers import AutoModel, AutoTokenizer

    rt = AutoTokenizer.from_pretrained(a.role_model)
    rm = AutoModel.from_pretrained(a.role_model, dtype=torch.float32).to(dev).eval()
    lefts, nexts = [], []
    for p in positions:
        r, c = divmod(int(p), a.ctx)
        lefts.append(tok.decode(toks[r, max(0, c - a.window):c + 1]))
        nexts.append(tok.decode(toks[r, c + 1:c + 2]) if c + 1 < lengths[r] else "")
    H = np.zeros((len(positions), rm.config.hidden_size), dtype=np.float32)
    buf = {}

    def hook(m, inp, out):
        buf["h"] = out[0] if isinstance(out, tuple) else out
        raise _Stop

    h = rm.h[a.role_layer - 1].register_forward_hook(hook)
    with torch.no_grad():
        for b0 in range(0, len(lefts), 256):
            enc = [e[-64:] or [rt.eos_token_id] for e in rt(lefts[b0:b0 + 256])["input_ids"]]
            n = torch.tensor([len(e) for e in enc])
            x = torch.full((len(enc), int(n.max())), rt.eos_token_id)
            for i, e in enumerate(enc):
                x[i, :len(e)] = torch.tensor(e)
            mask = (torch.arange(x.shape[1])[None] < n[:, None]).long()
            try:
                rm(input_ids=x.to(dev), attention_mask=mask.to(dev))
            except _Stop:
                pass
            last = buf["h"][torch.arange(len(enc), device=dev), (n - 1).to(dev)]
            H[b0:b0 + len(enc)] = last.float().cpu().numpy()
    h.remove()
    first = [e[:1] for e in rt(nexts)["input_ids"]]
    has = np.array([bool(t) and bool(e) for t, e in zip(nexts, first)])
    wte = rm.get_input_embeddings().weight.detach().float().cpu().numpy()
    del rm
    free(dev)

    def standard(X):
        return unit_norm((X - X.mean(0)) / (X.std(0) + 1e-6))

    Nx = np.zeros((len(positions), wte.shape[1]), dtype=np.float32)
    if has.any():
        Nx[has] = standard(wte[[e[0] for e, ok in zip(first, has) if ok]])
    J = np.hstack([np.sqrt(1 - a.next_weight) * standard(H), np.sqrt(a.next_weight) * Nx])
    Z = PCA(min(a.role_dims, J.shape[1], len(J)), random_state=a.seed).fit_transform(J)
    return unit_norm(Z)


def combine(sem, lex, w):
    """Unit-norm genre embeddings: semantic and lexical parts weighted w : 1 - w."""
    return np.hstack([np.sqrt(w) * sem.astype(np.float32), np.sqrt(1 - w) * lex.astype(np.float32)])


def spaces(a, cs, w=None):
    """The joined space jobs are found in, and the slice of it holding each test space."""
    G = combine(cs["sem"], cs["lex"], a.sem_weight if w is None else w)
    R = cs["role"].astype(np.float32)
    D = np.hstack([np.sqrt(1 - a.role_weight) * G, np.sqrt(a.role_weight) * R])
    return D, {"genre": slice(0, G.shape[1]), "role": slice(G.shape[1], D.shape[1])}


def participation_ratio(X):
    X = X - X.mean(0)
    s = np.linalg.svd(X, compute_uv=False) ** 2
    return float(s.sum() ** 2 / (s ** 2).sum()) if s.sum() > 0 else 0.0


def clusters(a, X):
    from sklearn.cluster import AgglomerativeClustering
    if len(X) < max(2, a.min_cluster):
        return np.zeros((0, X.shape[1])), []
    lab = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                  distance_threshold=a.cluster_thresh).fit_predict(X)
    cents, members = [], []
    for l in np.unique(lab):
        m = np.where(lab == l)[0]
        if len(m) >= a.min_cluster:
            c = X[m].mean(0)
            cents.append(c / np.linalg.norm(c))
            members.append(m)
    return np.array(cents).reshape(-1, X.shape[1]), members


def jaccard(x, y):
    return len(np.intersect1d(x, y)) / max(len(np.union1d(x, y)), 1)


def dedupe(a, toks, lengths, idx, vals, n):
    """The first n positions, by activation, whose window tokens no earlier one already had:
    templated text (licence headers, generated maths) repeats windows word for word."""
    out, seen = [], set()
    for p, v in zip(idx.tolist(), vals.tolist()):
        if not np.isfinite(v):
            break
        r, c = divmod(p, a.ctx)
        key = toks[r, max(0, c - a.window):min(lengths[r], c + 1 + a.after)].tobytes()
        if key not in seen:
            seen.add(key)
            out.append(p)
            if len(out) == n:
                break
    return np.array(out, dtype=np.int64)


def load_units(a, d_out, toks, lengths):
    """Sampled units at each checkpoint: top-k and top-2k positions with repeated windows
    dropped, the stored top-2k, the top-k cutoff (k-th activation), and probe activations."""
    S = all_steps(a)
    with np.load(os.path.join(d_out, f"step{S[0]}.npz")) as f:
        d = f["idx"].shape[0]
    seed = a.seed if a.neuron_seed is None else a.neuron_seed
    neurons = np.sort(np.random.default_rng(seed).choice(d, size=min(a.n_neurons, d), replace=False))
    data = {}
    for s in S:  # read each array once: NpzFile re-reads the file on every access
        with np.load(os.path.join(d_out, f"step{s}.npz")) as f:
            vals, idx = f["vals"][neurons], f["idx"][neurons]
            probe = f["probe"][:, neurons].astype(np.float32)
        wide = [dedupe(a, toks, lengths, idx[u], vals[u], 2 * a.k) for u in range(len(neurons))]
        data[s] = {"ids": [w[:a.k] for w in wide], "wide": wide, "vals": vals, "idx": idx,
                   "cut": vals[:, a.k - 1], "probe": probe}
    return S, neurons, data


def stability(a, S, data):
    """First step index of each unit's unbroken stable run ending at the final checkpoint.
    A link holds on the same probe tokens' activations, or on overlap of the top-k positions
    (the probe can miss sparse units)."""
    N = len(data[S[0]]["ids"])
    corr, jacc = {}, {}
    for t0, t1 in zip(S, S[1:]):
        p0, p1 = data[t0]["probe"], data[t1]["probe"]
        p0 = (p0 - p0.mean(0)) / (p0.std(0) + 1e-6)
        p1 = (p1 - p1.mean(0)) / (p1.std(0) + 1e-6)
        corr[t1] = (p0 * p1).mean(0)
        jacc[t1] = np.array([jaccard(data[t0]["ids"][u], data[t1]["ids"][u]) for u in range(N)])

    def run_start(linked):
        st = np.zeros(N, dtype=int)
        for u in range(N):
            i = len(S) - 1
            while i > 0 and linked(S[i], u):
                i -= 1
            st[u] = i
        return st

    by_corr = lambda s, u: corr[s][u] > a.stable_corr
    by_jacc = lambda s, u: jacc[s][u] > a.stable_jacc
    start = run_start(lambda s, u: by_corr(s, u) or by_jacc(s, u))
    return start, {
        "full_run": {"corr": int((run_start(by_corr) == 0).sum()), "jacc": int((run_start(by_jacc) == 0).sum()),
                     "either": int((start == 0).sum()), "of": N},
        "links_broken": {str(t): {"corr": int((corr[t] <= a.stable_corr).sum()),
                                  "jacc": int((jacc[t] <= a.stable_jacc).sum()),
                                  "both": int(((corr[t] <= a.stable_corr) & (jacc[t] <= a.stable_jacc)).sum())}
                         for t in S[1:]},
        "median_corr": {str(t): round(float(np.median(corr[t])), 3) for t in S[1:]},
        "median_jacc": {str(t): round(float(np.median(jacc[t])), 3) for t in S[1:]},
        "run_start_counts": {str(S[i]): int((start == i).sum()) for i in range(len(S))},
    }


def context_space(a, tok, toks, lengths, sources, d_out, S, data):
    """Every context the analysis or the controls can touch, sorted, with its Pile source and
    frozen genre and role embeddings (cached in d_out): each unit's top-k at every step and
    top-2k at the last, a random corpus sample, and single-token pseudo-neurons."""
    N = len(data[S[0]]["ids"])
    flat = np.flatnonzero(valid(a, lengths).ravel())
    ids = toks.ravel()
    rng = np.random.default_rng(a.seed + 1)
    rand = rng.choice(flat, size=min(a.n_random_ctx, len(flat)), replace=False)
    order = np.argsort(-np.bincount(ids[flat]), kind="stable")
    order = order[order != tok.eos_token_id]
    tokens = {}
    for r in TOKEN_RANKS:
        occ = flat[ids[flat] == order[r]]
        if len(occ) >= a.k:
            tokens[int(order[r])] = np.sort(rng.choice(occ, a.k, replace=False))
    units = [data[s]["ids"][u] for s in S for u in range(N)] + [data[S[-1]]["wide"][u] for u in range(N)]
    pos = np.unique(np.concatenate(units + [rand] + list(tokens.values())))
    names, codes = np.unique(sources, return_inverse=True)
    cs = {"pos": pos, "src": codes[pos // a.ctx], "src_names": names.tolist(),
          "random": rand, "tokens": tokens}

    path = os.path.join(d_out, "contexts.npz")
    meta = json.dumps({k: getattr(a, k) for k in ("window", "after", "lex_span", "seed", "role_model",
                                                    "role_layer", "next_weight", "role_dims")})
    if os.path.exists(path):
        with np.load(path) as f:
            if str(f["meta"]) == meta and np.array_equal(f["pos"], pos):
                cs.update(sem=f["sem"], lex=f["lex"], role=f["role"])
                return cs
    full, short = window_texts(a, tok, toks, pos)
    sem, lex = embed(a, device(a), full, short)
    role = role_embed(a, device(a), tok, toks, lengths, pos)
    cs.update(sem=sem.astype(np.float16), lex=lex.astype(np.float16), role=role.astype(np.float16))
    np.savez(path, pos=pos, meta=meta, sem=cs["sem"], lex=cs["lex"], role=cs["role"])
    return cs


def unit_rows(S, data, pos):
    """Rows of the context space holding each unit's top-k at each step."""
    N = len(data[S[0]]["ids"])
    return [[np.searchsorted(pos, data[s]["ids"][u]) for s in S] for u in range(N)]


def unit_jobs(a, D, rows, src):
    """A unit's jobs at one step: clusters of its contexts (members index rows), each with its
    main Pile source."""
    _, members = clusters(a, D[rows])
    return members, rows, [int(np.bincount(src[rows[m]]).argmax()) for m in members]


def bare(rows):
    """A step's entry when only its contexts matter (the step being tested)."""
    return [], rows, []


def nearest_job(D, sl, job, rows):
    """Index of the job nearest the centroid of `rows` in slice sl, its distance, the centroid."""
    members, prow, _ = job
    c = unit_norm(D[rows, sl].mean(0))
    P = unit_norm(np.array([D[prow[m], sl].mean(0) for m in members]))
    dist = 1 - P @ c
    j = int(dist.argmin())
    return j, float(dist[j]), c


class NullPool:
    """Every unit's jobs at one checkpoint, for the lineage null: their centroids in each test
    space and their main Pile sources."""

    def __init__(self, jobs, D, slices):
        self.jobs, self.D, self.slices, self.at = jobs, D, slices, {}

    def dists(self, si, c, space, src, own):
        """Each other unit's nearest job to centroid c, in that space, among its jobs at step
        index si from source src."""
        if si not in self.at:
            js = [(u, m, s) for u, j in enumerate(self.jobs) if j[si] is not None
                  for m, s in zip(j[si][0], j[si][2])]
            cents = {k: unit_norm(np.array([self.D[self.jobs[u][si][1][m], sl].mean(0)
                                            for u, m, _ in js]).reshape(len(js), sl.stop - sl.start))
                     for k, sl in self.slices.items()}
            self.at[si] = (np.array([u for u, _, _ in js], dtype=int),
                           np.array([s for _, _, s in js], dtype=int), cents)
        owner, srcs, cents = self.at[si]
        keep = (srcs == src) & (owner != own)
        d = np.full(len(self.jobs), np.inf)
        np.minimum.at(d, owner[keep], 1 - cents[space][keep] @ c)
        return d[np.isfinite(d)]


def pool(a, D, jobs, i0, si):
    """Every context a unit had over steps i0..si, once each, clustered together.
    born = the first of those steps at which the context was in the unit's top-k."""
    rows = np.concatenate([jobs[r][1] for r in range(i0, si + 1)])
    born = np.concatenate([np.full(len(jobs[r][1]), r) for r in range(i0, si + 1)])
    rows, first = np.unique(rows, return_index=True)  # steps are concatenated in order
    _, members = clusters(a, D[rows])
    return rows, born[first], members


def detect(a, D, jobs, i0, si, pools=None):
    """New jobs at step index si: with all of the unit's contexts from i0 through si
    clustered together, the clusters made mostly (new_frac) of contexts first seen at si.
    A sprout next to its parent counts as long as it forms its own cluster; it doesn't have
    to lie outside everything the unit ever did. Returns (rows by rank at si, best rank).
    pools caches the clustering, which new_frac doesn't change."""
    if pools is not None and si in pools:
        rows, born, members = pools[si]
    else:
        rows, born, members = pool(a, D, jobs, i0, si)
        if pools is not None:
            pools[si] = rows, born, members
    rank = {r: i for i, r in enumerate(jobs[si][1].tolist())}
    out = []
    for m in members:
        new = m[born[m] == si]
        if len(new) < a.min_cluster or len(new) < a.new_frac * len(m):
            continue  # mostly old contexts: an existing job, maybe grown
        ranks = np.array([rank[r] for r in rows[new].tolist()])
        order = np.argsort(ranks)
        out.append((rows[new][order], int(ranks[order][0])))
    return out


def lineage(a, D, slices, job, rows, src, null, ps, own):
    """Test one new job (contexts `rows`, main source `src`) against the unit's jobs at step
    index ps, separately in each test space: the nearest of them against each other unit's
    nearest same-source job there. None when the unit had no job at ps."""
    if not len(job[0]):
        return None
    out = {}
    for k, sl in slices.items():
        j, d, c = nearest_job(D, sl, job, rows)
        d_other = null.dists(ps, c, k, src, own)
        t = {"parent": j, "d": d, "n_null": len(d_other), "p": None, "label": None}
        if len(d_other) >= a.min_null:
            # under the null p is uniform: p >= 0.05 is "no evidence of lineage",
            # and ~5% of truly unrelated jobs still land in "adjacent" by chance
            t["p"] = float((d_other <= d).mean())
            t["label"] = "adjacent" if t["p"] < 0.05 else "no_lineage"
        out[k] = t
    return out


def label_counts(events, space):
    labs = [e[space]["label"] for e in events if e.get(space) and e[space]["label"]]
    return {l: labs.count(l) for l in ("adjacent", "no_lineage")}


def adjacent_share(a, rng, events, space):
    """Adjacent share with a 95% CI, resampling neurons (events of one neuron aren't independent)."""
    by = {}
    for e in events:
        if e.get(space) and e[space]["label"]:
            by.setdefault(e["neuron"], []).append(e[space]["label"] == "adjacent")
    if not by:
        return None
    adj = np.array([sum(v) for v in by.values()])
    tot = np.array([len(v) for v in by.values()])
    draws = rng.integers(0, len(adj), size=(a.boot, len(adj)))
    lo, hi = np.quantile(adj[draws].sum(1) / tot[draws].sum(1), [0.025, 0.975])
    return {"share": round(float(adj.sum() / tot.sum()), 3), "ci95": [round(float(lo), 3), round(float(hi), 3)],
            "n_events": int(tot.sum()), "n_neurons": len(adj)}


def pick_decoy(D, jobs, N, u, ps, rows, rng):
    """A decoy parent picked exactly as the real one is: a random other unit with a job at step
    index ps, and its job nearest the new job (contexts `rows`) in the joined space. Without
    lineage the two candidates are then exchangeable. (A decoy matched to the new job's source,
    unlike the real parent, drew readers by genre: they chose it 69% of the time.)"""
    others = [v for v in range(N) if v != u and jobs[v][ps] is not None and len(jobs[v][ps][0])]
    if not others:
        return None, None
    v = int(rng.choice(others))
    j, _, _ = nearest_job(D, slice(None), jobs[v][ps], rows)
    members, prow, _ = jobs[v][ps]
    return v, prow[members[j]]


def redecoy(a):
    """Recompute the blind test's decoys in existing reports, from the cached context space
    (no model runs). Run with the flags analyze used."""
    tok, toks, lengths, sources = load_corpus(a)
    for L in a.layers:
        for basis in a.bases:
            d_out = out_dir(a, L, basis)
            path = os.path.join(d_out, tagged(a, "report"))
            with open(path) as fh:
                r = json.load(fh)
            S, neurons, data = load_units(a, d_out, toks, lengths)
            cs = context_space(a, tok, toks, lengths, sources, d_out, S, data)
            D, _ = spaces(a, cs)
            jobs = [[unit_jobs(a, D, x, cs["src"]) for x in ru] for ru in unit_rows(S, data, cs["pos"])]
            col = {int(n): i for i, n in enumerate(neurons)}
            rd = np.random.default_rng(a.seed + 7)
            picks = []
            for e in r["events"]:
                if e["kind"] == "dated":
                    rows = np.searchsorted(cs["pos"], e["new_pos"])
                    picks.append((e, *pick_decoy(D, jobs, len(neurons), col[e["neuron"]],
                                                 S.index(e["parent_step"]), rows, rd)))
            need = np.array(sorted({x for _, v, dr in picks if v is not None for x in dr[:5].tolist()}), dtype=int)
            text = dict(zip(need.tolist(), window_texts(a, tok, toks, cs["pos"][need])[0]))
            for e, v, dr in picks:
                e.pop("decoy_neuron", None)
                e.pop("decoy_examples", None)
                if v is not None:
                    e["decoy_neuron"] = int(neurons[v])
                    e["decoy_examples"] = [text[x] for x in dr[:5].tolist()]
            with open(path, "w") as fh:
                json.dump(r, fh, indent=1, ensure_ascii=False)
            print(f"L{L}/{basis}: decoys for {sum(v is not None for _, v, _ in picks)} of {len(picks)} dated jobs")


def handoff_summary(a, dated):
    """Parents that collapsed (below collapse_share of the unit's cutoff at the last checkpoint)
    against those that stayed: the taker's share of its own cutoff on the parent's contexts at
    the end, and its gain since the parent's step. A handoff shows as collapsed parents' takers
    gaining more."""
    if not dated:
        return None
    groups = {"collapsed": [r for r in dated if r["parent_history"][-1] < a.collapse_share],
              "stayed": [r for r in dated if r["parent_history"][-1] >= a.collapse_share]}
    med = lambda xs, k: round(float(np.median([r["handoff"][k] for r in xs])), 3) if xs else None
    return {g: {"n": len(xs), "taker_share_median": med(xs, "share"), "taker_gain_median": med(xs, "gain"),
                "taker_gained_over_0.5": sum(r["handoff"]["gain"] > 0.5 for r in xs)}
            for g, xs in groups.items()}


def layer_acts(model, layers, x):
    """Post-GELU MLP activations at each of `layers` for token batch x (stops after the deepest)."""
    buf = {}

    def grab(L):
        def hook(m, inp):
            buf[L] = inp[0]
            if L == layers[-1]:
                raise _Stop
        return hook

    hooks = [model.gpt_neox.layers[L].mlp.dense_4h_to_h.register_forward_pre_hook(grab(L))
             for L in layers]
    try:
        model.gpt_neox(x)
    except _Stop:
        pass
    finally:
        for h in hooks:
            h.remove()
    return buf


def histories(a, toks, lengths, layer, basis, S, data, neurons, items, full=False):
    """For each item (unit index, step index, positions in the unit's top-k at that step): the
    unit's mean activation over the positions at every checkpoint, as a share of its top-k
    cutoff there, and the same over 200 random positions. Reruns only the chunks involved.
    err is the largest difference from collect's stored activations at the item's step, as a
    share of the cutoff: a check that the rerun matches. With full, also every unit's mean
    activation over each item's positions at every checkpoint, shape (checkpoints, units)."""
    from transformers import GPTNeoXForCausalLM

    out = {"hist": [], "quiet": [], "err": 0.0, "full": [] if full else None}
    if not items:
        return out
    dev = device(a)
    dt = getattr(torch, a.dtype or ("float16" if dev == "cuda" else "float32"))
    flat = np.flatnonzero(valid(a, lengths).ravel())
    quiet = np.sort(np.random.default_rng(a.seed + 5).choice(flat, size=200, replace=False))
    units = np.array(sorted({u for u, _, _ in items}), dtype=int)
    col = {u: i for i, u in enumerate(units.tolist())}
    pos = np.unique(np.concatenate([quiet] + [p for _, _, p in items]))
    A = np.zeros((len(S), len(pos), len(units)), dtype=np.float32)
    chunks = np.unique(pos // a.ctx)
    tt = torch.from_numpy(toks).long()
    cols = torch.as_tensor(neurons[units], device=dev)
    Q, F = None, None
    for si, s in enumerate(S):
        model = GPTNeoXForCausalLM.from_pretrained(
            a.model, revision=f"step{s}", dtype=dt).to(dev).eval()
        d = model.gpt_neox.layers[layer].mlp.dense_4h_to_h.in_features
        if basis == "random" and Q is None:  # the rotation collect used
            g = torch.Generator().manual_seed(a.seed)
            Q = torch.linalg.qr(torch.randn(d, d, generator=g))[0].to(dev, dt)
        if full and F is None:
            F = np.zeros((len(S), len(pos), d), dtype=np.float16)
        with torch.no_grad():
            for b0 in range(0, len(chunks), a.batch):
                rb = chunks[b0:b0 + a.batch]
                h = layer_acts(model, [layer], tt[rb].to(dev))[layer]
                sel = np.flatnonzero(np.isin(pos // a.ctx, rb))
                h = h[torch.as_tensor(np.searchsorted(rb, pos[sel] // a.ctx), device=dev),
                      torch.as_tensor(pos[sel] % a.ctx, device=dev)]  # (positions, d)
                if not torch.isfinite(h).all():
                    raise SystemExit(f"non-finite activations at layer {layer}, step {s}; "
                                     "try --dtype float32")
                if basis == "random":
                    h = h @ Q
                A[si, sel] = h[:, cols].float().cpu().numpy()
                if F is not None:
                    F[si, sel] = h.half().cpu().numpy()
        del model
        free(dev)
        print(f"  history L{layer}/{basis} step {s}: {len(chunks)} chunks")
    cut = np.array([[data[s]["cut"][u] for u in units] for s in S])
    qi = np.searchsorted(pos, quiet)
    for u, si, p in items:
        i, c = np.searchsorted(pos, p), col[u]
        out["hist"].append(A[:, i, c].mean(1) / cut[:, c])
        out["quiet"].append(A[:, qi, c].mean(1) / cut[:, c])
        kept = dict(zip(data[S[si]]["idx"][u].tolist(), data[S[si]]["vals"][u].tolist()))
        out["err"] = max(out["err"], float(np.max(np.abs(A[si, i, c] - [kept[x] for x in p.tolist()])))
                         / cut[si, c])
        if full:
            out["full"].append(F[:, i].astype(np.float32).mean(1))
    return out


def analyze(a):
    tok, toks, lengths, sources = load_corpus(a)
    for L in a.layers:
        for basis in a.bases:
            analyze_one(a, tok, toks, lengths, sources, L, basis)
    print("compare the adjacent shares with the random basis, and read the 'no_lineage' jobs by "
          "hand: each is either superposition or a lineage the embeddings missed.")


def analyze_one(a, tok, toks, lengths, sources, layer, basis):
    d_out = out_dir(a, layer, basis)
    S, neurons, data = load_units(a, d_out, toks, lengths)
    N = len(neurons)
    main = [i for i, s in enumerate(S) if s in a.steps]
    start, stab = stability(a, S, data)
    cs = context_space(a, tok, toks, lengths, sources, d_out, S, data)
    pos, src, names = cs["pos"], cs["src"], cs["src_names"]
    D, slices = spaces(a, cs)
    rows = unit_rows(S, data, pos)
    PR = np.array([[participation_ratio(D[r, slices["genre"]]) for r in ru] for ru in rows])
    jobs = [[unit_jobs(a, D, r, src) for r in ru] for ru in rows]
    null = NullPool(jobs, D, slices)

    # 1. new jobs at the analysis steps, within each unit's stable run there
    found = []
    for u in range(N):
        i0 = max(start[u], main[0])
        for si in range(i0 + 1, len(S)):
            found += [{"u": u, "si": si, "rows": r, "best_rank": br} for r, br in detect(a, D, jobs[u], i0, si)]

    # 2. birth: the first checkpoint from which the job's contexts average at least birth_share
    #    of the cutoff through the step it was found (early on, random contexts already sit
    #    near 0.1, so a single crossing can be noise)
    H = histories(a, toks, lengths, layer, basis, S, data, neurons,
                  [(f["u"], f["si"], pos[f["rows"]]) for f in found])
    err = H["err"]
    for f, h, q in zip(found, H["hist"], H["quiet"]):
        f["history"], f["quiet"] = h, q
        f["birth"] = next((t for t in range(f["si"] + 1) if (h[t:f["si"] + 1] >= a.birth_share).all()), f["si"])

    # 3. the parent comes from the checkpoint just before birth. New jobs of one unit and step
    #    with the same birth and the same parent are one job.
    groups, events = {}, []
    for f in found:
        u, b = f["u"], f["birth"]
        if b == 0:
            f["kind"] = "old"  # already active at the first checkpoint: born before the window
        elif not len(jobs[u][b - 1][0]):
            f["kind"] = "no_parent"  # the unit had no job just before this one was born
        else:
            j, d, _ = nearest_job(D, slice(None), jobs[u][b - 1], f["rows"])
            if d > a.min_new_dist:
                groups.setdefault((u, f["si"], b, j), []).append(f)
                continue
            f["kind"] = "too_close"
        events.append(f)
    for (u, si, b, j), fs in groups.items():
        rank = {r: i for i, r in enumerate(jobs[u][si][1].tolist())}
        rws = np.array(sorted({r for f in fs for r in f["rows"].tolist()}, key=rank.get), dtype=int)
        s_main = int(np.bincount(src[rws]).argmax())
        members, prow, _ = jobs[u][b - 1]
        events.append({"u": u, "si": si, "birth": b, "rows": rws, "kind": "dated", "source": s_main,
                       "best_rank": min(f["best_rank"] for f in fs), "n_clusters": len(fs),
                       "history": np.average([f["history"] for f in fs], axis=0,
                                             weights=[len(f["rows"]) for f in fs]),
                       "quiet": fs[0]["quiet"], "parent_rows": prow[members[j]],
                       "test": lineage(a, D, slices, jobs[u][b - 1], rws, s_main, null, b - 1, u)})
    dated_ev = [e for e in events if e["kind"] == "dated"]

    # 4. what became of each dated job's parent (its nearest job in the joined space): its own
    #    history, and the other unit firing most on its contexts at the last checkpoint (the
    #    taker), with how much the taker gained on them since the parent's step. Any coherent
    #    set of contexts has some unit firing on it, so the comparison is collapsed parents'
    #    takers against the takers of parents that stayed.
    H2 = histories(a, toks, lengths, layer, basis, S, data, neurons,
                   [(e["u"], e["birth"] - 1, pos[e["parent_rows"]]) for e in dated_ev], full=True)
    cut_all = []  # every unit's top-k cutoff at each checkpoint
    for s in S:
        with np.load(os.path.join(d_out, f"step{s}.npz")) as f:
            cut_all.append(f["vals"][:, a.k - 1])
    cut_all = np.array(cut_all)
    for e, h, v in zip(dated_ev, H2["hist"], H2["full"] or []):
        share = v / cut_all
        share[:, neurons[e["u"]]] = -np.inf
        o = int(share[-1].argmax())
        e["parent_history"] = h
        e["handoff"] = (o, float(share[-1, o]), float(share[-1, o] - share[e["birth"] - 1, o]))

    # 5. a decoy parent for the blind two-choice test
    rd = np.random.default_rng(a.seed + 7)
    for e in dated_ev:
        v, drows = pick_decoy(D, jobs, N, e["u"], e["birth"] - 1, e["rows"], rd)
        if v is not None:
            e["decoy_unit"], e["decoy_rows"] = v, drows

    def parent_rows(e, k):
        members, prow, _ = jobs[e["u"]][e["birth"] - 1]
        return prow[members[e["test"][k]["parent"]]]

    need = {r for e in events for r in e["rows"][:5].tolist()}
    need |= {r for e in dated_ev for k in slices for r in parent_rows(e, k)[:5].tolist()}
    need |= {r for e in dated_ev for key in ("parent_rows", "decoy_rows") if key in e for r in e[key][:5].tolist()}
    need = np.array(sorted(need), dtype=int)
    text = dict(zip(need.tolist(), window_texts(a, tok, toks, pos[need])[0]))
    out = []
    for e in sorted(events, key=lambda e: (e["kind"] != "dated", e["u"], e["si"])):
        u, b = e["u"], e["birth"]
        r = {"neuron": int(neurons[u]), "step": int(S[e["si"]]), "kind": e["kind"], "birth": int(S[b]),
             "size": len(e["rows"]), "best_rank": e["best_rank"],  # 0 = holds the unit's top activation
             "source": names[int(np.bincount(src[e["rows"]]).argmax())],
             "history": [round(float(x), 3) for x in e["history"]],
             "quiet": [round(float(x), 3) for x in e["quiet"]],
             "examples": [text[x] for x in e["rows"][:5].tolist()], "new_pos": pos[e["rows"]].tolist()}
        if e["kind"] == "dated":
            r.update(parent_step=int(S[b - 1]), stable=bool(start[u] <= b - 1), n_clusters=e["n_clusters"],
                     parent_examples=[text[x] for x in e["parent_rows"][:5].tolist()],
                     parent_pos=pos[e["parent_rows"]].tolist(),
                     parent_history=[round(float(x), 3) for x in e["parent_history"]],
                     handoff={"neuron": e["handoff"][0], "share": round(e["handoff"][1], 3),
                              "gain": round(e["handoff"][2], 3)})
            if "decoy_rows" in e:
                r.update(decoy_neuron=int(neurons[e["decoy_unit"]]),
                         decoy_examples=[text[x] for x in e["decoy_rows"][:5].tolist()])
            for k, t in e["test"].items():
                r[k] = {"d_parent": round(t["d"], 3), "p": None if t["p"] is None else round(t["p"], 3),
                        "label": t["label"], "n_null": t["n_null"],
                        "parent_examples": [text[x] for x in parent_rows(e, k)[:5].tolist()]}
        out.append(r)

    rng = np.random.default_rng(a.seed + 3)
    dated = [r for r in out if r["kind"] == "dated"]
    both = [r for r in dated if all(r[k]["label"] for k in slices)]
    adj = lambda r, k: r[k]["label"] == "adjacent"
    summary = {
        "kinds": {k: sum(r["kind"] == k for r in out) for k in ("dated", "old", "no_parent", "too_close")},
        "births": {str(s): sum(r["birth"] == s for r in out) for s in S},
        **{k: {"counts": label_counts(dated, k), "adjacent_share": adjacent_share(a, rng, dated, k),
               "stable_from_parent": label_counts([r for r in dated if r["stable"]], k)} for k in slices},
        "genre_x_role": {"both": sum(adj(r, "genre") and adj(r, "role") for r in both),
                         "role_only": sum(adj(r, "role") and not adj(r, "genre") for r in both),
                         "genre_only": sum(adj(r, "genre") and not adj(r, "role") for r in both),
                         "neither": sum(not adj(r, "genre") and not adj(r, "role") for r in both)},
        "by_source": {n: {k: label_counts([r for r in dated if r["source"] == n], k) for k in slices}
                      for n in sorted({r["source"] for r in dated})},
        "handoff": handoff_summary(a, dated),
        "max_err_vs_collect": round(max(err, H2["err"]), 6),
    }
    report = {
        "model": a.model, "layer": layer, "basis": basis, "steps": S, "analysis_steps": [S[i] for i in main],
        "args": vars(a), "neurons": neurons.tolist(), "stability": stab, "summary": summary,
        "median_PR_by_step": dict(zip(map(str, S), np.median(PR, 0).round(2).tolist())),
        "run_start": [int(S[i]) for i in start], "PR": PR.round(2).tolist(), "events": out,
    }
    with open(os.path.join(d_out, tagged(a, "report")), "w") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False)
    print(f"L{layer}/{basis}: jobs {summary['kinds']}, births {summary['births']}")
    for k in slices:
        print(f"  {k}: {summary[k]['counts']}, adjacent share {summary[k]['adjacent_share']} "
              f"(parent within the stable run: {summary[k]['stable_from_parent']})")
    print(f"  genre x role: {summary['genre_x_role']}; handoff: {summary['handoff']}; "
          f"max error vs collect {summary['max_err_vs_collect']:.1e} of the cutoff")


# ----------------------------------------------------------------- controls
def nearest(d, n):
    """Indices of the n smallest entries of d, nearest first."""
    idx = np.argpartition(d, n - 1)[:n]
    return idx[np.argsort(d[idx])]


def around(E, taken, row, n):
    """The n contexts nearest context `row`, skipping the taken ones: one coherent planted job."""
    d = 1 - E @ E[row]
    d[taken] = np.inf
    return nearest(d, n)


def planted_test(a, D, slices, jobs, i0, null, own, planted, pools, src):
    """The new job at the last step made mostly of planted contexts, if any, tested against
    the unit's jobs at the step before (plants are born at the last step)."""
    si = len(jobs) - 1
    best, n_best = None, 0
    for rws, _ in detect(a, D, jobs, i0, si, pools):
        n = int(np.isin(rws, planted).sum())
        if 2 * n >= len(rws) and n > n_best:
            best, n_best = rws, n
    if best is None:
        return None
    return lineage(a, D, slices, jobs[si - 1], best, int(np.bincount(src[best]).argmax()), null, si - 1, own)


def score_plants(plants, found, ranks, names):
    """Per test space: detection and false-positive rate of superposition plants, and power of
    lineage plants by rank. A plant counts as detected when its job is found and testable."""
    def rate(x, n):
        return round(x / n, 3) if n else None

    out = {}
    for k in names:
        res = [f[k] if f is not None and f[k]["label"] else None for f in found]
        sup = [t for (kind, _, _, _), t in zip(plants, res) if kind == "superposition"]
        det = [t for t in sup if t is not None]
        s = {"superposition": {"n": len(sup), "detected": rate(len(det), len(sup)),
                               "fpr": rate(sum(t["label"] == "adjacent" for t in det), len(det)),
                               "p_hist_10bins": np.histogram([t["p"] for t in det], bins=10,
                                                             range=(0, 1))[0].tolist()},
             "lineage": {}}
        for r in ranks:
            lin = [t for (kind, lvl, _, _), t in zip(plants, res) if kind == "lineage" and lvl == r]
            det = [t for t in lin if t is not None]
            adj = sum(t["label"] == "adjacent" for t in det)
            s["lineage"][str(r)] = {"n": len(lin), "detected": rate(len(det), len(lin)),
                                    "adjacent_if_detected": rate(adj, len(det)), "power": rate(adj, len(lin))}
        powers = [v["power"] for v in s["lineage"].values() if v["power"] is not None]
        s["mean_power"] = round(float(np.mean(powers)), 3) if powers else 0.0
        out[k] = s
    out["mean_power"] = round(float(np.mean([out[k]["mean_power"] for k in names])), 3)
    return out


def controls(a):
    tok, toks, lengths, sources = load_corpus(a)
    for L in a.layers:
        controls_one(a, tok, toks, lengths, sources, L)


def controls_one(a, tok, toks, lengths, sources, layer):
    """Planted answers in this layer's real context space, scored over the parameter grid.
    No real neuron's lineage label is computed into the output, so the pick is blind."""
    d_out = out_dir(a, layer, "neuron")
    S, neurons, data = load_units(a, d_out, toks, lengths)
    N, last = len(neurons), len(S) - 1
    main = [i for i, s in enumerate(S) if s in a.steps]
    first = max(main[0], len(S) - a.controls_span) if a.controls_span else main[0]
    start, stab = stability(a, S, data)
    i0s = np.maximum(start, first)
    print(f"L{layer}: stable over all steps {stab['full_run']}, run starts {stab['run_start_counts']}")
    cs = context_space(a, tok, toks, lengths, sources, d_out, S, data)
    pos, src = cs["pos"], cs["src"]
    rows = unit_rows(S, data, pos)
    wide = [np.searchsorted(pos, data[S[-1]]["wide"][u]) for u in range(N)]
    # no-change: odd and even ranks of the last checkpoint's top-2k, as if consecutive steps
    split = [(w[1::2], w[0::2]) for w in wide if len(w) == 2 * a.k]

    # plants, built once in the default geometry so every grid point sees the same contexts.
    # The bottom plant_size of a unit's last-step top-k are replaced by a coherent block of
    # contexts none of which the unit ever had. Superposition: seeded anywhere. Lineage:
    # seeded at the r-th nearest free context to one of its jobs at the step before.
    D0, _ = spaces(a, cs)
    rng = np.random.default_rng(a.seed + 2)
    plants = []  # (kind, rank, unit, planted rows)
    for u in range(N):
        if i0s[u] > last - 1:
            continue  # not stable into the last step
        taken = np.zeros(len(pos), dtype=bool)
        taken[np.concatenate(rows[u] + [wide[u]])] = True
        free_rows = np.flatnonzero(~taken)
        for _ in range(a.plant_reps):
            plants.append(("superposition", None, u, around(D0, taken, rng.choice(free_rows), a.plant_size)))
        members, prow, _ = unit_jobs(a, D0, rows[u][last - 1], src)
        for j in np.argsort([-len(m) for m in members], kind="stable")[:a.plant_reps]:
            d = 1 - D0 @ unit_norm(D0[prow[members[j]]].mean(0))
            d[taken] = np.inf
            ranked = nearest(d, max(a.plant_ranks) + 1)
            for r in a.plant_ranks:
                plants.append(("lineage", r, u, around(D0, taken, ranked[r], a.plant_size)))
    del D0
    print(f"  {len(split)} no-change units, {sum(p[0] == 'superposition' for p in plants)} superposition "
          f"and {sum(p[0] == 'lineage' for p in plants)} lineage plants")

    grid, pr = [], {}
    for w in a.grid_sem_weight:
        D, slices = spaces(a, cs, w)
        draw = np.random.default_rng(a.seed + 4)
        rand_sets = [np.searchsorted(pos, draw.choice(cs["random"], a.k, replace=False)) for _ in range(20)]
        pr[str(w)] = {k: {
            "random_contexts": round(float(np.median([participation_ratio(D[x, sl]) for x in rand_sets])), 2),
            "single_token": {tok.decode([t]): round(participation_ratio(D[np.searchsorted(pos, p), sl]), 2)
                             for t, p in cs["tokens"].items()},
            "units_median_by_step": {str(S[si]): round(float(np.median(
                [participation_ratio(D[ru[si], sl]) for ru in rows])), 2) for si in main},
        } for k, sl in slices.items()}
        for ct in a.grid_cluster_thresh:
            b = argparse.Namespace(**{**vars(a), "sem_weight": w, "cluster_thresh": ct})
            jobs = [[unit_jobs(b, D, rows[u][si], src) if si >= first else None for si in range(len(S))]
                    for u in range(N)]
            null = NullPool(jobs, D, slices)
            nc = [[unit_jobs(b, D, o, src), bare(e)] for o, e in split]
            pjobs = [bare(np.concatenate([rows[u][last][:a.k - a.plant_size], pr_rows]))
                     for _, _, u, pr_rows in plants]
            with_jobs = round(float(np.mean([len(jobs[u][last - 1][0]) > 0 for u in range(N)])), 3)
            # pooled clusterings depend on (w, ct) only: reuse them across new_frac
            real_pools, nc_pools = [{} for _ in range(N)], [{} for _ in nc]
            plant_pools = [{} for _ in plants]
            for f in a.grid_new_frac:
                b.new_frac = f
                g = {"sem_weight": w, "cluster_thresh": ct, "new_frac": f, "units_with_jobs": with_jobs}
                # label-blind: how many new jobs real units gain, for scale against no-change
                real = sum(len(detect(b, D, jobs[u], i0s[u], si, real_pools[u]))
                           for u in range(N) for si in range(i0s[u] + 1, len(S)))
                g["real_new_per_transition"] = round(real / max(int((last - i0s).sum()), 1), 3)
                false = sum(len(detect(b, D, nc[i], 0, 1, nc_pools[i])) for i in range(len(nc)))
                g["no_change_new_per_unit"] = round(false / max(len(nc), 1), 3)
                found = [planted_test(b, D, slices, jobs[u][:last] + [pj], i0s[u], null, u, pr_rows, pp, src)
                         for (_, _, u, pr_rows), pj, pp in zip(plants, pjobs, plant_pools)]
                g.update(score_plants(plants, found, a.plant_ranks, slices))
                grid.append(g)
                print(f"  w={w} ct={ct} f={f}: units with jobs {with_jobs}, no-change "
                      f"{g['no_change_new_per_unit']}/unit (real {g['real_new_per_transition']}/transition)  " +
                      "  ".join(f"{k}: fpr {g[k]['superposition']['fpr']} power "
                                f"{[v['power'] for v in g[k]['lineage'].values()]}" for k in slices) +
                      f"  mean {g['mean_power']}")

    rule = (f"most mean lineage power (over both spaces) among settings with superposition fpr <= "
            f"{a.max_fpr} in each space and no-change <= {a.max_false_new} new jobs per unit")
    ok = [g for g in grid if all((g[k]["superposition"]["fpr"] or 0) <= a.max_fpr for k in slices)
          and g["no_change_new_per_unit"] <= a.max_false_new]
    chosen = max(ok, key=lambda g: g["mean_power"]) if ok else None
    report = {"model": a.model, "layer": layer, "steps": S, "args": vars(a), "stability": stab,
              "rule": rule, "chosen": chosen, "pr_baselines": pr, "grid": grid}
    with open(os.path.join(d_out, tagged(a, "controls")), "w") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False)
    print("  PR baselines:", json.dumps(pr))
    if chosen:
        print(f"  chosen ({rule}): --sem-weight {chosen['sem_weight']} "
              f"--cluster-thresh {chosen['cluster_thresh']} --new-frac {chosen['new_frac']}")
    else:
        print(f"  no setting passed ({rule}); see {tagged(a, 'controls')}")


if __name__ == "__main__":
    args = parse()
    {"collect": collect, "controls": controls, "analyze": analyze, "redecoy": redecoy}[args.stage](args)
