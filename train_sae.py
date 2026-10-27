#!/usr/bin/env python
"""
train_sae.py

TopK sparse autoencoders (Gao et al. 2024) on the last checkpoint's MLP activations, post-GELU,
the space neurons and random directions live in, one per layer. They test whether the traces
faded jobs leave in a layer are features a dictionary finds (fade_followup.py sae).

Trained on pile-10k documents the study's corpus does not use, every 512-token chunk of each,
so no test context is seen in training. Activations stream from the model into a shuffled
buffer on the device and are never written to disk; only the dictionaries are saved
(fp16, about 0.5 GB per layer at 8x expansion).

  python train_sae.py --layers 6 12 18
"""
import argparse
import json
import os
import time

import numpy as np
import torch

import neuron_lineage as nl


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="EleutherAI/pythia-410m-deduped")
    p.add_argument("--step", type=int, default=143000)
    p.add_argument("--layers", type=int, nargs="+", default=[6, 12, 18])
    p.add_argument("--dataset", default="NeelNanda/pile-10k")
    p.add_argument("--split", default="train")
    p.add_argument("--corpus-docs", type=int, default=3906, help="documents the study's corpus takes; skipped")
    p.add_argument("--ctx", type=int, default=512)
    p.add_argument("--window", type=int, default=24, help="positions before this are left out, as in the corpus")
    p.add_argument("--tokens", type=int, default=12_288_000, help="training tokens per layer (epochs over the data if fewer)")
    p.add_argument("--eval-tokens", type=int, default=250_000)
    p.add_argument("--expansion", type=int, default=8)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--k-aux", type=int, default=512)
    p.add_argument("--dead-tokens", type=int, default=2_000_000, help="a latent unfired this long is dead")
    p.add_argument("--batch", type=int, default=4096)
    p.add_argument("--lr", type=float, default=None, help="default 2e-4 / sqrt(latents / 2^14)")
    p.add_argument("--buffer", type=int, default=500_000, help="tokens per layer held and shuffled at once")
    p.add_argument("--chunk-batch", type=int, default=32)
    p.add_argument("--device", default=None)
    p.add_argument("--out", default="lineage_out/sae")
    p.add_argument("--save-every", type=int, default=500, help="steps between checkpoints (resumed on restart)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def chunks(a, tok):
    """Every ctx-token chunk of the documents after the corpus's (same skip rule as nl.corpus),
    EOS-padded, with their lengths. Cached as a small npz."""
    path = os.path.join(a.out, "chunks.npz")
    if os.path.exists(path):
        with np.load(path) as f:
            return f["toks"], f["lengths"]
    from datasets import load_dataset
    rows, lens, skipped = [], [], 0
    for ex in load_dataset(a.dataset, split=a.split, streaming=True):
        if skipped < a.corpus_docs:
            skipped += len(tok(ex["text"][:10 * a.ctx])["input_ids"][:a.ctx]) > a.window
            continue
        ids = tok(ex["text"])["input_ids"]
        for s in range(0, len(ids), a.ctx):
            c = ids[s:s + a.ctx]
            if len(c) <= a.window:
                break
            rows.append(c + [tok.eos_token_id] * (a.ctx - len(c)))
            lens.append(len(c))
    toks, lengths = np.array(rows, dtype=np.int32), np.array(lens, dtype=np.int32)
    os.makedirs(a.out, exist_ok=True)
    np.savez(path, toks=toks, lengths=lengths)
    return toks, lengths


class TopK(torch.nn.Module):
    def __init__(self, d, m, k):
        super().__init__()
        W = torch.randn(d, m)
        W /= W.norm(dim=0, keepdim=True)
        self.W_enc = torch.nn.Parameter(W.clone())
        self.W_dec = torch.nn.Parameter(W.T.clone())  # rows kept unit norm
        self.b_enc = torch.nn.Parameter(torch.zeros(m))
        self.b_dec = torch.nn.Parameter(torch.zeros(d))
        self.k = k

    def pre(self, x):
        return (x - self.b_dec) @ self.W_enc + self.b_enc

    def codes(self, pre, k):
        v, i = pre.topk(k, dim=-1)
        return torch.zeros_like(pre).scatter_(-1, i, v.relu())

    def encode(self, x):
        return self.codes(self.pre(x), self.k)

    def forward(self, x):
        z = self.encode(x)
        return z @ self.W_dec + self.b_dec, z


def stream(a, model, dev, toks, lengths, order):
    """Yields (n, d) fp16 activation blocks per layer, one block per chunk batch, cycling over
    `order` (chunk indices) forever, valid positions only."""
    c = np.arange(a.ctx)[None]
    while True:
        for b0 in range(0, len(order), a.chunk_batch):
            rb = order[b0:b0 + a.chunk_batch]
            m = torch.as_tensor((c >= a.window) & (c < lengths[rb][:, None]), device=dev)
            with torch.no_grad():
                buf = nl.layer_acts(model, sorted(a.layers), torch.as_tensor(toks[rb], device=dev).long())
            yield {L: buf[L][m].half() for L in a.layers}


def main():
    a = parse()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    from transformers import AutoTokenizer, GPTNeoXForCausalLM

    tok = AutoTokenizer.from_pretrained(a.model)
    os.makedirs(a.out, exist_ok=True)
    toks, lengths = chunks(a, tok)
    n_valid = np.maximum(lengths - a.window, 0)
    n_eval = int(np.searchsorted(np.cumsum(n_valid[::-1]), a.eval_tokens)) + 1
    train_idx, eval_idx = np.arange(len(toks) - n_eval), np.arange(len(toks) - n_eval, len(toks))
    print(f"{len(toks)} chunks, {n_valid.sum():,} usable tokens; {n_valid[eval_idx].sum():,} held out for eval", flush=True)

    dev = nl.device(a)
    model = GPTNeoXForCausalLM.from_pretrained(a.model, revision=f"step{a.step}", dtype=torch.float32).to(dev).eval()
    d = model.gpt_neox.layers[a.layers[0]].mlp.dense_4h_to_h.in_features
    m = d * a.expansion
    lr = float(a.lr or 2e-4 / np.sqrt(m / 2 ** 14))
    src = stream(a, model, dev, toks, lengths, rng.permutation(train_idx))

    # one buffer per layer, allocated once and refilled in place: building it from a list of
    # blocks and concatenating doubles the peak, and the MPS allocator keeps that peak
    buf = {L: torch.empty((a.buffer + a.chunk_batch * a.ctx, d), dtype=torch.float16, device=dev) for L in a.layers}

    def fill():
        n = 0
        while n < a.buffer:
            blk = next(src)
            k = len(blk[a.layers[0]])
            for L in a.layers:
                buf[L][n:n + k] = blk[L]
            n += k
        del blk
        nl.free(dev)
        return n

    def chunked(L, n, fn, rows=65536):
        return sum(fn(buf[L][i:i + rows].float()) for i in range(0, n, rows))

    n_buf = fill()
    sae, opt, scale, since = {}, {}, {}, {}
    ckpt_path = os.path.join(a.out, "ckpt.pt")
    ck = torch.load(ckpt_path, map_location=dev) if os.path.exists(ckpt_path) else None
    for L in a.layers:
        sq = chunked(L, n_buf, lambda x: x.pow(2).sum().item())
        scale[L] = float(np.sqrt(d * n_buf / sq))  # inputs scaled to mean squared norm d
        sae[L] = TopK(d, m, a.k).to(dev)
        with torch.no_grad():
            sae[L].b_dec.copy_(chunked(L, n_buf, lambda x: x.sum(0)) * scale[L] / n_buf)
        opt[L] = torch.optim.Adam(sae[L].parameters(), lr=lr, eps=6.25e-10)
        since[L] = torch.zeros(m, device=dev)
        if ck is not None:
            sae[L].load_state_dict(ck["sae"][L])
            opt[L].load_state_dict(ck["opt"][L])
            scale[L], since[L] = ck["scale"][L], ck["since"][L]
    nl.free(dev)
    n_steps = a.tokens // a.batch
    warm, decay_from = 100, int(0.8 * n_steps)
    print(f"{len(a.layers)} SAEs, {d} -> {m} latents, k={a.k}, lr={lr:.2e}, {n_steps} steps of {a.batch} tokens", flush=True)

    step, t0, log = (ck["step"], time.time(), ck["log"]) if ck is not None else (0, time.time(), [])
    step0 = step
    if ck is not None:
        print(f"resumed from {ckpt_path} at step {step}", flush=True)
    del ck
    while step < n_steps:
        perm = torch.randperm(n_buf, device=dev)
        for b0 in range(0, len(perm) - a.batch + 1, a.batch):
            if step >= n_steps:
                break
            f = min(1.0, (step + 1) / warm) * (1.0 if step < decay_from else (n_steps - step) / (n_steps - decay_from))
            idx = perm[b0:b0 + a.batch]
            rec = {}
            for L in a.layers:
                s = sae[L]
                for g in opt[L].param_groups:
                    g["lr"] = lr * f
                x = buf[L][idx].float() * scale[L]
                pre = s.pre(x)
                z = s.codes(pre, a.k)
                xh = z @ s.W_dec + s.b_dec
                var = (x - x.mean(0)).pow(2).sum(1).mean()
                mse = (xh - x).pow(2).sum(1).mean() / var
                loss = mse
                since[L] += a.batch
                since[L][(z > 0).any(0)] = 0
                dead = since[L] > a.dead_tokens
                if dead.any():  # AuxK: dead latents reconstruct the residual; fixed k, as the MPS
                    # backend compiles and keeps a graph for every new size (live latents score -inf -> 0)
                    za = s.codes(pre.masked_fill(~dead, -torch.inf), a.k_aux)
                    err = (x - xh).detach()
                    loss = loss + ((za @ s.W_dec - err).pow(2).sum(1).mean() / err.pow(2).sum(1).mean()) / 32
                opt[L].zero_grad(set_to_none=True)
                loss.backward()
                opt[L].step()
                with torch.no_grad():
                    s.W_dec /= s.W_dec.norm(dim=1, keepdim=True)
                rec[L] = (1 - mse.item(), int(dead.sum()))
            step += 1
            if step % 100 == 0 or step == n_steps:
                el = time.time() - t0
                msg = " | ".join(f"L{L} fve {v:.3f} dead {dd}" for L, (v, dd) in rec.items())
                print(f"step {step}/{n_steps} {msg} | {el / 60:.1f} min, "
                      f"eta {el / (step - step0) * (n_steps - step) / 60:.0f} min", flush=True)
                log.append({"step": step, **{f"L{L}": v for L, v in rec.items()}})
                nl.free(dev)
            if step % a.save_every == 0 and step < n_steps:
                torch.save({"step": step, "log": log, "sae": {L: sae[L].state_dict() for L in a.layers},
                            "opt": {L: opt[L].state_dict() for L in a.layers}, "scale": scale, "since": since},
                           ckpt_path + ".tmp")
                os.replace(ckpt_path + ".tmp", ckpt_path)
        if step < n_steps:
            n_buf = fill()
    del buf
    nl.free(dev)

    for L in a.layers:  # saved before evaluation, so an evaluation failure loses no training
        torch.save({k: v.detach().half().cpu() for k, v in sae[L].state_dict().items()} | {"scale": scale[L], "k": a.k},
                   os.path.join(a.out, f"L{L}.pt"))
    if step0 < n_steps:
        torch.save({"step": step, "log": log, "sae": {L: sae[L].state_dict() for L in a.layers},
                    "opt": {L: opt[L].state_dict() for L in a.layers}, "scale": scale, "since": since}, ckpt_path)

    # held-out eval: variance explained, dead latents, next-token loss with the SAE spliced in.
    # Shapes stay fixed (full batches; the SAE runs on every position and only valid ones are
    # kept) and the cache is cleared after each batch: the MPS backend keeps memory per shape.
    c = np.arange(a.ctx)[None]
    batches = [eval_idx[b0:b0 + a.chunk_batch] for b0 in range(0, len(eval_idx) - a.chunk_batch + 1, a.chunk_batch)]
    metrics = {}
    for L in a.layers:
        s = sae[L]
        fired = torch.zeros(m, dtype=torch.bool, device=dev)
        mu, n_mu = torch.zeros(d, device=dev), 0
        with torch.no_grad():
            for rb in batches:  # mean activation: the mean-ablation baseline, and the FVE centre
                h = nl.layer_acts(model, [L], torch.as_tensor(toks[rb], device=dev).long())[L]
                ok = torch.as_tensor((c >= a.window) & (c < lengths[rb][:, None]), device=dev)
                mu += (h * ok[..., None]).sum((0, 1))
                n_mu += int(ok.sum())
                nl.free(dev)
        mu /= n_mu
        sums, n_tok, se, tot = {"clean": 0.0, "sae": 0.0, "mean": 0.0}, 0, 0.0, 0.0
        for rb in batches:
            x_in = torch.as_tensor(toks[rb], device=dev).long()
            msk = torch.as_tensor((c >= a.window) & (c < lengths[rb][:, None] - 1), device=dev)
            n_tok += int(msk.sum())
            for mode in ("clean", "sae", "mean"):
                def hook(mod, inp, mode=mode):
                    if mode == "clean":
                        return None
                    h = inp[0]
                    if mode == "sae":
                        xh, z = s(h.reshape(-1, d) * scale[L])
                        fired.logical_or_(((z > 0) & msk.reshape(-1, 1)).any(0))
                        new = (xh / scale[L]).view_as(h)
                    else:
                        new = mu.expand_as(h)
                    return (torch.where(msk[..., None], new, h),)
                hk = model.gpt_neox.layers[L].mlp.dense_4h_to_h.register_forward_pre_hook(hook)
                with torch.no_grad():
                    lp = torch.log_softmax(model(x_in).logits[:, :-1].float(), -1).gather(-1, x_in[:, 1:, None])[..., 0]
                    sums[mode] += float((-lp * msk[:, :-1]).sum())
                hk.remove()
                nl.free(dev)
            with torch.no_grad():
                x = nl.layer_acts(model, [L], x_in)[L] * scale[L]
                xh, _ = s(x.reshape(-1, d))
                ok = torch.as_tensor((c >= a.window) & (c < lengths[rb][:, None]), device=dev).float()
                se += float(((xh.view_as(x) - x).pow(2).sum(-1) * ok).sum())
                tot += float(((x - mu * scale[L]).pow(2).sum(-1) * ok).sum())
            nl.free(dev)
        ce = {k: v / n_tok for k, v in sums.items()}
        metrics[L] = {"fve": round(1 - se / tot, 4), "dead_frac": round(1 - float(fired.float().mean()), 4),
                      "ce": {k: round(v, 4) for k, v in ce.items()},
                      "loss_recovered": round((ce["mean"] - ce["sae"]) / (ce["mean"] - ce["clean"]), 4)}
        print(f"L{L}: {metrics[L]}", flush=True)
    with open(os.path.join(a.out, "metrics.json"), "w") as fh:
        json.dump({"args": vars(a), "metrics": metrics, "log": log}, fh, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
