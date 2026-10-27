#!/usr/bin/env python
"""
followup_stats.py

Two checks on the main run, from its reports and controls alone (no model runs):

  power    Role against genre after correcting each for its own detection rate. A test that
           catches more planted sprouts reports a larger excess for the same true excess, so
           each layer's adjacent share is converted to a true sprouting share,
           f = (observed - false positives) / (power - false positives), before comparing
           neurons with random directions. Bootstrap over units within each layer.
  windows  Adjacent share by birth window. If role-linked jobs are no more concentrated early
           than new jobs in general, "sprouting happens early" only restates "most change
           happens early".

  sizes    The main test, per layer and pooled, with the neuron bootstrap, per-unit rates,
           the corrected sprouting range and the power-corrected excess, for each Pythia size
           run with the same rules (lineage_out = 410m, lineage_out_1.4b = 1.4b).
  fade     Summaries of fade_followup.py's three tests, when their files exist: the probe on
           the untrained network against the last checkpoint, the ablation against its
           size-matched random kick, and the best SAE latent against the best neuron.

  python followup_stats.py        # writes lineage_out/followup_stats.json
"""
import json
import os

import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu, wilcoxon

OUT = "lineage_out"
LAYERS = (6, 12, 18)
WINDOWS = [("256–512", 256, 512), ("1,000–2,000", 1000, 2000), ("4,000–8,000", 4000, 8000), ("16,000+", 16000, 10 ** 9)]


def events(L, basis, out=OUT):
    r = json.load(open(f"{out}/L{L}{'' if basis == 'neuron' else '_rand'}/report_main.json"))
    return [e for e in r["events"] if e["kind"] == "dated" and e["genre"]["label"] and e["role"]["label"]]


def tallies(evs):
    """Per unit: role adjacent, genre adjacent, tested."""
    t = {}
    for e in evs:
        x = t.setdefault(e["neuron"], np.zeros(3))
        x += [e["role"]["label"] == "adjacent", e["genre"]["label"] == "adjacent", 1]
    return np.array(list(t.values()))


def power(n_boot=2000, seed=0, out=OUT):
    rng = np.random.default_rng(seed)
    ctrl = {L: json.load(open(f"{out}/L{L}/controls_main.json"))["grid"][0] for L in LAYERS}
    U = {(L, b): tallies(events(L, b, out)) for L in LAYERS for b in ("neuron", "random")}

    def true_excess(sample, space, which):
        i = {"role": 0, "genre": 1}[space]
        f, w = {"neuron": 0.0, "random": 0.0}, {"neuron": 0.0, "random": 0.0}
        for L in LAYERS:
            c = ctrl[L][space]
            pw = {"mean": c["mean_power"], "low": min(v["power"] for v in c["lineage"].values()),
                  "high": max(v["power"] for v in c["lineage"].values())}[which]
            for b in ("neuron", "random"):
                u = sample[(L, b)]
                obs = u[:, i].sum() / u[:, 2].sum()
                f[b] += (obs - c["superposition"]["fpr"]) / (pw - c["superposition"]["fpr"]) * u[:, 2].sum()
                w[b] += u[:, 2].sum()
        return f["neuron"] / w["neuron"] - f["random"] / w["random"]

    out = {"layers": {}}
    for L in LAYERS:
        out["layers"][L] = {s: {"fpr": ctrl[L][s]["superposition"]["fpr"], "mean_power": ctrl[L][s]["mean_power"],
                                "observed_excess": round(float(U[(L, "neuron")][:, i].sum() / U[(L, "neuron")][:, 2].sum()
                                                               - U[(L, "random")][:, i].sum() / U[(L, "random")][:, 2].sum()), 4)}
                            for i, s in enumerate(("role", "genre"))}
    boots = {"role": [], "genre": []}
    for _ in range(n_boot):
        sample = {k: u[rng.integers(0, len(u), len(u))] for k, u in U.items()}
        for s in boots:
            boots[s].append(true_excess(sample, s, "mean"))
    ci = lambda v: [round(float(np.quantile(v, q)), 4) for q in (0.025, 0.975)]
    pooled = lambda b, i: sum(U[(L, b)][:, i].sum() for L in LAYERS) / sum(U[(L, b)][:, 2].sum() for L in LAYERS)
    for i, s in enumerate(("role", "genre")):
        out[s] = {"observed_excess": round(float(pooled("neuron", i) - pooled("random", i)), 4),
                  "corrected_excess": round(true_excess(U, s, "mean"), 4), "ci": ci(boots[s]),
                  "range_by_power": [round(true_excess(U, s, "high"), 4), round(true_excess(U, s, "low"), 4)]}
    d = np.array(boots["role"]) - np.array(boots["genre"])
    out["role_minus_genre"] = {"corrected": round(out["role"]["corrected_excess"] - out["genre"]["corrected_excess"], 4),
                               "ci": ci(d), "share_boot_above_0": round(float((d > 0).mean()), 3)}
    return out


def windows():
    out = {}
    for b in ("neuron", "random"):
        evs = [e for L in LAYERS for e in events(L, b)]
        rows = []
        for name, lo, hi in WINDOWS:
            w = [e for e in evs if lo <= e["birth"] <= hi]
            role = sum(e["role"]["label"] == "adjacent" for e in w)
            cross = sum(e["role"]["label"] == "adjacent" and e["genre"]["label"] != "adjacent" for e in w)
            rows.append({"window": name, "n": len(w), "role": role, "cross": cross,
                         "role_share": round(role / len(w), 4) if w else None})
        n, role, cross = (sum(r[k] for r in rows) for k in ("n", "role", "cross"))
        early = [r for r in rows if r["window"] in ("256–512", "1,000–2,000")]
        ne, re_, ce = (sum(r[k] for r in early) for k in ("n", "role", "cross"))
        out[b] = {"rows": rows, "early_share": {"all_new": round(ne / n, 3), "role_adjacent": round(re_ / role, 3),
                                                "cross_genre": round(ce / cross, 3)},
                  "early_vs_late_role_rate": {"early": round(re_ / ne, 4), "late": round((role - re_) / (n - ne), 4),
                                              "p_two_sided": round(float(fisher_exact([[re_, ne - re_], [role - re_, n - ne - role + re_]])[1]), 4)}}
    # neuron advantage within each window
    out["excess_by_window"] = [{"window": rn["window"], "neuron": rn["role_share"], "random": rr["role_share"],
                                "p": round(float(fisher_exact([[rn["role"], rn["n"] - rn["role"]], [rr["role"], rr["n"] - rr["role"]]],
                                                              alternative="greater")[1]), 4)}
                               for rn, rr in zip(out["neuron"]["rows"], out["random"]["rows"])]
    return out


SIZES = {"410m": "lineage_out", "1.4b": "lineage_out_1.4b"}


def sizes(n_boot=2000, seed=0):
    out, boots = {}, {}
    for name, d in SIZES.items():
        if not all(os.path.exists(f"{d}/L{L}{s}/report_main.json") for L in LAYERS for s in ("", "_rand")):
            continue
        rng = np.random.default_rng(seed)
        U = {(L, b): tallies(events(L, b, d)) for L in LAYERS for b in ("neuron", "random")}
        rows = {}
        for L in [*LAYERS, "pooled"]:
            Ls = LAYERS if L == "pooled" else (L,)
            r = {}
            for b in ("neuron", "random"):
                u = np.concatenate([U[(l, b)] for l in Ls])
                n = u[:, 2].sum()
                r[b] = {"n": int(n), "role": round(float(u[:, 0].sum() / n), 4), "genre": round(float(u[:, 1].sum() / n), 4)}
            for i, k in enumerate(("role", "genre")):
                a_, b_ = (np.concatenate([U[(l, x)] for l in Ls]) for x in ("neuron", "random"))
                r[f"p_{k}"] = float(fisher_exact([[a_[:, i].sum(), a_[:, 2].sum() - a_[:, i].sum()],
                                                   [b_[:, i].sum(), b_[:, 2].sum() - b_[:, i].sum()]], alternative="greater")[1])
            rows[str(L)] = r
        # neuron bootstrap of the pooled role excess, and per-unit rates (512 units per layer and basis)
        A = {b: np.concatenate([U[(l, b)] for l in LAYERS]) for b in ("neuron", "random")}
        bs = []
        for _ in range(n_boot):
            sh = {b: (lambda x: x[:, 0].sum() / x[:, 2].sum())(u[rng.integers(0, len(u), len(u))]) for b, u in A.items()}
            bs.append(sh["neuron"] - sh["random"])
        units = 512 * len(LAYERS)
        per_unit = {b: {"role_adjacent": round(float(A[b][:, 0].sum() / units), 3),
                        "unparented": round(float((A[b][:, 2].sum() - A[b][:, 0].sum()) / units), 3),
                        "tested": round(float(A[b][:, 2].sum() / units), 3)} for b in A}
        ctrl = [json.load(open(f"{d}/L{L}/controls_main.json"))["grid"][0] for L in LAYERS]
        fpr = [c["role"]["superposition"]["fpr"] for c in ctrl]
        pw = [v["power"] for c in ctrl for v in c["role"]["lineage"].values()]
        corr = {b: {"lo": round(max(0.0, (rows["pooled"][b]["role"] - max(fpr)) / (max(pw) - max(fpr))), 4),
                    "hi": round((rows["pooled"][b]["role"] - min(fpr)) / (min(pw) - min(fpr)), 4)} for b in ("neuron", "random")}
        pc = power(n_boot, seed, d)
        out[name] = {"layers": rows, "boot_role_excess": [round(float(np.median(bs)), 4), round(float(np.quantile(bs, .025)), 4),
                                                          round(float(np.quantile(bs, .975)), 4)],
                     "per_unit": per_unit, "corrected_range": corr,
                     "controls": [{"layer": L, "fpr": {k: c[k]["superposition"]["fpr"] for k in ("genre", "role")},
                                   "mean_power": {k: round(c[k]["mean_power"], 3) for k in ("genre", "role")},
                                   "no_change": c["no_change_new_per_unit"]} for L, c in zip(LAYERS, ctrl)],
                     "power_corrected": {k: pc[k] for k in ("role", "genre", "role_minus_genre")}}
        boots[name] = np.array(bs)
    if len(boots) == 2:  # larger minus smaller, independent bootstraps
        d = boots["1.4b"] - boots["410m"]
        out["difference"] = {"role_excess": round(float(np.median(d)), 4),
                             "ci": [round(float(np.quantile(d, .025)), 4), round(float(np.quantile(d, .975)), 4)],
                             "share_above_0": round(float((d > 0).mean()), 3)}
    return out


def q3(v):
    v = np.asarray(v, dtype=float)
    v = v[np.isfinite(v)]
    return [round(float(np.percentile(v, q)), 4) for q in (25, 50, 75)] if len(v) else None


def fade():
    out = {}
    end = json.load(open(f"{OUT}/fade_probe.json"))["jobs"]
    at_end = lambda j: j["at"][str(j["end"])]
    if os.path.exists(f"{OUT}/fade_step0.json"):
        z = json.load(open(f"{OUT}/fade_step0.json"))
        ctrl = json.load(open(f"{OUT}/fade_control.json"))["sets"]
        res = {}
        for b in ("neuron", "random"):
            for k in ("faded", "stayed", "random"):
                pairs = [(at_end(e), s["step0"]) for e, s in zip(end, z["jobs"]) if e["basis"] == b and e["kind"] == k]
                gain = [x["probe"] - y["probe"] for x, y in pairs if y["probe"] is not None]
                res[f"{b}_{k}"] = {"n": len(gain), **{f"{m}_step0": q3([y[m] for _, y in pairs if y[m] is not None])
                                                       for m in ("probe", "best", "own")},
                                   **{f"{m}_end": q3([x[m] for x, _ in pairs if x[m] is not None]) for m in ("probe", "best", "own")},
                                   "probe_gain": q3(gain), "share_gain_above_0": round(float(np.mean(np.array(gain) > 0)), 3)}
        sg = [c["probe"] - s["step0"]["probe"] for c, s in zip(ctrl, z["sets"]) if s["step0"]["probe"] is not None]
        res["coherent"] = {"n": len(sg), "probe_step0": q3([s["step0"]["probe"] for s in z["sets"]]),
                           "probe_end": q3([c["probe"] for c in ctrl]), "probe_gain": q3(sg)}
        fg = [at_end(e)["probe"] - s["step0"]["probe"] for e, s in zip(end, z["jobs"])
              if e["basis"] == "neuron" and e["kind"] == "faded" and s["step0"]["probe"] is not None]
        res["no_valid_fold"] = sum(s["step0"]["probe"] is None for s in z["jobs"] + z["sets"])
        # at the same starting level: gain ~ step-0 AUC + faded, faded jobs against non-job sets.
        # Faded jobs start higher, so the raw comparison is biased against them and this one,
        # by regression to the mean in a noisy baseline, towards them.
        f0 = [s["step0"]["probe"] for e, s in zip(end, z["jobs"])
              if e["basis"] == "neuron" and e["kind"] == "faded" and s["step0"]["probe"] is not None]
        c0 = [s["step0"]["probe"] for s in z["sets"] if s["step0"]["probe"] is not None]
        A = np.c_[np.ones(len(f0) + len(c0)), np.r_[f0, c0], np.r_[np.ones(len(f0)), np.zeros(len(c0))]]
        g = np.r_[fg, sg]
        coef, *_ = np.linalg.lstsq(A, g, rcond=None)
        r = g - A @ coef
        se = np.sqrt(np.diag(r @ r / (len(g) - 3) * np.linalg.inv(A.T @ A)))
        res["faded_vs_coherent_adjusted"] = {"excess_gain": round(float(coef[2]), 4), "t": round(float(coef[2] / se[2]), 2),
                                             "raw_excess": round(float(np.median(fg) - np.median(sg)), 4)}
        res["faded_vs_coherent_gain_p"] = round(float(mannwhitneyu(fg, sg, alternative="greater").pvalue), 4)
        out["step0"] = res
    for tag in ("ablate", "ablate_all"):
        if not os.path.exists(f"{OUT}/fade_{tag}.json"):
            continue
        items = json.load(open(f"{OUT}/fade_{tag}.json"))["items"]
        res = {}
        for k in ("faded", "stayed", "coherent"):
            it = [i for i in items if i["kind"] == k and "pos" in i]
            real, kick = np.array([i["pos"]["kl_real"] for i in it]), np.array([i["pos"]["kl_kick"] for i in it])
            dr, dk = np.array([i["pos"]["dloss_real"] for i in it]), np.array([i["pos"]["dloss_kick"] for i in it])
            ok = kick > 0
            r = {"n": len(it), "kl_real": q3(real), "kl_kick": q3(kick), "ratio": q3(real[ok] / kick[ok]),
                 "share_real_above_kick": round(float(np.mean(real > kick)), 3),
                 "p_real_vs_kick": round(float(wilcoxon(real, kick, alternative="greater").pvalue), 6),
                 "p_dloss_real_vs_kick": round(float(wilcoxon(dr[np.isfinite(dr + dk)], dk[np.isfinite(dr + dk)],
                                                              alternative="greater").pvalue), 6),
                 "kl_real_neg": q3([i["neg"]["kl_real"] for i in it if "neg" in i]),
                 "dloss_real": q3([i["pos"]["dloss_real"] for i in it]), "dloss_kick": q3([i["pos"]["dloss_kick"] for i in it]),
                 "shift": q3([i["pos"]["shift"] for i in it])}
            if k != "coherent":
                r["kl_own"] = q3([i["pos"]["kl_own"] for i in it])
                r["dloss_own"] = q3([i["pos"]["dloss_own"] for i in it])
            res[k] = r
        ratio = lambda k: [i["pos"]["kl_real"] / i["pos"]["kl_kick"] for i in items
                           if i["kind"] == k and "pos" in i and i["pos"]["kl_kick"] > 0]
        res["p_faded_vs_coherent_ratio"] = round(float(mannwhitneyu(ratio("faded"), ratio("coherent"), alternative="greater").pvalue), 4)
        res["p_faded_vs_stayed_ratio"] = round(float(mannwhitneyu(ratio("faded"), ratio("stayed"), alternative="two-sided").pvalue), 4)
        if tag == "ablate_all":
            for k in ("faded", "stayed", "coherent"):
                it = [i for i in items if i["kind"] == k and "pos" in i]
                dr = np.array([i["pos"]["dspan_real"] for i in it]); dk = np.array([i["pos"]["dspan_kick"] for i in it])
                ok = np.isfinite(dr + dk)
                res[k].update({"dspan_real": q3(dr), "dspan_kick": q3(dk),
                               "p_dspan_real_vs_kick": round(float(wilcoxon(dr[ok], dk[ok], alternative="greater").pvalue), 4)})
                if k != "coherent":
                    res[k]["dspan_own"] = q3([i["pos"]["dspan_own"] for i in it])
        out[tag] = res
    if os.path.exists(f"{OUT}/fade_sae.json"):
        z = json.load(open(f"{OUT}/fade_sae.json"))
        items = z["items"]
        res = {"latents": z["latents"]}
        for b in ("neuron", "random"):
            for k in ("faded", "stayed", "random", "coherent"):
                it = [i for i in items if i["basis"] == b and i["kind"] == k and i["sae"] is not None]
                if not it:
                    continue
                d = np.array([i["sae"] - i["unit"] for i in it])
                res[f"{b}_{k}"] = {"n": len(it), **{m: q3([i[m] for i in it]) for m in ("probe", "unit", "sae", "random_m")},
                                   "sae_minus_unit": q3(d), "p_sae_vs_unit": round(float(wilcoxon(d, alternative="greater").pvalue), 6)
                                   if np.any(d != 0) else None,
                                   "share_sae_ge_0.89": round(float(np.mean([i["sae"] >= 0.89 for i in it])), 3),
                                   "fires_pos": q3([i["sae_fires_pos"] for i in it]), "fires_neg": q3([i["sae_fires_neg"] for i in it])}
        if os.path.exists(os.path.join("lineage_out/sae", "metrics.json")):
            res["quality"] = json.load(open("lineage_out/sae/metrics.json"))["metrics"]
        out["sae"] = res
    return out


def main():
    res = {"power": power(), "windows": windows(), "fade": fade(), "sizes": sizes()}
    with open(f"{OUT}/followup_stats.json", "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
