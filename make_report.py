#!/usr/bin/env python
"""
make_report.py

Builds report/neuron_lineage_report.html from the study's result files: the main-run
reports and controls, the blind test, the decoy distances, the neuron 1038 trace and,
when present, the faded-job probe. Every number on the page is read from those files
here and embedded as data; the page draws its charts from that data.

  python make_report.py
"""
import json
import math
import os

import numpy as np
from scipy.stats import fisher_exact

OUT = "lineage_out"
LAYERS = (6, 12, 18)


def wilson(k, n):
    if not n:
        return [0, 0]
    p, z = k / n, 1.96
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(c - h, 4), round(c + h, 4)]


def p_one_sided(a, n, b, m):
    return float(fisher_exact([[a, n - a], [b, m - b]], alternative="greater")[1])


def main_run():
    rows, pooled = [], {b: {"role": 0, "genre": 0, "role_only": 0, "n": 0} for b in ("neuron", "random")}
    units = {"neuron": {}, "random": {}}
    for L in LAYERS:
        row = {"layer": L}
        for b, sub in (("neuron", ""), ("random", "_rand")):
            r = json.load(open(f"{OUT}/L{L}{sub}/report_main.json"))
            s = r["summary"]
            n = sum(s["genre_x_role"].values())
            cnt = {"role": s["role"]["counts"]["adjacent"], "genre": s["genre"]["counts"]["adjacent"],
                   "role_only": s["genre_x_role"]["role_only"]}
            row[b] = {"n": n, "kinds": s["kinds"], **{k: [v, round(v / n, 4), wilson(v, n)] for k, v in cnt.items()}}
            for k, v in cnt.items():
                pooled[b][k] += v
            pooled[b]["n"] += n
            for e in r["events"]:  # per-neuron tallies for the bootstrap
                if e["kind"] == "dated" and e["genre"]["label"] and e["role"]["label"]:
                    g, ro = e["genre"]["label"] == "adjacent", e["role"]["label"] == "adjacent"
                    t = units[b].setdefault((L, e["neuron"]), np.zeros(4))
                    t += [ro, g, ro and not g, 1]
        row["p"] = {k: p_one_sided(row["neuron"][k][0], row["neuron"]["n"], row["random"][k][0], row["random"]["n"])
                    for k in ("role", "genre", "role_only")}
        rows.append(row)
    pool = {"layer": "Pooled"}
    for b in ("neuron", "random"):
        n = pooled[b]["n"]
        pool[b] = {"n": n, **{k: [pooled[b][k], round(pooled[b][k] / n, 4), wilson(pooled[b][k], n)]
                              for k in ("role", "genre", "role_only")}}
    pool["p"] = {k: p_one_sided(pool["neuron"][k][0], pool["neuron"]["n"], pool["random"][k][0], pool["random"]["n"])
                 for k in ("role", "genre", "role_only")}
    rows.append(pool)
    rng = np.random.default_rng(0)
    U = {b: np.array(list(v.values())) for b, v in units.items()}
    boot = {}
    for i, k in enumerate(("role", "genre", "role_only")):
        d = []
        for _ in range(2000):
            sh = {b: (lambda x: x[:, i].sum() / x[:, 3].sum())(u[rng.integers(0, len(u), len(u))]) for b, u in U.items()}
            d.append(sh["neuron"] - sh["random"])
        boot[k] = [round(float(np.median(d)), 4), round(float(np.quantile(d, .025)), 4), round(float(np.quantile(d, .975)), 4)]
    return rows, boot


def controls():
    out = []
    for L in LAYERS:
        g = json.load(open(f"{OUT}/L{L}/controls_main.json"))["grid"][0]
        out.append({"layer": L, **{k: {"fpr": g[k]["superposition"]["fpr"],
                                        "power": [v["power"] for v in g[k]["lineage"].values()]}
                                   for k in ("genre", "role")}, "with_jobs": g["units_with_jobs"]})
    return out


def corrected(rows, ctrl):
    obs = {b: rows[-1][b]["role"][1] for b in ("neuron", "random")}
    fpr = [c["role"]["fpr"] for c in ctrl]
    pw = [p for c in ctrl for p in c["role"]["power"]]
    lo_f, hi_f, lo_p, hi_p = min(fpr), max(fpr), min(pw), max(pw)
    return {b: {"obs": round(o, 4), "lo": round(max(0, (o - hi_f) / (hi_p - hi_f)), 4),
                "hi": round((o - lo_f) / (lo_p - lo_f), 4)} for b, o in obs.items()}


def blind():
    key = json.load(open("blind_main/key.json"))
    ans = {}
    for i in range(1, 7):
        for x in json.load(open(f"blind_main/answers_{i}.json")):
            ans[x["id"]] = x
    out = {}
    for b in ("neuron", "random"):
        for strat in ("adjacent", "no_lineage"):
            sel = [v for i, v in key.items() if v["basis"] == b and v["role"] == strat and i in ans]
            k = sum(ans[i]["choice"] == v["answer"] for i, v in key.items() if v in sel)
            out[f"{b}_{strat}"] = [k, len(sel), wilson(k, len(sel))]
    dd = json.load(open("blind_main/decoy_distance.json"))
    non = [x for x in dd if x["role"] == "no_lineage"]
    closer = [x for x in non if x["dist"]["joined"][1] < x["dist"]["joined"][0]]
    far = [x for x in non if x["dist"]["joined"][1] >= x["dist"]["joined"][0]]
    out["decoy"] = {"n": len(non), "decoy_closer": len(closer),
                    "right_when_parent_closer": [sum(x["correct"] for x in far), len(far)],
                    "right_when_decoy_closer": [sum(x["correct"] for x in closer), len(closer)]}
    return out


def probe():
    path = f"{OUT}/fade_probe.json"
    if not os.path.exists(path):
        return None
    from scipy.stats import mannwhitneyu
    J = json.load(open(path))["jobs"]
    q3 = lambda v: [round(float(np.nanpercentile(v, q)), 3) for q in (25, 50, 75)]
    at = lambda j, when, m: j["at"][str(j[when])][m]
    d_loss = lambda j: at(j, "end", "rel_loss") - at(j, "before", "rel_loss")
    rand = [j for j in J if j["kind"] == "random"]
    out = {"chance": round(float(np.percentile([at(j, w, "probe") for j in rand for w in ("before", "end")], 95)), 3),
           "groups": {}, "loss": {}, "carried": {}, "outcomes": {}, "by_layer": {}, "n": {}}
    c = json.load(open(f"{OUT}/fade_control.json"))["sets"]
    out["control"] = {"probe": q3([x["probe"] for x in c]), "best": q3([x["best"] for x in c]), "n": len(c),
                      "probe95": round(float(np.percentile([x["probe"] for x in c], 95)), 3)}
    for b in ("neuron", "random"):
        for kind in ("faded", "stayed", "random"):
            js = [j for j in J if j["kind"] == kind and j["basis"] == b]
            out["n"][f"{b}_{kind}"] = len(js)
            out["groups"][f"{b}_{kind}"] = {
                when: {m: q3([at(j, when, m) for j in js if j[when] is not None and at(j, when, m) is not None])
                       for m in ("probe", "best", "own", "rel_loss")
                       if any(j[when] is not None and at(j, when, m) is not None for j in js)}
                for when in ("before", "fade", "end") if any(j[when] is not None for j in js)}
        f = [j for j in J if j["kind"] == "faded" and j["basis"] == b]
        s = [j for j in J if j["kind"] == "stayed" and j["basis"] == b]
        df = np.array([d_loss(j) for j in f]); ds = np.array([d_loss(j) for j in s])
        df, ds = df[np.isfinite(df)], ds[np.isfinite(ds)]
        out["loss"][b] = {"faded": q3(df), "stayed": q3(ds), "n": [len(df), len(ds)],
                          "p": round(float(mannwhitneyu(df, ds, alternative="greater").pvalue), 3)}
        # outcome bars, set after seeing the data: one unit carries the job if its best unit reads it as
        # well as a stayed parent's own unit does; only the layer reads it if the probe beats 95% of
        # coherent non-job sets; otherwise the loss rose if it rose more than 90% of stayed parents'
        own_stay = float(np.median([at(j, "end", "own") for j in s]))
        loss_bar = float(np.percentile(ds, 90))
        out["carried"][b] = {"bar": round(own_stay, 3), "loss_bar": round(loss_bar, 3)}
        for kind, js in (("faded", f), ("stayed", s)):
            o = {"carried": 0, "spread": 0, "worse": 0, "same": 0}
            for j in js:
                d = d_loss(j)
                o["carried" if at(j, "end", "best") >= own_stay else "spread" if at(j, "end", "probe") >= out["control"]["probe95"]
                  else "worse" if np.isfinite(d) and d > loss_bar else "same"] += 1
            out["outcomes"][f"{b}_{kind}"] = o
            out["by_layer"][f"{b}_{kind}"] = {L: {m: round(float(np.median([at(j, "end", m) for j in js if j["layer"] == L])), 2)
                                                  for m in ("probe", "best", "own")} for L in LAYERS}
    return out


def main():
    rows, boot = main_run()
    ctrl = controls()
    data = {
        "layers": rows, "boot": boot, "controls": ctrl, "corrected": corrected(rows, ctrl),
        "blind": blind(),
        "n1038": json.load(open("case_study/neuron1038.json")), "probe": probe(),
        "followup": json.load(open(f"{OUT}/followup_stats.json")) if os.path.exists(f"{OUT}/followup_stats.json") else None,
    }
    html = open("report/template.html").read().replace("/*DATA*/null", json.dumps(data))
    if data["probe"] and os.path.exists("report/probe_section.html"):
        html = html.replace("<!--PROBE-->", open("report/probe_section.html").read())
    os.makedirs("report", exist_ok=True)
    with open("report/neuron_lineage_report.html", "w") as fh:
        fh.write(html)
    print("wrote report/neuron_lineage_report.html", "with probe" if data["probe"] else "without probe")
    print(json.dumps({"pooled": rows[-1], "boot": boot, "corrected": data["corrected"], "probe": data["probe"]}, indent=1)[:4000])


if __name__ == "__main__":
    main()
