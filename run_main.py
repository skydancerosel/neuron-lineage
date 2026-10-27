#!/usr/bin/env python
"""
run_main.py

The main run of neuron_lineage.py, with the setting fixed from the pilot's
blind controls: collect every checkpoint for every layer (one pass per
checkpoint), then per layer the planted-answer controls at that one setting (a
validation, not a search) and the analysis on both bases. A step whose output
already exists is skipped, so the run can be restarted. Other Pythia sizes take the
same rules unchanged, in their own output folder.

  python run_main.py
  python run_main.py --model EleutherAI/pythia-1.4b-deduped --out lineage_out_1.4b
"""
import argparse
import os
import subprocess
import sys

LAYERS = [12, 6, 18]  # 12 first: the pilot's layer
STEPS = ["--early-steps", "128", "256", "512",
         "--steps", "1000", "2000", "4000", "8000", "16000", "32000", "64000", "143000"]
UNITS = ["--n-neurons", "512", "--neuron-seed", "2"]  # a sample the pilot never saw
FIXED = ["--sem-weight", "0.5", "--cluster-thresh", "0.65", "--new-frac", "0.75"]
GRID = ["--grid-sem-weight", "0.5", "--grid-cluster-thresh", "0.65", "--grid-new-frac", "0.75"]


def run(*args):
    print(">>", " ".join(args), flush=True)
    subprocess.run([sys.executable, "neuron_lineage.py", *args], check=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="EleutherAI/pythia-410m-deduped")
    p.add_argument("--out", default="lineage_out")
    a = p.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    M = ["--model", a.model, "--out", a.out]
    run("collect", "--layers", *map(str, LAYERS), *STEPS, *M)
    for L in LAYERS:
        if not os.path.exists(f"{a.out}/L{L}/controls_main.json"):
            run("controls", "--layers", str(L), *STEPS, *UNITS, *GRID, "--controls-span", "3", "--tag", "main", *M)
        if not os.path.exists(f"{a.out}/L{L}_rand/report_main.json"):
            run("analyze", "--layers", str(L), *STEPS, *UNITS, *FIXED, "--tag", "main", *M)
