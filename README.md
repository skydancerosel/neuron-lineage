# neuron-lineage

Do neurons grow new jobs from old ones? A test, on Pythia training checkpoints, of the "recursive generalization" claim in Nick Cammarata's essay [Neural Networks Are Already Clean](https://already-clean.goodfire.com): that a neuron's job sprouts a neighbouring job, which sprouts another, until the ends look unrelated.

## Result

Sometimes, and more often than random directions do, but not often. In Pythia-410m and 1.4b, a neuron's new job sits next to one of its older jobs about 4 points more often than in random directions of the same layer, or about 10 points once corrected for what the test misses. Even so, only about 5–28% of new jobs sprout that way.

| Model | Jobs tested (neurons / random) | Role-adjacent: neurons vs random | p | Genre-adjacent: neurons vs random | p |
|---|---|---|---|---|---|
| Pythia-410m | 4,025 / 3,152 | 10.4% vs 6.5% | 5 × 10⁻⁹ | 6.5% vs 4.8% | 0.002 |
| Pythia-1.4b | 2,278 / 1,088 | 10.1% vs 5.5% | 3 × 10⁻⁶ | 6.0% vs 4.2% | 0.021 |

Pooled over MLP layers 6, 12 and 18, one-sided Fisher p. The difference between the two sizes (+0.8 points, 95% CI −1.6 to 3.1) could be zero.

Three claims from an earlier draft failed follow-up checks:

- **Role, not topic.** Corrected for each test's detection rate, the role and genre excesses are about the same (9.7 and 7.6 points at 410m).
- **Sprouting happens early.** It does, but so does most of everything: 69% of role-adjacent jobs are born at steps 256–2,000, against 72% of all new jobs.
- **Jobs a neuron drops stay hidden in the layer.** The untrained network already reads faded jobs at 0.76 AUC (0.90 trained), and no SAE feature reads them better than the best neuron (0.79 against 0.78).

Everything else (per-layer numbers, planted-answer controls, the blind-reader test, the follow-ups) is in `report/neuron_lineage_report.html` (open it in a browser) and `lineage_out/followup_stats.json`.

## Method

1. **Top contexts.** For 512 sampled neurons per layer, and 512 random directions in the same space as the baseline, the 200 places in a fixed 2M-token Pile sample (`NeelNanda/pile-10k`) where each fires most, at 11 checkpoints from step 128 to 143,000; at most 2 per document.
2. **Jobs.** Each context is embedded in two frozen spaces: *genre* (sentence embedding plus character n-grams) and *role* (GPT-2 medium's layer-12 state at the firing token, plus the next token). A job is a cluster of at least 10 contexts.
3. **New jobs.** A cluster made mostly (75%) of contexts first seen at a checkpoint.
4. **Birth and parent.** The new job's contexts are rerun through every checkpoint to date its birth; its parent is the unit's nearest job at the checkpoint before.
5. **Lineage test.** The parent is *adjacent* if it is closer to the new job than 95% of other units' nearest jobs from the same kind of text.

Settings were fixed on a pilot with a different neuron sample. Planted answers measured the test's false-positive rate and power on every layer.

## Repository

| Path | What it is |
|---|---|
| `neuron_lineage.py` | The pipeline: `collect`, `controls`, `analyze`, `redecoy` |
| `run_main.py` | The main run with the fixed settings, any Pythia size |
| `blind_test.py` | Blind two-choice test: `make` writes items, `score` scores answers |
| `case_study.py` | Traces one neuron's jobs checkpoint by checkpoint |
| `fade_probe.py`, `fade_control.py` | Where faded jobs go: whole-layer probe, best unit, own unit, coherent non-job sets |
| `train_sae.py` | TopK sparse autoencoders on held-out text |
| `fade_followup.py` | Untrained-network probe (`step0`), ablations (`ablate`), SAE match (`sae`) |
| `followup_stats.py` | Power-corrected comparison, birth windows, size comparison, follow-up summaries |
| `make_report.py` | Builds `report/neuron_lineage_report.html` from the result files |
| `lineage_out/`, `lineage_out_1.4b/` | Results for 410m and 1.4b: per layer `report_main.json` (every new job, its parent and labels) and `controls_main.json`; `L*_rand/` is the random-direction baseline |
| `blind_main/` | Blind-test items, answers, key and scores |
| `case_study/` | The neuron 1038 trace (from the pilot run) |

Not included, and rebuilt by the scripts: activations (`*.npz`), SAE weights, model checkpoints and the pilot runs.

## Running it

```bash
pip install -r requirements.txt
python run_main.py
python run_main.py --model EleutherAI/pythia-1.4b-deduped --out lineage_out_1.4b
python fade_probe.py --layers 6 12 18
python fade_control.py --layers 6 12 18
python train_sae.py --layers 6 12 18
python fade_followup.py step0
python fade_followup.py ablate
python fade_followup.py ablate --scope all
python fade_followup.py sae
python followup_stats.py
python make_report.py
```

Checkpoints and the corpus download from the Hugging Face Hub on first use. On Apple MPS the models run in fp32, since fp16 overflows in Pythia's attention output; on CUDA fp16 is the default. On one Apple-silicon Mac with 137 GB of memory, the 410m run took roughly 6 hours, 1.4b roughly 14, and the SAEs about 3. Each model's 11 checkpoints take 18–31 GB of disk, and each main run writes 10–14 GB.

Tested with torch 2.9.0, transformers 4.57.1, datasets 4.5.0, sentence-transformers 5.1.1, scikit-learn 1.5.1, scipy 1.13.1 and numpy 1.26.4.
