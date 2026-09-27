# STEC-HGB reproducibility package (camera-ready version)

This package accompanies the paper **“STEC: A Topology-Aware Semantic-Temporal Framework for Explainable Alarm Prioritization in Smart Buildings”** (ICDSAIA 2026, paper ID 253). It contains the deterministic synthetic benchmark generator, feature extraction, model training, evaluation, robustness tests, aggregate outputs, and the generated feature tables used in the manuscript.

## Scope and scientific status

- All records are synthetic event metadata; no personal, biometric, video, access-credential, or operational security data are included.
- The benchmark is intended to test the internal logic of cross-system event correlation under controlled site shift, missing events, and timestamp jitter.
- It is not evidence of deployment efficacy. A prospective, multi-building shadow-mode evaluation remains required.
- Random seeds and site-held-out partitions are fixed in `run_experiment.py`.

## Environment

Tested with Python 3.12 and the package versions listed in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
python run_experiment.py
```

The script writes outputs to `results/` by default. To select another directory:

```bash
STEC_OUTPUT_DIR=/path/to/output python run_experiment.py
```

A complete run takes approximately 20–30 seconds on a typical modern CPU. It does not require a GPU or network access.

## Fixed evaluation design

- Training: sites 0–6, 19,600 windows.
- Validation: site 7, 4,800 windows; used only for decision-threshold selection.
- Test: sites 8–11, 11,200 windows.
- Default corruption: 11% event missingness and 7-second timestamp jitter.
- Primary metric: area under the precision-recall curve (AUPRC).
- Secondary metrics: AUROC, precision, recall, F1, Brier score, and false alarms per 1,000 normal windows.
- Uncertainty: stratified bootstrap confidence intervals on the held-out test set.

## Contents

- `run_experiment.py`: benchmark generation, feature extraction, baselines, STEC-HGB, ablations, bootstrap analysis, robustness tests, and plots.
- `results/train_features.csv`, `validation_features.csv`, `test_features.csv`: generated feature tables and labels for the fixed partitions.
- `results/main_results.csv`: primary model comparison.
- `results/ablation_results.csv`: component ablations.
- `results/bootstrap_ci.json`: bootstrap confidence intervals.
- `results/robustness_results.csv`: missing-event and timestamp-jitter stress tests.
- `results/incident_recall.csv`: recall by synthetic incident family.
- `results/permutation_importance.csv`: held-out permutation importance.
- `results/*.png`: plots generated from the tabular outputs.
- `MANIFEST.sha256`: SHA-256 checksums for integrity verification.

## Integrity check

From the package root:

```bash
sha256sum -c MANIFEST.sha256
```

## Expected headline result

With the pinned environment and fixed seeds, `results/main_results.csv` should report an AUPRC close to 0.934 and an F1 close to 0.873 for STEC-HGB on the site-held-out test set. Minor floating-point differences may occur across operating systems or library builds.

## Correction in the camera-ready version

The submitted version of `extract_features` paired event systems and confidences in arrival order with times and zones in system-grouped order inside the pairwise term (Eq. 1 of the paper). This mis-paired events in most incident windows. The corrected code uses one consistent ordering. With the error, STEC-HGB reported 0.912 AUPRC / 0.834 F1 / 11.5 FAR; the corrected implementation gives 0.934 / 0.873 / 9.2. Baselines without pairwise features are unchanged. τ and λ are now named constants (`TAU`, `LAMBDA`).

## Revision experiments (reviewer requests)

```bash
python -m pip install -r requirements.txt     # includes scipy and a CPU build of torch
python revision_experiments.py                # ~45 min on 2 CPU cores; writes revision_results/
python timing.py                              # inference-cost comparison; writes revision_results/timing.json
```

- `E1_site_level.csv`: per-site results and 12-fold leave-one-site-out comparison (Wilcoxon signed-rank test in `revision_summary.json`).
- `E2_sensitivity.csv`: tau, lambda, pair-weight and sequence-template perturbations.
- `E3_generator_shift.csv`: slow/dispersed incidents, unseen incident families, correlated upstream faults, site drift.
- `E3_leave_one_family_out.csv`: recall on each incident family when it is removed from training.
- `E4_gnn_baseline.csv`: Event-GNN learned-graph baseline (with and without zone distance), three seeds.
- `revision_summary.json`: summary including the evidence-path deletion test (E5).
