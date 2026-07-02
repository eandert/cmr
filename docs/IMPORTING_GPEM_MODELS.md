# Importing GPEM Error Models — Simulation

How to import a detector's **fitted** GPEM error model from `mmdetection3d` into
this repo so the SUMO simulator can use it. This page covers the **simulation**
detectors (DETR3D, BEVFusion, CenterPoint). The V2V4Real and DAIR domains follow
the same shape and will be documented here later.

> **Scope split.** *Generating* the fit inputs (running the detector, evaluating
> per-bin + polar errors, writing `regression_models.txt`) lives in the
> **mmdetection3d** repo and is documented there
> (`3detector_error_analysis.md`, `V2V4REAL_TO_CMR_README.md`) — that repo is
> co-published with this one. **This page starts from the point where an
> `error_analysis/` directory already exists** and covers only the CMR-side
> import.

---

## 1. What GPEM ingests, and why it must be a *fit*

GPEM feeds calibrated measurement covariance into the fusion filters. Its whole
premise is a **smooth parameterized fit** of a detector's error as a function of
operating conditions — it transports across conditions, scales, and gives the
Kalman filter continuous, well-conditioned covariance. **It is not a bin lookup.**

At runtime, `mode='polar'` evaluates the smooth polar fit

```
std(r, θ) = (a + b·r) · (1 + c·cos θ + d·cos 2θ)
```

from the `polar_std_a..d` coefficient columns in the detector's main CSV
(`error_model.get_std → predict_std_polar`). If those columns are absent the
model raises `ErrorModelCoverageError` and the sweeps crash (surfacing
confusingly as a TraCI `peer shutdown`). So **every polar-mode detector must carry
`polar_std_*` coefficients** — which means it must be imported through the
converter below, *not* through `import_mmdet_error_models.py` (see §6).

---

## 2. Prerequisites

- `mmdetection3d` checked out as a **sibling** of this repo (`../mmdetection3d`).
- For each detector, an `error_analysis/` directory produced by mmdetection3d's
  `tools/evaluate_errors.py`.

That's all. The polar fit (`*_std_polar:`) is emitted by mmdetection3d's own
pipeline — it is that repo's job to guarantee, not something you hand-check here.
The import step below **verifies every covariance mode loads** (including
`polar`), so a missing fit fails loudly and immediately rather than silently.

---

## 3. The simulation manifest

| CMR detector (`error_model_name`) | mmdetection3d `error_analysis` source | CMR destination |
|-----------------------------------|----------------------------------------|-----------------|
| `detr3d`      | `work_dirs/old/detr3d_100m_full/error_analysis`      | `src/data/sensor_models/detr3d.csv` (+ `_distributions`, `_polar_distributions`) |
| `bev_fusion`  | `work_dirs/old/bevfusion_100m_full/error_analysis`   | `src/data/sensor_models/bev_fusion.csv` (+ …) |
| `centerpoint` | `work_dirs/old/centerpoint_100m_full/error_analysis` | `src/data/sensor_models/centerpoint.csv` (+ …) |

This manifest is encoded in `scripts/import_sim_gpem_models.py` (`SIM_MANIFEST`).
The `*_100m_full` characterisations are the canonical full-range fits and are the
ones that carry the polar std coefficients.

---

## 4. Import — the one command

```bash
python3 scripts/import_sim_gpem_models.py
```

That's it. It imports all three detectors, verifies every covariance mode loads,
and prints a smoke command. Options: `--only detr3d` to import one,
`--dry-run` to print the exact commands without running them.

For each detector it runs the **two-step chain** (this is *exactly* what the one
command does — run these directly if you prefer):

```bash
# Step 1 — fit → CMR schema (emits polar_std_a..d + distance/polar bins)
python3 ../mmdetection3d/tools/regression_to_cmr_csv.py \
    --error-analysis ../mmdetection3d/work_dirs/old/detr3d_100m_full/error_analysis \
    --out-dir src/data/sensor_models \
    --name detr3d

# Step 2 — append the count-weighted static "overall bin" rows the static loader
#          requires (regression_to_cmr_csv does not emit them)
python3 scripts/synthesize_overall_bins.py detr3d
```

Repeat with `bevfusion_100m_full`/`bev_fusion` and
`centerpoint_100m_full`/`centerpoint`. (The script also normalises CRLF→LF, which
the converter emits.)

Each detector produces three files in `src/data/sensor_models/`:

| File | Used for |
|------|----------|
| `{name}.csv` | regression coefficients: linear, quadratic, **`polar_std_*`** (→ R matrix) |
| `{name}_distributions.csv` | distance-binned distributions for error injection + the static overall bin |
| `{name}_polar_distributions.csv` | polar (range × angle) bins for the data-driven birth gate |

> **Consistency note.** Import all three files for a detector from the **same**
> `error_analysis` source. Injection samples come from the bins and the covariance
> comes from the coefficients; if they come from different characterisations the
> injected noise won't match the predicted covariance.

---

## 5. Verify end-to-end

```bash
# (a) every covariance mode loads for every detector
python3 scripts/import_sim_gpem_models.py --only detr3d   # prints "✓ ... all modes load"

# (b) a fast smoke on the intersection map — both sweep types should complete
python3 src/experiment_runner.py --suite gpem_distribution_sweep \
    --runs 1 --map fast_city --warmup 40 --record 80 -j 8 --output results/SMOKE_dist
python3 src/experiment_runner.py --suite gpem_penetration_sweep \
    --runs 1 --map fast_city --warmup 40 --record 80 -j 8 --output results/SMOKE_pen
# expect "Succeeded: 8 / Failed: 0"; the summary.json carries 6 filters × 5 modes,
# including the *_gpem_polar streams.
```

Delete the `results/SMOKE_*` dirs afterwards — they are throwaway.

---

## 6. History: the retired `import_mmdet_error_models.py`

The simulator detectors were once imported by `scripts/import_mmdet_error_models.py`,
which re-parsed the fit files inside this repo and wrote the *bins* but **not** the
`polar_std_*` coefficients — so polar mode had no fit to read and fell back to a bin
lookup. That divergence is what broke the sweeps after the error-model refactor.
It has been **retired to `scripts/old/`**: `import_sim_gpem_models.py` replaces it,
delegating to the single canonical fitter (`regression_to_cmr_csv.py`) so there is
no second, incomplete copy of the conversion logic to drift.

---

## 7. Other domains (later)

- **V2V4Real** — already automated end-to-end by
  `scripts/build_v2v4real_input_bucket.py` (its `gpem` step calls
  `regression_to_cmr_csv.py`, so those detectors already carry polar
  coefficients). A dedicated section will be added here.
- **DAIR** — polar coefficients are not yet generated for the `dair_pp_*` models;
  `run_dair_evaluation.py` currently excludes polar streams. To be done.
