# CMR — Cooperative Monitoring and Recovery

[![tests](https://github.com/eandert/cmr/actions/workflows/tests.yml/badge.svg)](https://github.com/eandert/cmr/actions/workflows/tests.yml)

**CMR** is a simulation framework for cooperative end-to-end monitoring and recovery in connected autonomous vehicle (CAV) networks. It implements two contributions: **GPEM (Generalizable Parameterized Error Model)** — dynamic sensor covariance estimation for V2X cooperative perception that adapts measurement uncertainty to real-time operating conditions — and **SABRE**, an adaptive-NIS gate on top of Covariance Intersection that retains CI's conservative multi-source fusion while rejecting clear outliers.

## Overview

In cooperative perception, multiple vehicles share sensor observations through V2X communication. The quality of these observations varies with factors such as sensor type, vehicle velocity, detection distance, and approach angle. CMR provides:

- **GPEM covariance modeling** — replaces static, hand-tuned covariance matrices with parameterized models learned from sensor characterization data, including polar (angle+distance) error profiles
- **Multi-filter fusion** — comparative evaluation across six Bayesian fusion filters (EKF, Covariance Intersection, Adaptive Kalman, Particle Filter, Batch ICI, and SABRE)
- **SABRE fusion** — adaptive NIS (normalized innovation squared) gating on top of Covariance Intersection: keeps CI's conservative multi-source fusion while rejecting clear outliers
- **Polar error models** — direction-aware detection error and miss rate modeling using 2D (angle × distance) bins from real detector evaluations
- **Trust-based anomaly detection** — per-participant scoring to identify and down-weight degraded or adversarial data sources
- **Multi-scenario evaluation** — six SUMO maps across intersection, city, highway, and rural environments (three full maps plus fast quick-signal variants) for testing generalization across road types
- **Reproducible experimentation** — automated experiment suites with parallel execution and standardized evaluation metrics (AMOTA, AMOTP, HOTA)

## Installation

### Prerequisites

- Python 3.10 (the tested/CI version; 3.11–3.12 not yet validated)
- [SUMO](https://eclipse.dev/sumo/) traffic simulator

### 1. Install SUMO

```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
```

### 2. Set up Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` covers the simulator and error-model pipeline. For the full, version-pinned environment used by the test suite and the real-data pipelines, install `requirements-dev.txt` (a superset) instead.

### 3. Run the tests

```bash
pip install -r requirements-dev.txt
pytest tests/
```

The suite is CPU-only (no GPU/CUDA required). Data- and artifact-gated tests skip automatically on a fresh clone. CI runs it on every push — see [`.github/workflows/tests.yml`](.github/workflows/tests.yml).

## Quick Start

Run a basic simulation:

```bash
source venv/bin/activate
python3 src/traci_interface.py
```

Run a GPEM evaluation experiment:

```bash
python3 src/experiment_runner.py --suite gpem_penetration_sweep --runs 10 --ff-min 6000 --ff-max 12000 -j 8
```

Run across different road environments:

```bash
python3 src/experiment_runner.py --suite gpem_distribution_sweep --runs 10 --ff-min 6000 --ff-max 12000 -j 8 --map city
python3 src/experiment_runner.py --suite gpem_distribution_sweep --runs 10 --ff-min 6000 --ff-max 12000 -j 8 --map highway
```

## Reproducing Paper Results

> The full versioned reproduction guide (clone → every paper-table number, including the real-data evaluations) is being finalized for publication. The SUMO-simulation reproduction is summarized below.

Each experiment runs 10 independent trials per configuration. The simulation timeline for each trial is:

1. **Fast-forward** (`--ff-list`) — SUMO advances the specified number of steps with no sensor processing, allowing traffic to reach steady state. The 10 FF values are evenly spaced across the 36,000-step (3600s) simulation to sample diverse traffic conditions with minimal recording overlap. The explicit list ensures exact reproducibility.
2. **Warmup** (`--warmup 100`) — 100 steps of full sensor/fusion processing to initialize filters, but metrics are not recorded.
3. **Recording** (`--record 6000`) — 6000 steps (~10 minutes at 0.1s step) of full processing with metric logging (AMOTA, AMOTP, HOTA).

Each trial produces 30 synchronized result streams (6 filters × 5 covariance modes) from a single shared SUMO simulation, ensuring fair comparison.

```bash
# Paper parameters for exact reproducibility
FF="3000,5990,8980,11970,14960,17950,20940,23930,26920,29900"  # 10 evenly spaced FF values
SEED=42  # Base random seed (per-run seed = base + run_id)

# Penetration sweep: varies CAV penetration from 1% to 100%
# Tests how GPEM performance scales with the number of connected vehicles
for map in city highway rural; do
  python3 src/experiment_runner.py \
    --suite gpem_penetration_sweep \
    --runs 10 --ff-list $FF --seed $SEED \
    -j 8 --warmup 100 --record 6000 \
    --map $map
done

# Distribution sweep: varies detector/localizer mix from homogeneous to heterogeneous
# Tests GPEM generalization across different sensor fleet compositions
for map in city highway rural; do
  python3 src/experiment_runner.py \
    --suite gpem_distribution_sweep \
    --runs 10 --ff-list $FF --seed $SEED \
    -j 8 --warmup 100 --record 6000 \
    --map $map
done
```

Results and plots are saved to `results/` automatically. To regenerate LaTeX tables:

```bash
python3 scripts/generate_latex_table.py results/<results_dir> --combined
```

### Real-data validation (preliminary)

Beyond the SUMO simulation above, GPEM+SABRE is validated on real cooperative-perception
datasets:

- **DAIR-V2X-Seq (SPD), 3D tracking.** On the same detections through DAIR's own
  `eval_tracking` (3D-IoU 0.25), fully-autotuned GPEM+SABRE scores **MOTA 0.7221 vs the
  late-fusion baseline 0.5486** (+0.174; ID-switches 764 → 36). Every covariance *and*
  lifecycle parameter is derived closed-form from the train-set calibration (no hand-set
  lifecycle). End-to-end reproduction of the off-repo data/detector pipeline lives in a
  separate DAIR-V2X integration repository (not part of this repo).
- **V2V4Real, 2-CAV tracking.** Staged S1–S4. 3D-IoU scoring of dumped tracks uses
  the full 6-DOF `lidar_pose` projection (raw drive required; `--legacy-constant-z`
  for the pre-2026-06-29 number).

## GPEM — Generalizable Parameterized Error Model

Traditional cooperative perception systems use fixed covariance matrices for measurement fusion. GPEM replaces these with parameterized models that capture how sensor error varies with operating conditions.

### Covariance Modes

| Mode | R Matrix (position) | R Matrix (heading) | Description |
|------|--------------------|--------------------|-------------|
| **Baseline** | Weighted avg MSE across all detectors | Fixed (0.1 rad²) | Fleet-averaged covariance, no distance dependence |
| **Static** | Per-detector avg MSE | Per-detector avg yaw MSE | Per-detector average, constant across distance |
| **GPEM Linear** | `bias(d)² + var(d)` via linear regression | `yaw_bias(d)² + yaw_var(d)` | Distance-dependent MSE from linear regression |
| **GPEM Quadratic** | `bias(d)² + var(d)` via quadratic regression | `yaw_bias(d)² + yaw_var(d)` | Distance-dependent MSE from quadratic regression |
| **GPEM Polar** | Smooth polar fit `((a+b·r)(1+c·cos θ+d·cos 2θ))²` | `yaw_bias(d)² + yaw_var(d)` | Range+angle-dependent std from the polar coefficient fit (most granular) |

### How Regressions Are Used

The evaluation pipeline produces five regression types per error dimension (radial, lateral, yaw, etc.):

| Regression | Formula | Used For |
|------------|---------|----------|
| **MAE** | `mae = intercept + slope × d` | Error magnitude reference |
| **Std** | `std = std_intercept + std_slope × d` | Error spread (√variance) |
| **Variance** | `var = var_intercept + var_slope × d` | R matrix diagonal (combined with bias²) |
| **Bias** | `bias = bias_intercept + bias_slope × d` | Systematic offset, contributes to MSE |
| **Bias Std** | `bias_std = ...` | Error sampling distribution: `normal(bias, bias_std)` |

The measurement covariance matrix R is constructed using the **bias-variance decomposition**:

```
R_diagonal = bias(d)² + variance(d) = MSE(d)
```

This captures both the systematic offset (bias) and random spread (variance) of detector errors. At close range where bias ≈ 0, R ≈ variance. At long range where bias grows, R correctly inflates to account for the systematic error that the filter cannot correct.

For heading measurements (yaw), the same MSE principle applies. Previous versions used a hardcoded `0.1 rad²` (~18° std); GPEM modes now use the detector's distance-dependent yaw MSE, which is typically 0.007–0.027 rad² (4–14× tighter).

GPEM models are fit from error characterization data for each detector (DETR3D, BEV Fusion, CenterPoint) and localizer (KISS-ICP, ORB-SLAM3). Polar mode evaluates a smooth per-axis fit `std(r,θ) = (a+b·r)(1+c·cos θ+d·cos 2θ)` for the R matrix; the polar *bins* (21 distance × 36 angle per error type at 100m range) feed the data-driven birth gate, not R.

### Error Model Pipeline

GPEM error models are **fitted** in [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
and imported into `src/data/sensor_models/`. For the **simulator** detectors
(DETR3D, BEVFusion, CenterPoint), the import is one command:

```bash
python3 scripts/import_sim_gpem_models.py            # fit → CMR CSVs (incl. polar_std) + verify
python3 scripts/import_localizer_binned.py           # localizer error models (velocity-binned)
```

**GPEM is a smooth fit, not a bin lookup** — `mode='polar'` reads the `polar_std_*`
coefficients, so detectors are imported through the fitter that emits them
(`import_sim_gpem_models.py` delegates to mmdetection3d's `regression_to_cmr_csv.py`
then appends the static overall bins, and verifies every mode loads). Full
step-by-step (simulation) — manifest, the two-step chain, and verification — lives
in **[docs/IMPORTING_GPEM_MODELS.md](docs/IMPORTING_GPEM_MODELS.md)**. (V2V4Real
detectors are imported by the bucket pipeline below.)

### Rebuilding a detector's GPEM model + birth gate

Each detector lineage lives in an *input bucket* under `data/v2v4real_inputs/ours/detectors/<bucket>/`, registered in `scripts/input_bucket/registry.py`. Rebuilding a bucket's GPEM model is a **single reproducible command**:

```bash
python scripts/build_v2v4real_input_bucket.py \
    --bucket ours/detectors/<bucket> --rebuild all --apply
```

This runs the full chain end-to-end, in order:

1. **import** — `cmr_export/` CSVs → `ab3dmot_detections/` KITTI-MOT `.txt`.
2. **gpem** — wraps mmdet3d's `evaluate_errors.py` (preds + config → error CSVs) and `regression_to_cmr_csv.py` (→ the GPEM `*.csv`, `*_distributions.csv`, `*_polar_distributions.csv`, `*_polar_calibration.csv` in the bucket's `gpem_calibration/`). The `error_analysis/` output is now **preserved** under `gpem_calibration/_error_analysis/<basename>/` (it holds the `polar_binned_errors.csv` the gate step consumes).
3. **gate** — `synthesize_overall_bins.py` appends the count-weighted *overall-bin* rows the static error-model loader requires (regression_to_cmr_csv does not emit them), then **`solve_amota_bayes_gate.py`** — the authoritative AMOTA-Bayes gate + log-odds lifecycle source the production config reads — writes the gate curves (α=96 polar, 97 static, 98 linear, 99 quadratic) and `recommended_lifecycle.csv` into `results/birth_gate_calibration/`. It reads each detector's `polar_binned_errors.csv` from the preserved `_error_analysis/` dir (passed via `--error-analysis-dir <basename>=...`). This step is merge-safe: it updates only the bucket's detectors and preserves every other detector's rows in the shared `birth_gate_curves.csv` / `recommended_gates.csv`. The production log_odds config sets `data_driven_gate_alpha=99.0` (AMOTA-Bayes quadratic) + `data_driven_lifecycle=True` (reads `recommended_lifecycle.csv`); the solver's `--w` FP-cost knob is the lever for later FP-precision tuning. (`derive_optimal_birth_gate.py` remains a separate standalone tp-preserving (α=1/2) tool the production config does *not* read — it is not run by this pipeline.)
4. **manifest / readme** — re-hash the bucket and regenerate its `README.md`.

Steps 3+4 were previously manual and easily skipped; folding them into `--rebuild all` makes the rebuild a single call. Use `--rebuild gpem gate` to refit without touching the manifest, and drop `--apply` for a dry-run that prints every command it would run.

**preds ↔ infos alignment.** `evaluate_errors.py` pairs `preds[i]` to ground truth `ann_info[i]` *positionally*. mmdet3d-native, order-preserving detectors (e.g. CenterPoint) align with the config's default `ann_file` automatically. OpenCOOD-produced detectors (e.g. PointPillar) do **not** — their `GpemSource` must set `ann_file_relpath` to a 1:1-aligned infos pkl, which the refit passes through as `--cfg-options test_dataloader.dataset.ann_file=<path>`. A length-mismatch between preds and the chosen `ann_file` is caught by a fail-hard guard in `evaluate_errors.py` rather than silently aligning to the wrong GT.

## Fusion Filters

CMR evaluates six Bayesian fusion filters. Each GPEM experiment runs all six in parallel via a triple-fusion mechanism — a single SUMO simulation produces 30 result streams (6 filters × 5 covariance modes):

| Filter | Description |
|--------|-------------|
| **EKF** | Extended Kalman Filter with Constant Turn-Rate and Velocity (CTRV) motion model |
| **CI** | Covariance Intersection — conservative multi-source fusion without cross-correlation assumptions |
| **AKF** | Adaptive Kalman Filter — extends EKF with online process noise adaptation via innovation monitoring |
| **PF** | Particle Filter — Sequential Importance Resampling (SIR), 500 particles |
| **BICI** | Batch Inverse Covariance Intersection — N-way simultaneous CI optimization that fuses all measurements at once, rather than pairwise |
| **SABRE** | Source-Adaptive Batch Reliability Estimation — BICI with per-source adaptive-NIS gating that rejects clear outliers while retaining CI's conservatism (the paper's second contribution) |

## Maps

| Map | Description | Size |
|-----|-------------|------|
| `fast_city` | Single signalized 4-way intersection — fastest quick-signal map | Fastest |
| `fast_highway` | Reduced-extent highway segment — fast quick-signal variant | Fast |
| `fast_rural` | Reduced-extent rural segment — fast quick-signal variant | Fast |
| `city` | Tempe, AZ urban grid (2×3 blocks with traffic lights) | Large |
| `highway` | I-10 freeway west of Phoenix (Buckeye area, motorway + side roads) | Medium |
| `rural` | Sparse roads south of Queen Creek, AZ | Large |

## Experiment Suites

### GPEM Evaluation

| Suite | Description |
|-------|-------------|
| `gpem_penetration_sweep` | Sweeps AV penetration rate from 1% to 100% with heterogeneous detector/localizer fleet |
| `gpem_penetration_sweep_accurate` | Same sweep with reduced localizer error (1/4 scale) |
| `gpem_penetration_sweep_static_matching` | Same sweep with static covariance used for Mahalanobis gating |
| `gpem_distribution_sweep` | Sweeps detector/localizer distribution from single-model to balanced multi-model fleet |

#### Penetration Sweep

Varies the connected vehicle penetration rate across 13 levels (1%, 2.5%, 5%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%) with a fixed heterogeneous fleet:
- **Detectors**: 33% BEV Fusion, 33% CenterPoint, 33% DETR3D
- **Localizers**: 50% CT_ICP (LiDAR), 50% ORB_SLAM3 (vision)

#### Distribution Sweep

Sweeps the detector and localizer mix across 8 steps, from homogeneous (100% DETR3D / 100% ORB_SLAM3) to balanced (33/33/33 detectors / 50-50 localizers), testing GPEM generalization across heterogeneous sensor fleets.

### Error Injection and Trust Scoring

| Suite | Description |
|-------|-------------|
| `error_sweep` | Comprehensive sweep of all error types at configurable severity |
| `trust_comparison` | Paired evaluation of trust scoring ON vs OFF for each error type |
| `sensor_comparison` | Baseline vs normal sensor configurations |
| `pointpillars` | PointPillars OS1-128 LiDAR evaluation |
| `error_injection` | Targeted error injection at specific levels |

#### Supported Error Types

| Error Type | Description |
|------------|-------------|
| `SINGLE_SENSOR_EXTRINSICS` | Extrinsic calibration offset on a single sensor |
| `LOCALIZATION` | Degraded ego-vehicle localization accuracy |
| `MALICIOUS_REMOVAL` | Adversarial suppression of valid detections |
| `MALICIOUS_ADDITION` | Adversarial injection of false detections |
| `HIGH_MISS_RATE` | Elevated detection miss rate |
| `DETECTION_LAG` | Temporal delay in detection delivery |

## Command-Line Reference

```
python3 src/experiment_runner.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--suite` | `pointpillars` | Experiment suite to run |
| `--runs` | `10` | Number of independent runs per configuration |
| `--ff-min` / `--ff-max` | — | Randomized fast-forward range (steps; skips sensor processing) |
| `--ff-list` | — | Explicit comma-separated FF steps per run (for exact reproducibility) |
| `--seed` | — | Base random seed for reproducibility (per-run seed = base + run_id) |
| `--warmup` | `600` | Warmup steps (full processing, metrics not recorded) |
| `--record` | `6000` | Recording steps (full processing with metric logging) |
| `-j, --parallel` | `1` | Number of simulations to run concurrently |
| `--output` | `results` | Output directory |
| `--map` | — | SUMO map: `city`, `highway`, or `rural` (plus `fast_*` quick-test variants) |
| `--variance-floor` | `0.0` | Minimum diagonal variance for GPEM covariance matrix |
| `--no-trust-scoring` | — | Disable per-participant trust scoring |
| `--resume` | — | Resume a crashed experiment from existing results directory |

### Examples

```bash
# GPEM penetration sweep with 10 runs, 8 parallel workers
python3 src/experiment_runner.py --suite gpem_penetration_sweep --runs 10 --ff-min 6000 --ff-max 12000 -j 8

# Distribution sweep on highway map
python3 src/experiment_runner.py --suite gpem_distribution_sweep --runs 10 --ff-min 6000 --ff-max 12000 -j 8 --map highway

# Resume a crashed run
python3 src/experiment_runner.py --suite gpem_penetration_sweep --runs 10 --ff-min 6000 --ff-max 18001 -j 8 --resume results/GPEM_Penetration_Sweep_2026-03-18_211028

# Generate LaTeX table from results
python3 scripts/generate_latex_table.py results/GPEM_Distribution_Sweep_2026-03-24_190657 --combined
```

## Output

Results are written to `results/<SuiteName>_<timestamp>/`:

| File | Contents |
|------|----------|
| `summary.json` | Aggregated metrics for all configurations |
| `suite_config.json` | Full experiment configuration for reproducibility |
| `*.png` | Auto-generated comparison plots (per-filter and cross-filter) |
| `<config>/run_XX.csv` | Per-run tracking telemetry |
| `<config>/aggregated.json` | Per-config aggregated statistics |

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **AMOTA** | Average Multi-Object Tracking Accuracy — primary tracking quality metric |
| **AMOTP** | Average Multi-Object Tracking Precision — localization error, lower is better (reported in meters for the SUMO sim; IoU-based for the real-data V2V4Real/DAIR pipeline) |
| **HOTA** | Higher-Order Tracking Accuracy — jointly evaluates detection and association |
| **DetA** | Detection Accuracy — detection quality component of HOTA |
| **AssA** | Association Accuracy — identity association component of HOTA |

## License

MIT — see [LICENSE](LICENSE) for details.
