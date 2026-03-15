# CMR — Cooperative Monitoring and Recovery

**CMR** is a simulation framework for cooperative end-to-end monitoring and recovery in connected autonomous vehicle (CAV) networks. It implements **GPEM (Generalized Parameterized Error Modeling)**, a method for dynamic sensor covariance estimation in V2X cooperative perception that adapts measurement uncertainty to real-time operating conditions.

## Overview

In cooperative perception, multiple vehicles share sensor observations through V2X communication. The quality of these observations varies with factors such as sensor type, vehicle velocity, and detection distance. CMR provides:

- **GPEM covariance modeling** — replaces static, hand-tuned covariance matrices with velocity-dependent models learned from sensor characterization data
- **Multi-filter fusion** — comparative evaluation across four Bayesian fusion filters (EKF, Covariance Intersection, Adaptive Kalman, Particle Filter)
- **Trust-based anomaly detection** — per-participant scoring to identify and down-weight degraded or adversarial data sources
- **Reproducible experimentation** — automated experiment suites with SUMO traffic simulation, parallel execution, and standardized evaluation metrics (AMOTA, AMOTP, HOTA)

## Installation

### Prerequisites

- Python 3.10+
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

## Quick Start

Run a basic simulation:

```bash
source venv/bin/activate
python3 src/traci_interface.py
```

Run a GPEM evaluation experiment:

```bash
cd src
python experiment_runner.py --suite gpem_penetration_sweep --runs 10 --ff-min 6000 --ff-max 12000 -j 16
```

## GPEM — Generalized Parameterized Error Modeling

Traditional cooperative perception systems use fixed covariance matrices for measurement fusion. GPEM replaces these with parameterized models that capture how sensor error varies with operating conditions.

### Covariance Modes

| Mode | Description |
|------|-------------|
| **Baseline** | Default identity-scaled covariance (no error model) |
| **Static** | Averaged covariance from sensor characterization (constant across conditions) |
| **GPEM Linear** | Velocity-dependent covariance via linear regression on per-bin error statistics |
| **GPEM Quadratic** | Velocity-dependent covariance via quadratic regression on per-bin error statistics |

GPEM models are fit from velocity-binned error characterization data for each detector (e.g., DETR3D, BEV_FUSION, CenterPoint) and localizer (e.g., KISS-ICP, ORB-SLAM3). At runtime, the measurement covariance matrix **R** is constructed from the model prediction at the current ego velocity, providing tighter covariance when conditions are favorable and appropriately inflated uncertainty otherwise.

## Fusion Filters

CMR evaluates four Bayesian fusion filters. Each GPEM experiment runs all four in parallel via a triple-fusion mechanism — a single SUMO simulation produces 16 result streams (4 filters x 4 covariance modes):

| Filter | Description |
|--------|-------------|
| **EKF** | Extended Kalman Filter with Constant Turn-Rate and Velocity (CTRV) motion model |
| **CI** | Covariance Intersection — conservative multi-source fusion without cross-correlation assumptions |
| **AKF** | Adaptive Kalman Filter — extends EKF with online process noise adaptation via innovation monitoring |
| **PF** | Particle Filter — Sequential Importance Resampling (SIR), 500 particles |

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
- **Detectors**: 33% BEV_FUSION, 33% CenterPoint, 33% DETR3D
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
python experiment_runner.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--suite` | `pointpillars` | Experiment suite to run |
| `--runs` | `10` | Number of independent runs per configuration |
| `--ff-min` / `--ff-max` | — | Randomized fast-forward range (steps; skips sensor processing) |
| `--warmup` | `600` | Warmup steps (full processing, metrics not recorded) |
| `--record` | `6000` | Recording steps (full processing with metric logging) |
| `-j, --parallel` | `1` | Number of simulations to run concurrently |
| `--output` | `results` | Output directory |
| `--map` | `single` | SUMO map (`single` or `tempe_2x3`) |
| `--variance-floor` | `0.0` | Minimum diagonal variance for GPEM covariance matrix |
| `--no-trust-scoring` | — | Disable per-participant trust scoring |

### Examples

```bash
# GPEM penetration sweep with 10 runs, 16 parallel workers
python experiment_runner.py --suite gpem_penetration_sweep --runs 10 --ff-min 6000 --ff-max 12000 -j 16

# GPEM penetration sweep with accurate localizer models
python experiment_runner.py --suite gpem_penetration_sweep_accurate --runs 10 --ff-min 6000 --ff-max 12000 -j 16

# GPEM distribution sweep
python experiment_runner.py --suite gpem_distribution_sweep --runs 10 --ff-min 6000 --ff-max 12000 -j 16

# Error sweep with custom affected-vehicle range
python experiment_runner.py --suite error_sweep --runs 10 --warmup 2000 --record 5000
```

## Output

Results are written to `results/<SuiteName>_<timestamp>/`:

| File | Contents |
|------|----------|
| `summary.json` | Aggregated metrics for all configurations |
| `summary.csv` | Tabular comparison with baseline improvement statistics |
| `suite_config.json` | Full experiment configuration for reproducibility |
| `*.png` | Auto-generated comparison plots (per-filter and cross-filter) |
| `<config>/run_XX.csv` | Per-run tracking telemetry |

### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **AMOTA** | Average Multi-Object Tracking Accuracy — primary tracking quality metric |
| **AMOTP** | Average Multi-Object Tracking Precision — localization error in meters (lower is better) |
| **HOTA** | Higher-Order Tracking Accuracy — jointly evaluates detection and association |
| **DetA** | Detection Accuracy — detection quality component of HOTA |
| **AssA** | Association Accuracy — identity association component of HOTA |

## License

MIT — see [LICENSE](LICENSE) for details.
