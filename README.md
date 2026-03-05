# cmr
Cooperative end-to-end monitoring and recovery for autonomous vehicles.

## Installation

### 1. Install SUMO

```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc
```

### 2. Set up Python virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Basic Simulation

```bash
source venv/bin/activate
python3 src/traci_interface.py
```

### Running Experiments

The experiment runner supports multiple experiment suites with configurable parameters.

```bash
cd src
source ../venv/bin/activate
python experiment_runner.py [OPTIONS]
```

#### Available Suites

| Suite | Description |
|-------|-------------|
| `sensor_comparison` | Compare baseline vs normal sensors |
| `pointpillars` | PointPillars OS1-128 LiDAR vs baseline |
| `error_injection` | Test specific error types at various levels |
| `error_sweep` | Comprehensive sweep of all error types at 1-10% CAV rates |
| `gpem_penetration_sweep` | GPEM vs Static Covariance over 1-100% AV penetration |
| `gpem_distribution_sweep` | GPEM vs Static Covariance over heterogeneous model mix |

#### Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--suite` | `pointpillars` | Experiment suite to run |
| `--runs` | `10` | Number of runs per configuration |
| `--ff-min` / `--ff-max` | None | Randomized Fast Forward range (skips sensor processing) |
| `--warmup` | `600` | Warmup steps (full processing, no logging) |
| `--record` | `6000` | Recording steps (full processing + logging) |
| `-j, --parallel` | `1` | Number of simulations to run concurrently |
| `--output` | `results` | Output directory for results |

#### Examples

**GPEM Penetration Sweep (1-100% AVs):**
```bash
python experiment_runner.py --suite gpem_penetration_sweep --runs 10 --ff-min 6000 --ff-max 12000 -j 16
```

**GPEM Distribution Sweep:**
```bash
python experiment_runner.py --suite gpem_distribution_sweep --runs 10 --ff-min 6000 --ff-max 12000 -j 16
```

**Error sweep with custom range (5-10% CAVs affected):**
```bash
python experiment_runner.py --suite error_sweep --runs 10 --warmup 2000 --record 5000 --error-pct-min 5 --error-pct-max 10
```

### GPEM Experiment Suites

Generalized Parameterized Error Modeling (GPEM) provides dynamic covariance estimation based on real-time factors like object distance and vehicle velocity. These suites compare GPEM against traditional static (averaged) covariance models.

#### 1. Penetration Sweep (`gpem_penetration_sweep`)
Varies the AV penetration rate from 1% to 100% to test how the parameterized model scales.
- **Detector Mix**: 33% BEV_FUSION, 33% CENTERPOINT, 33% DETR3D
- **Localizer Mix**: 50% CT_ICP, 50% ORB_SLAM3
- **Auto-Plot**: Generates `gpem_comparison_plot.png`

#### 2. Distribution Sweep (`gpem_distribution_sweep`)
Sweeps the distribution of error models from 100% legacy models to a balanced split of high-performance models.
- **Purpose**: Tests generalizability across heterogeneous fleets.
- **Auto-Plot**: Generates `gpem_distribution_sweep_plot.png`

### Output & Metrics

Results are saved to `results/<SuiteName>_<timestamp>/`:
- `summary.json` / `summary.csv` - Overall metrics and GPEM improvement stats.
- `*.png` - Automatically generated comparison plots.
- `suite_config.json` - Full configuration for reproducibility.
- `<config_name>/run_XX.csv` - Raw per-run telemetry and tracking errors.

#### Key Metrics
- **AMOTA** - Average Multi-Object Tracking Accuracy (primary tracking metric).
- **TTC Violations** - Safety metric: Time-to-collision threshold violations.
- **MSE Violations** - Safety metric: Minimum Safety Envelope violations.
- **Improvement %** - Statistical delta between Static and GPEM models.
