# Sensor Model Data Files

This directory contains CSV files defining sensor error models.

## File Structure

Each sensor model consists of **two CSV files**:

1. **`{model_name}.csv`** - Regression parameters for error prediction
2. **`{model_name}_distributions.csv`** - Best-fit distributions for error sampling

## Regression Parameters File

Defines linear regression for predicting error magnitude:

```csv
# Comment lines start with #
error_type,intercept,slope,unit,notes
radial,0.014547,0.000788,meters,Range error towards/away from sensor
lateral,0.023388,0.000279,meters,Cross-range error left/right
yaw,0.045834,-0.000485,radians,Heading angle error
...
```

### Required Error Types

| Error Type | Description | Unit |
|------------|-------------|------|
| `radial` | Range error (towards/away from sensor) | meters |
| `lateral` | Cross-range error (left/right in ground plane) | meters |
| `vertical` | Elevation error (up/down) | meters |
| `yaw` | Heading angle error | radians |
| `width` | Bounding box width error | meters |
| `length` | Bounding box length error | meters |
| `box_height` | Bounding box height error | meters |
| `miss_rate` | Probability of NOT detecting (0-1) | probability |

### Linear Model

```
abs_error = intercept + slope * distance
std = 1.253 * abs_error  (half-normal to std conversion)
```

## Distribution Bins File

Defines best-fit distributions for each distance bin (for sampling actual errors):

```csv
# Format: error_type,dist_min,dist_max,distribution,param1,param2,param3
error_type,dist_min,dist_max,distribution,param1,param2,param3
distal,1,6,logistic,0.0237,0.0824,
distal,6,11,student_t,3.28,-0.0044,0.0658
distal,11,16,normal,0.0070,0.0675,
...
```

### Distribution Types

| Distribution | param1 | param2 | param3 |
|-------------|--------|--------|--------|
| `normal` | μ (mean) | σ (std) | - |
| `laplace` | μ (location) | b (scale) | - |
| `logistic` | μ (location) | s (scale) | - |
| `student_t` | ν (df) | μ (location) | σ (scale) |

### Error Types for Distributions

| Error Type | Description |
|------------|-------------|
| `distal` | Radial/range error |
| `perpendicular` | Lateral/cross-range error |
| `height` | Vertical error |

## Adding a New Sensor

1. Create regression file: `my_sensor.csv`
2. Create distributions file: `my_sensor_distributions.csv`
3. Add to `detector_type.py`:
   ```python
   MY_SENSOR = (7, [...], [...], [...], [...], "my_sensor")
   ```
4. Use in config:
   ```python
   config["detector_types"] = ["MY_SENSOR"]
   ```

## Available Models

- `pointpillars_os1_128.csv` - PointPillars detector with Ouster OS1-128 LiDAR
  - `pointpillars_os1_128_distributions.csv` - Best-fit error distributions