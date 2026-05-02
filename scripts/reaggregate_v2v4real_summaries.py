#!/usr/bin/env python3
"""Re-aggregate summary.json from per-scenario run_01.csv files.

Useful when the aggregator schema was updated AFTER the suite ran but the
per-scenario CSVs already have the new fields written. Doesn't re-run any
pipeline — just walks the result dirs and re-builds summary.json.
"""
import csv, json, sys
from pathlib import Path
from statistics import mean, stdev

REPO = Path(__file__).resolve().parent.parent

# Map from CSV global_metrics fields to summary.json keys.
# Rate metrics that have a matching GT-count weight field — _mean is GT-weighted.
# Other scalars (max_recall_reached) fall back to simple mean.
SCALAR_FIELDS = [
    "paper_amota", "paper_amotp", "paper_samota", "paper_mota",
    "paper_mt", "paper_ml",
    "paper_per_ego_amota", "paper_per_ego_amotp", "paper_per_ego_samota",
    "paper_per_ego_mota", "paper_per_ego_mt", "paper_per_ego_ml",
    "v2v4real_amota", "v2v4real_amotp", "v2v4real_samota",
    "v2v4real_mota", "v2v4real_mt", "v2v4real_ml",
    "v2v4real_max_recall_reached",
    "v2v4real_mg_amota", "v2v4real_mg_amotp", "v2v4real_mg_samota",
    "v2v4real_mg_mota", "v2v4real_mg_max_recall_reached",
    # HOTA components (not GT-weighted; simple mean across scenarios)
    "hota", "deta", "assa",
]
COUNT_FIELDS = [
    "paper_ids_total", "paper_tp_total", "paper_fp_total", "paper_fn_total",
    "paper_gt_total", "paper_n_unique_gts",
    "paper_per_ego_ids_total", "paper_per_ego_tp_total", "paper_per_ego_fp_total",
    "paper_per_ego_fn_total", "paper_per_ego_gt_total",
    "v2v4real_tp_total", "v2v4real_fp_total", "v2v4real_gt_total",
    "v2v4real_mg_tp_total", "v2v4real_mg_fp_total", "v2v4real_mg_gt_total",
]

# Maps scalar field → GT count field used to weight its _mean.
WEIGHT_FOR = {
    "paper_amota":          "paper_gt_total",
    "paper_amotp":          "paper_gt_total",
    "paper_samota":         "paper_gt_total",
    "paper_mota":           "paper_gt_total",
    "paper_mt":             "paper_gt_total",
    "paper_ml":             "paper_gt_total",
    "paper_per_ego_amota":  "paper_per_ego_gt_total",
    "paper_per_ego_amotp":  "paper_per_ego_gt_total",
    "paper_per_ego_samota": "paper_per_ego_gt_total",
    "paper_per_ego_mota":   "paper_per_ego_gt_total",
    "paper_per_ego_mt":     "paper_per_ego_gt_total",
    "paper_per_ego_ml":     "paper_per_ego_gt_total",
    "v2v4real_amota":       "v2v4real_gt_total",
    "v2v4real_amotp":       "v2v4real_gt_total",
    "v2v4real_samota":      "v2v4real_gt_total",
    "v2v4real_mota":        "v2v4real_gt_total",
    "v2v4real_mt":          "v2v4real_gt_total",
    "v2v4real_ml":          "v2v4real_gt_total",
    "v2v4real_mg_amota":    "v2v4real_mg_gt_total",
    "v2v4real_mg_amotp":    "v2v4real_mg_gt_total",
    "v2v4real_mg_samota":   "v2v4real_mg_gt_total",
    "v2v4real_mg_mota":     "v2v4real_mg_gt_total",
}


def _wmean(values, weights):
    tot = sum(weights)
    if tot <= 0:
        return mean(values) if values else 0.0
    return sum(v * w for v, w in zip(values, weights)) / tot


def load_run_csv(path: Path) -> dict:
    """Read run_01.csv (Metric,Value rows) into a flat dict."""
    out = {}
    with open(path) as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) < 2: continue
            k, v = row[0], row[1]
            try: out[k] = float(v)
            except ValueError: out[k] = v
    return out


def reaggregate(run_dir: Path):
    scen_dirs = [d for d in run_dir.iterdir() if d.is_dir() and "_av_" not in d.name and d.name not in ("test","train","val","unknown")]
    splits = {}
    for scen in scen_dirs:
        # Split name from prefix
        prefix = scen.name.split("__", 1)[0]
        splits.setdefault(prefix, []).append(scen)

    for split, scen_list in splits.items():
        per_config: dict = {}
        for scen in scen_list:
            for stream_dir in sorted(scen.iterdir()):
                if not stream_dir.is_dir() or not stream_dir.name.endswith("_av_100.0pct"):
                    continue
                csv_path = stream_dir / "run_01.csv"
                if not csv_path.exists(): continue
                d = load_run_csv(csv_path)
                # Strip the `global_` prefix from CSV column names to match
                # in-memory schema.
                gm = {k.replace("global_", "", 1): v for k, v in d.items()
                      if k.startswith("global_")}
                # Also pick up HOTA fields (stored without prefix, as [0,1] fractions).
                # Multiply by 100 to match the percentage scale of all other metrics.
                for hf in ("hota", "deta", "assa"):
                    if hf in d:
                        gm[hf] = 100.0 * d[hf]
                per_config.setdefault(stream_dir.name, []).append(gm)

        if not per_config:
            continue

        results = {}
        for name, runs in per_config.items():
            n = len(runs)
            agg = {"num_runs": n}
            for f in SCALAR_FIELDS:
                vals = [r.get(f, 0.0) for r in runs]
                if f in WEIGHT_FOR:
                    wts = [r.get(WEIGHT_FOR[f], 0) for r in runs]
                    agg[f"{f}_mean"] = _wmean(vals, wts)
                else:
                    agg[f"{f}_mean"] = mean(vals) if vals else 0.0
                agg[f"{f}_std"] = stdev(vals) if n > 1 else 0.0
            for f in COUNT_FIELDS:
                vals = [int(r.get(f, 0)) for r in runs]
                agg[f] = sum(vals)
            results[name] = agg

        out = {"suite_name": "V2V4Real_Eval_Reaggregated",
               "split": split,
               "total_runs": len(scen_list),
               "results": results}
        out_dir = run_dir / split
        out_dir.mkdir(exist_ok=True)
        out_path = out_dir / "summary_v2v4real_protocol.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"  wrote {out_path}")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        runs = [Path(p) for p in sys.argv[1:]]
    else:
        runs = sorted((REPO / "results").glob("V2V4Real_*_FINAL*"))
    for r in runs:
        if r.is_dir():
            print(f"\n=== {r.name} ===")
            reaggregate(r)
