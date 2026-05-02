#!/usr/bin/env python3
"""Quick A/B: 'AB3DMOT recipe' approximation vs our log_odds pipeline.

Runs ONE scenario through the 30-stream replay under three configs:
  1. our headline:  log_odds + gate 0.3 + GPEM-quad         (current best)
  2. AB3DMOT-ish:   ab3dmot lifecycle + combined NMS 0.15 +  baseline cov +
                    score 0.20 + no P(TP) gate
  3. AB3DMOT-ish + GPEM-static (mid-ground; AB3DMOT lifecycle but our
                    static cov instead of flat 0.5·I)

The headline stream from each is which best-AMOTA stream surfaces; we
read merged AMOTA off `triple_paper_metrics`. Designed as a small
footprint (single core per config sequentially) so it doesn't fight
the training-sweep for compute.

Usage:
    python scripts/single_scenario_ab3dmot_recipe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from v2v4real_replay import run_scenario  # noqa: E402

SCENARIO = Path("/home/rave/test/mmdetection3d/work_dirs/cmr_export/test__Day19__testoutput_CAV_data_2022-03-15-10-29-43_3")

BASE_CFG = {
    "detector_name":      "pointpillar_v2v4real_score02_{vehicle}",
    "localizer_name":     "rtk_v2v4real",
    "max_range":          50.0,
    "use_static_matching": True,
    "self_report_egos":   True,
}

CONFIGS = [
    ("our headline (log_odds, gate 0.3, GPEM)",  {
        **BASE_CFG,
        "score_threshold":   0.30,
        "p_tp_birth_gate":   0.3,
        "lifecycle_mode":    "log_odds",
    }),
    ("AB3DMOT-recipe (baseline cov, NMS 0.15, score 0.20)",  {
        **BASE_CFG,
        "score_threshold":   0.20,
        "p_tp_birth_gate":   None,
        "lifecycle_mode":    "ab3dmot",
        "ab3dmot_min_hits":  3,
        "ab3dmot_max_age":   2,
        "combined_nms_iou":  0.15,
    }),
    ("AB3DMOT-recipe + GPEM-static (NMS 0.15, score 0.20)",  {
        **BASE_CFG,
        "score_threshold":   0.20,
        "p_tp_birth_gate":   None,
        "lifecycle_mode":    "ab3dmot",
        "ab3dmot_min_hits":  3,
        "ab3dmot_max_age":   2,
        "combined_nms_iou":  0.15,
    }),
]

# For each config, we'll surface the relevant stream(s) — for AB3DMOT-recipe
# the headline is `baseline_av_100.0pct` (flat cov ≈ fixed R), for the GPEM
# variants it's the matching cov mode.
FOCUS_STREAMS = ["baseline", "static", "gpem_quadratic",
                 "ci_baseline", "ci_static", "ci_gpem_quadratic",
                 "bici_baseline", "bici_gpem_quadratic",
                 "sabre_baseline", "sabre_gpem_quadratic",
                 # New AB3DMOT-Kalman streams (filter_class=AB3DMOTKalman):
                 "ab3dmot_baseline", "ab3dmot_static", "ab3dmot_gpem_quadratic"]


def fmt(x: float) -> str:
    return f"{x*100:>6.2f}"


def main() -> None:
    results = {}
    for label, cfg in CONFIGS:
        print(f"  running: {label} ...", flush=True)
        m = run_scenario(SCENARIO, cfg)
        results[label] = m

    def amota_merged(m, key):
        return m.get("triple_paper_metrics", {}).get(key, {}).get("amota", 0.0)
    def amota_per_ego(m, key):
        return m.get("triple_paper_metrics_per_ego", {}).get(key, {}).get("amota", 0.0)

    print()
    print(f"{'Stream':<24s} | " + " | ".join(f"{lbl[:30]:>30s}" for lbl, _ in CONFIGS))
    print("-" * (27 + 33 * len(CONFIGS)))
    for stream in FOCUS_STREAMS:
        row = [stream]
        for label, _ in CONFIGS:
            m = results[label]
            mrg = amota_merged(m, stream)
            ego = amota_per_ego(m, stream)
            row.append(f"  M={fmt(mrg)} E={fmt(ego)}")
        print(f"{row[0]:<24s} |" + "".join(row[1:]))
    print()
    print("AB3DMOT recipe target: ~29.28 merged AMOTA.")
    print("M = merged AMOTA(%), E = per-ego AMOTA(%) — both ↑ better.")


if __name__ == "__main__":
    main()
