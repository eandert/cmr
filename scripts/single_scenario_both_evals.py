#!/usr/bin/env python3
"""Quick: run ONE scenario through our calibrated pipeline,
report AMOTA under BOTH evaluators (standard + V2V4Real-protocol)."""
import sys
from pathlib import Path
sys.path.insert(0, "src")

from v2v4real_replay import run_scenario

SCEN = Path("/home/rave/test/mmdetection3d/work_dirs/cmr_export_pointpillar_score02/test__Day19__testoutput_CAV_data_2022-03-15-09-54-40_0")

CFG = {
    "detector_name": "pointpillar_v2v4real_score02_{vehicle}",
    "localizer_name": "rtk_v2v4real",
    "score_threshold": 0.30,
    "max_range": 50.0,
    "p_tp_birth_gate": 0.3,
    "lifecycle_mode": "log_odds",
    "use_static_matching": True,
    "self_report_egos": True,
}

print(f"Running {SCEN.name[:55]}...", flush=True)
m = run_scenario(SCEN, CFG)

std = m.get("triple_paper_metrics", {})
v2v = m.get("triple_v2v4real_protocol", {})

streams = ["baseline", "static", "gpem_quadratic",
           "ci_baseline", "ci_gpem_quadratic",
           "akf_gpem_quadratic", "bici_gpem_quadratic",
           "sabre_gpem_quadratic"]

print()
print(f"{'Stream':<28s}  {'Std AMOTA':>10s}  {'V2V Proto AMOTA':>16s}  {'Δ':>6s}")
print("-" * 70)
for s in streams:
    std_amota = std.get(s, {}).get("amota", 0.0) * 100
    v2v_amota = v2v.get(s, {}).get("amota", 0.0) * 100
    delta = v2v_amota - std_amota
    print(f"{s:<28s}  {std_amota:>10.2f}  {v2v_amota:>16.2f}  {delta:>+6.2f}")
print()
print("Std AMOTA = AB3DMOT-nuScenes 2m center-distance gate (FPs counted)")
print("V2V Proto AMOTA = IoU @ 0.25 + ignore_unmatched_fps=True (V2V4Real protocol)")
