#!/usr/bin/env python3
"""Single-scenario lifecycle/gate sweep — fast iteration tool.

Runs ONE V2V4Real scenario through the 30-stream replay pipeline under
several (lifecycle_mode, p_tp_birth_gate) combinations and prints the
merged AMOTA / per-ego AMOTA per stream side-by-side. Designed to take
~30-60s per config (one scenario, 30 streams) for fast iteration on
calibration-related changes — far faster than a full 9-scenario suite.

Usage:
    python scripts/single_scenario_lifecycle_sweep.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from v2v4real_replay import run_scenario  # noqa: E402

# Smallest test scenario — 114 frames, runs in ~30s.
SCENARIO = Path("/home/rave/test/mmdetection3d/work_dirs/cmr_export/test__Day19__testoutput_CAV_data_2022-03-15-10-29-43_3")

CONFIGS = [
    # tape-score decay A/B — does keeping the latest matched det.score
    # across coast frames (no decay) recover AMOTA the same way it does
    # for AB3DMOT? Was suspected from "FPs factored out" insight.
    ("decay 0.5 (prior)",  {"lifecycle_mode": "log_odds", "p_tp_birth_gate": 0.3,
                             "tape_score_miss_decay": 0.5}),
    ("decay 1.0 (no decay)", {"lifecycle_mode": "log_odds", "p_tp_birth_gate": 0.3,
                              "tape_score_miss_decay": 1.0}),
    ("decay 0.7",           {"lifecycle_mode": "log_odds", "p_tp_birth_gate": 0.3,
                             "tape_score_miss_decay": 0.7}),
    ("decay 0.9",           {"lifecycle_mode": "log_odds", "p_tp_birth_gate": 0.3,
                             "tape_score_miss_decay": 0.9}),
]

BASE_CFG = {
    "detector_name": "pointpillar_v2v4real_score02_{vehicle}",
    "localizer_name": "rtk_v2v4real",
    "score_threshold": 0.30,
    "max_range": 50.0,
    "use_static_matching": True,
    "self_report_egos": True,
}

STREAMS = [
    "baseline",          "gpem_linear",      "gpem_quadratic",
    "ci_baseline",       "ci_gpem_quadratic",
    "akf_baseline",      "akf_gpem_quadratic",
    "bici_baseline",     "bici_gpem_quadratic",
    "sabre_baseline",    "sabre_gpem_quadratic",
]


def fmt(x: float) -> str:
    return f"{x*100:>6.2f}"


def main() -> None:
    results = {}
    for label, override in CONFIGS:
        cfg = {**BASE_CFG, **override}
        print(f"  running: {label} ...", flush=True)
        m = run_scenario(SCENARIO, cfg)
        results[label] = m

    # Pull merged + per-ego AMOTA for selected streams.
    def amota(per_stream_metrics, stream_key):
        if stream_key not in per_stream_metrics:
            return 0.0
        return per_stream_metrics[stream_key].get("amota", 0.0)

    print()
    print(f"{'Stream':<30s} | " + " | ".join(f"{lbl:>22s}" for lbl, _ in CONFIGS))
    print("-" * (33 + 25 * len(CONFIGS)))
    for stream in STREAMS:
        row = [stream]
        for label, _ in CONFIGS:
            m = results[label]
            merged  = amota(m.get("triple_paper_metrics", {}),         stream)
            per_ego = amota(m.get("triple_paper_metrics_per_ego", {}), stream)
            row.append(f"  M={fmt(merged)} E={fmt(per_ego)}")
        print(f"{row[0]:<30s} |" + "".join(row[1:]))
    print()
    print("Notation: M = merged AMOTA(%), E = per-ego AMOTA(%) — both ↑ better.")


if __name__ == "__main__":
    main()
