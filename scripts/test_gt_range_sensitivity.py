#!/usr/bin/env python3
"""Quick test: how does AMOTA shift if we apply KITTI-style range filtering
to GT (and tracker output) post-hoc? Mirrors V2V4Real's likely use of
KITTI min_height filter, which implicitly drops GT cars beyond ~43m.

Re-uses the v2v4real_replay pipeline machinery but evaluates the merged
match tape under several `max_gt_range_m` settings.
"""
import sys, math
from pathlib import Path
sys.path.insert(0, "src")

from v2v4real_replay import run_scenario
from metrics.v2v4real_metrics import compute_ab3dmot_metrics

# Pick all 9 test scenarios
SCEN_ROOT = Path("/home/rave/test/mmdetection3d/work_dirs/cmr_export_pointpillar_score02")
test_scenarios = sorted([d for d in SCEN_ROOT.iterdir()
                         if d.is_dir() and d.name.startswith("test__")])

CONFIG = {
    "detector_name": "pointpillar_v2v4real_score02_{vehicle}",
    "localizer_name": "rtk_v2v4real",
    "score_threshold": 0.30,
    "max_range": 50.0,
    "p_tp_birth_gate": 0.3,
    "lifecycle_mode": "log_odds",
    "use_static_matching": True,
    "self_report_egos": True,
}

# Re-evaluation caps (roughly: KITTI min_height=25 ≈ 43m at typical focal length)
CAPS = [None, 70, 50, 43, 35, 25, 20]

# Streams to inspect
STREAMS = ["akf_gpem_quadratic", "ci_gpem_quadratic", "bici_gpem_quadratic",
           "sabre_gpem_quadratic"]


def filter_tape(tape, ego_pose_per_frame, max_range_m):
    """Drop tracks AND gts beyond max_range_m of any ego in that frame."""
    if max_range_m is None:
        return tape
    out = []
    for f in tape:
        fid = f["frame_idx"]
        egos = ego_pose_per_frame.get(fid, [])
        def in_range(x, y):
            if not egos: return True
            return any(math.hypot(x-ex, y-ey) <= max_range_m for ex, ey in egos)
        out.append({
            "frame_idx": fid,
            "tracks": [t for t in f["tracks"] if in_range(t[1], t[2])],
            "gts":    [g for g in f["gts"]    if in_range(g[1], g[2])],
        })
    return out


def main():
    aggregated = {s: [] for s in STREAMS}
    ego_pose_per_frame_all = {}

    for scen_dir in test_scenarios:
        print(f"  running {scen_dir.name[:55]} ...", flush=True)
        m = run_scenario(scen_dir, CONFIG)
        # Pull the merged match tape per stream
        tapes = m.get("match_tape_per_stream", {})
        # Ego positions per frame for this scenario
        from v2v4real_replay import load_scenario
        s = load_scenario(scen_dir)
        for fid in s.frame_ids:
            ego_xys = []
            for vid, ego in s.frames[fid]["vehicles"].items():
                ego_xys.append((ego["x"], ego["y"]))
            ego_pose_per_frame_all[(scen_dir.name, fid)] = ego_xys

        for stream in STREAMS:
            tape = tapes.get(stream, {}).get("merged", [])
            for entry in tape:
                # Annotate with scenario name so range lookups work
                aggregated[stream].append({
                    "frame_idx": (scen_dir.name, entry["frame_idx"]),
                    "tracks": entry["tracks"],
                    "gts":    entry["gts"],
                })

    print()
    print(f"{'Stream':<28s}  {'no cap':>7s}  {'70m':>6s}  {'50m':>6s}  {'43m':>6s}  {'35m':>6s}  {'25m':>6s}  {'20m':>6s}")
    for stream in STREAMS:
        row = [stream]
        for cap in CAPS:
            tape = filter_tape(aggregated[stream], ego_pose_per_frame_all, cap)
            metrics = compute_ab3dmot_metrics(tape)
            amota = metrics["amota"] * 100
            row.append(f"{amota:6.2f}")
        print(f"{row[0]:<28s}  " + "  ".join(f"{v:>6s}" for v in row[1:]))
    print()
    print("'no cap' = current evaluator (gt_range=100m).")
    print("'43m' ≈ KITTI min_height=25 cutoff at typical 720px focal length.")


if __name__ == "__main__":
    main()
