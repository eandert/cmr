#!/usr/bin/env python3
"""Dump our tuned tracker output to KITTI MOT format and evaluate via the same
path that V2V4Real/AB3DMOT use, for apples-to-apples HOTA / AMOTA comparison.

Why: our run_suite computes HOTA via triple_global_hota in v2v4real_replay.py;
V2V4Real's published 75.59 HOTA comes from compute_hota_iou in
metrics.v2v4real_metrics evaluated on KITTI MOT tracking files. Different
eval paths produce different HOTA. To compare apples-to-apples we must
either run our tracks through the precomputed-tracks evaluator OR run their
tracks through ours. This script does the former.

Pipeline:
  1. Load 9 test scenarios from the cmr_export_cobevt_se export
  2. For each scenario, run the tuned config (lifecycle=log_odds,
     score_thr=0.0) with record_tape_for=[stream] to capture per-frame
     tracks in CMR global UTM frame
  3. Transform each track back to cav-0 (tesla) LiDAR frame, then KITTI
     camera-frame columns (inverse of cobevt_dets_to_cmr_export)
  4. Write 18-col KITTI MOT files (0000.txt … 0008.txt) at <out>/data_0/
  5. Invoke evaluate_precomputed_tracks.py on those files vs the V2V4Real
     official KITTI GT to compute HOTA/AMOTA

Output: stdout with the same metric breakdown evaluate_precomputed_tracks
prints; numbers are directly comparable to AB3DMOT's V2V=38.77, HOTA=75.59.

Usage:
    python scripts/dump_tracks_to_kitti_mot.py \\
        --stream sabre_static_av_100.0pct \\
        --out-dir /tmp/our_kitti_tracks
"""
import argparse
import csv
import glob
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

import v2v4real_replay  # noqa: E402

GT_DIR = REPO / "data" / "v2v4real_inputs" / "baselines" / "paper_gt" / "kitti_labels"
EVAL_SCRIPT = REPO / "scripts" / "evaluate_precomputed_tracks.py"


def _default_export() -> Path:
    """Default --export-dir: legacy cobevt_se export under the developer's
    mmdet3d clone. Resolved lazily so importing this module doesn't require
    paths.local.yaml — only running its CLI does.
    """
    sys.path.insert(0, str(REPO / "src"))
    import local_paths as _LOCAL_PATHS  # noqa: E402
    return _LOCAL_PATHS.get("mmdet3d_root") / "work_dirs/cmr_export_cobevt_se"

# Legacy constant vertical (used only with --legacy-constant-z, kept for
# bit-identical reproduction of the pre-2026-06-29 number). The KITTI MOT line
# is `frame id type 0 0 0 0 0 0 0 h w l X Y Z ry score`; Y is the camera-down
# coordinate of the box CENTER. lidar_z = -1.87 + h/2 = -0.945 lands every box
# at the GT-mean vertical → ~0.68 single-pass 3D-IoU recall on CPft.
#
# This constant was a WORKAROUND, not a fix. The real bug (root-caused
# 2026-06-29, results/GT_MERGE_RTK/Z_6DOF_FIX_PLAN.md): the cmr_export ego pose
# was flattened to yaw-only, so the previous yaw-only global→KITTI projection
# put a range×tilt error (~1° per-frame roll/pitch) into the vertical — which a
# constant can't fix and a naive per-object z made *worse* (it fought the tilt).
# The CPft detector z is in fact excellent (paired residual 3 cm bias / 19 cm
# std vs GT). The default path now uses the FULL raw 6-DOF rotation
# (world_to_kitti_full, validated to <1e-3 m on GT) + the track's real per-object
# z/h → ~0.85 recall. See global_to_kitti().
DEFAULT_H = 1.85
DEFAULT_LIDAR_Z = -1.87

# Raw V2V4Real test split: per-frame `lidar_pose` 4x4 carries the FULL rotation
# (roll/pitch/yaw) the cmr_export dropped. cav0 = tesla = dir "0". Mount the
# drive (e.g. SanDisk @ /media/rave/eddie_drive1) before running the default path.
DEFAULT_RAW_BASE = Path("/media/rave/eddie_drive1/v2v4real/test")


def load_tesla_poses(scenario_dir: Path) -> dict[int, tuple[float, float, float, float]]:
    """Read ego_state.csv → {frame_id: (x, y, z, yaw)} for tesla (cav-0)."""
    poses = {}
    with open(scenario_dir / "ego_state.csv") as f:
        for r in csv.DictReader(f):
            if r["vehicle_id"] != "tesla":
                continue
            poses[int(r["frame_id"])] = (
                float(r["x"]), float(r["y"]), float(r["z"]), float(r["yaw"])
            )
    return poses


def load_cav0_rotation(scen_name: str, raw_base: Path) -> dict[int, "np.ndarray"]:
    """frame -> tesla (cav0) LiDAR→world 3x3 rotation from the raw V2V4Real
    ``lidar_pose`` 4x4. The cmr_export only kept x,y,z,yaw, dropping pitch/roll;
    the full rotation is required to land tracks on the official KITTI vertical
    (a yaw-only projection puts a range×tilt error into Y). Frame index = sorted
    yaml order (matches cmr_export frame_id, verified). Same loader as
    build_merged_gt.load_tesla_rotation — kept local so this script stays runnable
    standalone.
    """
    scen_raw = scen_name.split("__", 2)[2]               # test__Day19__<raw> -> <raw>
    files = sorted(glob.glob(str(raw_base / scen_raw / "0" / "*.yaml")))
    if not files:
        raise SystemExit(
            f"no raw V2V4Real yaml for '{scen_raw}' under {raw_base} — is the drive "
            f"mounted? (use --legacy-constant-z to dump without the 6-DOF fix)")
    out: dict[int, np.ndarray] = {}
    for i, f in enumerate(files):
        d = yaml.load(open(f), Loader=yaml.Loader)        # full loader: numpy-tagged pose
        out[i] = np.asarray(d["lidar_pose"], dtype=float)[:3, :3]
    return out


def global_to_kitti(
    track: tuple, tesla_pose: tuple[float, float, float, float],
    R: "np.ndarray | None" = None,
) -> tuple[float, float, float, float, float, float, float]:
    """Track tuple is (id, x_global, y_global, w, l, yaw_global, score[, height, z]).
    Returns (kitti_x, kitti_y_cam, kitti_z_fwd, kitti_ry, w, l, h) for a KITTI MOT row.

    Inverse of cobevt_dets_to_cmr_export.py. Two modes:

    * **Full 6-DOF (default, R given):** ``lid = R^T (p_world − p_ego)`` with the raw
      LiDAR→world rotation R, then KITTI (X, Y, Z) = (lid_x, lid_z + h/2, lid_y).
      This is `build_merged_gt.world_to_kitti_full` (validated to <1e-3 m vs the
      official paper_gt). Uses the track's real per-object z (idx 8) and height
      (idx 7); falls back to the GT-mean constant only when a track never saw a 3D
      detection (z/height is None). Gate-0 measured the resulting CPft vertical
      residual at 0.15 m std (1% > 0.5 m) → ~0.85 3D-IoU recall.

    * **Legacy (R is None, --legacy-constant-z):** the old yaw-only projection with
      a constant DEFAULT_LIDAR_Z + h/2 vertical → ~0.68 recall. Kept only for
      bit-identical reproduction of the pre-2026-06-29 number; do not use for new
      results (it carries the dropped-tilt artifact described on DEFAULT_LIDAR_Z).
    """
    x_g, y_g, w, l, yaw_g = track[1], track[2], track[3], track[4], track[5]
    cx, cy, cz, cyaw = tesla_pose
    # Per-object height / vertical, with constant fallback when the track never
    # matched a 3D detection (height/z still None on the tape).
    h = track[7] if len(track) > 7 and track[7] is not None else DEFAULT_H
    z_g = track[8] if len(track) > 8 and track[8] is not None else None

    if R is None:
        # Legacy yaw-only + constant vertical.
        cos_y, sin_y = math.cos(cyaw), math.sin(cyaw)
        dx, dy = x_g - cx, y_g - cy
        kitti_x = cos_y * dx + sin_y * dy
        kitti_z_fwd = -sin_y * dx + cos_y * dy
        kitti_y_cam = DEFAULT_LIDAR_Z + h / 2
        yaw_lidar = yaw_g - cyaw
        return (kitti_x, kitti_y_cam, kitti_z_fwd, yaw_lidar, w, l, h)

    # Full 6-DOF inverse. When z is None, use the GT-mean world z so the box still
    # lands at the legacy vertical for that (rare) track; otherwise use the real z.
    z_use = z_g if z_g is not None else (cz + DEFAULT_LIDAR_Z)
    lid = R.T @ np.array([x_g - cx, y_g - cy, z_use - cz])
    kitti_x = lid[0]
    kitti_z_fwd = lid[1]                 # forward distance from cav-0
    kitti_y_cam = (lid[2] + h / 2) if z_g is not None else (DEFAULT_LIDAR_Z + h / 2)
    yaw_lidar = (yaw_g - cyaw + math.pi) % (2 * math.pi) - math.pi
    return (kitti_x, kitti_y_cam, kitti_z_fwd, yaw_lidar, w, l, h)


def write_kitti_mot(
    out_path: Path, tape: list[dict], poses: dict,
    rotations: "dict[int, np.ndarray] | None" = None,
) -> int:
    """Write 18-col KITTI MOT file from match-tape entries.

    KITTI MOT line:
      frame track_id type 0 0 0 0 0 0 0 h w l X Y Z ry score

    rotations: frame → raw 3x3 LiDAR→world rotation for the full 6-DOF projection
    (default path). Pass None for the legacy yaw-only + constant-z projection.
    """
    n_lines = 0
    with open(out_path, "w") as f:
        for entry in tape:
            frame = entry["frame_idx"]
            pose = poses.get(frame)
            if pose is None:
                continue
            R = None if rotations is None else rotations.get(frame)
            if rotations is not None and R is None:
                # No raw pose for this frame — skip rather than mis-project a box
                # with the wrong (or stale) rotation.
                continue
            for track in entry["tracks"]:
                # Full track tuple is (id, x, y, w, l, yaw, score[, height, z]);
                # global_to_kitti reads the trailing vertical fields itself.
                tid, score = int(track[0]), float(track[6])
                kx, ky, kz, kry, kw, kl, kh = global_to_kitti(track, pose, R)
                # Standard 18-col KITTI MOT
                f.write(
                    f"{frame} {tid} Car 0 0 "
                    f"0.000000 0.000000 0.000000 0.000000 0.000000 "
                    f"{kh:.6f} {kw:.6f} {kl:.6f} "
                    f"{kx:.6f} {ky:.6f} {kz:.6f} {kry:.6f} {score:.6f}\n"
                )
                n_lines += 1
    return n_lines


def build_replay_cfg(stream: str, autotune: bool = True) -> dict:
    """Auto-tuned config matching cobevt_dets_cmr_se_dd_lc_static (Tier 1.5).

    autotune=True (default): auto-gate (data_driven_gate_alpha=97.0) +
      auto-lifecycle (data_driven_lifecycle=True). This is the headline
      Tier 1.5 config that gave us 38.80 V2V AMOTA on cobevt.
    autotune=False: legacy manual p_tp_birth_gate=0.25 baseline.
    """
    cfg = {
        "detector_name": "cobevt_tracker_{vehicle}",
        "localizer_name": "rtk_v2v4real",
        "detector_max_range": 100.0,
        "variance_floor": None,
        "score_threshold": 0.0,
        "vehicle_classes": ["car", "truck", "bus", "construction_vehicle"],
        "use_static_matching": True,
        "self_report_egos": False,
        "lifecycle_mode": "log_odds",
        "record_tape_for": [stream],
    }
    if autotune:
        cfg.update({
            "p_tp_birth_gate": 0.5,            # overridden by auto-gate
            "data_driven_gate_alpha": 97.0,    # static fit
            "data_driven_gate_k": 10.0,
            "data_driven_lifecycle": True,     # Tier 1.5
            "per_cov_lifecycle": {"baseline": "ab3dmot"},
        })
    else:
        cfg["p_tp_birth_gate"] = 0.25          # legacy manual default
    return cfg


def build_replay_cfg_from_benchmark(config_name: str, stream: str):
    """Pull cfg + export_dir from run_v2v4real_benchmark.BENCHMARK_CONFIGS.

    Lets us dump any auto-tuned benchmark config (u_pp_cp_lf_dd_*, etc.)
    without re-specifying its params here. Returns (cfg_dict, export_dir).
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_bench", REPO / "scripts" / "run_v2v4real_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    matches = [c for c in mod.BENCHMARK_CONFIGS if c.get("name") == config_name]
    if not matches:
        raise SystemExit(f"config not found in BENCHMARK_CONFIGS: {config_name}")
    bench = matches[0]
    cfg = dict(bench.get("params", {}))
    cfg["record_tape_for"] = [stream]
    if "localizer_name" not in cfg:
        cfg["localizer_name"] = "rtk_v2v4real"
    if "vehicle_classes" not in cfg:
        cfg["vehicle_classes"] = ["car", "truck", "bus", "construction_vehicle"]
    if "use_static_matching" not in cfg:
        cfg["use_static_matching"] = True
    if "self_report_egos" not in cfg:
        cfg["self_report_egos"] = False
    export_dir = Path(bench["export_dir"])
    return cfg, export_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stream", default="baseline",
                    help="Stream key whose tracks to dump. Defaults to "
                         "'baseline' (EKF + uncalibrated cov, no GPEM) "
                         "for direct comparison with AB3DMOT's published baseline. "
                         "Ignored when --streams is given.")
    ap.add_argument("--streams", type=str, default=None,
                    help="Comma-separated stream keys to dump in one replay pass. "
                         "Output goes to <out-dir>/<stream>/data_0/. Eliminates "
                         "the N-passes overhead when comparing many filters.")
    ap.add_argument("--export-dir", type=Path, default=None,
                    help="CMR export dir with test__* scenario subdirs "
                         "(default: paths.local.yaml::mmdet3d_root / "
                         "work_dirs/cmr_export_cobevt_se)")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Output dir for KITTI MOT files (data_0/0000.txt…). "
                         "With --streams, one subdir per stream is created.")
    ap.add_argument("--gt-dir", type=Path, default=GT_DIR,
                    help="V2V4Real KITTI GT dir for evaluation")
    ap.add_argument("--skip-eval", action="store_true",
                    help="Just dump tracks, don't run evaluator")
    ap.add_argument("--no-autotune", action="store_true",
                    help="Use legacy manual p_tp_birth_gate=0.25 instead of Tier 1.5 auto-tune")
    ap.add_argument("--benchmark-config", type=str, default=None,
                    help="Pull cfg + export_dir from BENCHMARK_CONFIGS by this name. "
                         "Lets us dump any auto-tuned config (u_pp_cp_lf_dd_*, etc.).")
    ap.add_argument("--raw-base", type=Path, default=DEFAULT_RAW_BASE,
                    help="Raw V2V4Real test split (per-frame lidar_pose yaml) for the "
                         "full 6-DOF vertical projection. Default: %(default)s")
    ap.add_argument("--legacy-constant-z", action="store_true",
                    help="Use the old yaw-only + constant-z projection (~0.68 CPft "
                         "3D-IoU recall) instead of the 6-DOF fix. For reproducing "
                         "the pre-2026-06-29 number only — carries the dropped-tilt "
                         "vertical artifact.")
    args = ap.parse_args()
    explicit_export_dir = args.export_dir   # explicit --export-dir wins over benchmark default

    # Determine which streams to record/dump. --streams (csv) overrides --stream.
    streams = ([s.strip() for s in args.streams.split(",") if s.strip()]
               if args.streams else [args.stream])

    # When --benchmark-config is given, pull cfg + export_dir from the benchmark
    # module so any auto-tuned config can be dumped. The benchmark's export_dir is
    # only a default — an explicit --export-dir overrides it (e.g. to point at the
    # rehydrated cp_finetune_100m view rather than a stale per-vehicle bucket path).
    if args.benchmark_config:
        cfg, bench_export_dir = build_replay_cfg_from_benchmark(
            args.benchmark_config, streams[0])
        args.export_dir = explicit_export_dir if explicit_export_dir is not None else bench_export_dir
    else:
        cfg = build_replay_cfg(streams[0], autotune=not args.no_autotune)
        if args.export_dir is None:
            args.export_dir = _default_export()
    cfg["record_tape_for"] = list(streams)

    if not args.export_dir.is_dir():
        raise SystemExit(
            f"export dir not found: {args.export_dir}\n"
            f"  (benchmark configs may carry a stale bucket path; pass --export-dir "
            f"explicitly, e.g. data/v2v4real_merged_views/cp_finetune_100m)")

    scenarios = sorted(d for d in args.export_dir.iterdir()
                       if d.is_dir() and d.name.startswith("test__"))
    if len(scenarios) != 9:
        raise SystemExit(f"Expected 9 test scenarios in {args.export_dir}, got {len(scenarios)}")

    # Per-stream output dirs: out_dir/<stream>/data_0/. If only one stream is
    # requested and it equals --stream's default, fall back to single-dir mode
    # for backwards-compat with callers that expect out_dir/data_0/.
    multi_mode = len(streams) > 1 or args.streams is not None
    out_data_per_stream: dict[str, Path] = {}
    for s in streams:
        d = (args.out_dir / s / "data_0") if multi_mode else (args.out_dir / "data_0")
        d.mkdir(parents=True, exist_ok=True)
        out_data_per_stream[s] = d

    print(f"Dumping tracks for stream(s): {streams}")
    print(f"  config: lifecycle={cfg['lifecycle_mode']}, score_thr={cfg['score_threshold']}")
    print(f"  vertical: {'LEGACY yaw-only + constant z' if args.legacy_constant_z else f'full 6-DOF (raw_base={args.raw_base})'}")
    print()

    for seq_idx, scen_dir in enumerate(scenarios):
        seq_id = f"{seq_idx:04d}"
        print(f"  [{seq_idx+1}/9] {scen_dir.name}", end="", flush=True)
        try:
            result = v2v4real_replay.run_scenario(scen_dir, cfg)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue

        poses = load_tesla_poses(scen_dir)
        rotations = (None if args.legacy_constant_z
                     else load_cav0_rotation(scen_dir.name, args.raw_base))
        tapes_by_stream = result.get("match_tape_per_stream", {}) or {}
        counts: list[tuple[str, int]] = []
        for s in streams:
            tape = tapes_by_stream.get(s, [])
            out_path = out_data_per_stream[s] / f"{seq_id}.txt"
            n = write_kitti_mot(out_path, tape, poses, rotations)
            counts.append((s, n))
        summary = " ".join(f"{s}={n}" for s, n in counts)
        print(f"  → {summary}")

    if args.skip_eval:
        if multi_mode:
            print(f"\nDone. Streams at {args.out_dir}/<stream>/data_0/")
            for s in streams:
                print(f"  {s}: {out_data_per_stream[s]}")
        else:
            print(f"\nDone. Tracks at {out_data_per_stream[streams[0]]}")
        return

    if multi_mode:
        raise SystemExit("Per-stream eval not implemented for --streams mode; "
                         "use --skip-eval and run AB3DMOT/precomputed eval "
                         "manually per stream.")
    out_data = out_data_per_stream[streams[0]]

    # Run evaluate_precomputed_tracks.py on the output
    print(f"\nRunning evaluate_precomputed_tracks on {out_data}")
    print("=" * 60)
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--tracking-dir", str(out_data),
        "--gt-dir", str(args.gt_dir),
        "--global-only",   # report aggregate, not per-sequence
    ]
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
