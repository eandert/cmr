"""Convert a detector bucket's cmr_export/ into AB3DMOT per-CAV KITTI .txt files.

Each detector bucket has::

    <bucket>/cmr_export/
        astuff/<scenario>/{detections.csv, ego_state.csv, gt_objects.csv, meta.yaml}
        tesla/<scenario>/...    (same structure)

This module writes::

    <bucket>/ab3dmot_detections/
        {seq}_astuff.txt          KITTI MOT detections in TESLA's KITTI frame
                                  (mount height = astuff's, because the rays
                                  originate from astuff's LiDAR)
        {seq}_tesla.txt           same, mount height = tesla's
        {seq}_poses.csv           per-(frame, source) RTK world poses
        {seq}_{src}_ranges.csv    per-row source-local Euclidean range, used
                                  by GPEM R lookup for sensor-correct std.

**Coordinate frame**: detections land in TESLA's KITTI frame (the "common
frame" convention AB3DMOT and the GT both use). This is required so the
tracker can associate detections across CAVs and so the scorer can match
tracker output against GT — both also live in tesla's frame.

**Per-source range for GPEM**: each detection's source-local range
(distance from the producing CAV's pose to the target's world position) is
written in the ``_ranges.csv`` side-car, NOT computed from the detection's
own ``(x, z)``. The GPEM injector should read the side-car to get the
correct per-source range for noise lookup. Until that injector fix lands
(see the data-pipeline TODO), GPEM falls back to tesla-frame range for
non-tesla detections — silently wrong for those, but at least the tracker
itself associates correctly.

Single-vehicle buckets (e.g. cp_zeroshot_100m has tesla only) just produce
``{seq}_tesla.txt`` files.

Reuses geometry helpers from ``augment_v2v4real_with_cav_self_reports``.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Iterable

import yaml

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "configs"))

from augment_v2v4real_with_cav_self_reports import (  # noqa: E402
    SELF_REPORT_SCORE,
    emit_kitti_det_row,
    read_poses,
    world_to_local_kitti,
)
from v2v4real_seq_map import SCENARIO_TO_SEQ, SEQ_TO_NUM_FRAMES  # noqa: E402

# Drop detections whose source-local Euclidean range exceeds this. 142m is
# the diagonal of the V2V4Real point-cloud grid; anything beyond is an
# inter-CAV-offset artifact (see augment_v2v4real_per_cav_local.py docstring).
MAX_SOURCE_LOCAL_RANGE_M = 142.0

SUPPORTED_VEHICLES = ("astuff", "tesla")


# Per-CAV cmr_export convention (verified June 2026): the upstream exporter
# writes one cmr_export tree per producing CAV (e.g. cmr_export_pp_astuff
# contains ONLY astuff's detections). Each tree's detections.csv is in
# "centered world" frame — the upstream exporter subtracts the SCENARIO
# CENTROID of that tree's vehicle from every position. CRITICAL: the centroid
# differs per tree (pp_astuff centers on astuff's pose centroid, pp_tesla
# centers on tesla's). The 2026-06-08 coord audit caught a (+54.35, -60.75)
# astuff-vs-tesla centroid delta that propagated straight into bucket
# coordinates and tanked AMOTA from ~47 to 2.23 on identical-tracker A/B.
#
# We rehydrate to a SHARED real-world frame on read (just add the tree's
# center_world back to every x/y/z) before mixing dets/poses across trees;
# then world_to_local_kitti operates on a coherent frame.


def _load_center_world(scen_dir: Path) -> tuple[float, float, float]:
    """Read the scenario centroid the upstream exporter subtracted from every
    coordinate in this tree. Raises if meta.yaml is missing or malformed —
    silent fallback would mask the very coordinate-drift bug this function exists
    to prevent.
    """
    meta_path = scen_dir / "meta.yaml"
    if not meta_path.is_file():
        raise ImportError_(
            f"missing meta.yaml at {meta_path}\n"
            f"  Fix: this cmr_export tree didn't run through the v0.4+ "
            f"exporter. Re-run mmdetection3d/tools/export_v2v4real_to_cmr.py "
            f"with the up-to-date exporter (writes center_world to meta.yaml)."
        )
    with open(meta_path) as f:
        meta = yaml.safe_load(f)
    cw = meta.get("center_world")
    if cw is None or len(cw) != 3:
        raise ImportError_(
            f"{meta_path} missing or invalid 'center_world' key (got {cw!r}).\n"
            f"  Fix: regenerate this cmr_export tree with the v0.4+ exporter."
        )
    return float(cw[0]), float(cw[1]), float(cw[2])


class ImportError_(RuntimeError):
    """Bucket import failure; message includes a one-line fix."""


def _normalize_scenario(name: str) -> str:
    """Map cmr_export's scenario folder name to SCENARIO_TO_SEQ's key.

    The legacy exporter prefixed scenario dirs with ``test__``; the newer
    per-vehicle exports drop that prefix. Both forms point at the same
    underlying scenario.
    """
    return name if name.startswith("test__") else "test__" + name


def _read_detections(det_csv: Path) -> dict[tuple[str, int], list[tuple]]:
    """Parse cmr_export's detections.csv into {(vehicle, frame): [det, ...]}.

    Detections are in WORLD frame. Dedups byte-identical duplicate rows by
    (vehicle, frame, det_idx) — the upstream exporter occasionally emits each
    detection twice.
    """
    seen: set[tuple[str, int, int]] = set()
    out: dict[tuple[str, int], list[tuple]] = {}
    with open(det_csv) as f:
        for r in csv.DictReader(f):
            v = r["vehicle_id"]
            fid = int(r["frame_id"])
            did = int(r.get("det_idx", "0"))
            if (v, fid, did) in seen:
                continue
            seen.add((v, fid, did))
            out.setdefault((v, fid), []).append((
                float(r["x"]), float(r["y"]), float(r["z"]),
                float(r["yaw"]),
                float(r["length"]), float(r["width"]), float(r["height"]),
                float(r["score"]),
            ))
    return out


def _scenarios_per_vehicle(cmr_export_root: Path) -> dict[str, dict[str, Path]]:
    """For each vehicle subdir, list scenarios it has.

    Returns ``{vehicle: {scenario_name: scen_dir}}``. Vehicles with no subdir
    in cmr_export/ are simply absent from the returned dict.
    """
    out: dict[str, dict[str, Path]] = {}
    for vehicle in SUPPORTED_VEHICLES:
        vdir = cmr_export_root / vehicle
        if not vdir.is_dir():
            continue
        scenarios: dict[str, Path] = {}
        for scen_dir in sorted(vdir.iterdir()):
            if not scen_dir.is_dir():
                continue
            scenarios[scen_dir.name] = scen_dir
        if scenarios:
            out[vehicle] = scenarios
    return out


def import_bucket_to_ab3dmot(
    bucket_root: Path,
    *,
    dry_run: bool,
) -> dict[str, int]:
    """Convert ``<bucket>/cmr_export/`` → ``<bucket>/ab3dmot_detections/``.

    Returns a dict of counters: ``{files_written, dets, self_reports, pose_rows}``.
    Raises ``ImportError_`` with a friendly message on any structural problem.
    """
    cmr_export = bucket_root / "cmr_export"
    ab3dmot_dir = bucket_root / "ab3dmot_detections"

    if not cmr_export.is_dir():
        raise ImportError_(
            f"missing cmr_export at {cmr_export}\n"
            f"  Fix: this bucket's upstream cmr_export hasn't been symlinked / "
            f"populated yet. See its README.md for the upstream source."
        )

    per_vehicle = _scenarios_per_vehicle(cmr_export)
    if not per_vehicle:
        raise ImportError_(
            f"no vehicle subdirs found under {cmr_export} (expected at least "
            f"one of {SUPPORTED_VEHICLES}). Fix the cmr_export symlinks first."
        )
    print(f"  vehicles present: {sorted(per_vehicle)} — common frame: tesla")

    # Val-coverage check: every val seq scenario must be present under at
    # least one vehicle's cmr_export subdir. The new per-vehicle exports
    # (May 2026) covered the train split only and silently dropped 8/9 val
    # scenarios — re-run mmdet3d/tools/export_v2v4real_to_cmr.py with the
    # val/test scenarios included, then re-symlink the bucket's cmr_export.
    union_names: set[str] = set()
    for scens in per_vehicle.values():
        union_names |= set(scens.keys())
    val_no_pfx = {scen[len("test__"):] for scen in SCENARIO_TO_SEQ}
    missing_val = sorted(
        scen for scen in val_no_pfx
        if scen not in union_names and ("test__" + scen) not in union_names
    )
    if missing_val:
        raise ImportError_(
            f"val coverage incomplete: {len(missing_val)}/{len(val_no_pfx)} "
            f"val scenarios are missing from cmr_export.\n"
            f"  Missing:\n    - " + "\n    - ".join(missing_val) +
            f"\n\n  Fix: re-run the upstream cmr_export step "
            f"(mmdetection3d/tools/export_v2v4real_to_cmr.py) with the val "
            f"scenarios included. Then re-link this bucket's cmr_export "
            f"subdir(s) to the updated upstream output."
        )

    if not dry_run:
        # Wipe old children but preserve the dir itself.
        if ab3dmot_dir.exists():
            for p in list(ab3dmot_dir.iterdir()):
                if p.is_symlink() or p.is_file():
                    p.unlink()
        ab3dmot_dir.mkdir(parents=True, exist_ok=True)

    counts = {"files_written": 0, "dets": 0, "self_reports": 0, "pose_rows": 0,
              "dets_dropped_by_range_cap": 0}

    # Iterate seqs 0..8; each seq corresponds to one scenario in V2V4Real val.
    for seq in sorted(SEQ_TO_NUM_FRAMES):
        seq_str = f"{seq:04d}"
        scen_with_prefix = next(k for k, v in SCENARIO_TO_SEQ.items() if v == seq)
        scen_no_prefix = scen_with_prefix[len("test__"):]

        # Gather per-vehicle scen dirs (normalize the prefix variants).
        scen_dirs: dict[str, Path] = {}
        for v, scenarios in per_vehicle.items():
            if scen_with_prefix in scenarios:
                scen_dirs[v] = scenarios[scen_with_prefix]
            elif scen_no_prefix in scenarios:
                scen_dirs[v] = scenarios[scen_no_prefix]

        if not scen_dirs:
            print(f"  seq {seq_str}: scenario not found in any vehicle subdir — skipping")
            continue

        # First: collect ALL poses across vehicles for this scenario, so we
        # can use tesla's pose as the common-frame anchor regardless of
        # which subdir we're reading from.
        #
        # CRITICAL: each tree's CSVs are in that-tree's centered frame. Rehydrate
        # to a shared real-world frame on read by ADDING the tree's center_world.
        # Without this, astuff/tesla coordinates differ by the centroid delta
        # (~81m on Day19 seq 0), and world_to_local_kitti silently puts astuff
        # dets at the wrong tesla-frame position.
        all_poses: dict[tuple[str, int], tuple[float, ...]] = {}
        all_dets: dict[tuple[str, int], list[tuple]] = {}
        for vehicle in sorted(scen_dirs):
            sdir = scen_dirs[vehicle]
            ego_csv = sdir / "ego_state.csv"
            det_csv = sdir / "detections.csv"
            if not ego_csv.exists() or not det_csv.exists():
                raise ImportError_(
                    f"seq {seq_str} {vehicle}: missing ego_state.csv or "
                    f"detections.csv under {sdir}. Re-run upstream cmr_export."
                )
            cx, cy, cz = _load_center_world(sdir)
            # Rehydrate poses: (x, y, z, yaw, l, w, h) → add center to x/y/z
            for key, row in read_poses(ego_csv).items():
                x, y, z, yaw, L, W, H = row
                all_poses[key] = (x + cx, y + cy, z + cz, yaw, L, W, H)
            # Rehydrate dets: (x, y, z, yaw, length, width, height, score)
            for key, dets in _read_detections(det_csv).items():
                rehydrated = []
                for d in dets:
                    dx, dy, dz, dyaw, dL, dW, dH, ds = d
                    rehydrated.append((dx + cx, dy + cy, dz + cz, dyaw,
                                        dL, dW, dH, ds))
                all_dets[key] = rehydrated

        # Determine which vehicle is the common-frame anchor. Convention:
        # tesla if present (matches AB3DMOT + GT). For single-vehicle buckets
        # (e.g. cp_zeroshot_100m has tesla only) the only vehicle IS tesla
        # so this is trivially correct.
        common_vehicle = "tesla" if any(v == "tesla" for (v, _) in all_poses) else next(
            iter({v for (v, _) in all_poses})
        )

        # For each vehicle, write a .txt with detections in the COMMON frame
        # (tesla's KITTI). The mount-height portion of each row still uses
        # THIS vehicle's mount (rays originate from this CAV's LiDAR). A
        # side-car _ranges.csv preserves per-source range for GPEM lookup.
        for vehicle in sorted(scen_dirs):
            frames_v = sorted({fid for (v_, fid) in all_poses if v_ == vehicle})

            det_path = ab3dmot_dir / f"{seq_str}_{vehicle}.txt"
            range_path = ab3dmot_dir / f"{seq_str}_{vehicle}_ranges.csv"
            if dry_run:
                counts["files_written"] += 2
                continue

            with open(det_path, "w") as fdet, open(range_path, "w") as frng:
                frng.write("frame,line_idx_in_frame,source_local_range_m,kind\n")
                for fid in frames_v:
                    self_pose = all_poses.get((vehicle, fid))
                    common_pose = all_poses.get((common_vehicle, fid))
                    if self_pose is None or common_pose is None:
                        continue
                    line_idx = 0

                    # 1. self-report: this CAV's location, expressed in the
                    # common frame, using its OWN mount height.
                    # Use common_vehicle's mount height (we're expressing this
                    # row in the common frame; the row's own producing-CAV
                    # identity is implicit in the file name). Passing the
                    # producing-CAV here adds a ~2m Y offset and silently
                    # breaks IoU matching against GT.
                    self_kitti = world_to_local_kitti(self_pose, common_pose, common_vehicle)
                    fdet.write(emit_kitti_det_row(fid, self_kitti) + "\n")
                    frng.write(f"{fid},{line_idx},0.000000,self\n")
                    line_idx += 1
                    counts["self_reports"] += 1

                    # 2. real detections from this CAV (world → common frame).
                    # The _ranges side-car records source-local Euclidean
                    # range (||target - this_CAV_pose||) for GPEM lookup —
                    # NOT the tesla-frame range that x,z would give.
                    sx, sy, sz = self_pose[0], self_pose[1], self_pose[2]
                    for d in all_dets.get((vehicle, fid), []):
                        target_world = d[:7]
                        score = d[7]
                        tx, ty, tz = target_world[0], target_world[1], target_world[2]
                        src_range = ((tx - sx) ** 2 + (ty - sy) ** 2 + (tz - sz) ** 2) ** 0.5
                        if src_range > MAX_SOURCE_LOCAL_RANGE_M:
                            counts["dets_dropped_by_range_cap"] += 1
                            continue
                        kitti = world_to_local_kitti(target_world, common_pose, common_vehicle)
                        fdet.write(emit_kitti_det_row(fid, kitti, score=score) + "\n")
                        frng.write(f"{fid},{line_idx},{src_range:.6f},det\n")
                        line_idx += 1
                        counts["dets"] += 1
            counts["files_written"] += 2

        # Per-seq poses file (one row per (frame, vehicle), world frame). The
        # tracker uses this at runtime to resolve cross-CAV geometry.
        pose_path = ab3dmot_dir / f"{seq_str}_poses.csv"
        if dry_run:
            counts["files_written"] += 1
            continue
        with open(pose_path, "w") as fp:
            fp.write("frame,source,x,y,z,yaw,length,width,height\n")
            seq_pose_rows = 0
            for (vehicle, fid), pose in sorted(all_poses.items(), key=lambda kv: (kv[0][1], kv[0][0])):
                x, y, z, yaw, l, w, h = pose
                fp.write(f"{fid},{vehicle},{x:.6f},{y:.6f},{z:.6f},{yaw:.6f},"
                         f"{l:.6f},{w:.6f},{h:.6f}\n")
                seq_pose_rows += 1
            counts["pose_rows"] += seq_pose_rows
        counts["files_written"] += 1

        print(f"  seq {seq_str}  vehicles={sorted(scen_dirs)}  "
              f"common_frame={common_vehicle}  "
              f"pose_rows={seq_pose_rows if not dry_run else '?'}")

    return counts
