#!/usr/bin/env python3
"""Build a drift-corrected, co-observation-merged V2V4Real GT ("Ours-merged").

The released V2V4Real GT (and the existing ``augmented_gt``) averages co-observed
boxes WITHOUT correcting the inter-CAV RTK pose drift, so a co-observed object is
smeared between the two CAVs' (mutually mis-localized) views. This script fixes
that: it applies the per-frame OU-smoothed drift t (from
``estimate_intercav_rtk_drift.py``) to bring astuff's GT into tesla's frame,
THEN merges co-observed boxes (plain centroid-average) and keeps singletons.

Per scenario:
* Cluster objects across the two CAVs with union-find over the per-frame
  ``ass_id`` links (astuff ``ass_id`` → matching tesla ``obj_id``), so each
  physical object carries ONE track id whether it is co-observed or seen by a
  single CAV in a given frame (no fake GT id-switches).
* astuff box (x,y) += (tx, ty) — translation-only drift; z, yaw, dims unchanged.
* Co-observed cluster present in a frame → centroid-average tesla box and the
  drift-corrected astuff box (mean position + dims, circular-mean yaw). Singleton
  → that (corrected, for astuff) box.
* Transform world → tesla-ego KITTI via the SAME ``world_to_local_kitti`` the
  augmenter uses (imported, not reimplemented) and emit KITTI MOT rows.

Population is THIRD-PARTY objects only (the egos drive drift estimation but are
not emitted), so the output is directly comparable to the official paper_gt.

``--no-drift`` builds the same merge with t=0 — used to validate the transform +
merge mechanics against paper_gt (should match closely) before trusting the
drift-corrected output.

Output: ``data/v2v4real_inputs/ours_merged_gt/kitti_labels/{0000..0008}.txt``.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "configs"))

from v2v4real_seq_map import SEQ_TO_SCENARIO, SEQ_TO_NUM_FRAMES  # noqa: E402
from augment_v2v4real_with_cav_self_reports import emit_kitti_gt_row  # noqa: E402

DEFAULT_EXPORT = REPO / "data" / "v2v4real_cmr_export"
DEFAULT_DRIFT = REPO / "results" / "GT_MERGE_RTK" / "drift"
DEFAULT_OUT = REPO / "data" / "v2v4real_inputs" / "ours_merged_gt" / "kitti_labels"
# Raw V2V4Real test split — needed for the FULL 4x4 lidar_pose (the cmr_export
# flattened poses to x,y,z,yaw and dropped pitch/roll). cav0 = tesla = dir "0".
DEFAULT_RAW_BASE = Path("/media/rave/eddie_drive1/v2v4real/test")


# ---------------------------------------------------------------------------
def load_gt_full(scen_dir: Path):
    """tesla[frame][objid]=(x,y,z,yaw,l,w,h); astuff[frame][objid]=(...,ass_id)."""
    tesla: dict[int, dict[int, tuple]] = defaultdict(dict)
    astuff: dict[int, dict[int, tuple]] = defaultdict(dict)
    with open(scen_dir / "gt_objects.csv") as f:
        for r in csv.DictReader(f):
            fr = int(r["frame_id"])
            box = (float(r["x"]), float(r["y"]), float(r["z"]), float(r["yaw"]),
                   float(r["length"]), float(r["width"]), float(r["height"]))
            if r["vehicle_id"] == "tesla":
                tesla[fr][int(r["obj_id"])] = box
            elif r["vehicle_id"] == "astuff":
                astuff[fr][int(r["obj_id"])] = box + (int(r["ass_id"]),)
    return tesla, astuff


def load_tesla_ego(scen_dir: Path) -> dict[int, tuple]:
    """frame -> (x,y,z,yaw,l,w,h) — the 7-tuple world_to_local_kitti expects
    (its l,w,h are ignored, but must be present to unpack)."""
    ego: dict[int, tuple] = {}
    with open(scen_dir / "ego_state.csv") as f:
        for r in csv.DictReader(f):
            if r["vehicle_id"] == "tesla":
                ego[int(r["frame_id"])] = (
                    float(r["x"]), float(r["y"]), float(r["z"]), float(r["yaw"]),
                    float(r["length"]), float(r["width"]), float(r["height"]))
    return ego


def load_tesla_rotation(scen: str, raw_base: Path) -> dict[int, "np.ndarray"]:
    """frame -> tesla (cav0) LiDAR→world 3x3 rotation from the raw V2V4Real
    ``lidar_pose`` 4x4. The cmr_export only kept x,y,z,yaw, dropping pitch/roll;
    the full rotation is required to land world boxes on the official KITTI Y
    (a 2x4 yaw-only approximation gives a range-growing vertical error). Frame
    index = sorted yaml order (matches cmr_export frame_id, verified)."""
    scen_raw = scen.split("__", 2)[2]               # test__Day19__<raw> -> <raw>
    files = sorted(glob.glob(str(raw_base / scen_raw / "0" / "*.yaml")))
    if not files:
        raise SystemExit(f"no raw yaml for {scen_raw} under {raw_base}")
    out: dict[int, np.ndarray] = {}
    for i, f in enumerate(files):
        d = yaml.load(open(f), Loader=yaml.Loader)   # full loader: numpy-tagged pose
        out[i] = np.asarray(d["lidar_pose"], dtype=float)[:3, :3]
    return out


def world_to_kitti_full(target_world: tuple, ego_world: tuple,
                        R: "np.ndarray") -> tuple:
    """World box -> tesla KITTI camera frame via the FULL 3D rotation.

    lidar = R^T (p_world - p_ego)   (R = LiDAR->world; delta is origin-invariant
    so cmr_export-world deltas compose with the raw-world rotation). Then the
    V2V4Real KITTI map (X,Y,Z) = (lidar_x, lidar_z + h/2, lidar_y). Reproduces
    the official paper_gt to <1e-3 m in X/Y/Z for tesla-observed boxes.
    """
    tx, ty, tz, tyaw, tl, tw, th = target_world
    ex, ey, ez, eyaw = ego_world[:4]
    lid = R.T @ np.array([tx - ex, ty - ey, tz - ez])
    kitti_X = lid[0]
    kitti_Z = lid[1]
    kitti_Y = lid[2] + th / 2.0
    kitti_ry = (tyaw - eyaw + math.pi) % (2 * math.pi) - math.pi
    return (kitti_X, kitti_Y, kitti_Z, kitti_ry, tl, tw, th)


def load_drift(scen: str, drift_dir: Path, use_drift: bool) -> dict[int, tuple]:
    """frame -> (tx, ty) OU-smoothed drift; zeros if --no-drift or file absent."""
    out: dict[int, tuple] = defaultdict(lambda: (0.0, 0.0))
    if not use_drift:
        return out
    path = drift_dir / f"{scen}.csv"
    if not path.exists():
        raise SystemExit(f"missing drift series {path}; run estimate_intercav_rtk_drift.py")
    with open(path) as f:
        for r in csv.DictReader(f):
            out[int(r["frame"])] = (float(r["tx_ou"]), float(r["ty_ou"]))
    return out


class UnionFind:
    def __init__(self):
        self.parent: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:          # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def build_clusters(tesla, astuff) -> dict:
    """Union ('t',objid) with ('a',objid) over all per-frame ass_id links; assign
    a stable per-scenario integer tid to each cluster root."""
    uf = UnionFind()
    for fr, objs in astuff.items():
        for objid, box in objs.items():
            uf.find(("a", objid))
            ass = box[7]
            if ass >= 0:
                uf.union(("a", objid), ("t", ass))
    for fr, objs in tesla.items():
        for objid in objs:
            uf.find(("t", objid))
    roots = sorted({uf.find(n) for n in uf.parent})
    root_tid = {root: i + 1 for i, root in enumerate(roots)}
    return {n: root_tid[uf.find(n)] for n in uf.parent}


def circular_mean(angles) -> float:
    return math.atan2(sum(math.sin(a) for a in angles) / len(angles),
                      sum(math.cos(a) for a in angles) / len(angles))


def merge_boxes(boxes) -> tuple:
    """Centroid-average a cluster's member world boxes (x,y,z,l,w,h mean; yaw
    circular-mean)."""
    n = len(boxes)
    xs = [sum(b[i] for b in boxes) / n for i in (0, 1, 2)]          # x,y,z
    yaw = circular_mean([b[3] for b in boxes])
    dims = [sum(b[i] for b in boxes) / n for i in (4, 5, 6)]        # l,w,h
    return (xs[0], xs[1], xs[2], yaw, dims[0], dims[1], dims[2])


# ---------------------------------------------------------------------------
def build_scenario_rows(seq: int, scen: str, export_root: Path, drift_dir: Path,
                        use_drift: bool, raw_base: Path):
    scen_dir = export_root / scen
    tesla, astuff = load_gt_full(scen_dir)
    ego = load_tesla_ego(scen_dir)
    rot = load_tesla_rotation(scen, raw_base)
    drift = load_drift(scen, drift_dir, use_drift)
    node_tid = build_clusters(tesla, astuff)
    n_frames = SEQ_TO_NUM_FRAMES[seq]
    rows: list[str] = []
    n_merged = n_tesla_only = n_astuff_only = 0
    for fr in range(n_frames):
        if fr not in ego or fr not in rot:
            continue
        tx, ty = drift[fr]
        cluster_boxes: dict[int, list] = defaultdict(list)
        cluster_src: dict[int, set] = defaultdict(set)
        for objid, box in tesla.get(fr, {}).items():
            tid = node_tid[("t", objid)]
            cluster_boxes[tid].append(box[:7])
            cluster_src[tid].add("t")
        for objid, box in astuff.get(fr, {}).items():
            tid = node_tid[("a", objid)]
            corr = (box[0] + tx, box[1] + ty) + box[2:7]   # drift-correct x,y only
            cluster_boxes[tid].append(corr)
            cluster_src[tid].add("a")
        for tid in sorted(cluster_boxes):
            merged = merge_boxes(cluster_boxes[tid])
            kitti = world_to_kitti_full(merged, ego[fr], rot[fr])
            rows.append(emit_kitti_gt_row(fr, tid, kitti))
            src = cluster_src[tid]
            if src == {"t", "a"}:
                n_merged += 1
            elif src == {"t"}:
                n_tesla_only += 1
            else:
                n_astuff_only += 1
    stats = dict(merged=n_merged, tesla_only=n_tesla_only, astuff_only=n_astuff_only,
                 clusters=len({v for v in node_tid.values()}))
    return rows, stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT)
    ap.add_argument("--drift-dir", type=Path, default=DEFAULT_DRIFT)
    ap.add_argument("--raw-base", type=Path, default=DEFAULT_RAW_BASE,
                    help="Raw V2V4Real test split (for the full 4x4 lidar_pose).")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-drift", action="store_true",
                    help="t=0 (validation against paper_gt; transform+merge only)")
    args = ap.parse_args()
    use_drift = not args.no_drift
    args.out.mkdir(parents=True, exist_ok=True)

    total = 0
    print(f"{'seq':>3s} {'scenario':40s} {'rows':>6s} {'merged':>7s} "
          f"{'tOnly':>6s} {'aOnly':>6s}")
    for seq in sorted(SEQ_TO_SCENARIO):
        scen = SEQ_TO_SCENARIO[seq]
        rows, st = build_scenario_rows(seq, scen, args.export_root, args.drift_dir,
                                       use_drift, args.raw_base)
        (args.out / f"{seq:04d}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        total += len(rows)
        print(f"{seq:>3d} {scen[7:47]:40s} {len(rows):>6d} {st['merged']:>7d} "
              f"{st['tesla_only']:>6d} {st['astuff_only']:>6d}")
    mode = "NO-DRIFT (validation)" if args.no_drift else "drift-corrected"
    print(f"\n[{mode}] wrote {total} GT rows across 9 sequences to {args.out}")


if __name__ == "__main__":
    main()
