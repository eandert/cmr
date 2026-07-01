#!/usr/bin/env python3
"""Run DMSTrack's CoBEVT detections through AB3DMOT-style tracking and
score under BOTH our standard evaluator and the V2V4Real protocol.

Killer comparison: DMSTrack's published 'CoBEVT+AB3DMOT (our impl)' AMOTA
is 37.16 — but their evaluator inherits KITTI's min_height=25 filter that
silently zeros LiDAR-only FPs. Does that 37.16 hold up when FPs count?

Inputs come from the cloned DMSTrack repo:
  - CoBEVT detections (KITTI-tracking format):
        DMSTrack/AB3DMOT/data/v2v4real/detection/cobevt_Car_val/{0000..0008}.txt
  - GT labels:
        DMSTrack/AB3DMOT/scripts/KITTI/v2v4real_val_label/{0000..0008}.txt

Note: DMSTrack's "val" split (9 sequences) corresponds to V2V4Real's
"test" split (their published 37.16 / 43.52 numbers are on these same 9
sequences).

KITTI det format columns (per row):
    frame, type, 0,0,0,0, score, h, w, l, x, y, z, ry, alpha
KITTI gt format columns:
    frame, gt_id, type, 0,...,0, h, w, l, x, y, z, ry
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

# Reuse the AB3DMOT-style tracker bits.
from v2v4real_ab3dmot_replica import CVKalman, _bev_nms, _hungarian_match
from metrics.v2v4real_metrics import (
    compute_ab3dmot_metrics,
    compute_ab3dmot_metrics_iou,
)

DMSTRACK_ROOT = (REPO / "third_party" / "AB3DMOT").resolve().parent
DEFAULT_DET_DIR = DMSTRACK_ROOT / "AB3DMOT/data/v2v4real/detection/cobevt_Car_val"
GT_DIR          = DMSTRACK_ROOT / "AB3DMOT/scripts/KITTI/v2v4real_val_label"
DET_DIR = DEFAULT_DET_DIR  # overridden in main()


def parse_kitti_det(path: Path) -> Dict[int, List[Dict]]:
    """Parse a CoBEVT KITTI-format det file → {frame_idx: [det_dict, ...]}."""
    out: Dict[int, List[Dict]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        toks = line.split(",")
        if len(toks) < 15:
            toks = line.split()
        frame = int(float(toks[0]))
        # cols: 0:frame 1:type 2-5:zeros 6:score 7:h 8:w 9:l 10:x 11:y 12:z 13:ry 14:alpha
        score = float(toks[6])
        h = float(toks[7]); w = float(toks[8]); l = float(toks[9])
        x = float(toks[10]); y = float(toks[11]); z = float(toks[12])
        ry = float(toks[13])
        out.setdefault(frame, []).append({
            "x": x, "y": y, "width": w, "length": l, "yaw": ry, "score": score,
        })
    return out


def parse_kitti_gt(path: Path) -> Dict[int, List[Tuple]]:
    """Parse a v2v4real_val_label KITTI-format GT file →
       {frame_idx: [(gt_id, x, y, w, l, yaw), ...]}."""
    out: Dict[int, List[Tuple]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        toks = line.split()
        # cols: 0:frame 1:gt_id 2:type 3-9:zeros 10:h 11:w 12:l 13:x 14:y 15:z 16:ry
        frame = int(float(toks[0]))
        gid   = int(float(toks[1]))
        typ   = toks[2]
        if typ.lower() not in ("car",):
            continue
        h = float(toks[10]); w = float(toks[11]); l = float(toks[12])
        x = float(toks[13]); y = float(toks[14]); z = float(toks[15])
        ry = float(toks[16])
        out.setdefault(frame, []).append((gid, x, y, w, l, ry))
    return out


def track_scenario(dets_per_frame: Dict[int, List[Dict]],
                   score_threshold: float = 0.20,
                   nms_iou: float = 0.15,
                   match_iou_gate: float = 0.01,
                   min_hits: int = 3,
                   max_age: int = 2) -> Dict[int, List[Tuple]]:
    """AB3DMOT-style tracking — same recipe as v2v4real_ab3dmot_replica.

    Returns {frame_idx: [(track_id, x, y, w, l, yaw, score), ...]} for
    confirmed (hits>=min_hits) tracks active in that frame.
    """
    tracks: List[CVKalman] = []
    next_id = 1
    out: Dict[int, List[Tuple]] = {}

    frames = sorted(dets_per_frame.keys())
    for fid in frames:
        time = fid * 0.1  # 10 Hz
        dets = [d for d in dets_per_frame[fid] if d["score"] >= score_threshold]
        dets = _bev_nms(dets, nms_iou)

        # Predict all live tracks forward.
        for tr in tracks:
            tr.predict_inplace(time)

        # Hungarian match.
        matches, um_t, um_d = _hungarian_match(tracks, dets, match_iou_gate)

        # Update matched.
        matched_t = set()
        for ti, di in matches:
            d = dets[di]
            tracks[ti].update(d["x"], d["y"], d["score"],
                              d["width"], d["length"], d["yaw"], time)
            matched_t.add(ti)

        # Increment misses for unmatched tracks.
        for ti in um_t:
            tracks[ti].misses_since_match += 1

        # Spawn new tracks for unmatched detections.
        for di in um_d:
            d = dets[di]
            tracks.append(CVKalman(d["x"], d["y"], time, d["score"],
                                    d["width"], d["length"], d["yaw"],
                                    "Car", next_id))
            next_id += 1

        # Kill aged-out tracks.
        tracks = [tr for tr in tracks if tr.misses_since_match < max_age]

        # Emit confirmed tracks for this frame.
        emit: List[Tuple] = []
        for tr in tracks:
            if tr.hits >= min_hits and tr.misses_since_match == 0:
                emit.append((tr.id, tr.x, tr.y, tr.width, tr.length,
                              tr.yaw, tr.score_initial))
        out[fid] = emit

    return out


def build_per_frame_tape(tracks_per_frame: Dict[int, List[Tuple]],
                          gts_per_frame: Dict[int, List[Tuple]]) -> List[Dict]:
    """Build the per_frame tape format expected by metrics.v2v4real_metrics."""
    all_frames = sorted(set(tracks_per_frame) | set(gts_per_frame))
    return [
        {"frame_idx": f,
         "tracks": tracks_per_frame.get(f, []),
         "gts":    gts_per_frame.get(f, [])}
        for f in all_frames
    ]


def run_one(scen_id: str, args) -> Tuple[Dict, Dict, int]:
    det_path = DET_DIR / f"{scen_id}.txt"
    gt_path  = GT_DIR  / f"{scen_id}.txt"
    dets = parse_kitti_det(det_path)
    gts  = parse_kitti_gt(gt_path)
    tracks = track_scenario(dets,
                             score_threshold=args.score_threshold,
                             nms_iou=args.nms_iou,
                             match_iou_gate=args.match_iou_gate,
                             min_hits=args.min_hits,
                             max_age=args.max_age)
    tape = build_per_frame_tape(tracks, gts)
    std_m = compute_ab3dmot_metrics(tape)
    v2v_m = compute_ab3dmot_metrics_iou(tape, iou_threshold=0.25,
                                         ignore_unmatched_fps=True)
    return std_m, v2v_m, sum(len(v) for v in gts.values())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--score-threshold", type=float, default=0.20)
    ap.add_argument("--nms-iou", type=float, default=0.15)
    ap.add_argument("--match-iou-gate", type=float, default=0.01)
    ap.add_argument("--min-hits", type=int, default=3)
    ap.add_argument("--max-age", type=int, default=2)
    args = ap.parse_args()

    scen_ids = [f"{i:04d}" for i in range(9)]

    print(f"=== DMSTrack CoBEVT+AB3DMOT through our evaluator ===")
    print(f"  9 sequences from {DET_DIR}")
    print(f"  GT from           {GT_DIR}")
    print(f"  AB3DMOT recipe: score>={args.score_threshold} NMS@{args.nms_iou} "
          f"IoU-gate={args.match_iou_gate} min_hits={args.min_hits} max_age={args.max_age}")
    print()
    print(f"{'seq':<6s}{'std_AMOTA':>11s}{'V2V_AMOTA':>11s}{'std_MOTA':>10s}{'V2V_MOTA':>10s}{'gt':>6s}")

    sums = {"std_a": 0, "v2v_a": 0, "std_m": 0, "v2v_m": 0,
            "std_sa": 0, "v2v_sa": 0, "std_p": 0, "v2v_p": 0}
    for sid in scen_ids:
        s, v, ng = run_one(sid, args)
        sums["std_a"]  += s["amota"]  * 100
        sums["v2v_a"]  += v["amota"]  * 100
        sums["std_m"]  += s["mota"]   * 100
        sums["v2v_m"]  += v["mota"]   * 100
        sums["std_sa"] += s["samota"] * 100
        sums["v2v_sa"] += v["samota"] * 100
        sums["std_p"]  += s["amotp"]  * 100
        sums["v2v_p"]  += v["amotp"]  * 100
        print(f"{sid:<6s}{s['amota']*100:>11.2f}{v['amota']*100:>11.2f}"
              f"{s['mota']*100:>10.2f}{v['mota']*100:>10.2f}{ng:>6d}")

    n = len(scen_ids)
    print()
    print(f"=== Mean across {n} sequences ===")
    print(f"               STD eval (FPs counted)   V2V proto (FPs ignored)")
    print(f"  AMOTA   :     {sums['std_a']/n:>10.2f}              {sums['v2v_a']/n:>10.2f}")
    print(f"  AMOTP   :     {sums['std_p']/n:>10.2f}              {sums['v2v_p']/n:>10.2f}")
    print(f"  sAMOTA  :     {sums['std_sa']/n:>10.2f}              {sums['v2v_sa']/n:>10.2f}")
    print(f"  MOTA    :     {sums['std_m']/n:>10.2f}              {sums['v2v_m']/n:>10.2f}")
    print()
    print(f"DMSTrack reported CoBEVT+AB3DMOT: AMOTA=37.16  sAMOTA=84.54  MOTA=84.14")
    print(f"V2V4Real paper Late Fusion:        AMOTA=29.28  sAMOTA=71.05")


if __name__ == "__main__":
    main()
