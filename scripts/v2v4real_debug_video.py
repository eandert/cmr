#!/usr/bin/env python3
"""
Render a debug video of one V2V4Real scenario.

Visualises, frame-by-frame:
  - GT bounding boxes in BLACK
  - astuff (Ford) detections in RED
  - tesla detections in GREEN
  - Ego poses as filled markers (Δ for astuff, ○ for tesla)
  - (Optional) cross-fused tracks in BLUE with track IDs

Useful for: spot-checking coordinate frames, eyeballing detector noise, seeing
where FPs come from, verifying that GT dedup looks sane.

Examples:
  # Detections-only video (fast, ~30s for a 200-frame scenario):
  scripts/v2v4real_debug_video.py \\
    data/v2v4real_cmr_export/test__Day19__testoutput_CAV_data_2022-03-15-10-29-43_3 \\
    --score 0.3 --classes car,truck,bus,construction_vehicle

  # PointPillar export, with fused tracks overlaid (slow — runs the replay):
  scripts/v2v4real_debug_video.py \\
    data/v2v4real_cmr_export_pointpillar/test__Day21__... \\
    --detector pointpillar_v2v4real_{vehicle} \\
    --with-tracks --score 0.3
"""
import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter

# Make src/ importable.
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))


def _read_csv(p: Path) -> List[Dict]:
    with open(p) as f:
        return list(csv.DictReader(f))


def _load_scenario(scenario_dir: Path) -> Dict:
    """Read the scenario CSVs into per-frame dicts."""
    ego = _read_csv(scenario_dir / "ego_state.csv")
    det = _read_csv(scenario_dir / "detections.csv")
    gt  = _read_csv(scenario_dir / "gt_objects.csv")

    frames: Dict[int, Dict] = {}
    for r in ego:
        fid = int(r["frame_id"])
        frames.setdefault(fid, {"egos": {}, "dets": {}, "gts": []})
        frames[fid]["egos"][r["vehicle_id"]] = {
            "x": float(r["x"]), "y": float(r["y"]),
            "yaw": float(r["yaw"]),
            "length": float(r["length"]), "width": float(r["width"]),
        }
    for r in det:
        fid = int(r["frame_id"])
        if fid not in frames:
            continue
        frames[fid]["dets"].setdefault(r["vehicle_id"], []).append({
            "x": float(r["x"]), "y": float(r["y"]),
            "yaw": float(r["yaw"]),
            "length": float(r["length"]), "width": float(r["width"]),
            "score": float(r["score"]),
            "label": r["label"],
        })
    for r in gt:
        fid = int(r["frame_id"])
        if fid not in frames:
            continue
        frames[fid]["gts"].append({
            "ass_id": int(r["ass_id"]),
            "obj_id": int(r["obj_id"]),
            "vehicle_id": r["vehicle_id"],
            "x": float(r["x"]), "y": float(r["y"]),
            "yaw": float(r["yaw"]),
            "length": float(r["length"]), "width": float(r["width"]),
            "label": r["label"],
        })

    return {"frames": frames, "frame_ids": sorted(frames.keys())}


def _bbox_corners(cx: float, cy: float, length: float, width: float, yaw: float) -> np.ndarray:
    """Build a 4-corner polygon (4×2) for a length-along-heading bbox."""
    hl, hw = length / 2.0, width / 2.0
    c, s = math.cos(yaw), math.sin(yaw)
    pts = [
        ( cx + (-hl)*c - (-hw)*s, cy + (-hl)*s + (-hw)*c ),
        ( cx + ( hl)*c - (-hw)*s, cy + ( hl)*s + (-hw)*c ),
        ( cx + ( hl)*c - ( hw)*s, cy + ( hl)*s + ( hw)*c ),
        ( cx + (-hl)*c - ( hw)*s, cy + (-hl)*s + ( hw)*c ),
    ]
    return np.array(pts)


def _draw_bbox(ax, corners: np.ndarray, color: str, lw: float = 1.2,
               alpha: float = 0.9, fill: bool = False) -> None:
    poly = mpatches.Polygon(corners, closed=True, edgecolor=color,
                            facecolor=color if fill else "none",
                            linewidth=lw, alpha=alpha)
    ax.add_patch(poly)


def _scenario_extent(frames: Dict, margin: float = 5.0) -> Tuple[float, float, float, float]:
    """Compute fixed axis bounds covering all entities across all frames."""
    xs, ys = [], []
    for fdata in frames.values():
        for e in fdata["egos"].values():
            xs.append(e["x"]); ys.append(e["y"])
        for vlist in fdata["dets"].values():
            for d in vlist:
                xs.append(d["x"]); ys.append(d["y"])
        for g in fdata["gts"]:
            xs.append(g["x"]); ys.append(g["y"])
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    return xmin - margin, xmax + margin, ymin - margin, ymax + margin


def _maybe_load_fused_tracks(
    scenario_dir: Path,
    stream_key: str,
    replay_cfg: Dict,
) -> Optional[Dict[int, List[Tuple[int, float, float, float]]]]:
    """Run the replay engine on this scenario and return per-frame fused
    tracks for the requested STREAM_KEY (e.g. 'bici_gpem_quadratic').

    Returns dict[frame_idx -> [(track_id, x, y, score), …]] or None if
    the replay couldn't run.
    """
    try:
        import sensor_fusion  # noqa: F401  — break circular import
        import v2v4real_replay
    except Exception as e:
        print(f"  (could not import replay engine: {e})", file=sys.stderr)
        return None

    cfg = dict(replay_cfg or {})
    cfg["record_tape_for"] = [stream_key]
    print(f"  Running replay to collect '{stream_key}' tracks (~30-60s)…")
    metrics = v2v4real_replay.run_scenario(scenario_dir, cfg)
    tape = metrics.get("match_tape_per_stream", {}).get(stream_key, [])
    if not tape:
        print(f"  (replay returned no tape for {stream_key})", file=sys.stderr)
        return None
    out: Dict[int, List[Tuple[int, float, float, float]]] = {}
    for entry in tape:
        out[int(entry["frame_idx"])] = list(entry.get("tracks", []))
    return out


def render(
    scenario_dir: Path,
    out_path: Path,
    score_threshold: float = 0.3,
    classes: Optional[List[str]] = None,
    fps: int = 10,
    figsize: Tuple[float, float] = (10, 10),
    show_score_labels: bool = True,
    title_extra: str = "",
    fused_tracks_per_frame: Optional[Dict[int, List[Tuple[int, float, float, float]]]] = None,
    fused_track_label: str = "",
) -> None:
    """Build the debug video and save to `out_path` (.mp4 or .gif)."""
    scenario = _load_scenario(scenario_dir)
    frame_ids = scenario["frame_ids"]
    frames = scenario["frames"]

    classes_set = set(classes) if classes else None

    xmin, xmax, ymin, ymax = _scenario_extent(frames)
    fig, ax = plt.subplots(figsize=figsize)

    def init():
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X (m, scenario-local)")
        ax.set_ylabel("Y (m, scenario-local)")
        return []

    def draw_frame(fid):
        ax.clear()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

        fdata = frames[fid]

        # GT in BLACK.
        for g in fdata["gts"]:
            corners = _bbox_corners(g["x"], g["y"], g["length"], g["width"], g["yaw"])
            _draw_bbox(ax, corners, color="black", lw=1.5, alpha=0.9, fill=False)

        # Detections — astuff in RED, tesla in GREEN.
        per_ego_color = {"astuff": "tab:red", "tesla": "tab:green"}
        for ego_id, dlist in fdata["dets"].items():
            color = per_ego_color.get(ego_id, "tab:orange")
            for d in dlist:
                if d["score"] < score_threshold:
                    continue
                if classes_set is not None and d["label"] not in classes_set:
                    continue
                corners = _bbox_corners(d["x"], d["y"], d["length"], d["width"], d["yaw"])
                _draw_bbox(ax, corners, color=color, lw=1.0, alpha=0.6, fill=False)
                if show_score_labels:
                    ax.text(d["x"], d["y"], f"{d['score']:.2f}",
                            color=color, fontsize=5, alpha=0.5,
                            ha="center", va="center")

        # Fused tracks (BLUE) — overlaid on top of detections so they're
        # visible. Each track is a small filled circle + track_id label.
        if fused_tracks_per_frame and fid in fused_tracks_per_frame:
            for (tid, tx, ty, tscore) in fused_tracks_per_frame[fid]:
                ax.scatter(tx, ty, s=80, marker="X", facecolor="tab:blue",
                           edgecolor="white", linewidth=1.0, alpha=0.95, zorder=8)
                ax.text(tx, ty + 1.0, f"#{int(tid)}",
                        color="tab:blue", fontsize=7, fontweight="bold",
                        ha="center", va="bottom",
                        bbox=dict(boxstyle="round,pad=0.15",
                                  facecolor="white", edgecolor="tab:blue", alpha=0.7))

        # Ego poses (filled markers + outlined bbox).
        for ego_id, e in fdata["egos"].items():
            color = per_ego_color.get(ego_id, "black")
            marker = "^" if ego_id == "astuff" else "o"
            ax.scatter(e["x"], e["y"], s=120, marker=marker,
                       facecolor=color, edgecolor="black",
                       linewidth=1.2, zorder=10)
            corners = _bbox_corners(e["x"], e["y"], e["length"], e["width"], e["yaw"])
            _draw_bbox(ax, corners, color=color, lw=2.0, alpha=1.0, fill=False)

        # Title with frame counter + counts.
        n_dets_total = sum(
            sum(1 for d in vlist
                if d["score"] >= score_threshold and (classes_set is None or d["label"] in classes_set))
            for vlist in fdata["dets"].values()
        )
        n_gt = len(fdata["gts"])
        title = f"{scenario_dir.name}  frame {fid}/{frame_ids[-1]}"
        if title_extra:
            title = f"{title} — {title_extra}"
        title += f"\nGT={n_gt}  dets(score≥{score_threshold:.1f})={n_dets_total}"
        ax.set_title(title, fontsize=10)

        # Legend (drawn each frame so it persists).
        legend_handles = [
            mpatches.Patch(edgecolor="black", facecolor="none", label="GT"),
            mpatches.Patch(edgecolor="tab:red", facecolor="none", label="astuff dets"),
            mpatches.Patch(edgecolor="tab:green", facecolor="none", label="tesla dets"),
            plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="tab:red",
                       markeredgecolor="black", markersize=10, label="astuff ego"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:green",
                       markeredgecolor="black", markersize=10, label="tesla ego"),
        ]
        if fused_tracks_per_frame:
            label = f"fused track ({fused_track_label})" if fused_track_label else "fused track"
            legend_handles.append(
                plt.Line2D([0], [0], marker="X", color="w",
                           markerfacecolor="tab:blue", markeredgecolor="white",
                           markersize=10, label=label)
            )
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
        return []

    anim = FuncAnimation(fig, draw_frame, frames=frame_ids, init_func=init,
                          blit=False, repeat=False)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".gif":
        writer = PillowWriter(fps=fps)
    else:
        # yuv420p for broad compatibility (default yuv444p is rejected by
        # browsers / Quicktime / many media players → "black video"). The
        # extra args force libx264 + yuv420p + faststart for streaming.
        writer = FFMpegWriter(
            fps=fps, bitrate=4000, codec="libx264",
            extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
        )

    print(f"Writing {len(frame_ids)} frames @ {fps}fps → {out_path}")
    anim.save(str(out_path), writer=writer, dpi=120)
    plt.close(fig)
    print(f"  ✓ done ({out_path.stat().st_size / 1024 / 1024:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenario_dir", type=Path,
                    help="Path to a scenario directory "
                         "(e.g. data/v2v4real_cmr_export/test__Day19__...)")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="Output path (.mp4 or .gif). "
                         "Default: results/debug_videos/<scenario>.mp4")
    ap.add_argument("--score", type=float, default=0.3,
                    help="Drop detections with score < this. Default 0.3.")
    ap.add_argument("--classes", default="car,truck,bus,construction_vehicle",
                    help="Comma-separated class filter, or 'all'. "
                         "Default: car,truck,bus,construction_vehicle.")
    ap.add_argument("--fps", type=int, default=10,
                    help="Output frame rate. V2V4Real is 10 Hz so default is real-time.")
    ap.add_argument("--no-score-labels", action="store_true",
                    help="Don't print the score next to each detection.")
    ap.add_argument("--title-extra", default="",
                    help="Optional extra string in the frame titles "
                         "(e.g. 'CenterPoint zero-shot, score>0.3').")
    ap.add_argument("--with-tracks", default=None,
                    help="STREAM_KEY to overlay fused tracks for "
                         "(e.g. 'bici_gpem_quadratic'). Triggers a full replay "
                         "of the scenario, ~30-60s. The matching --detector / "
                         "--score-threshold are passed through.")
    ap.add_argument("--detector-name", default=None,
                    help="Detector model name passed to the replay (when "
                         "--with-tracks is set). Auto-derived from the export "
                         "dir if omitted: pointpillar_v2v4real_{vehicle} for "
                         "the PP export, centerpoint_100m_on_v2v4real otherwise.")
    ap.add_argument("--no-self-report", action="store_true",
                    help="Disable V2X self-report in the replay (each ego "
                         "broadcasts its own RTK pose to fusion). On by "
                         "default. Useful for A/B-comparing track outputs.")
    args = ap.parse_args()

    classes = None if args.classes.strip().lower() == "all" else \
              [c.strip() for c in args.classes.split(",") if c.strip()]
    if args.out is None:
        args.out = (_REPO / "results" / "debug_videos" /
                    f"{args.scenario_dir.name}.mp4")

    fused = None
    if args.with_tracks:
        # Pick the right detector for the export dir if not specified.
        det_name = args.detector_name
        if det_name is None:
            if "pointpillar" in str(args.scenario_dir).lower():
                det_name = "pointpillar_v2v4real_{vehicle}"
            else:
                det_name = "centerpoint_100m_on_v2v4real"
        replay_cfg = {
            "detector_name":   det_name,
            "score_threshold": args.score,
            "vehicle_classes": classes,
            "self_report_egos": not args.no_self_report,
        }
        fused = _maybe_load_fused_tracks(args.scenario_dir, args.with_tracks, replay_cfg)

    render(
        scenario_dir=args.scenario_dir,
        out_path=args.out,
        score_threshold=args.score,
        classes=classes,
        fps=args.fps,
        show_score_labels=not args.no_score_labels,
        title_extra=args.title_extra,
        fused_tracks_per_frame=fused,
        fused_track_label=args.with_tracks or "",
    )


if __name__ == "__main__":
    main()
