"""
V2V4Real evaluation suite runner.

Iterates over scenarios in a V2V4Real CSV export, replays each through the
30-stream fusion pipeline (`v2v4real_replay.run_scenario`), aggregates results
per split (train/val/test/unknown), and writes per-split `summary.json` files
in the same shape as `experiment_runner.py`'s output.

Standalone — does NOT import or extend ExperimentRunner / run_simulation.
"""
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List, Optional, Tuple


SPLITS = ("train", "val", "test", "unknown")


# --------------------------------------------------------------------------
# Scenario discovery
# --------------------------------------------------------------------------

def discover_scenarios(export_dir: Path) -> List[Tuple[str, str, Path]]:
    """
    Walk export_dir and return (split, scenario_name, scenario_path) tuples,
    sorted by (split, scenario_name).

    Split is parsed from the directory prefix `<split>__<scenario_name>`.
    Directories without a recognised prefix are labeled 'unknown'.
    """
    out: List[Tuple[str, str, Path]] = []
    for child in sorted(Path(export_dir).iterdir()):
        if not child.is_dir():
            continue
        if child.name == "sensor_models":
            continue
        # Must look like a scenario directory.
        if not (child / "ego_state.csv").exists():
            continue
        name = child.name
        if "__" in name:
            split, scenario = name.split("__", 1)
            if split not in SPLITS:
                split = "unknown"
        else:
            split = "unknown"
            scenario = name
        out.append((split, scenario, child))
    return out


# --------------------------------------------------------------------------
# Worker (one process per scenario)
# --------------------------------------------------------------------------

def _replay_one(args: Tuple[str, Path, Dict]) -> Dict:
    """ProcessPoolExecutor-friendly worker. Imports v2v4real_replay lazily so
    each child process loads its own CMR module state."""
    scenario_name, scenario_dir, replay_cfg = args
    # Late import (forked workers may not have these in scope at fork time).
    sys.path.insert(0, str(Path(__file__).parent))
    import v2v4real_replay  # noqa: E402

    metrics = v2v4real_replay.run_scenario(scenario_dir, replay_cfg)
    metrics["_scenario_name"] = scenario_name
    metrics["_scenario_dir"] = str(scenario_dir)
    return metrics


# --------------------------------------------------------------------------
# Per-scenario CSV (mirrors experiment_runner._save_run_result)
# --------------------------------------------------------------------------

def _split_metrics_to_run_csvs(scenario_metrics: Dict, scenario_dir: Path) -> Dict[str, Dict]:
    """
    Convert one run_scenario() return dict into 30 per-stream RunResult-like
    dicts, then return them keyed by config_name.

    Each per-stream dict has the same fields experiment_runner.RunResult writes
    via _save_run_result (see experiment_runner.py:1234-1261). The schema is:
      config_name, run_id, total_steps, recorded_steps, wall_time_seconds,
      avg_cav_amota, avg_cis_amota, avg_global_amota,
      hota, deta, assa,
      cav_*, cis_*, global_* metrics.
    """
    triple_metrics = scenario_metrics.get("triple_global_metrics", {})
    triple_amota_n = scenario_metrics.get("triple_global_amota_frames_count", {})
    triple_amotp_n = scenario_metrics.get("triple_global_amotp_frames_count", {})
    triple_hota = scenario_metrics.get("triple_global_hota", {})
    triple_paper = scenario_metrics.get("triple_paper_metrics", {})
    triple_paper_pe = scenario_metrics.get("triple_paper_metrics_per_ego", {})
    triple_v2v4real = scenario_metrics.get("triple_v2v4real_protocol", {})
    triple_v2v4real_mg = scenario_metrics.get("triple_v2v4real_merged_protocol", {})

    # Same canonical 30-key list used by experiment_runner.py:1125-1130.
    method_keys = [
        "baseline", "static", "gpem_linear", "gpem_quadratic", "gpem_polar",
        "ci_baseline", "ci_static", "ci_gpem_linear", "ci_gpem_quadratic", "ci_gpem_polar",
        "akf_baseline", "akf_static", "akf_gpem_linear", "akf_gpem_quadratic", "akf_gpem_polar",
        "pf_baseline", "pf_static", "pf_gpem_linear", "pf_gpem_quadratic", "pf_gpem_polar",
        "bici_baseline", "bici_static", "bici_gpem_linear", "bici_gpem_quadratic", "bici_gpem_polar",
        "sabre_baseline", "sabre_static", "sabre_gpem_linear", "sabre_gpem_quadratic", "sabre_gpem_polar",
    ]
    # Output names: V2V4Real has no penetration sweep, so we pin AV rate at 100.0%
    # so existing plotting code reads the structure correctly.
    output_names = [
        "baseline_av_100.0pct", "static_cov_av_100.0pct",
        "gpem_linear_av_100.0pct", "gpem_quadratic_av_100.0pct", "gpem_polar_av_100.0pct",
        "ci_baseline_av_100.0pct", "ci_static_av_100.0pct",
        "ci_gpem_linear_av_100.0pct", "ci_gpem_quadratic_av_100.0pct", "ci_gpem_polar_av_100.0pct",
        "akf_baseline_av_100.0pct", "akf_static_av_100.0pct",
        "akf_gpem_linear_av_100.0pct", "akf_gpem_quadratic_av_100.0pct", "akf_gpem_polar_av_100.0pct",
        "pf_baseline_av_100.0pct", "pf_static_av_100.0pct",
        "pf_gpem_linear_av_100.0pct", "pf_gpem_quadratic_av_100.0pct", "pf_gpem_polar_av_100.0pct",
        "bici_baseline_av_100.0pct", "bici_static_av_100.0pct",
        "bici_gpem_linear_av_100.0pct", "bici_gpem_quadratic_av_100.0pct", "bici_gpem_polar_av_100.0pct",
        "sabre_baseline_av_100.0pct", "sabre_static_av_100.0pct",
        "sabre_gpem_linear_av_100.0pct", "sabre_gpem_quadratic_av_100.0pct", "sabre_gpem_polar_av_100.0pct",
    ]

    iterations = scenario_metrics.get("iterations", 0)
    wall_time = scenario_metrics.get("wall_time_seconds", 0.0)

    per_stream: Dict[str, Dict] = {}
    for key, out_name in zip(method_keys, output_names):
        gm = triple_metrics.get(key, {})
        n_amota = max(triple_amota_n.get(key, 1), 1)
        n_amotp = max(triple_amotp_n.get(key, 1), 1)
        avg_amota = gm.get("amota", 0.0) / n_amota
        avg_amotp = gm.get("amotp", 0.0) / n_amotp
        h = triple_hota.get(key, {})
        p = triple_paper.get(key, {})
        ppe = triple_paper_pe.get(key, {})

        # AB3DMOT-protocol metrics, in V2V4Real-leaderboard format (percent).
        # Two views:
        #   paper_*          merged GT (spatial dedup, cross-fusion natural).
        #   paper_per_ego_*  per-ego GT (V2V4Real exact, paper-comparable).
        # Stored in global_metrics with `paper_*` / `paper_per_ego_*` prefixes
        # so RunResult.load_from_csv carries them through unchanged.
        paper_fields = {
            # Merged-GT view (ours).
            "paper_amota":      p.get("amota_pct", 0.0),
            "paper_amotp":      p.get("amotp_pct", 0.0),
            "paper_samota":     p.get("samota_pct", 0.0),
            "paper_mota":       p.get("mota_pct", 0.0),
            "paper_mt":         p.get("mt_pct", 0.0),
            "paper_ml":         p.get("ml_pct", 0.0),
            "paper_tp_total":   p.get("tp_total", 0),
            "paper_fp_total":   p.get("fp_total", 0),
            "paper_fn_total":   p.get("fn_total", 0),
            "paper_ids_total":  p.get("ids_total", 0),
            "paper_gt_total":   p.get("gt_total", 0),
            "paper_n_unique_gts": p.get("n_unique_gts", 0),
            # Per-ego view (V2V4Real-exact, leaderboard-comparable).
            "paper_per_ego_amota":     ppe.get("amota_pct", 0.0),
            "paper_per_ego_amotp":     ppe.get("amotp_pct", 0.0),
            "paper_per_ego_samota":    ppe.get("samota_pct", 0.0),
            "paper_per_ego_mota":      ppe.get("mota_pct", 0.0),
            "paper_per_ego_mt":        ppe.get("mt_pct", 0.0),
            "paper_per_ego_ml":        ppe.get("ml_pct", 0.0),
            "paper_per_ego_tp_total":  ppe.get("tp_total", 0),
            "paper_per_ego_fp_total":  ppe.get("fp_total", 0),
            "paper_per_ego_fn_total":  ppe.get("fn_total", 0),
            "paper_per_ego_ids_total": ppe.get("ids_total", 0),
            "paper_per_ego_gt_total":  ppe.get("gt_total", 0),
        }
        # V2V4Real-protocol view (DMSTrack-style: 3D/BEV-IoU @ 0.25 +
        # FP-ignore via min_height filter on LiDAR-only data). Apples-to-
        # apples with the published Table 4 numbers.
        v2v = triple_v2v4real.get(key, {})
        paper_fields.update({
            "v2v4real_amota":   100.0 * v2v.get("amota", 0.0),
            "v2v4real_amotp":   100.0 * v2v.get("amotp", 0.0),
            "v2v4real_samota":  100.0 * v2v.get("samota", 0.0),
            "v2v4real_mota":    100.0 * v2v.get("mota", 0.0),
            "v2v4real_mt":      100.0 * v2v.get("mt", 0.0),
            "v2v4real_ml":      100.0 * v2v.get("ml", 0.0),
            "v2v4real_tp_total": v2v.get("tp_total", 0),
            "v2v4real_fp_total": v2v.get("fp_total", 0),
            "v2v4real_gt_total": v2v.get("gt_total", 0),
            "v2v4real_max_recall_reached": v2v.get("max_recall_reached", 0.0),
        })
        # V2V protocol on merged GT (apples-to-apples with V2V4Real official 38.77).
        v2v_mg = triple_v2v4real_mg.get(key, {})
        paper_fields.update({
            "v2v4real_mg_amota":  100.0 * v2v_mg.get("amota", 0.0),
            "v2v4real_mg_amotp":  100.0 * v2v_mg.get("amotp", 0.0),
            "v2v4real_mg_samota": 100.0 * v2v_mg.get("samota", 0.0),
            "v2v4real_mg_mota":   100.0 * v2v_mg.get("mota", 0.0),
            "v2v4real_mg_tp_total": v2v_mg.get("tp_total", 0),
            "v2v4real_mg_fp_total": v2v_mg.get("fp_total", 0),
            "v2v4real_mg_gt_total": v2v_mg.get("gt_total", 0),
            "v2v4real_mg_max_recall_reached": v2v_mg.get("max_recall_reached", 0.0),
        })

        per_stream[out_name] = {
            "config_name": out_name,
            "run_id": 1,
            "total_steps": iterations,
            "recorded_steps": iterations,
            "wall_time_seconds": wall_time,
            "avg_cav_amota": 0.0,
            "avg_cis_amota": 0.0,
            "avg_global_amota": avg_amota,
            "avg_global_amotp": avg_amotp,
            "hota": h.get("hota", 0.0),
            "deta": h.get("deta", 0.0),
            "assa": h.get("assa", 0.0),
            "global_metrics": {
                **gm, "amota": avg_amota, "amotp": avg_amotp, **paper_fields
            },
            "cav_metrics": {},
            "cis_metrics": {},
        }
    return per_stream


def _write_run_csv(run_csv_path: Path, run_dict: Dict) -> None:
    """Write a per-stream RunResult to <run_csv_path>/run_01.csv. Matches
    experiment_runner.py:1238-1261 exactly so RunResult.load_from_csv can
    parse it without modification."""
    run_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(run_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        w.writerow(["config_name", run_dict["config_name"]])
        w.writerow(["run_id", run_dict["run_id"]])
        w.writerow(["total_steps", run_dict["total_steps"]])
        w.writerow(["recorded_steps", run_dict["recorded_steps"]])
        w.writerow(["wall_time_seconds", run_dict["wall_time_seconds"]])
        w.writerow(["avg_cav_amota", run_dict["avg_cav_amota"]])
        w.writerow(["avg_cis_amota", run_dict["avg_cis_amota"]])
        w.writerow(["avg_global_amota", run_dict["avg_global_amota"]])
        w.writerow(["hota", run_dict["hota"]])
        w.writerow(["deta", run_dict["deta"]])
        w.writerow(["assa", run_dict["assa"]])
        for k, v in run_dict.get("cav_metrics", {}).items():
            w.writerow([f"cav_{k}", v])
        for k, v in run_dict.get("cis_metrics", {}).items():
            w.writerow([f"cis_{k}", v])
        for k, v in run_dict.get("global_metrics", {}).items():
            w.writerow([f"global_{k}", v])


# --------------------------------------------------------------------------
# Per-split aggregation (mirrors experiment_runner._aggregate_results +
# _save_summary)
# --------------------------------------------------------------------------

def _aggregate_per_config(per_scenario: List[Dict]) -> Dict:
    """Aggregate a list of per-scenario run-dicts (same config_name) into the
    same 22-field schema produced by experiment_runner._save_summary."""
    n = len(per_scenario)
    if n == 0:
        return {
            "num_runs": 0,
            "avg_global_amota_mean": 0.0, "avg_global_amota_std": 0.0,
            "avg_global_amotp_mean": 0.0, "avg_global_amotp_std": 0.0,
            "avg_cis_amota_mean": 0.0,    "avg_cis_amota_std": 0.0,
            "avg_cis_amotp_mean": 0.0,    "avg_cis_amotp_std": 0.0,
            "avg_cav_amota_mean": 0.0,    "avg_cav_amota_std": 0.0,
            "avg_cav_amotp_mean": 0.0,    "avg_cav_amotp_std": 0.0,
            "avg_hota_mean": 0.0, "avg_hota_std": 0.0,
            "avg_deta_mean": 0.0, "avg_deta_std": 0.0,
            "avg_assa_mean": 0.0, "avg_assa_std": 0.0,
            "avg_perception_anomalies_mean": 0.0,
            "avg_perception_anomalies_std": 0.0,
            "total_anomaly_detections": 0,
        }
    g_amota = [r["avg_global_amota"] for r in per_scenario]
    g_amotp = [r["avg_global_amotp"] for r in per_scenario]
    hotas   = [r["hota"]             for r in per_scenario]
    detas   = [r["deta"]             for r in per_scenario]
    assas   = [r["assa"]             for r in per_scenario]
    # AB3DMOT paper metrics: pulled out of global_metrics by name (in percent).
    # Merged-GT view (ours).
    paper_amota  = [r["global_metrics"].get("paper_amota", 0.0)  for r in per_scenario]
    paper_amotp  = [r["global_metrics"].get("paper_amotp", 0.0)  for r in per_scenario]
    paper_samota = [r["global_metrics"].get("paper_samota", 0.0) for r in per_scenario]
    paper_mota   = [r["global_metrics"].get("paper_mota", 0.0)   for r in per_scenario]
    paper_mt     = [r["global_metrics"].get("paper_mt", 0.0)     for r in per_scenario]
    paper_ml     = [r["global_metrics"].get("paper_ml", 0.0)     for r in per_scenario]
    paper_ids    = [r["global_metrics"].get("paper_ids_total", 0) for r in per_scenario]
    paper_tp     = [r["global_metrics"].get("paper_tp_total", 0) for r in per_scenario]
    paper_fp     = [r["global_metrics"].get("paper_fp_total", 0) for r in per_scenario]
    paper_fn     = [r["global_metrics"].get("paper_fn_total", 0) for r in per_scenario]
    paper_gt     = [r["global_metrics"].get("paper_gt_total", 0) for r in per_scenario]
    paper_uniq_gt= [r["global_metrics"].get("paper_n_unique_gts", 0) for r in per_scenario]
    # Per-ego view (V2V4Real-exact, leaderboard-comparable).
    pe_amota     = [r["global_metrics"].get("paper_per_ego_amota", 0.0)  for r in per_scenario]
    pe_amotp     = [r["global_metrics"].get("paper_per_ego_amotp", 0.0)  for r in per_scenario]
    pe_samota    = [r["global_metrics"].get("paper_per_ego_samota", 0.0) for r in per_scenario]
    pe_mota      = [r["global_metrics"].get("paper_per_ego_mota", 0.0)   for r in per_scenario]
    pe_mt        = [r["global_metrics"].get("paper_per_ego_mt", 0.0)     for r in per_scenario]
    pe_ml        = [r["global_metrics"].get("paper_per_ego_ml", 0.0)     for r in per_scenario]
    pe_ids       = [r["global_metrics"].get("paper_per_ego_ids_total", 0) for r in per_scenario]
    pe_tp        = [r["global_metrics"].get("paper_per_ego_tp_total", 0) for r in per_scenario]
    pe_fp        = [r["global_metrics"].get("paper_per_ego_fp_total", 0) for r in per_scenario]
    pe_fn        = [r["global_metrics"].get("paper_per_ego_fn_total", 0) for r in per_scenario]
    pe_gt        = [r["global_metrics"].get("paper_per_ego_gt_total", 0) for r in per_scenario]
    # V2V4Real-protocol view (DMSTrack KITTI-eval: IoU @ 0.25 + FP-ignore via min_height filter).
    v2v_amota   = [r["global_metrics"].get("v2v4real_amota", 0.0)   for r in per_scenario]
    v2v_amotp   = [r["global_metrics"].get("v2v4real_amotp", 0.0)   for r in per_scenario]
    v2v_samota  = [r["global_metrics"].get("v2v4real_samota", 0.0)  for r in per_scenario]
    v2v_mota    = [r["global_metrics"].get("v2v4real_mota", 0.0)    for r in per_scenario]
    v2v_mt      = [r["global_metrics"].get("v2v4real_mt", 0.0)      for r in per_scenario]
    v2v_ml      = [r["global_metrics"].get("v2v4real_ml", 0.0)      for r in per_scenario]
    v2v_tp      = [r["global_metrics"].get("v2v4real_tp_total", 0)  for r in per_scenario]
    v2v_fp      = [r["global_metrics"].get("v2v4real_fp_total", 0)  for r in per_scenario]
    v2v_gt      = [r["global_metrics"].get("v2v4real_gt_total", 0)  for r in per_scenario]
    v2v_maxr    = [r["global_metrics"].get("v2v4real_max_recall_reached", 0.0) for r in per_scenario]
    # V2V protocol on merged GT (apples-to-apples with V2V4Real Table 4 baseline).
    mg_amota    = [r["global_metrics"].get("v2v4real_mg_amota", 0.0)   for r in per_scenario]
    mg_amotp    = [r["global_metrics"].get("v2v4real_mg_amotp", 0.0)   for r in per_scenario]
    mg_samota   = [r["global_metrics"].get("v2v4real_mg_samota", 0.0)  for r in per_scenario]
    mg_mota     = [r["global_metrics"].get("v2v4real_mg_mota", 0.0)    for r in per_scenario]
    mg_tp       = [r["global_metrics"].get("v2v4real_mg_tp_total", 0)  for r in per_scenario]
    mg_fp       = [r["global_metrics"].get("v2v4real_mg_fp_total", 0)  for r in per_scenario]
    mg_gt       = [r["global_metrics"].get("v2v4real_mg_gt_total", 0)  for r in per_scenario]
    mg_maxr     = [r["global_metrics"].get("v2v4real_mg_max_recall_reached", 0.0) for r in per_scenario]

    def _std(xs): return stdev(xs) if len(xs) > 1 else 0.0

    def _wmean(values, weights):
        """GT-weighted mean. Falls back to simple mean when weights sum to 0."""
        tot = sum(weights)
        if tot <= 0:
            return mean(values) if values else 0.0
        return sum(v * w for v, w in zip(values, weights)) / tot

    return {
        "num_runs": n,
        "avg_global_amota_mean": mean(g_amota),
        "avg_global_amota_std":  _std(g_amota),
        "avg_global_amotp_mean": mean(g_amotp),
        "avg_global_amotp_std":  _std(g_amotp),
        "avg_cis_amota_mean": 0.0, "avg_cis_amota_std": 0.0,
        "avg_cis_amotp_mean": 0.0, "avg_cis_amotp_std": 0.0,
        "avg_cav_amota_mean": 0.0, "avg_cav_amota_std": 0.0,
        "avg_cav_amotp_mean": 0.0, "avg_cav_amotp_std": 0.0,
        "avg_hota_mean": mean(hotas), "avg_hota_std": _std(hotas),
        "avg_deta_mean": mean(detas), "avg_deta_std": _std(detas),
        "avg_assa_mean": mean(assas), "avg_assa_std": _std(assas),
        # V2V4Real-paper-protocol metrics (AB3DMOT). All scalars in percent
        # to match the V2V4Real leaderboard format.
        # Merged-GT view (ours, default). All rate metrics are GT-count weighted.
        "paper_amota_mean":  _wmean(paper_amota,  paper_gt),  "paper_amota_std":  _std(paper_amota),
        "paper_amotp_mean":  _wmean(paper_amotp,  paper_gt),  "paper_amotp_std":  _std(paper_amotp),
        "paper_samota_mean": _wmean(paper_samota, paper_gt),  "paper_samota_std": _std(paper_samota),
        "paper_mota_mean":   _wmean(paper_mota,   paper_gt),  "paper_mota_std":   _std(paper_mota),
        "paper_mt_mean":     _wmean(paper_mt,     paper_gt),  "paper_mt_std":     _std(paper_mt),
        "paper_ml_mean":     _wmean(paper_ml,     paper_gt),  "paper_ml_std":     _std(paper_ml),
        "paper_ids_total":   int(sum(paper_ids)),
        "paper_tp_total":    int(sum(paper_tp)),
        "paper_fp_total":    int(sum(paper_fp)),
        "paper_fn_total":    int(sum(paper_fn)),
        "paper_gt_total":    int(sum(paper_gt)),
        "paper_n_unique_gts":int(sum(paper_uniq_gt)),
        # Per-ego view (V2V4Real-exact). These are the numbers that go
        # in the leaderboard table. GT-count weighted.
        "paper_per_ego_amota_mean":  _wmean(pe_amota,  pe_gt),  "paper_per_ego_amota_std":  _std(pe_amota),
        "paper_per_ego_amotp_mean":  _wmean(pe_amotp,  pe_gt),  "paper_per_ego_amotp_std":  _std(pe_amotp),
        "paper_per_ego_samota_mean": _wmean(pe_samota, pe_gt),  "paper_per_ego_samota_std": _std(pe_samota),
        "paper_per_ego_mota_mean":   _wmean(pe_mota,   pe_gt),  "paper_per_ego_mota_std":   _std(pe_mota),
        "paper_per_ego_mt_mean":     _wmean(pe_mt,     pe_gt),  "paper_per_ego_mt_std":     _std(pe_mt),
        "paper_per_ego_ml_mean":     _wmean(pe_ml,     pe_gt),  "paper_per_ego_ml_std":     _std(pe_ml),
        "paper_per_ego_ids_total":   int(sum(pe_ids)),
        "paper_per_ego_tp_total":    int(sum(pe_tp)),
        "paper_per_ego_fp_total":    int(sum(pe_fp)),
        "paper_per_ego_fn_total":    int(sum(pe_fn)),
        "paper_per_ego_gt_total":    int(sum(pe_gt)),
        # V2V4Real-protocol view (DMSTrack KITTI-eval: IoU @ 0.25 + FP-ignore).
        # Apples-to-apples with V2V4Real Table 4. NOT methodologically clean
        # (the FP-ignore quirk inflates AMOTA — see docs/V2V4REAL_REPRODUCIBILITY_NOTES.md).
        # GT-count weighted.
        "v2v4real_amota_mean":  _wmean(v2v_amota,  v2v_gt),  "v2v4real_amota_std":  _std(v2v_amota),
        "v2v4real_amotp_mean":  _wmean(v2v_amotp,  v2v_gt),  "v2v4real_amotp_std":  _std(v2v_amotp),
        "v2v4real_samota_mean": _wmean(v2v_samota, v2v_gt),  "v2v4real_samota_std": _std(v2v_samota),
        "v2v4real_mota_mean":   _wmean(v2v_mota,   v2v_gt),  "v2v4real_mota_std":   _std(v2v_mota),
        "v2v4real_mt_mean":     _wmean(v2v_mt,     v2v_gt),  "v2v4real_mt_std":     _std(v2v_mt),
        "v2v4real_ml_mean":     _wmean(v2v_ml,     v2v_gt),  "v2v4real_ml_std":     _std(v2v_ml),
        "v2v4real_tp_total":    int(sum(v2v_tp)),
        "v2v4real_fp_total":    int(sum(v2v_fp)),
        "v2v4real_gt_total":    int(sum(v2v_gt)),
        "v2v4real_max_recall_reached_mean": mean(v2v_maxr),
        # V2V4Real protocol on MERGED GT (apples-to-apples with their published
        # 38.77 baseline — they use single-instance KITTI GT, ≈ our merged GT).
        "v2v4real_mg_amota_mean":  _wmean(mg_amota,  mg_gt),  "v2v4real_mg_amota_std":  _std(mg_amota),
        "v2v4real_mg_amotp_mean":  _wmean(mg_amotp,  mg_gt),  "v2v4real_mg_amotp_std":  _std(mg_amotp),
        "v2v4real_mg_samota_mean": _wmean(mg_samota, mg_gt),  "v2v4real_mg_samota_std": _std(mg_samota),
        "v2v4real_mg_mota_mean":   _wmean(mg_mota,   mg_gt),  "v2v4real_mg_mota_std":   _std(mg_mota),
        "v2v4real_mg_tp_total":    int(sum(mg_tp)),
        "v2v4real_mg_fp_total":    int(sum(mg_fp)),
        "v2v4real_mg_gt_total":    int(sum(mg_gt)),
        "v2v4real_mg_max_recall_reached_mean": mean(mg_maxr),
        "avg_perception_anomalies_mean": 0.0,
        "avg_perception_anomalies_std":  0.0,
        "total_anomaly_detections": 0,
    }


def _write_split_summary(split_dir: Path, split_name: str, scenarios_in_split: List[Dict]) -> None:
    """Write one summary.json for a given split."""
    split_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "suite_name": "V2V4Real_Eval",
        "split": split_name,
        "completed_at": datetime.now().isoformat(),
        "total_runs": len(scenarios_in_split),
        "results": {},
    }
    if not scenarios_in_split:
        with open(split_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return

    # Pivot: per-config-name list of per-scenario run dicts.
    per_config: Dict[str, List[Dict]] = {}
    for scen in scenarios_in_split:
        per_stream = scen["per_stream"]
        for config_name, run_dict in per_stream.items():
            per_config.setdefault(config_name, []).append(run_dict)

    for config_name, runs in per_config.items():
        summary["results"][config_name] = _aggregate_per_config(runs)

    with open(split_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------

def run_suite(
    export_dir: Path,
    output_dir: Path,
    splits: Optional[List[str]] = None,
    workers: int = 8,
    detector_name: str = "centerpoint_100m_on_v2v4real",
    localizer_name: str = "rtk_v2v4real",
    detector_max_range: float = 100.0,
    variance_floor: Optional[float] = None,
    score_threshold: float = 0.3,
    use_static_matching: bool = True,
    vehicle_classes: Optional[List[str]] = None,
    class_thresholds: Optional[Dict[str, float]] = None,
    self_report_egos: bool = True,
    p_tp_birth_gate: Optional[float] = None,
    lifecycle_mode: str = "legacy",
    ab3dmot_min_hits: int = 3,
    ab3dmot_max_age: int = 2,
    per_ego_nms_iou: float = 1.0,
    combined_nms_iou: float = 1.0,
    enable_inactive_preservation: bool = False,
    inactive_grace_s: float = 1.0,
    confirm_log_odds: Optional[float] = None,
    kill_log_odds: Optional[float] = None,
    log_lr_miss: Optional[float] = None,
    mahal_weight: float = 0.6,
    iou_weight: float = 0.4,
    mahal_gate: float = 13.82,
    tape_score_miss_decay: Optional[float] = None,
) -> None:
    """Run the suite and write per-split summary.json files."""
    export_dir = Path(export_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = discover_scenarios(export_dir)
    if splits:
        wanted = set(splits)
        scenarios = [(sp, sn, sd) for (sp, sn, sd) in scenarios if sp in wanted]

    if not scenarios:
        print(f"  ✗ No scenarios found in {export_dir} (splits filter: {splits})")
        return

    # Default vehicle-class filter (drops barriers/cones/peds — see
    # docs/V2V4REAL_EVALUATION.md "Class filtering" section).
    if vehicle_classes is None:
        vehicle_classes = ["car", "truck", "bus", "construction_vehicle"]

    print("=" * 70)
    print(f"V2V4Real Evaluation Suite")
    print(f"  Export dir:    {export_dir}")
    print(f"  Output dir:    {output_dir}")
    print(f"  Detector:      {detector_name}")
    print(f"  Localizer:     {localizer_name}")
    print(f"  Max range:     {detector_max_range:.1f} m")
    print(f"  Score thresh:  {score_threshold}")
    print(f"  Vehicle clsss: {sorted(vehicle_classes) if vehicle_classes else 'ALL (no filter)'}")
    if class_thresholds:
        print(f"  Class thresh:  {class_thresholds}")
    print(f"  Workers:       {workers}")
    print(f"  Scenarios:     {len(scenarios)}")
    print("=" * 70)
    by_split: Dict[str, int] = {}
    for sp, _, _ in scenarios:
        by_split[sp] = by_split.get(sp, 0) + 1
    for sp in SPLITS:
        if sp in by_split:
            print(f"    {sp:8s} {by_split[sp]} scenarios")

    # Save the suite config for reproducibility.
    suite_config = {
        "export_dir": str(export_dir),
        "output_dir": str(output_dir),
        "detector_name": detector_name,
        "localizer_name": localizer_name,
        "detector_max_range": detector_max_range,
        "variance_floor": variance_floor,
        "score_threshold": score_threshold,
        "vehicle_classes": sorted(vehicle_classes) if vehicle_classes else None,
        "class_thresholds": class_thresholds,
        "use_static_matching": use_static_matching,
        "p_tp_birth_gate": p_tp_birth_gate,
        "lifecycle_mode": lifecycle_mode,
        "ab3dmot_min_hits": ab3dmot_min_hits,
        "ab3dmot_max_age":  ab3dmot_max_age,
        "per_ego_nms_iou": per_ego_nms_iou,
        "combined_nms_iou": combined_nms_iou,
        "enable_inactive_preservation": enable_inactive_preservation,
        "inactive_grace_s": inactive_grace_s,
        "confirm_log_odds": confirm_log_odds,
        "kill_log_odds": kill_log_odds,
        "log_lr_miss": log_lr_miss,
        "mahal_weight": mahal_weight,
        "iou_weight": iou_weight,
        "mahal_gate": mahal_gate,
        "tape_score_miss_decay": tape_score_miss_decay,
        "splits_filter": splits,
        "n_scenarios": len(scenarios),
    }
    with open(output_dir / "suite_config.json", "w") as f:
        json.dump(suite_config, f, indent=2)

    replay_cfg = {
        "detector_name": detector_name,
        "localizer_name": localizer_name,
        "detector_max_range": detector_max_range,
        "variance_floor": variance_floor,
        "score_threshold": score_threshold,
        "vehicle_classes": list(vehicle_classes) if vehicle_classes else None,
        "class_thresholds": class_thresholds,
        "use_static_matching": use_static_matching,
        "self_report_egos": self_report_egos,
        "p_tp_birth_gate": p_tp_birth_gate,
        "lifecycle_mode": lifecycle_mode,
        "ab3dmot_min_hits": ab3dmot_min_hits,
        "ab3dmot_max_age":  ab3dmot_max_age,
        "per_ego_nms_iou": per_ego_nms_iou,
        "combined_nms_iou": combined_nms_iou,
        "enable_inactive_preservation": enable_inactive_preservation,
        "inactive_grace_s": inactive_grace_s,
        "confirm_log_odds": confirm_log_odds,
        "kill_log_odds": kill_log_odds,
        "log_lr_miss": log_lr_miss,
        "mahal_weight": mahal_weight,
        "iou_weight": iou_weight,
        "mahal_gate": mahal_gate,
        "tape_score_miss_decay": tape_score_miss_decay,
    }

    # Per-split scenario buckets, populated as workers finish.
    split_results: Dict[str, List[Dict]] = {sp: [] for sp in SPLITS}
    failures: List[Tuple[str, str]] = []

    t0 = time.time()
    # One worker per scenario.
    work = [
        (f"{split}__{name}", path, replay_cfg)
        for split, name, path in scenarios
    ]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_replay_one, w): w for w in work}
        completed = 0
        for fut in as_completed(futures):
            scenario_label = futures[fut][0]
            try:
                metrics = fut.result()
            except Exception as e:
                completed += 1
                failures.append((scenario_label, str(e)))
                print(f"  [{completed}/{len(work)}] ✗ {scenario_label}: {e}")
                continue

            per_stream = _split_metrics_to_run_csvs(metrics, Path(metrics["_scenario_dir"]))
            split = scenario_label.split("__", 1)[0]
            scenario_name = scenario_label.split("__", 1)[1]

            # Write per-config run_01.csv files for this scenario.
            scenario_out_dir = output_dir / scenario_label
            scenario_out_dir.mkdir(parents=True, exist_ok=True)
            for config_name, run_dict in per_stream.items():
                _write_run_csv(scenario_out_dir / config_name / "run_01.csv", run_dict)

            split_results[split].append({
                "scenario_name": scenario_name,
                "scenario_label": scenario_label,
                "per_stream": per_stream,
                "iterations": metrics.get("iterations", 0),
                "wall_time_seconds": metrics.get("wall_time_seconds", 0.0),
            })

            completed += 1
            elapsed = time.time() - t0
            eta = (elapsed / completed) * (len(work) - completed) if completed else 0
            print(
                f"  [{completed}/{len(work)}] ✓ {scenario_label} "
                f"({metrics.get('iterations', 0)} frames, "
                f"{metrics.get('wall_time_seconds', 0.0):.1f}s) "
                f"[elapsed {elapsed/60:.1f}m, ETA {eta/60:.1f}m]"
            )

    # Per-split summary.json files.
    for split in SPLITS:
        if not split_results[split]:
            continue
        _write_split_summary(output_dir / split, split, split_results[split])

    # Top-level meta.json.
    meta = {
        "completed_at": datetime.now().isoformat(),
        "total_scenarios": len(scenarios),
        "succeeded": sum(len(v) for v in split_results.values()),
        "failed": len(failures),
        "failures": failures,
        "wall_time_seconds": time.time() - t0,
        "config": suite_config,
    }
    with open(output_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print()
    print("=" * 70)
    print(f"  Succeeded: {meta['succeeded']}/{meta['total_scenarios']}")
    print(f"  Failed:    {meta['failed']}")
    print(f"  Wall time: {meta['wall_time_seconds']/60:.1f} min")
    print(f"  Output:    {output_dir}")
    print("=" * 70)
