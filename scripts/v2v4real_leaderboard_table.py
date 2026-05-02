#!/usr/bin/env python3
"""
Dump a V2V4Real leaderboard-format comparison table.

Reads one or more `results/V2V4Real_*/{split}/summary.json` files and prints
a wide table with three AMOTA variants and HOTA for every row.

Columns
-------
  V2V-AMOTA  : V2V4Real protocol (per-ego, FP-ignore) — apples-to-apples with paper
  PE-AMOTA   : per-ego strict, FPs fully counted
  MG-AMOTA   : merged-GT (our cooperative eval, n/a for single-ego baselines)
  HOTA / AssA: standard eval (FPs fully counted)
  + AMOTP, sAMOTA, MOTA, MT, ML, Cost(MB), IDS, GT

Usage:
  scripts/v2v4real_leaderboard_table.py <results_dir> [<results_dir> ...] \\
      [--split test] [--md|--latex|--ascii]

  # Auto-discover all result dirs in a benchmark dir:
  scripts/v2v4real_leaderboard_table.py \\
      --benchmark-dir results/V2V4Real_BENCHMARK_FINAL
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# V2V4Real paper Table 4 ("DMSTrack" paper, V2V4Real version) — the rows we
# print as reference at the top of the table.
PAPER_ROWS: List[Tuple[str, float, float, float, float, float, float, str]] = [
    # (method,           AMOTA, AMOTP, sAMOTA, MOTA,  MT,    ML,    cost_MB)
    ("No Fusion",        16.08, 41.60, 53.84, 43.46, 29.41, 60.18, "0"),
    ("Late Fusion",      29.28, 51.08, 71.05, 59.89, 45.25, 31.22, "0.003"),
    ("Early Fusion",     26.19, 48.15, 67.34, 60.87, 40.95, 32.13, "0.96"),
    ("F-Cooper",         23.29, 43.11, 65.63, 58.34, 35.75, 38.91, "0.20"),
    ("AttFuse",          28.64, 50.48, 73.21, 63.03, 46.38, 28.05, "0.20"),
    ("V2VNet",           30.48, 54.28, 75.53, 64.85, 48.19, 27.83, "0.20"),
    ("V2X-ViT",          30.85, 54.32, 74.01, 64.82, 45.93, 26.47, "0.20"),
    ("CoBEVT",           32.12, 55.61, 77.65, 63.75, 47.29, 30.32, "0.20"),
    ("CoBEVT (*)",       37.16, 57.20, 84.54, 84.14, 57.07, 15.83, "0.20"),
    ("DMSTrack",         43.52, 57.94, 91.50, 88.32, 68.35, 13.19, "0.0073"),
]


# Filter group prefixes: which stream-name patterns belong to which filter.
FILTER_GROUPS = {
    "EKF":   {"baseline": "baseline_av_100.0pct",         "static": "static_cov_av_100.0pct",
              "lin":     "gpem_linear_av_100.0pct",     "quad":    "gpem_quadratic_av_100.0pct",
              "polar":   "gpem_polar_av_100.0pct"},
    "CI":    {"baseline": "ci_baseline_av_100.0pct",      "static": "ci_static_av_100.0pct",
              "lin":     "ci_gpem_linear_av_100.0pct",  "quad":    "ci_gpem_quadratic_av_100.0pct",
              "polar":   "ci_gpem_polar_av_100.0pct"},
    "AKF":   {"baseline": "akf_baseline_av_100.0pct",     "static": "akf_static_av_100.0pct",
              "lin":     "akf_gpem_linear_av_100.0pct", "quad":    "akf_gpem_quadratic_av_100.0pct",
              "polar":   "akf_gpem_polar_av_100.0pct"},
    "PF":    {"baseline": "pf_baseline_av_100.0pct",      "static": "pf_static_av_100.0pct",
              "lin":     "pf_gpem_linear_av_100.0pct",  "quad":    "pf_gpem_quadratic_av_100.0pct",
              "polar":   "pf_gpem_polar_av_100.0pct"},
    "BICI":  {"baseline": "bici_baseline_av_100.0pct",    "static": "bici_static_av_100.0pct",
              "lin":     "bici_gpem_linear_av_100.0pct","quad":    "bici_gpem_quadratic_av_100.0pct",
              "polar":   "bici_gpem_polar_av_100.0pct"},
    "SABRE": {"baseline": "sabre_baseline_av_100.0pct",   "static": "sabre_static_av_100.0pct",
              "lin":     "sabre_gpem_linear_av_100.0pct","quad":   "sabre_gpem_quadratic_av_100.0pct",
              "polar":   "sabre_gpem_polar_av_100.0pct"},
}


_DETECTOR_PRETTY = {
    "centerpoint_100m_on_v2v4real":                   "CenterPoint (ZS)",
    "centerpoint_54m_v2v4real_finetune":               "CPft",
    "centerpoint_54m_v2v4real_finetune_{vehicle}":     "CPft",
    "pointpillar_v2v4real":                            "PointPillar (LF)",
    "pointpillar_v2v4real_{vehicle}":                  "PointPillar (LF)",
    "pointpillar_v2v4real_score02":                    "PointPillar",
    "pointpillar_v2v4real_score02_{vehicle}":          "PointPillar",
    "centerpoint":                "CenterPoint(sim)",
    "bev_fusion":                 "BEVFusion(sim)",
    "detr3d":                     "DETR3D(sim)",
}


def _parse_label_from_path(path: Path) -> str:
    cfg_path = path / "suite_config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            det = cfg.get("detector_name", "")
            if det in _DETECTOR_PRETTY:
                return _DETECTOR_PRETTY[det]
            stripped = det.replace("_{vehicle}", "")
            if stripped in _DETECTOR_PRETTY:
                return _DETECTOR_PRETTY[stripped]
            if det:
                return det
        except Exception:
            pass
    name = path.name
    if name.startswith("V2V4Real_"):
        parts = name[len("V2V4Real_"):].rsplit("_", 2)
        return parts[0] if parts else name
    return name


def _bandwidth_cost_mb(suite_config: Dict, export_dir: Optional[Path] = None) -> Optional[float]:
    if export_dir is None:
        return None
    try:
        from v2v4real_bandwidth_calc import (
            count_detections_per_frame,
            compute_bandwidth,
        )
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from v2v4real_bandwidth_calc import count_detections_per_frame, compute_bandwidth

    score_threshold = suite_config.get("score_threshold", 0.30)
    classes = suite_config.get("vehicle_classes")

    total_counts: Dict = {}
    for scenario in Path(export_dir).iterdir():
        if not scenario.is_dir() or not (scenario / "detections.csv").exists():
            continue
        c = count_detections_per_frame(scenario, score_threshold, classes)
        for fid, n in c.items():
            total_counts[(scenario.name, fid)] = n

    if not total_counts:
        return None

    counts_for_compute = {i: v for i, v in enumerate(total_counts.values())}
    res = compute_bandwidth(counts_for_compute, n_egos=2, fps=10.0)
    return res.get("strategy_A_MB_per_frame")


def _best_gpem_and_baseline(
    summary_results: Dict, filter_group: str
) -> Tuple[Optional[Tuple[str, Dict]], Optional[Tuple[str, Dict]]]:
    """Return (best_gpem_entry, baseline_entry) for one filter group.

    best GPEM = argmax paper_per_ego_amota_mean over (lin, quad, polar).
    """
    cfg = FILTER_GROUPS.get(filter_group)
    if not cfg:
        return None, None

    base_key = cfg["baseline"]
    base = (base_key, summary_results.get(base_key)) if base_key in summary_results else None

    gpem_modes = [("lin", cfg["lin"]), ("quad", cfg["quad"]), ("polar", cfg["polar"])]
    best: Optional[Tuple[str, Dict]] = None
    for label, key in gpem_modes:
        r = summary_results.get(key)
        if r is None:
            continue
        val = r.get("paper_per_ego_amota_mean", 0.0)
        if best is None or val > best[1].get("paper_per_ego_amota_mean", 0.0):
            best = (label, r)

    return best, base


# ---------------------------------------------------------------------------
# Unified row building
# ---------------------------------------------------------------------------

def _fv(v: Optional[float], fmt: str = ".2f") -> str:
    """Format optional float; None → '—'."""
    if v is None:
        return "—"
    return format(float(v), fmt)


def _fi(v: Optional[int]) -> str:
    """Format optional int; None → '—'."""
    if v is None:
        return "—"
    return str(int(v))


def _row_paper(method: str, am: float, amp: float, sam: float,
               mota: float, mt: float, ml: float, cost: str) -> Dict:
    """Unified row dict for a V2V4Real paper reference row."""
    return {
        "method": method, "section": "paper",
        "v2v_amota": am, "pe_amota": None, "mg_amota": None,
        "hota": None, "assa": None,
        "amotp": amp, "samota": sam, "mota": mota, "mt": mt, "ml": ml,
        "cost": cost, "ids": None, "gt": None,
    }


def _row_from_summary(r: Dict, method: str, section: str, cost: str) -> Dict:
    """Unified row dict from summary.json (our configs).

    HOTA is stored as [0,1] fraction in summary.json → multiply ×100.
    """
    hota_raw = r.get("avg_hota_mean")
    assa_raw = r.get("avg_assa_mean")
    return {
        "method": method, "section": section,
        "v2v_amota": r.get("v2v4real_amota_mean"),
        "pe_amota":  r.get("paper_per_ego_amota_mean"),
        "mg_amota":  r.get("paper_amota_mean"),
        "hota":  hota_raw * 100.0 if hota_raw is not None else None,
        "assa":  assa_raw * 100.0 if assa_raw is not None else None,
        "amotp":  r.get("paper_per_ego_amotp_mean"),
        "samota": r.get("paper_per_ego_samota_mean"),
        "mota":   r.get("paper_per_ego_mota_mean"),
        "mt":     r.get("paper_per_ego_mt_mean"),
        "ml":     r.get("paper_per_ego_ml_mean"),
        "cost":   cost,
        "ids":    r.get("paper_per_ego_ids_total"),
        "gt":     r.get("paper_per_ego_gt_total"),
    }


def _row_from_protocol(r: Dict, method: str, section: str, cost: str) -> Dict:
    """Unified row dict from summary_v2v4real_protocol.json (precomputed/dmstrack).

    HOTA is already in percentage in protocol JSONs.
    MG-AMOTA is n/a for single-ego approaches.
    """
    return {
        "method": method, "section": section,
        "v2v_amota": r.get("v2v4real_amota_mean"),
        "pe_amota":  r.get("paper_per_ego_amota_mean"),
        "mg_amota":  None,
        "hota":  r.get("hota_mean"),
        "assa":  r.get("assa_mean"),
        "amotp":  r.get("paper_per_ego_amotp_mean"),
        "samota": r.get("paper_per_ego_samota_mean"),
        "mota":   r.get("paper_per_ego_mota_mean"),
        "mt":     r.get("paper_per_ego_mt_mean"),
        "ml":     r.get("paper_per_ego_ml_mean"),
        "cost":   cost,
        "ids":    r.get("paper_per_ego_ids_total"),
        "gt":     r.get("paper_per_ego_gt_total"),
    }


# ---------------------------------------------------------------------------
# ASCII / Markdown / LaTeX formatters
# ---------------------------------------------------------------------------

_W_METHOD = 30
_ASCII_HEADER = (
    f"{'Method':<{_W_METHOD}s}  "
    f"{'V2V-AMOTA':>10s}  {'PE-AMOTA':>8s}  {'MG-AMOTA':>8s}  "
    f"{'HOTA':>7s}  {'AssA':>7s}  "
    f"{'AMOTP':>9s}  {'sAMOTA':>9s}  {'MOTA':>8s}  "
    f"{'MT':>6s}  {'ML':>6s}  {'Cost(MB)':>10s}  "
    f"{'IDS':>6s}  {'GT':>7s}"
)


def _format_row_ascii(row: Dict) -> str:
    return (
        f"{row['method']:<{_W_METHOD}s}  "
        f"{_fv(row['v2v_amota']):>10s}  "
        f"{_fv(row['pe_amota']):>8s}  "
        f"{_fv(row['mg_amota']):>8s}  "
        f"{_fv(row['hota']):>7s}  "
        f"{_fv(row['assa']):>7s}  "
        f"{_fv(row['amotp']):>9s}  "
        f"{_fv(row['samota']):>9s}  "
        f"{_fv(row['mota']):>8s}  "
        f"{_fv(row['mt']):>6s}  "
        f"{_fv(row['ml']):>6s}  "
        f"{str(row['cost'] or '—'):>10s}  "
        f"{_fi(row['ids']):>6s}  "
        f"{_fi(row['gt']):>7s}"
    )


def _format_row_md(row: Dict) -> str:
    return (
        f"| {row['method']} | "
        f"{_fv(row['v2v_amota'])} | {_fv(row['pe_amota'])} | {_fv(row['mg_amota'])} | "
        f"{_fv(row['hota'])} | {_fv(row['assa'])} | "
        f"{_fv(row['amotp'])} | {_fv(row['samota'])} | {_fv(row['mota'])} | "
        f"{_fv(row['mt'])} | {_fv(row['ml'])} | {row['cost'] or '—'} | "
        f"{_fi(row['ids'])} | {_fi(row['gt'])} |"
    )


def _format_row_latex(row: Dict) -> str:
    return (
        f"{row['method']} & {_fv(row['v2v_amota'])} & {_fv(row['pe_amota'])} & {_fv(row['mg_amota'])} & "
        f"{_fv(row['hota'])} & {_fv(row['assa'])} & "
        f"{_fv(row['amotp'])} & {_fv(row['samota'])} & {_fv(row['mota'])} & "
        f"{_fv(row['mt'])} & {_fv(row['ml'])} & {row['cost'] or '--'} & "
        f"{_fi(row['ids'])} & {_fi(row['gt'])} \\\\"
    )


# ---------------------------------------------------------------------------
# Benchmark-dir helper (precomputed/dmstrack configs)
# ---------------------------------------------------------------------------

_BENCHMARK_KEY = "dmstrack_av_100.0pct"


def _load_benchmark_rows(benchmark_dir: Path, split: str) -> List[Dict]:
    """Scan a benchmark dir for evaluated-baseline configs (dmstrack/precomputed).

    Returns list of unified row dicts for configs whose protocol summary has
    the dmstrack_av_100.0pct key and show_in_table is not False.
    """
    rows = []
    if not benchmark_dir.is_dir():
        return rows
    for cfg_dir in sorted(benchmark_dir.iterdir()):
        if not cfg_dir.is_dir():
            continue
        sj = cfg_dir / split / "summary_v2v4real_protocol.json"
        if not sj.exists():
            continue
        with open(sj) as f:
            d = json.load(f)
        r = d.get("results", {}).get(_BENCHMARK_KEY)
        if r is None:
            continue
        if not d.get("show_in_table", True):
            continue
        label = d.get("suite_name", cfg_dir.name)
        rows.append(_row_from_protocol(r, label, "evaluated_baseline", "0.20"))
    return rows


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------

_CSV_COLS = [
    "method", "section",
    "v2v_amota", "pe_amota", "mg_amota",
    "hota", "assa",
    "amotp", "samota", "mota", "mt", "ml",
    "cost_mb", "ids", "gt",
]


def _to_csv_row(row: Dict) -> Dict:
    """Convert unified row dict to flat CSV-ready dict."""
    return {
        "method":    row["method"],
        "section":   row["section"],
        "v2v_amota": "" if row["v2v_amota"] is None else round(float(row["v2v_amota"]), 4),
        "pe_amota":  "" if row["pe_amota"]  is None else round(float(row["pe_amota"]),  4),
        "mg_amota":  "" if row["mg_amota"]  is None else round(float(row["mg_amota"]),  4),
        "hota":      "" if row["hota"]      is None else round(float(row["hota"]),      4),
        "assa":      "" if row["assa"]       is None else round(float(row["assa"]),       4),
        "amotp":     "" if row["amotp"]     is None else round(float(row["amotp"]),     4),
        "samota":    "" if row["samota"]     is None else round(float(row["samota"]),    4),
        "mota":      "" if row["mota"]      is None else round(float(row["mota"]),      4),
        "mt":        "" if row["mt"]        is None else round(float(row["mt"]),        4),
        "ml":        "" if row["ml"]        is None else round(float(row["ml"]),        4),
        "cost_mb":   row["cost"] or "",
        "ids":       "" if row["ids"] is None else int(row["ids"]),
        "gt":        "" if row["gt"]  is None else int(row["gt"]),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results_dirs", nargs="*", type=Path,
                    help="One or more results/V2V4Real_*/ directories")
    ap.add_argument("--export-dir", type=Path, default=None,
                    help="V2V4Real CMR export root used to compute Cost(MB) per row.")
    ap.add_argument("--split", default="test", help="Split to read (default: test)")
    ap.add_argument("--filter-groups", default="EKF,CI,AKF,PF,BICI,SABRE",
                    help="Which CMR filter families to include in the table.")
    ap.add_argument("--benchmark-dir", type=Path, default=None,
                    help="results/V2V4Real_BENCHMARK_*/ dir containing evaluated "
                         "baselines (dmstrack/precomputed configs). Their rows are "
                         "injected between the paper rows and the CMR rows.")
    ap.add_argument("--no-paper", action="store_true",
                    help="Skip the V2V4Real paper-reference rows at the top.")
    ap.add_argument("--csv", type=Path, default=None,
                    help="Also write results to this CSV file.")
    fmt = ap.add_mutually_exclusive_group()
    fmt.add_argument("--ascii", action="store_const", const="ascii",
                    dest="fmt", help="Plain ASCII (default).")
    fmt.add_argument("--md",    action="store_const", const="md",
                    dest="fmt", help="Markdown table.")
    fmt.add_argument("--latex", action="store_const", const="latex",
                    dest="fmt", help="LaTeX tabular row format.")
    ap.set_defaults(fmt="ascii")
    args = ap.parse_args()

    # Auto-discover "our" config subdirs when --benchmark-dir given without explicit paths.
    if args.benchmark_dir and not args.results_dirs:
        args.results_dirs = sorted(
            d for d in args.benchmark_dir.iterdir()
            if d.is_dir() and (d / args.split / "summary.json").exists()
        )

    # Default CSV path: <benchmark-dir>/leaderboard.csv.
    csv_path = args.csv
    if csv_path is None and args.benchmark_dir:
        csv_path = args.benchmark_dir / "leaderboard.csv"

    filter_groups = [g.strip() for g in args.filter_groups.split(",") if g.strip()]

    fmt_row = {"ascii": _format_row_ascii,
               "md":    _format_row_md,
               "latex": _format_row_latex}[args.fmt]

    all_rows: List[Dict] = []
    sep = "-" * len(_ASCII_HEADER)

    if args.fmt == "ascii":
        print(_ASCII_HEADER)
        print(sep)
    elif args.fmt == "md":
        print("| Method | V2V-AMOTA | PE-AMOTA | MG-AMOTA | HOTA | AssA | AMOTP | sAMOTA | MOTA | MT | ML | Cost(MB) | IDS | GT |")
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    # 1. V2V4Real paper reference rows.
    if not args.no_paper:
        for method, am, amp, sam, mota, mt, ml, cost in PAPER_ROWS:
            row = _row_paper(method, am, amp, sam, mota, mt, ml, cost)
            print(fmt_row(row))
            all_rows.append(row)
        if args.fmt == "ascii":
            print(sep)

    # 2. Evaluated baselines from benchmark dir (dmstrack/precomputed configs).
    if args.benchmark_dir:
        bm_rows = _load_benchmark_rows(args.benchmark_dir, args.split)
        for row in bm_rows:
            print(fmt_row(row))
            all_rows.append(row)
        if bm_rows and args.fmt == "ascii":
            print(sep)

    # 3. Our CMR rows: for each results dir, for each filter group.
    for rdir in args.results_dirs:
        sj = rdir / args.split / "summary.json"
        if not sj.exists():
            print(f"# missing: {sj}", file=sys.stderr)
            continue
        with open(sj) as f:
            summary = json.load(f)
        results = summary.get("results", {})
        det_label = _parse_label_from_path(rdir)

        suite_cfg = {}
        suite_cfg_path = rdir / "suite_config.json"
        if suite_cfg_path.exists():
            with open(suite_cfg_path) as f:
                suite_cfg = json.load(f)
        cost_mb = _bandwidth_cost_mb(suite_cfg, args.export_dir)
        cost_str = f"{cost_mb:.4f}" if cost_mb is not None else "—"

        for fg in filter_groups:
            best, base = _best_gpem_and_baseline(results, fg)
            if base is not None:
                method = f"{det_label} + {fg} (baseline)"
                row = _row_from_summary(base[1], method, "ours", cost_str)
                print(fmt_row(row))
                all_rows.append(row)
            if best is not None:
                method = f"{det_label} + {fg} + GPEM-{best[0]}"
                row = _row_from_summary(best[1], method, "ours", cost_str)
                print(fmt_row(row))
                all_rows.append(row)
        if args.fmt == "ascii":
            print(sep)

    if csv_path and all_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_CSV_COLS)
            w.writeheader()
            w.writerows(_to_csv_row(r) for r in all_rows)
        print(f"\n  Saved to {csv_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
