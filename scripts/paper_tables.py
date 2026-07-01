#!/usr/bin/env python3
"""Generate paper-ready tables from the experiment registry.

Reads the registry (configs/v2v4real_experiments.py), runs paper_metrics on
the requested experiments (if not already computed), and emits one canonical
markdown file with:

  - **Anchor table**: paper baselines (LF, CoBEVT, DMSTrack) compared to
    published reference AMOTAs from V2V4Real / DMSTrack papers.
  - **Headline table**: our paper's headline cooperative-tracker config(s).
  - **Ablation table**: S1 → S2 → S3 progression per detector × filter × mode.

Every row labels the experiment by its registered NAME (not the raw
result_sha) so a reviewer can reproduce any cell via:

    python scripts/v2v4real_runner.py --experiment <name>
    python scripts/paper_tables.py --roles <role>

Usage:
    # Default: render all three tables for the paper.
    python scripts/paper_tables.py --out results/paper_tables.md

    # Just anchors:
    python scripts/paper_tables.py --roles v2v4real_paper_anchor dmstrack_paper_anchor

    # Just one experiment:
    python scripts/paper_tables.py --experiments DMS_S3NoNorm_sabre_quadratic_headline

The script CACHES metric results in-memory; if you re-run, configs that have
already been scored this session are read from cache. Configs whose tracker
output doesn't exist on disk yet are skipped with a warning (run them first
via v2v4real_runner.py).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "configs"))

from eval_config import V2V4REAL_RESULTS_ROOT  # noqa: E402
from v2v4real_experiments import (   # noqa: E402
    EXPERIMENTS, ExperimentConfig, get as get_experiment, list_all,
)
from gt_integrity import parse_seqmap  # noqa: E402
from eval_config import SEQMAP_VAL  # noqa: E402
from paper_metrics import compute_stage, STAGES  # noqa: E402


def _result_dir_exists(cfg: ExperimentConfig) -> bool:
    """True if the tracker output for this experiment is on disk."""
    d = V2V4REAL_RESULTS_ROOT / cfg.result_dir_name() / "data_0"
    return d.is_dir() and any(d.glob("*.txt"))


def _worker_init() -> None:
    """Cap each child's BLAS/OpenMP fan-out so N workers × M threads doesn't oversubscribe."""
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def _stage_worker(task) -> Tuple[str, str, Optional[Dict], Optional[str]]:
    """Pool worker. task = (result_dir_name, exp_name, protocol, gt_subdir, label, seqmap).
    Returns (exp_name, label, metrics_dict_or_None, err_or_None)."""
    result_dir_name, exp_name, protocol, gt_subdir, label, seqmap = task
    try:
        m = compute_stage(result_dir_name, protocol, gt_subdir, seqmap)
        return exp_name, label, m, None
    except Exception as e:
        return exp_name, label, None, str(e)


def _gather_metrics(cfgs: List[ExperimentConfig], seqmap: Dict,
                    workers: int = 8) -> Dict[str, Dict[str, Dict]]:
    """For each experiment × stage, compute paper_metrics in parallel.
    Returns: {experiment_name: {stage_label: metrics_dict}}.
    Configs without tracker output on disk are skipped with a printed warning.
    """
    runnable: List[ExperimentConfig] = []
    for cfg in cfgs:
        if not _result_dir_exists(cfg):
            print(f"[SKIP] {cfg.name}: no tracker output on disk "
                  f"(run via v2v4real_runner.py first)")
            continue
        runnable.append(cfg)

    if not runnable:
        return {}

    # Build the full task list — every (experiment × stage) cross-product.
    tasks: List[Tuple] = []
    for cfg in runnable:
        for protocol, gt_subdir, label in STAGES:
            tasks.append((cfg.result_dir_name(), cfg.name,
                          protocol, gt_subdir, label, seqmap))

    n_total = len(tasks)
    n_workers = max(1, min(workers, n_total))
    print(f"[paper_tables] {n_total} (experiment × stage) work items, {n_workers} worker(s)",
          flush=True)

    out: Dict[str, Dict[str, Dict]] = {cfg.name: {} for cfg in runnable}
    if n_workers == 1:
        for task in tasks:
            exp_name, label, m, err = _stage_worker(task)
            if err:
                print(f"  [WARN] {exp_name}::{label} failed: {err}", flush=True)
            elif m is not None:
                out[exp_name][label] = m
                print(f"  [{exp_name}::{label}] AMOTA={m['amota']*100:.2f}", flush=True)
        return out

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers, initializer=_worker_init) as pool:
        done = 0
        for exp_name, label, m, err in pool.imap_unordered(_stage_worker, tasks):
            done += 1
            if err:
                print(f"  [{done:2d}/{n_total}] [WARN] {exp_name}::{label} failed: {err}",
                      flush=True)
            elif m is not None:
                out[exp_name][label] = m
                print(f"  [{done:2d}/{n_total}] {exp_name} :: {label} :: "
                      f"AMOTA={m['amota']*100:.2f} MOTA={m['mota']*100:.2f} "
                      f"HOTA={m['hota']*100:.2f}", flush=True)
    return out


def _stage_short(label: str) -> str:
    """Compact column header from a long stage label."""
    return {
        "V2V (FP-ignore, base GT)": "V2V",
        "OG+FP (FP-counted, base GT)": "OG+FP",
        "Ours (FP-counted, +ego GT)": "Ours",
        "Ours-merged (FP-counted, +ego GT, dup-merged)": "Ours-merged",
    }.get(label, label)


def _format_pct(x: Optional[float], fmt: str = "{:.2f}") -> str:
    if x is None:
        return "—"
    return fmt.format(x * 100.0)


def _render_anchor_table(metrics: Dict[str, Dict[str, Dict]],
                          anchor_cfgs: List[ExperimentConfig]) -> str:
    """Per-detector anchor table with published reference comparison."""
    lines = [
        "## Published-baseline anchor table",
        "",
        "Compares our clean evaluator's V2V column (DMSTrack-protocol, FP-ignore) "
        "and OG+FP column (V2V4Real-paper-protocol, FP-counted) to the published "
        "reference AMOTA for each detector.",
        "",
        "| experiment | published ref | source | our V2V | our OG+FP | our Ours | our Ours-merged |",
        "|---|---|---|---|---|---|---|",
    ]
    for cfg in anchor_cfgs:
        per = metrics.get(cfg.name, {})
        ref = f"{cfg.reference_amota:.2f}" if cfg.reference_amota else "—"
        src = cfg.reference_source or "—"
        v2v = _format_pct(per.get("V2V (FP-ignore, base GT)", {}).get("amota"))
        ogfp = _format_pct(per.get("OG+FP (FP-counted, base GT)", {}).get("amota"))
        ours = _format_pct(per.get("Ours (FP-counted, +ego GT)", {}).get("amota"))
        ours_m = _format_pct(per.get("Ours-merged (FP-counted, +ego GT, dup-merged)", {}).get("amota"))
        lines.append(f"| `{cfg.name}` | {ref} | {src} | {v2v} | {ogfp} | {ours} | {ours_m} |")
    return "\n".join(lines) + "\n"


def _render_headline_table(metrics: Dict[str, Dict[str, Dict]],
                            headline_cfgs: List[ExperimentConfig]) -> str:
    """Headline cooperative-tracker table — full metrics for each headline experiment."""
    lines = [
        "## Headline cooperative-tracker results",
        "",
        "Our paper's headline numbers. Each row is one registered experiment; all "
        "stages share the same fixed evaluator (proven bit-identical to clean "
        "`evaluate.py` at 1.11e-16).",
        "",
        "| experiment | stage | AMOTA | MOTA | HOTA | DetA | AssA | GT_total | FP | IDS |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for cfg in headline_cfgs:
        per = metrics.get(cfg.name, {})
        for protocol, gt_subdir, label in STAGES:
            m = per.get(label)
            if m is None:
                continue
            lines.append(
                f"| `{cfg.name}` | {_stage_short(label)} | "
                f"{m['amota']*100:.2f} | {m['mota']*100:.2f} | "
                f"{m['hota']*100:.2f} | {m['deta']*100:.2f} | {m['assa']*100:.2f} | "
                f"{m['gt_total']} | {m['fp_total']} | {m['ids']} |"
            )
    return "\n".join(lines) + "\n"


def _render_ablation_table(metrics: Dict[str, Dict[str, Dict]],
                            ablation_cfgs: List[ExperimentConfig],
                            stage_filter: str = "V2V (FP-ignore, base GT)") -> str:
    """Compact ablation table: one row per experiment, one stage shown."""
    lines = [
        f"## Ablation table — {_stage_short(stage_filter)} column",
        "",
        "Compact view of the S1 → S2 → S3 progression. One stage shown; switch "
        "with `--ablation-stage <label>`.",
        "",
        "| experiment | AMOTA | MOTA | HOTA | DetA | AssA | IDS |",
        "|---|---|---|---|---|---|---|",
    ]
    for cfg in ablation_cfgs:
        per = metrics.get(cfg.name, {})
        m = per.get(stage_filter)
        if m is None:
            continue
        lines.append(
            f"| `{cfg.name}` | "
            f"{m['amota']*100:.2f} | {m['mota']*100:.2f} | "
            f"{m['hota']*100:.2f} | {m['deta']*100:.2f} | {m['assa']*100:.2f} | "
            f"{m['ids']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--roles", nargs="+", default=None,
                   choices=["v2v4real_paper_anchor", "dmstrack_paper_anchor",
                            "ablation", "headline", "side_quest"],
                   help="restrict to experiments in these paper_role(s). "
                        "Default: all anchors + headlines.")
    g.add_argument("--experiments", nargs="+", default=None,
                   help="list of explicit experiment names")
    g.add_argument("--all", action="store_true",
                   help="every registered experiment (warning: long)")
    ap.add_argument("--ablation-stage", default="V2V (FP-ignore, base GT)",
                    help="stage label to show in the ablation table")
    ap.add_argument("--out", default="results/paper_tables.md",
                    help="output markdown path")
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel workers for metric computation (default 8). "
                         "Each (experiment × stage) is one work item.")
    args = ap.parse_args()

    # Choose which configs to render.
    if args.experiments:
        cfgs = [get_experiment(n) for n in args.experiments]
    elif args.all:
        cfgs = list(EXPERIMENTS.values())
    elif args.roles:
        cfgs = [c for c in EXPERIMENTS.values() if c.paper_role in args.roles]
    else:
        # Default: anchors + headline.
        cfgs = [c for c in EXPERIMENTS.values()
                if c.paper_role in ("v2v4real_paper_anchor", "dmstrack_paper_anchor",
                                    "headline")]

    if not cfgs:
        print("no experiments selected")
        return 1

    seqmap = parse_seqmap(SEQMAP_VAL)
    metrics = _gather_metrics(cfgs, seqmap, workers=args.workers)
    if not metrics:
        print("no metrics computed — run experiments via v2v4real_runner.py first")
        return 1

    # Group by role for rendering.
    by_role: Dict[str, List[ExperimentConfig]] = {}
    for cfg in cfgs:
        if cfg.name in metrics:
            by_role.setdefault(cfg.paper_role, []).append(cfg)

    out_parts: List[str] = [
        "# V2V4Real paper tables — auto-generated from registry",
        "",
        "All rows reference registered experiments in `configs/v2v4real_experiments.py`. "
        "Reproduce any cell with:",
        "",
        "```bash",
        "python scripts/v2v4real_runner.py --experiment <name>",
        "python scripts/paper_tables.py --experiments <name>",
        "```",
        "",
    ]

    anchors = (by_role.get("v2v4real_paper_anchor", []) +
               by_role.get("dmstrack_paper_anchor", []))
    if anchors:
        out_parts.append(_render_anchor_table(metrics, anchors))
        out_parts.append("")

    headlines = by_role.get("headline", [])
    if headlines:
        out_parts.append(_render_headline_table(metrics, headlines))
        out_parts.append("")

    ablations = by_role.get("ablation", [])
    if ablations:
        out_parts.append(_render_ablation_table(metrics, ablations, args.ablation_stage))
        out_parts.append("")

    side_quests = by_role.get("side_quest", [])
    if side_quests:
        out_parts.append(_render_headline_table(metrics, side_quests))
        out_parts.append("")

    out_path = REPO / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_parts))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
