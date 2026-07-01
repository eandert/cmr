#!/usr/bin/env python3
"""Decompose the V2V4Real co-observed GT offset into RTK drift + annotation diff.

When both CAVs annotate the same object, their GT box centres differ. That
inter-CAV GT offset (tesla_gt − astuff_gt) has two distinct sources:

* **RTK localization drift** — the two CAVs disagree on where *they* are, which
  rigidly shifts everything. We measure it cleanly via ego cross-observation
  (``estimate_intercav_rtk_drift.py``): the other CAV's detection of an ego vs
  that ego's RTK-exact pose — no annotation involved.
* **Annotation difference** — the two CAVs disagree on where the *box* is for the
  same car (different visible faces / occlusion → different centres), even with
  perfect localization.

Because the ego signal isolates the RTK part, the annotation part is just the
remainder:  annotation_diff = GT_offset − RTK_drift  (per scenario, as vectors).

This emits the per-scenario decomposition table (CSV + Markdown) to
``results/GT_MERGE_RTK/``. Reads the per-CAV GT (data/v2v4real_cmr_export) and the
OU-smoothed drift series produced by ``estimate_intercav_rtk_drift.py``.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "configs"))

from v2v4real_seq_map import SEQ_TO_SCENARIO, SEQ_TO_NUM_FRAMES  # noqa: E402
import build_merged_gt as bmg  # noqa: E402

OUT = REPO / "results" / "GT_MERGE_RTK"


def decompose():
    rows = []
    for seq, scen in sorted(SEQ_TO_SCENARIO.items()):
        sd = bmg.DEFAULT_EXPORT / scen
        tesla, astuff = bmg.load_gt_full(sd)
        node = bmg.build_clusters(tesla, astuff)
        drift = bmg.load_drift(scen, bmg.DEFAULT_DRIFT, use_drift=True)
        gt_offs, rtk = [], []
        for fr in range(SEQ_TO_NUM_FRAMES[seq]):
            tb = {node[("t", o)]: b[:2] for o, b in tesla.get(fr, {}).items()}
            ab = {node[("a", o)]: b[:2] for o, b in astuff.get(fr, {}).items()}
            co = set(tb) & set(ab)
            if not co:
                continue
            gt_offs.append(np.mean([[tb[t][0] - ab[t][0], tb[t][1] - ab[t][1]]
                                    for t in co], axis=0))
            rtk.append(drift[fr])
        gt = np.mean(gt_offs, axis=0)
        rt = np.mean(rtk, axis=0)
        ann = gt - rt
        rows.append(dict(
            seq=seq, scenario=scen[7:], n_coobs_frames=len(gt_offs),
            gt_off_x=gt[0], gt_off_y=gt[1], gt_off_mag=float(np.hypot(*gt)),
            rtk_x=rt[0], rtk_y=rt[1], rtk_mag=float(np.hypot(*rt)),
            ann_x=ann[0], ann_y=ann[1], ann_mag=float(np.hypot(*ann)),
            dominant="RTK" if np.hypot(*rt) >= np.hypot(*ann) else "annotation"))
    return rows


def write_csv(rows, path: Path):
    cols = ["seq", "scenario", "n_coobs_frames", "gt_off_x", "gt_off_y",
            "gt_off_mag", "rtk_x", "rtk_y", "rtk_mag", "ann_x", "ann_y",
            "ann_mag", "dominant"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v)
                        for k, v in r.items()})


def write_md(rows, path: Path):
    n_rtk = sum(r["dominant"] == "RTK" for r in rows)
    lines = [
        "# V2V4Real inter-CAV GT offset: RTK drift vs annotation difference",
        "",
        "When both CAVs (astuff, tesla) annotate the same object, their GT box",
        "centres disagree. That co-observed offset `tesla_gt − astuff_gt` splits into",
        "two sources, separated using the ego-cross-observation RTK measurement:",
        "",
        "- **RTK drift** — inter-CAV localization disagreement (rigid frame shift),",
        "  measured no-oracle from each ego's detection-of-the-other vs its RTK pose.",
        "- **Annotation difference** — viewpoint/occlusion disagreement on the box for",
        "  the same car; the remainder `GT_offset − RTK_drift`. (Box dims agree → same",
        "  car, not misassociation.)",
        "",
        "| seq | scenario | co-obs frames | GT offset (m) | RTK drift (m) | annotation diff (m) | dominant |",
        "|----:|----------|--------------:|--------------:|--------------:|--------------------:|----------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['seq']} | {r['scenario'][:30]} | {r['n_coobs_frames']} "
            f"| {r['gt_off_mag']:.2f} | {r['rtk_mag']:.2f} | {r['ann_mag']:.2f} "
            f"| {r['dominant']} |")
    gt_all = np.mean([r["gt_off_mag"] for r in rows])
    rtk_all = np.mean([r["rtk_mag"] for r in rows])
    ann_all = np.mean([r["ann_mag"] for r in rows])
    lines += [
        f"| — | **mean** | — | **{gt_all:.2f}** | **{rtk_all:.2f}** | **{ann_all:.2f}** | — |",
        "",
        f"**Takeaways.** RTK drift is real and cleanly measurable (0.04–1.09 m, yaw <0.6°),",
        f"but it is the *minority* of the inter-CAV GT offset in most scenarios: it dominates",
        f"in {n_rtk}/{len(rows)} (seq 0, 7), while the larger, viewpoint-driven annotation",
        f"difference dominates the rest (up to ~1.9 m, e.g. seq 1). Magnitudes are scenario-mean",
        f"vectors; the ego signal is what makes the split possible (without it the raw GT offset",
        f"would be mistaken for drift).",
        "",
        "_Generated by `scripts/rtk_vs_annotation_decomposition.py` from the per-CAV GT",
        "(`data/v2v4real_cmr_export`) + the OU-smoothed drift (`estimate_intercav_rtk_drift.py`)._",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = decompose()
    write_csv(rows, OUT / "rtk_vs_annotation_decomposition.csv")
    write_md(rows, OUT / "rtk_vs_annotation_decomposition.md")
    # echo the table
    print(f"{'seq':>3} {'GToff':>6} {'RTK':>6} {'annot':>6}  dominant")
    for r in rows:
        print(f"{r['seq']:>3} {r['gt_off_mag']:>6.2f} {r['rtk_mag']:>6.2f} "
              f"{r['ann_mag']:>6.2f}  {r['dominant']}")
    print(f"\nWrote rtk_vs_annotation_decomposition.{{csv,md}} to {OUT}")


if __name__ == "__main__":
    main()
