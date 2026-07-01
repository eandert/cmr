#!/usr/bin/env python3
"""Estimate the inter-CAV RTK relative-pose drift on V2V4Real, per scenario.

Two CAVs (``astuff``, ``tesla``) each localize with their own RTK; their pose
solutions are mutually inconsistent by a slowly-varying error (the drift). We
recover that drift directly from the released per-CAV data, with no external
oracle.

THE BIG SIGNAL — ego cross-observation. Each CAV's own RTK pose (``ego_state``)
is exact truth; the *other* CAV's nearest detection (``detections.csv``) to it
is where that CAV *thinks* the ego is. The offset measures the drift with no
annotation ambiguity — only a little detector noise on a large, close, easily
detected target. Using BOTH directions and differencing them cancels common-mode
detector bias, leaving the relative translation t = δ_tesla − δ_astuff: i.e. the
transform that maps an astuff-placed point to where tesla would place it is
simply g(p) = p + t.

We deliberately estimate TRANSLATION ONLY. A per-frame rotation fit (forced
through the annotation-noisy third-party objects) inflated the translation by
Δθ×lever-arm and was unstable at the sub-degree level the real yaw drift lives
at. Instead we report yaw as a per-scenario DIAGNOSTIC and validate the
translational drift against the co-observed third-party objects, but do not fold
their annotation noise into the applied correction.

VALIDATION — co-observed third-party objects. Objects annotated by both CAVs
(``gt_objects.csv`` ``ass_id`` on astuff rows → matching tesla ``obj_id``) give
an independent drift read. We report (a) their mean offset vs the ego estimate,
and (b) the residual after applying the ego translation — small ⇒ the ego drift
also explains the third-party objects (no meaningful rotation); large ⇒ that
scenario's third-party offset is annotation/geometry, not bulk drift.

The per-frame raw translation is smoothed per scenario with a mean-reverting
Ornstein–Uhlenbeck RTS smoother (localization error is persistent OU drift, not
IID), weighting each frame by its own measurement uncertainty. We emit BOTH the
raw and OU-smoothed series so the difference can be inspected; the merge step
(build_merged_gt) consumes the OU-smoothed drift.

Outputs (under ``results/GT_MERGE_RTK/drift/``):
* ``<scenario>.csv`` — per-frame raw and OU-smoothed (tx, ty) + diagnostics.
* ``summary.csv``     — one row per scenario (coverage, raw vs OU, validation).
"""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
DEFAULT_EXPORT = REPO / "data" / "v2v4real_cmr_export"
DEFAULT_OUT = REPO / "results" / "GT_MERGE_RTK" / "drift"

# --- matching knobs (documented; no hidden defaults) ------------------------
EGO_GATE_M = 3.0          # max dist from an ego's RTK pose to call a detection "it".
                          # A real detection of the (large, close, clear) other ego
                          # lands within true-drift (≤~1.5 m) + detector noise (~0.5 m)
                          # ≈ 2 m; 3 m rejects the 3–8 m confusions with nearby cars
                          # that otherwise inflate low-coverage scenarios (e.g. Day20
                          # 11-51-42_0: 1.56 m spurious → 0.55 m, matching third-party).
OBS_VAR_FLOOR_M2 = 0.04   # floor on per-frame obs variance (0.2 m)² for the OU smoother
ONE_EGO_VAR_MULT = 4.0    # inflate obs var when only one ego direction is available
TP_FALLBACK_VAR_MULT = 9.0  # inflate obs var when a frame has only third-party pairs


@dataclass
class Frame:
    """Matched (A=astuff-frame, T=tesla-frame) point pairs for one frame."""
    ego_A: list = field(default_factory=list)
    ego_T: list = field(default_factory=list)
    tp_A: list = field(default_factory=list)
    tp_T: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Loading. One scenario folder = both CAVs already in one shared center_world
# frame (per scripts/input_bucket/merged_scenario_view.py) — no rehydration.
# ---------------------------------------------------------------------------
def _load_ego(scen_dir: Path) -> dict[int, dict[str, tuple[float, float, float]]]:
    ego: dict[int, dict[str, tuple[float, float, float]]] = defaultdict(dict)
    with open(scen_dir / "ego_state.csv") as f:
        for r in csv.DictReader(f):
            ego[int(r["frame_id"])][r["vehicle_id"]] = (
                float(r["x"]), float(r["y"]), float(r["yaw"]))
    return ego


def _load_dets(scen_dir: Path) -> dict[int, dict[str, np.ndarray]]:
    raw: dict[int, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    with open(scen_dir / "detections.csv") as f:
        for r in csv.DictReader(f):
            raw[int(r["frame_id"])][r["vehicle_id"]].append(
                (float(r["x"]), float(r["y"])))
    return {fr: {v: np.asarray(pts, float) for v, pts in vd.items()}
            for fr, vd in raw.items()}


def _load_gt(scen_dir: Path):
    """tesla_by_frame[frame][obj_id]=(x,y); astuff_assoc[frame]=[(ass_id,x,y),...]."""
    tesla: dict[int, dict[int, tuple[float, float]]] = defaultdict(dict)
    astuff: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    with open(scen_dir / "gt_objects.csv") as f:
        for r in csv.DictReader(f):
            fr = int(r["frame_id"]); x = float(r["x"]); y = float(r["y"])
            if r["vehicle_id"] == "tesla":
                tesla[fr][int(r["obj_id"])] = (x, y)
            elif r["vehicle_id"] == "astuff" and int(r["ass_id"]) >= 0:
                astuff[fr].append((int(r["ass_id"]), x, y))
    return tesla, astuff


def _nearest(pts: np.ndarray, x: float, y: float):
    d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
    i = int(np.argmin(d))
    return float(pts[i, 0]), float(pts[i, 1]), float(d[i])


def build_frames(scen_dir: Path) -> dict[int, Frame]:
    """Per frame: ego cross-observation pairs and third-party co-observed pairs.

    Each pair is (A in astuff's drifted frame, T in tesla's frame) for the same
    physical thing, so g: astuff→tesla satisfies g(A)≈T.
    """
    ego = _load_ego(scen_dir)
    dets = _load_dets(scen_dir)
    tesla_gt, astuff_gt = _load_gt(scen_dir)
    frames: dict[int, Frame] = {}
    for fr, vd in ego.items():
        if "astuff" not in vd or "tesla" not in vd:
            continue
        F = Frame()
        ax, ay, _ = vd["astuff"]
        tx, ty, _ = vd["tesla"]
        fdets = dets.get(fr, {})
        # ego pair 1: astuff observes tesla -> A=astuff's det of tesla, T=tesla RTK
        if len(fdets.get("astuff", [])):
            dx, dy, dist = _nearest(fdets["astuff"], tx, ty)
            if dist <= EGO_GATE_M:
                F.ego_A.append((dx, dy)); F.ego_T.append((tx, ty))
        # ego pair 2: tesla observes astuff -> A=astuff RTK, T=tesla's det of astuff
        if len(fdets.get("tesla", [])):
            dx, dy, dist = _nearest(fdets["tesla"], ax, ay)
            if dist <= EGO_GATE_M:
                F.ego_A.append((ax, ay)); F.ego_T.append((dx, dy))
        # third-party co-observed (ass_id link)
        tgt = tesla_gt.get(fr, {})
        for ass, gx, gy in astuff_gt.get(fr, []):
            t = tgt.get(ass)
            if t is not None:
                F.tp_A.append((gx, gy)); F.tp_T.append(t)
        if F.ego_A or F.tp_A:
            frames[fr] = F
    return frames


def frame_translation(F: Frame):
    """Ego-driven per-frame translation t=δ_tesla−δ_astuff and its obs variance.

    Both ego directions averaged (cancels common-mode detector bias). Falls back
    to the third-party centroid offset only if no ego match this frame.
    Returns (tx, ty, obs_var, n_ego, n_tp).
    """
    n_ego = len(F.ego_A); n_tp = len(F.tp_A)
    if n_ego:
        A = np.asarray(F.ego_A); T = np.asarray(F.ego_T)
        per_pair = T - A                      # one translation estimate per pair
        t = per_pair.mean(0)
        if n_ego >= 2:
            # variance of the 2-direction mean ~ |t1-t2|²/4 (floored)
            d = per_pair[0] - per_pair[1]
            var = max(float(d @ d) / 4.0, OBS_VAR_FLOOR_M2)
        else:
            var = OBS_VAR_FLOOR_M2 * ONE_EGO_VAR_MULT
        return float(t[0]), float(t[1]), var, n_ego, n_tp
    # third-party fallback (rare): centroid offset, low confidence
    A = np.asarray(F.tp_A); T = np.asarray(F.tp_T)
    t = (T - A).mean(0)
    return float(t[0]), float(t[1]), OBS_VAR_FLOOR_M2 * TP_FALLBACK_VAR_MULT, n_ego, n_tp


def kabsch_theta(A: np.ndarray, T: np.ndarray) -> float:
    """Residual rotation (rad) mapping A→T (2D Kabsch). For the yaw diagnostic."""
    cA = A.mean(0); cT = T.mean(0)
    H = (A - cA).T @ (T - cT)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, d]) @ U.T
    return math.atan2(R[1, 0], R[0, 0])


# ---------------------------------------------------------------------------
# Ornstein–Uhlenbeck RTS smoother (mean-reverting AR(1) state-space, scalar).
# State d[t]=μ+φ(d[t-1]-μ)+w, w~N(0,q); obs y[t]=d[t]+v, v~N(0,r[t]). Frames
# without a measurement carry r=inf (predict only). Zero-phase (RTS smoother).
# ---------------------------------------------------------------------------
def ou_rts_smooth(y: np.ndarray, r: np.ndarray) -> np.ndarray:
    obs = np.isfinite(y) & np.isfinite(r)
    if obs.sum() < 2:
        return np.full_like(y, float(np.mean(y[obs])) if obs.any() else 0.0)
    mu = float(np.average(y[obs], weights=1.0 / r[obs]))
    z = y[obs] - mu
    if len(z) > 2 and np.dot(z[:-1], z[:-1]) > 0:
        phi = float(np.dot(z[1:], z[:-1]) / np.dot(z[:-1], z[:-1]))
    else:
        phi = 0.0
    phi = min(max(phi, 0.0), 0.999)
    var_y = float(np.var(z))
    mean_r = float(np.mean(r[obs]))
    q = max(var_y - mean_r, 1e-6) * (1.0 - phi ** 2)
    p_stat = q / (1.0 - phi ** 2) if phi < 1 else var_y

    n = len(y)
    xf = np.zeros(n); Pf = np.zeros(n); xp = np.zeros(n); Pp = np.zeros(n)
    x_prev, P_prev = 0.0, p_stat
    for t in range(n):
        xp[t] = phi * x_prev
        Pp[t] = phi ** 2 * P_prev + q
        if np.isfinite(y[t]) and np.isfinite(r[t]):
            K = Pp[t] / (Pp[t] + r[t])
            xf[t] = xp[t] + K * ((y[t] - mu) - xp[t])
            Pf[t] = (1.0 - K) * Pp[t]
        else:
            xf[t] = xp[t]; Pf[t] = Pp[t]
        x_prev, P_prev = xf[t], Pf[t]
    xs = xf.copy()
    for t in range(n - 2, -1, -1):
        C = phi * Pf[t] / Pp[t + 1] if Pp[t + 1] > 0 else 0.0
        xs[t] = xf[t] + C * (xs[t + 1] - xp[t + 1])
    return xs + mu


# ---------------------------------------------------------------------------
def estimate_scenario(scen_dir: Path):
    frames = build_frames(scen_dir)
    if not frames:
        return None
    frs = sorted(frames)
    fmin, fmax = frs[0], frs[-1]
    n = fmax - fmin + 1
    tx = np.full(n, np.nan); ty = np.full(n, np.nan); rvar = np.full(n, np.inf)
    n_ego = np.zeros(n, int); n_tp = np.zeros(n, int)
    tp_off = []          # third-party (T-A) over all pairs, for validation
    tp_resid = []        # |T-(A+t_frame)| over tp pairs, after ego correction
    rot_A, rot_T = [], []  # ego-corrected tp pairs aggregated, for yaw diagnostic
    for fr in frs:
        F = frames[fr]
        x, y, var, ne, nt = frame_translation(F)
        i = fr - fmin
        tx[i], ty[i], rvar[i], n_ego[i], n_tp[i] = x, y, var, ne, nt
        if F.tp_A:
            A = np.asarray(F.tp_A); T = np.asarray(F.tp_T)
            tp_off.append(T - A)
            Ac = A + np.array([x, y])           # ego-translation-corrected astuff
            tp_resid.append(np.hypot(*(T - Ac).T))
            rot_A.append(Ac); rot_T.append(T)
    tx_ou = ou_rts_smooth(tx, rvar)
    ty_ou = ou_rts_smooth(ty, rvar)

    # diagnostics
    tp_off = np.vstack(tp_off) if tp_off else np.empty((0, 2))
    tp_resid = np.concatenate(tp_resid) if tp_resid else np.empty(0)
    yaw_deg = float("nan")
    if rot_A and sum(len(a) for a in rot_A) >= 3:
        yaw_deg = math.degrees(kabsch_theta(np.vstack(rot_A), np.vstack(rot_T)))
    return dict(frame=np.arange(fmin, fmax + 1), tx=tx, ty=ty, rvar=rvar,
                tx_ou=tx_ou, ty_ou=ty_ou, n_ego=n_ego, n_tp=n_tp,
                tp_off=tp_off, tp_resid=tp_resid, yaw_deg=yaw_deg,
                n_frames=len(frs), n_total=n)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export-root", type=Path, default=DEFAULT_EXPORT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    scen_dirs = sorted(d for d in args.export_root.iterdir()
                       if d.is_dir() and d.name.startswith("test__"))
    if not scen_dirs:
        raise SystemExit(f"no test__ scenarios under {args.export_root}")
    args.out.mkdir(parents=True, exist_ok=True)

    summary = []
    hdr = (f"{'scenario':46s} {'frm':>4s} {'egoCov':>7s} {'raw|t|':>7s} "
           f"{'ou|t|':>6s} {'meanT':>13s} {'tpOff':>6s} {'tpResid':>7s} {'yaw°':>5s}")
    print(hdr); print("-" * len(hdr))
    for sd in scen_dirs:
        res = estimate_scenario(sd)
        if res is None:
            print(f"{sd.name[:46]:46s}  (no frames)"); continue
        obs = np.isfinite(res["tx"])
        raw_t = float(np.mean(np.hypot(res["tx"][obs], res["ty"][obs])))
        ou_t = float(np.mean(np.hypot(res["tx_ou"], res["ty_ou"])))
        mean_t = (float(np.mean(res["tx"][obs])), float(np.mean(res["ty"][obs])))
        tp_off_mag = float(np.hypot(*res["tp_off"].mean(0))) if len(res["tp_off"]) else float("nan")
        tp_resid = float(np.mean(res["tp_resid"])) if len(res["tp_resid"]) else float("nan")
        ego_cov = f"{int(np.sum(res['n_ego']>0))}/{res['n_frames']}"
        print(f"{sd.name[:46]:46s} {res['n_frames']:>4d} {ego_cov:>7s} {raw_t:>7.3f} "
              f"{ou_t:>6.3f} ({mean_t[0]:+.2f},{mean_t[1]:+.2f}) {tp_off_mag:>6.3f} "
              f"{tp_resid:>7.3f} {res['yaw_deg']:>5.2f}")
        with open(args.out / f"{sd.name}.csv", "w", newline="") as f:
            wri = csv.writer(f)
            wri.writerow(["frame", "tx_raw", "ty_raw", "tx_ou", "ty_ou",
                          "obs_var", "n_ego", "n_tp"])
            for j in range(res["n_total"]):
                fin = np.isfinite(res["tx"][j])
                wri.writerow([
                    int(res["frame"][j]),
                    f"{res['tx'][j]:.5f}" if fin else "",
                    f"{res['ty'][j]:.5f}" if fin else "",
                    f"{res['tx_ou'][j]:.5f}", f"{res['ty_ou'][j]:.5f}",
                    f"{res['rvar'][j]:.5f}" if np.isfinite(res['rvar'][j]) else "",
                    int(res["n_ego"][j]), int(res["n_tp"][j])])
        summary.append((sd.name, res["n_frames"], ego_cov, f"{raw_t:.4f}",
                        f"{ou_t:.4f}", f"{mean_t[0]:.4f}", f"{mean_t[1]:.4f}",
                        f"{tp_off_mag:.4f}", f"{tp_resid:.4f}", f"{res['yaw_deg']:.4f}"))

    with open(args.out / "summary.csv", "w", newline="") as f:
        wri = csv.writer(f)
        wri.writerow(["scenario", "n_frames", "ego_coverage", "raw_trans_mag_m",
                      "ou_trans_mag_m", "mean_tx_m", "mean_ty_m",
                      "thirdparty_offset_mag_m", "thirdparty_resid_after_ego_m",
                      "scenario_yaw_deg"])
        wri.writerows(summary)
    print(f"\nWrote {len(summary)} per-scenario CSVs + summary.csv to {args.out}")


if __name__ == "__main__":
    main()
