"""Regression guard for the canonical AB3DMOT/V2V4Real AMOTP definition.

AB3DMOT's `evaluate.py` defines MOTP = total_cost / tp = **mean 3D-IoU over
true positives** (`evaluate.py:1024`, `total_cost += 1 - cost = IoU`), and
AMOTP = mean MOTP over the 40-point recall sweep (`evaluate.py:1196`). There is
NO division by the IoU gate. A previous implementation divided the mean cost by
`1 - iou_threshold = 0.75`, which made our AMOTP ~11 pts low (46.47 vs
DMSTrack's published 57.94 on the same tracks). These tests pin the metric to
mean-IoU so the `/0.75` gate normalization cannot silently return.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from metrics.v2v4real_metrics import (  # noqa: E402
    compute_ab3dmot_metrics_iou,
    match_frame_3d_iou,
)

IOU_GATE = 0.25
N_FRAMES = 10


def _make_tape(track_dx: float):
    """One GT (id=1) present in every frame; one always-matched track (id=1)
    offset laterally by `track_dx` so the 3D-IoU is a known partial value."""
    tape = []
    for f in range(N_FRAMES):
        gt = (1, 0.0, 0.0, 2.0, 4.0, 0.0, 1.5, 0.0)            # id,x,y,w,l,yaw,h,z
        trk = (1, track_dx, 0.0, 2.0, 4.0, 0.0, 0.9, 1.5, 0.0)  # +score @ idx6
        tape.append({"frame_idx": f, "tracks": [trk], "gts": [gt]})
    return tape


def _matched_cost(tape) -> float:
    """The (1 - IoU) cost the matcher assigns to the single TP per frame."""
    m, _, _ = match_frame_3d_iou(tape[0]["tracks"], tape[0]["gts"], IOU_GATE)
    assert len(m) == 1, "fixture must produce exactly one match"
    return m[0][2]


def _amotp(tape, rematch):
    return compute_ab3dmot_metrics_iou(
        tape, iou_threshold=IOU_GATE, match_3d=True,
        ab3dmot_rematching=rematch,
    )["amotp"]


@pytest.mark.parametrize("dx", [0.5, 1.0])
def test_plain_sweep_amotp_equals_mean_iou(dx):
    """Plain recall sweep: at full recall with constant match IoU every recall
    threshold contributes MOTP = 1 - cost, so AMOTP == mean IoU == 1 - cost.
    The reverted `/0.75` form would instead give 1 - cost/0.75."""
    tape = _make_tape(dx)
    cost = _matched_cost(tape)
    assert cost > 1e-6, "fixture must be an imperfect (partial-IoU) match"

    amotp = _amotp(tape, rematch=False)
    assert amotp == pytest.approx(1.0 - cost, abs=1e-9)
    gate_normalized = 1.0 - cost / (1.0 - IOU_GATE)
    assert abs(amotp - gate_normalized) > 1e-6, (
        "AMOTP must NOT be the /0.75 gate-normalized value (the reverted bug)")


@pytest.mark.parametrize("dx", [0.5, 1.0])
def test_rematch_amotp_scales_as_mean_iou(dx):
    """AB3DMOT-rematching path (the Table-1 scorer): the recall-sweep length is
    governed by getThresholds, but MOTP per reached threshold is still mean IoU.
    A partial match therefore scales the perfect-match AMOTP by exactly (1-cost)
    — NOT by the gate-normalized (1 - cost/0.75)."""
    cost = _matched_cost(_make_tape(dx))
    assert cost > 1e-6

    amotp_perfect = _amotp(_make_tape(0.0), rematch=True)
    amotp_partial = _amotp(_make_tape(dx), rematch=True)
    assert amotp_perfect > 0.0

    assert amotp_partial / amotp_perfect == pytest.approx(1.0 - cost, abs=1e-9)
    gate_ratio = 1.0 - cost / (1.0 - IOU_GATE)
    assert abs(amotp_partial / amotp_perfect - gate_ratio) > 1e-6


def test_amotp_perfect_match_is_one():
    """Identical track/GT boxes → IoU 1 → AMOTP 1.0 (sanity bound)."""
    tape = _make_tape(0.0)
    m = compute_ab3dmot_metrics_iou(tape, iou_threshold=IOU_GATE, match_3d=True)
    assert m["amotp"] == pytest.approx(1.0, abs=1e-9)
