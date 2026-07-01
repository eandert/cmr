"""Adversarial tests for the V2V4Real ground-truth pipeline.

These attack both the base (frozen, official) V2V4Real label set and the
``with_cav`` augmented variant. They are intentionally hostile: each test
mutates the label tree in a different way and asserts that the integrity
machinery in ``cmr/src/gt_integrity.py`` (plus its pinned manifest in
``tests/fixtures/gt_checksums.json``) fails loudly.

Run from repo root:
    pytest tests/test_gt_integrity.py -v
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

# Conventional sys.path setup: src + scripts + configs.
REPO = Path(__file__).resolve().parent.parent
for sub in ("src", "scripts", "configs"):
    sys.path.insert(0, str(REPO / sub))

from eval_config import (  # noqa: E402
    BASE_GT_LABEL_DIR,
    WITH_CAV_GT_LABEL_DIR,
    SEQMAP_VAL,
    TESLA_GT_ID,
    ASTUFF_GT_ID,
    MAX_ABS_POS_M,
)
from gt_integrity import (  # noqa: E402
    check_invariants,
    compute_manifest,
    parse_seqmap,
    verify_manifest,
)

PINNED_MANIFEST_PATH = REPO / "tests" / "fixtures" / "gt_checksums.json"

# The base GT lives under third_party/AB3DMOT (a peer-repo symlink), absent in a
# fresh clone / CI. Skip the whole module then rather than fail on missing files.
pytestmark = pytest.mark.skipif(
    not Path(BASE_GT_LABEL_DIR).exists() or not any(Path(BASE_GT_LABEL_DIR).glob("*.txt")),
    reason="base GT labels absent (third_party/AB3DMOT not present — fresh clone/CI)",
)


# ── Helpers ─────────────────────────────────────────────────────────────────
def _load_pinned() -> dict[str, str]:
    return json.loads(PINNED_MANIFEST_PATH.read_text())["checksums"]


def _load_seqmap():
    return parse_seqmap(SEQMAP_VAL)


def _write_valid_row(frame: int = 0, tid: int = 1, cls: str = "Car",
                    h: float = 1.7, w: float = 2.0, l: float = 4.5,
                    x: float = 0.0, y: float = 0.0, z: float = 10.0,
                    ry: float = 0.0,
                    bbox=(0, 0, 0, 0)) -> str:
    """Build one well-formed KITTI MOT 17-col GT row (space-separated)."""
    b1, b2, b3, b4 = bbox
    # Columns: frame tid type trunc occ alpha bb1 bb2 bb3 bb4 h w l X Y Z ry
    return (f"{frame} {tid} {cls} 0 0 0 "
            f"{b1} {b2} {b3} {b4} "
            f"{h} {w} {l} {x} {y} {z} {ry}")


def _seed_seqmap(tmp_path: Path) -> dict:
    """Build a tiny seqmap covering only 0000 so synthetic single-file dirs
    don't trip 'sequence in seqmap but no label file' bookkeeping."""
    return {"0000": (0, 100)}


# ── Base-GT checksum & invariant tests ──────────────────────────────────────
def test_base_gt_checksum_pinned():
    """Every file in BASE_GT_LABEL_DIR matches the pinned sha256 manifest."""
    pinned = _load_pinned()
    problems = verify_manifest(BASE_GT_LABEL_DIR, pinned)
    assert problems == [], f"checksum drift: {problems}"


def test_base_gt_invariants_pass():
    """Structural invariants are clean on the frozen official labels."""
    seqmap = _load_seqmap()
    problems = check_invariants(BASE_GT_LABEL_DIR, seqmap)
    assert problems == [], f"invariant violations: {problems}"


def test_tampered_label_fails_checksum(tmp_path):
    """Flipping a single byte in 0000.txt must surface as sha256 mismatch."""
    pinned = _load_pinned()
    src = BASE_GT_LABEL_DIR / "0000.txt"
    dst = tmp_path / "0000.txt"
    # Copy all 9 so the only complaint is the tampered one.
    for p in sorted(BASE_GT_LABEL_DIR.glob("*.txt")):
        shutil.copy2(p, tmp_path / p.name)
    # Flip the very first byte ('0' → '1') without changing file length.
    data = bytearray(dst.read_bytes())
    data[0] = ord("9") if data[0] != ord("9") else ord("0")
    dst.write_bytes(bytes(data))

    problems = verify_manifest(tmp_path, pinned)
    mismatch_lines = [p for p in problems if "0000.txt" in p and "sha256 mismatch" in p]
    assert mismatch_lines, f"expected sha256 mismatch on 0000.txt, got: {problems}"


def test_missing_label_fails_checksum(tmp_path):
    """Dropping 0008.txt must surface as MISSING."""
    pinned = _load_pinned()
    for p in sorted(BASE_GT_LABEL_DIR.glob("*.txt")):
        if p.name == "0008.txt":
            continue
        shutil.copy2(p, tmp_path / p.name)

    problems = verify_manifest(tmp_path, pinned)
    missing_lines = [p for p in problems if "0008.txt" in p and "MISSING" in p]
    assert missing_lines, f"expected MISSING for 0008.txt, got: {problems}"


def test_extra_label_fails_checksum(tmp_path):
    """An unexpected 0009.txt must surface as UNEXPECTED extra."""
    pinned = _load_pinned()
    for p in sorted(BASE_GT_LABEL_DIR.glob("*.txt")):
        shutil.copy2(p, tmp_path / p.name)
    (tmp_path / "0009.txt").write_text(_write_valid_row() + "\n")

    problems = verify_manifest(tmp_path, pinned)
    extra_lines = [p for p in problems if "0009.txt" in p and "UNEXPECTED" in p]
    assert extra_lines, f"expected UNEXPECTED for 0009.txt, got: {problems}"


# ── Invariant attacks ───────────────────────────────────────────────────────
def test_nonzero_2d_bbox_invariant_fails(tmp_path):
    """A row with nonzero 2D bbox (col 9, "y2"=10) must be flagged with file:line."""
    seqmap = _seed_seqmap(tmp_path)
    bad = _write_valid_row(bbox=(0, 0, 0, 10))
    (tmp_path / "0000.txt").write_text(bad + "\n")

    problems = check_invariants(tmp_path, seqmap)
    hits = [p for p in problems if "0000.txt:1" in p and "nonzero 2D bbox" in p]
    assert hits, f"expected nonzero-2D-bbox flag with file:line, got: {problems}"


def test_negative_dim_invariant_fails(tmp_path):
    """h=-1.0 must be flagged as non-positive dim, with file:line."""
    seqmap = _seed_seqmap(tmp_path)
    bad = _write_valid_row(h=-1.0)
    (tmp_path / "0000.txt").write_text(bad + "\n")

    problems = check_invariants(tmp_path, seqmap)
    hits = [p for p in problems if "0000.txt:1" in p and "non-positive dim" in p]
    assert hits, f"expected non-positive-dim flag with file:line, got: {problems}"


def test_duplicate_frame_tid_fails(tmp_path):
    """Same (frame, tid) twice (non-whitelisted) must surface as duplicate."""
    seqmap = _seed_seqmap(tmp_path)
    rows = [_write_valid_row(frame=0, tid=1),
            _write_valid_row(frame=0, tid=1, x=1.0)]
    (tmp_path / "0000.txt").write_text("\n".join(rows) + "\n")

    problems = check_invariants(tmp_path, seqmap)
    hits = [p for p in problems if "duplicate" in p and "frame=0" in p and "track_id=1" in p]
    assert hits, f"expected duplicate (frame, track_id) flag, got: {problems}"


def test_class_other_than_car_fails(tmp_path):
    """A Pedestrian row must be flagged as class not in allowed_classes."""
    seqmap = _seed_seqmap(tmp_path)
    bad = _write_valid_row(cls="Pedestrian")
    (tmp_path / "0000.txt").write_text(bad + "\n")

    problems = check_invariants(tmp_path, seqmap)
    hits = [p for p in problems if "Pedestrian" in p and "not in" in p]
    assert hits, f"expected class-not-in-allowed flag, got: {problems}"


def test_position_sanity_rail(tmp_path):
    """|X|=1e6 must be flagged as |position| > MAX_ABS_POS_M."""
    seqmap = _seed_seqmap(tmp_path)
    bad = _write_valid_row(x=1e6)
    (tmp_path / "0000.txt").write_text(bad + "\n")

    problems = check_invariants(tmp_path, seqmap)
    hits = [p for p in problems if f"|position| > {MAX_ABS_POS_M}" in p]
    assert hits, f"expected position-sanity-rail flag, got: {problems}"


# ── with_cav variant ────────────────────────────────────────────────────────
def test_with_cav_invariants_with_extras_pass():
    """If the with_cav GT exists, it must be clean WHEN the synthetic 90001/90002
    IDs are whitelisted — proving the augmentation respects every invariant
    except the (frame, tid) recurrence (which is by design)."""
    if not Path(WITH_CAV_GT_LABEL_DIR).exists():
        pytest.skip(f"{WITH_CAV_GT_LABEL_DIR} not present")
    seqmap = _load_seqmap()
    problems = check_invariants(
        WITH_CAV_GT_LABEL_DIR, seqmap,
        extra_track_ids=(TESLA_GT_ID, ASTUFF_GT_ID),
    )
    assert problems == [], f"with_cav invariants violated: {problems}"


def test_with_cav_invariants_without_extras_fails(tmp_path):
    """The ``extra_track_ids`` whitelist must be load-bearing: when the
    augmented GT carries duplicate (frame, tid) rows for the synthetic CAV
    IDs, ``extra_track_ids=()`` must surface them as duplicate and the
    whitelisted call must not.

    REAL-DATA FINDING (see report): the live
    ``v2v4real_val_label_with_cav`` set actually contains each synthetic
    90001/90002 ID exactly ONCE per frame, so ``extra_track_ids`` is
    inert today (no key collisions to suppress). To still exercise the
    whitelist mechanism we synthesise a duplicate row in tmp_path."""
    seqmap = {"0000": (0, 10)}
    rows = []
    for fid in range(3):
        rows.append(_write_valid_row(frame=fid, tid=1))             # real GT
        rows.append(_write_valid_row(frame=fid, tid=TESLA_GT_ID))   # CAV self
        rows.append(_write_valid_row(frame=fid, tid=TESLA_GT_ID,    # DUPLICATE
                                    x=1.0))
    (tmp_path / "0000.txt").write_text("\n".join(rows) + "\n")

    # extras whitelisted: duplicate is suppressed.
    clean = check_invariants(tmp_path, seqmap,
                             extra_track_ids=(TESLA_GT_ID, ASTUFF_GT_ID))
    assert clean == [], f"whitelist failed to suppress duplicate: {clean}"

    # extras empty: must surface as duplicate (frame, tid) for 90001.
    problems = check_invariants(tmp_path, seqmap, extra_track_ids=())
    dup_hits = [p for p in problems
                if "duplicate" in p and f"track_id={TESLA_GT_ID}" in p]
    assert dup_hits, (
        f"expected duplicate (frame, track_id={TESLA_GT_ID}) when extras=(); "
        f"got: {problems}"
    )
