"""V2V4Real ground-truth integrity: checksums + structural invariants.

The base labels (``v2v4real_val_label/``) are the official, frozen
V2V4Real/DMSTrack release. We never regenerate them — we only pin them with
sha256 and assert structural invariants so any silent corruption (a dropped
file, a re-export with a different coordinate twist, an editor touching a row)
fails loudly.

Used by both ``scripts/verify_eval_stack.py`` and ``tests/test_gt_integrity.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

from eval_config import (
    N_LABEL_COLS as N_COLS,
    BBOX2D_COLS,
    DIM_COLS,
    POS_COLS,
    MAX_ABS_POS_M as MAX_ABS_POS,
)


class GTIntegrityError(AssertionError):
    """Raised when GT checksums or structural invariants fail."""


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def label_files(label_dir: Path) -> List[Path]:
    return sorted(Path(label_dir).glob("*.txt"))


def compute_manifest(label_dir: Path) -> Dict[str, str]:
    """Map of ``{filename: sha256}`` for every label file in ``label_dir``."""
    return {p.name: sha256_of(p) for p in label_files(label_dir)}


def parse_seqmap(seqmap_path: Path) -> Dict[str, Tuple[int, int]]:
    """Parse ``<seq> <name> <start> <end>`` → ``{seq4: (start, end)}`` (inclusive)."""
    out: Dict[str, Tuple[int, int]] = {}
    for line in Path(seqmap_path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        f = line.split()
        seq = "%04d" % int(f[0])
        out[seq] = (int(f[2]), int(f[3]))
    return out


def verify_manifest(label_dir: Path, manifest: Dict[str, str]) -> List[str]:
    """Return a list of human-readable mismatches against a pinned manifest.

    Empty list == every file present and byte-identical to the manifest.
    """
    problems: List[str] = []
    actual = compute_manifest(label_dir)
    for name, want in manifest.items():
        got = actual.get(name)
        if got is None:
            problems.append(f"{name}: MISSING from {label_dir}")
        elif got != want:
            problems.append(f"{name}: sha256 mismatch (pinned {want[:12]}…, got {got[:12]}…)")
    for name in actual:
        if name not in manifest:
            problems.append(f"{name}: UNEXPECTED extra file in {label_dir}")
    return problems


def check_invariants(label_dir: Path, seqmap: Dict[str, Tuple[int, int]],
                     allowed_classes=("Car",), extra_track_ids=()) -> List[str]:
    """Structural invariants for V2V4Real GT label files.

    ``extra_track_ids`` whitelists synthetic IDs (e.g. 90001/90002 CAV self
    reports in the with_cav variant) so the same checker covers both dirs.
    Returns a list of problems; empty == clean.
    """
    problems: List[str] = []
    seen_seqs = set()
    for p in label_files(label_dir):
        seq = p.stem
        seen_seqs.add(seq)
        if seq not in seqmap:
            problems.append(f"{p.name}: sequence not in seqmap")
            continue
        start, end = seqmap[seq]
        frames_seen = set()
        seen_keys = set()
        for ln, line in enumerate(p.read_text().splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            f = line.split()
            if len(f) != N_COLS:
                problems.append(f"{p.name}:{ln}: {len(f)} cols, expected {N_COLS}")
                continue
            try:
                frame = int(float(f[0]))
                tid = int(float(f[1]))
                vals = [float(x) for x in f[6:]]
            except ValueError:
                problems.append(f"{p.name}:{ln}: non-numeric field")
                continue
            if any(not math.isfinite(v) for v in vals):
                problems.append(f"{p.name}:{ln}: non-finite numeric field")
            if f[2] not in allowed_classes:
                problems.append(f"{p.name}:{ln}: class {f[2]!r} not in {allowed_classes}")
            frames_seen.add(frame)
            if not (start <= frame <= end):
                problems.append(f"{p.name}:{ln}: frame {frame} outside seqmap [{start},{end}]")
            # 2D bbox must be all-zero in V2V4Real (LiDAR-only; the min_height
            # eval filter keys off this — nonzero here would silently drop dets).
            if any(float(f[c]) != 0.0 for c in BBOX2D_COLS):
                problems.append(f"{p.name}:{ln}: nonzero 2D bbox {[f[c] for c in BBOX2D_COLS]}")
            # Positive dimensions.
            if any(float(f[c]) <= 0.0 for c in DIM_COLS):
                problems.append(f"{p.name}:{ln}: non-positive dim h/w/l {[f[c] for c in DIM_COLS]}")
            # Position magnitude sanity rail.
            if any(abs(float(f[c])) > MAX_ABS_POS for c in POS_COLS):
                problems.append(f"{p.name}:{ln}: |position| > {MAX_ABS_POS}")
            # Unique (frame, track_id) — except whitelisted synthetic CAV IDs,
            # which legitimately recur once per frame.
            key = (frame, tid)
            if key in seen_keys and tid not in extra_track_ids:
                problems.append(f"{p.name}:{ln}: duplicate (frame={frame}, track_id={tid})")
            seen_keys.add(key)
        # Frame contiguity: every frame in [start, end] present at least once?
        # V2V4Real GT can legitimately have empty frames (no cars visible), so
        # we only assert no frame EXCEEDS the seqmap and the file is non-empty.
        if not frames_seen:
            problems.append(f"{p.name}: no rows parsed")
    for seq in seqmap:
        if seq not in seen_seqs:
            problems.append(f"sequence {seq} in seqmap but no label file")
    return problems


def write_manifest(label_dir: Path, out_path: Path, seqmap_path: Path = None) -> Dict:
    """Write a checksum manifest (with optional invariant snapshot) to JSON."""
    manifest = {
        "label_dir": str(label_dir),
        "checksums": compute_manifest(label_dir),
    }
    if seqmap_path is not None:
        seqmap = parse_seqmap(seqmap_path)
        problems = check_invariants(label_dir, seqmap)
        manifest["invariants_clean"] = (len(problems) == 0)
        manifest["invariant_problems"] = problems
    Path(out_path).write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest
