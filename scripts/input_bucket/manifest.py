"""MANIFEST.json reader/writer for v2v4real_inputs/ buckets.

A bucket's manifest is the contract between the on-disk bytes and every
downstream consumer (tests, readme generator, fetch_baselines, build script).
It records sha256 + size + row count for every file in the bucket, plus
provenance metadata about the upstream artifact the bytes were derived from.

Schema is versioned via ``SCHEMA_VERSION``. Bumping it requires migrating
every committed MANIFEST.json in the tree.

Example bucket directory walked here::

    v2v4real_inputs/baselines/dmstrack_pp/
      ab3dmot_detections/   ← walked recursively (symlinks resolved)
      gpem_calibration/     ← walked recursively
      MANIFEST.json         ← we write this
      README.md             ← not walked; that's the human-readable doc
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


SCHEMA_VERSION = 1

# Files at the bucket root that are NOT bucket data (they describe it).
_BUCKET_META_FILES = frozenset({"MANIFEST.json", "README.md", "build.log"})


@dataclass
class BucketUpstream:
    """Where a bucket's bytes were derived from (for reproducibility)."""

    source_kind: str = ""              # e.g. "pointpillar_v0.4", "dmstrack_release"
    source_repo_url: str = ""          # https URL, no trailing /
    source_commit_sha_or_tag: str = "" # short SHA or release tag
    source_artifact_description: str = ""  # human note

    def to_dict(self) -> dict:
        return {
            "source_kind": self.source_kind,
            "source_repo_url": self.source_repo_url,
            "source_commit_sha_or_tag": self.source_commit_sha_or_tag,
            "source_artifact_description": self.source_artifact_description,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BucketUpstream":
        return cls(**{k: d.get(k, "") for k in (
            "source_kind", "source_repo_url",
            "source_commit_sha_or_tag", "source_artifact_description",
        )})


@dataclass
class FileEntry:
    relpath: str
    sha256: str
    size: int
    rows: int  # 0 if not a row-oriented file (e.g. yaml, json, pdf)


@dataclass
class Manifest:
    bucket: str  # e.g. "baselines/dmstrack_pp"
    build_timestamp_utc: str
    upstream: BucketUpstream
    files: list[FileEntry] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "bucket": self.bucket,
            "build_timestamp_utc": self.build_timestamp_utc,
            "upstream": self.upstream.to_dict(),
            "files": {
                e.relpath: {"sha256": e.sha256, "size": e.size, "rows": e.rows}
                for e in sorted(self.files, key=lambda e: e.relpath)
            },
        }


def _sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _count_rows(path: Path) -> int:
    """Count newline-terminated rows for text-like files; 0 for everything else."""
    suffix = path.suffix.lower()
    if suffix in (".txt", ".csv", ".tsv"):
        try:
            with path.open("rb") as f:
                return sum(1 for _ in f)
        except OSError:
            return 0
    return 0


def _walk_bucket_files(bucket_root: Path) -> list[Path]:
    """Return every file under bucket_root that should be in the manifest.

    Walks all subdirs of the bucket (resolving symlinks), gathering files.
    Excludes meta files at the bucket root (MANIFEST.json, README.md, build.log)
    so the manifest doesn't try to track itself.

    Handles directory-symlinks (e.g. ``cmr_export/astuff -> /external/path``) by
    following them and recursing into the target.
    """
    out: list[Path] = []
    visited_dirs: set[Path] = set()

    def _walk(d: Path) -> None:
        try:
            entries = sorted(d.iterdir())
        except (PermissionError, FileNotFoundError):
            return
        for p in entries:
            if p.is_symlink():
                target = p.resolve()
                if target.is_dir():
                    if target in visited_dirs:
                        continue
                    visited_dirs.add(target)
                    _walk(p)
                elif target.is_file():
                    out.append(p)
                continue
            if p.is_dir():
                if p in visited_dirs:
                    continue
                visited_dirs.add(p)
                _walk(p)
            elif p.is_file():
                if p.parent == bucket_root and p.name in _BUCKET_META_FILES:
                    continue
                out.append(p)

    _walk(bucket_root)
    return out


def compute_manifest(
    bucket_root: Path,
    bucket_label: str,
    upstream: BucketUpstream,
    now_utc: Optional[datetime] = None,
) -> Manifest:
    """Walk the bucket's data subdirs, sha256 every file, return a Manifest."""
    if now_utc is None:
        now_utc = datetime.now(timezone.utc)
    entries: list[FileEntry] = []
    for p in _walk_bucket_files(bucket_root):
        relpath = str(p.relative_to(bucket_root))
        entries.append(FileEntry(
            relpath=relpath,
            sha256=_sha256_of(p),
            size=p.stat().st_size,
            rows=_count_rows(p),
        ))
    return Manifest(
        bucket=bucket_label,
        build_timestamp_utc=now_utc.replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        upstream=upstream,
        files=entries,
    )


def write_manifest(bucket_root: Path, m: Manifest) -> Path:
    path = bucket_root / "MANIFEST.json"
    path.write_text(json.dumps(m.to_dict(), indent=2) + "\n")
    return path


def read_manifest(bucket_root: Path) -> Manifest:
    """Load a bucket's MANIFEST.json. Raises if missing or schema mismatch."""
    path = bucket_root / "MANIFEST.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run "
            f"`scripts/build_v2v4real_input_bucket.py --bucket <name> --rebuild manifest --apply`."
        )
    d = json.loads(path.read_text())
    if d.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path} has schema_version={d.get('schema_version')!r} but this tool understands "
            f"SCHEMA_VERSION={SCHEMA_VERSION}. Either bump the schema migration or use the right tool."
        )
    files = [
        FileEntry(relpath=rp, sha256=meta["sha256"], size=meta["size"], rows=meta.get("rows", 0))
        for rp, meta in sorted(d["files"].items())
    ]
    return Manifest(
        schema_version=d["schema_version"],
        bucket=d["bucket"],
        build_timestamp_utc=d["build_timestamp_utc"],
        upstream=BucketUpstream.from_dict(d["upstream"]),
        files=files,
    )


def verify_against_disk(bucket_root: Path) -> list[str]:
    """Return a list of integrity violations for the bucket; empty list = OK.

    Each string in the returned list describes one problem with enough detail
    that a future-you (or CI) can diagnose without re-reading the script.
    """
    problems: list[str] = []
    try:
        m = read_manifest(bucket_root)
    except FileNotFoundError as e:
        return [str(e)]
    except ValueError as e:
        return [str(e)]
    expected = {e.relpath: e for e in m.files}
    on_disk = {str(p.relative_to(bucket_root)): p for p in _walk_bucket_files(bucket_root)}
    for rp in sorted(set(expected) - set(on_disk)):
        problems.append(f"missing on disk: {rp}")
    for rp in sorted(set(on_disk) - set(expected)):
        problems.append(f"extra on disk (not in manifest): {rp}")
    for rp in sorted(set(expected) & set(on_disk)):
        e = expected[rp]
        p = on_disk[rp]
        actual_sha = _sha256_of(p)
        if actual_sha != e.sha256:
            problems.append(f"sha256 mismatch on {rp}: manifest={e.sha256[:12]}... disk={actual_sha[:12]}...")
        actual_size = p.stat().st_size
        if actual_size != e.size:
            problems.append(f"size mismatch on {rp}: manifest={e.size} disk={actual_size}")
    return problems
