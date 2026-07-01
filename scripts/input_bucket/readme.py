"""Per-bucket README.md renderer.

The README is generated from the bucket's MANIFEST.json + the registry entry —
single source of truth, never drifts from disk. Re-render by running::

    python scripts/build_v2v4real_input_bucket.py --bucket <name> --rebuild readme --apply
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .manifest import Manifest, read_manifest
from .registry import BucketSpec, BUCKETS


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024
    return f"{n:.1f} TB"


def _summarize_files(m: Manifest) -> str:
    by_subdir: dict[str, list] = {}
    for e in m.files:
        sub = e.relpath.split("/", 1)[0]
        by_subdir.setdefault(sub, []).append(e)
    lines = []
    for sub in sorted(by_subdir):
        entries = by_subdir[sub]
        total_size = sum(e.size for e in entries)
        total_rows = sum(e.rows for e in entries)
        row_note = f", {total_rows:,} rows" if total_rows else ""
        lines.append(f"- **`{sub}/`** — {len(entries)} file(s), {_human_size(total_size)}{row_note}")
    return "\n".join(lines) if lines else "_(empty — bucket needs to be built)_"


def _sample_data_row(bucket_root: Path, m: Manifest) -> Optional[str]:
    """Find a small text file in the bucket and return its first 3 lines."""
    for e in m.files:
        if e.rows == 0 or e.size > 5_000_000:
            continue
        p = bucket_root / e.relpath
        if not p.exists():
            continue
        try:
            with p.open() as f:
                head = [next(f).rstrip() for _ in range(3)]
            return f"```\n# from {e.relpath}\n" + "\n".join(head) + "\n```"
        except (StopIteration, OSError):
            continue
    return None


def render_readme(bucket_root: Path, spec: BucketSpec, m: Optional[Manifest]) -> str:
    """Render the bucket README.md content. Returns the text; caller writes it."""
    tier_label = "Tier 1 — bytes live inside cmr" if spec.tier == "tier1" \
                 else "Tier 2 — bytes fetched at setup (gitignored)"

    upstream = []
    if spec.source_repo_url:
        upstream.append(f"- **Source**: [{spec.source_repo_url}]({spec.source_repo_url})")
    if spec.source_commit_sha_or_tag:
        upstream.append(f"- **Pinned at**: `{spec.source_commit_sha_or_tag}`")
    if spec.source_kind:
        upstream.append(f"- **Source kind**: `{spec.source_kind}`")
    if spec.source_artifact_description:
        upstream.append(f"- **Artifact**: {spec.source_artifact_description}")
    upstream_md = "\n".join(upstream) if upstream else "_(no external source; generated in-repo)_"

    file_summary = _summarize_files(m) if m else "_(no manifest yet)_"
    sample = _sample_data_row(bucket_root, m) if m else None
    sample_section = f"\n## Sample row\n\n{sample}\n" if sample else ""

    if spec.consumed_by_configs:
        configs_md = "\n".join(f"- `{name}`" for name in spec.consumed_by_configs)
    else:
        configs_md = "_(no configs registered yet; populate after Phase F rewires `configs/v2v4real_experiments.py` to use `detector_bucket`)_"

    notes_md = ("\n".join(f"> {n}" for n in spec.notes)) if spec.notes else ""
    notes_section = f"\n## Notes\n\n{notes_md}\n" if notes_md else ""

    if spec.tier == "tier1":
        regen_cmd = (
            f"python scripts/build_v2v4real_input_bucket.py "
            f"--bucket {spec.label} --apply"
        )
    else:
        regen_cmd = f"python scripts/fetch_baselines.py --bucket {spec.label} --apply"

    return f"""# `{spec.label}/` — {spec.short_description}

**{tier_label}**

{spec.long_description}

## Provenance

{upstream_md}

## Contents

{file_summary}
{sample_section}
## Consumed by

{configs_md}
{notes_section}
## Regenerate

```bash
{regen_cmd}
```

## Integrity

This bucket's bytes are pinned by `MANIFEST.json` (sha256 of every file). Run:

```bash
pytest tests/test_input_bucket_integrity.py -k {spec.label.replace("/", "_")}
```

to verify on-disk bytes match the manifest.
"""
