"""Freeze and guard checks for the sealed holdout (docs/protocol.md §7).

``write_freeze`` records every model checkpoint's sha256 (plus the protocol and holdout
manifest hashes) once all GNN leakage batteries pass. ``check_guard`` is run by
``score_holdout.py`` before anything is scored: it refuses without a freeze, on any hash
mismatch, or when a holdout result already exists.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CHUNK = 1 << 20


class GuardError(SystemExit):
    """Raised when the holdout may not be scored (exits with a message)."""


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file, read in chunks."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def battery_path(cfg: dict[str, Any], key: str, split: str) -> Path:
    """Where the leakage battery result of one checkpoint and split lives."""
    return Path(cfg["battery_file_pattern"].format(key=key, split=split))


def write_freeze(cfg: dict[str, Any]) -> dict[str, Any]:
    """Create freeze.json; refuses if it exists, a checkpoint is missing or a battery fails."""
    freeze_path = Path(cfg["freeze_file"])
    if freeze_path.exists():
        raise GuardError(f"{freeze_path} already exists; the freeze is final")
    problems = []
    entries = []
    for m in cfg["models"]:
        ckpt = Path(m["checkpoint"])
        if not ckpt.exists():
            problems.append(f"{m['key']}: checkpoint {ckpt} missing")
            continue
        if m["kind"] in ("hetero_gat", "graphsage"):
            for split in cfg["battery_splits"]:
                bp = battery_path(cfg, m["key"], split)
                if not bp.exists() or not json.loads(bp.read_text(encoding="utf-8"))["gate_pass"]:
                    problems.append(f"{m['key']}: no passing leakage battery on {split} ({bp})")
        entries.append(
            {"key": m["key"], "checkpoint": ckpt.as_posix(), "sha256": sha256_file(ckpt)}
        )
    if problems:
        raise GuardError("Cannot freeze:\n  " + "\n  ".join(problems))
    doc = {
        "frozen_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "protocol_sha256": sha256_file(Path(cfg["protocol_file"])),
        "holdout_manifest_sha256": sha256_file(Path(cfg["manifest_out"])),
        "models": entries,
    }
    freeze_path.parent.mkdir(parents=True, exist_ok=True)
    with freeze_path.open("x", encoding="utf-8") as fh:  # exclusive: never overwrite
        fh.write(json.dumps(doc, indent=2) + "\n")
    return doc


def check_guard(cfg: dict[str, Any]) -> dict[str, Any]:
    """All checks that must pass before the holdout is scored. Returns the freeze document."""
    freeze_path = Path(cfg["freeze_file"])
    if not freeze_path.exists():
        raise GuardError(f"No freeze at {freeze_path}: run scripts/freeze_models.py first")
    if Path(cfg["results_out"]).exists():
        raise GuardError(f"{cfg['results_out']} exists: the holdout has already been scored")
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))

    frozen = {e["key"]: e for e in freeze["models"]}
    problems = []
    for m in cfg["models"]:
        entry = frozen.get(m["key"])
        if entry is None:
            problems.append(f"{m['key']}: not listed in the freeze")
            continue
        ckpt = Path(m["checkpoint"])
        if not ckpt.exists() or sha256_file(ckpt) != entry["sha256"]:
            problems.append(f"{m['key']}: checkpoint missing or sha256 differs from the freeze")
    extra = set(frozen) - {m["key"] for m in cfg["models"]}
    if extra:
        problems.append(f"freeze lists models not in the config: {sorted(extra)}")

    manifest_path = Path(cfg["manifest_out"])
    if sha256_file(manifest_path) != freeze["holdout_manifest_sha256"]:
        problems.append("holdout manifest changed since the freeze")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for item in [manifest["holdout_raw"], *manifest["holdout_graphs"]]:
        path = Path(item["path"])
        if not path.exists() or sha256_file(path) != item["sha256"]:
            problems.append(f"holdout file {path} missing or differs from the manifest")
    if sha256_file(Path(cfg["protocol_file"])) != freeze["protocol_sha256"]:
        problems.append("docs/protocol.md changed since the freeze")
    if problems:
        raise GuardError("Refusing to score the holdout:\n  " + "\n  ".join(problems))
    return freeze
