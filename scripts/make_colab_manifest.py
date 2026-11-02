"""Write the size and SHA-256 manifest of the data files Colab training needs.

The notebook ``notebooks/colab_train.ipynb`` copies these files from Google Drive and refuses
to train if any checksum differs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch_geometric
import yaml

CHUNK_BYTES = 1 << 20


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    """Hash the configured files and write the manifest."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/colab.yaml")
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    entries = []
    for rel in cfg["files"]:
        path = Path(rel)
        if not path.exists():
            raise FileNotFoundError(f"Missing data file: {path}")
        size = path.stat().st_size
        entries.append(
            {
                "path": rel,
                "bytes": size,
                "mb": round(size / 1e6, 1),
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        # Graph files are pickled PyG objects: load them with the same torch-geometric version.
        "created_with": {
            "torch": torch.__version__,
            "torch_geometric": torch_geometric.__version__,
        },
        "total_mb": round(sum(e["bytes"] for e in entries) / 1e6, 1),
        "files": entries,
    }
    out = Path(cfg["manifest_out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
