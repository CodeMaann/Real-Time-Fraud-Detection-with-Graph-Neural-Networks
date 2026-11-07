"""Freeze the models to be scored on the holdout (docs/protocol.md §7).

Writes docs/verification/freeze.json with every checkpoint's sha256, the protocol hash and
the holdout-manifest hash. Refuses if a checkpoint is missing, if any GNN checkpoint lacks a
passing leakage battery on val, test and holdout, or if a freeze already exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.holdout_guard import write_freeze


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="configs/holdout.yaml")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    print(json.dumps(write_freeze(cfg), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
