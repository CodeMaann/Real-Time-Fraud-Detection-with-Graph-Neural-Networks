"""Smoke tests — verify the package installs and config loads correctly."""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_package_imports() -> None:
    """The top-level package should be importable and expose a version."""
    import fraudgnn

    assert hasattr(fraudgnn, "__version__")
    assert isinstance(fraudgnn.__version__, str)


def test_subpackages_import() -> None:
    """Every declared subpackage should be importable."""


def test_config_loads_default() -> None:
    """``load_config`` should parse the default YAML without errors."""
    from fraudgnn.config import load_config

    cfg = load_config(DEFAULT_CONFIG)
    assert cfg.project_name == "fraud-gnn"
    assert cfg.data.num_transactions == 500_000
    assert cfg.data.fraud_ratio == pytest.approx(0.008)
    assert cfg.training.seed == 42


def test_config_missing_file_raises() -> None:
    """``load_config`` should raise ``FileNotFoundError`` for bad paths."""
    from fraudgnn.config import load_config

    with pytest.raises(FileNotFoundError):
        load_config("nonexistent.yaml")


def test_config_validation_rejects_bad_values(tmp_path: Path) -> None:
    """Pydantic validation should reject out-of-range values."""
    from pydantic import ValidationError

    from fraudgnn.config import AppConfig

    with pytest.raises(ValidationError):
        AppConfig(data={"fraud_ratio": 1.5})  # type: ignore[arg-type]
