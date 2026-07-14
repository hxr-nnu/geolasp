from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent


def _path_from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value) if value else default


DEFAULT_DATA_ROOT = _path_from_env("GEOLASP_DATA_ROOT", REPO_ROOT / "data")
DEFAULT_SCANREFER_ROOT = _path_from_env("SCANREFER_ROOT", DEFAULT_DATA_ROOT / "scanrefer")
DEFAULT_PROCESSED_ROOT = _path_from_env("GEOLASP_PROCESSED_ROOT", DEFAULT_DATA_ROOT / "processed")
DEFAULT_LAYOUT_DIR = _path_from_env("GEOLASP_LAYOUT_DIR", DEFAULT_PROCESSED_ROOT / "layouts")
DEFAULT_OUTPUT_DIR = _path_from_env("GEOLASP_OUTPUT_DIR", DEFAULT_PROCESSED_ROOT / "outputs")
DEFAULT_CONSTRAINT_CACHE_DIR = _path_from_env("GEOLASP_CONSTRAINT_CACHE_DIR", DEFAULT_PROCESSED_ROOT / "constraints")
DEFAULT_POINT_FEATURE_DIR = _path_from_env("GEOLASP_POINT_FEATURE_DIR", DEFAULT_OUTPUT_DIR / "point_features")
DEFAULT_SPATIALLM_ROOT = _path_from_env(
    "SPATIALLM_ROOT",
    _path_from_env("GEOLASP_SPATIALLM_ROOT", REPO_ROOT.parent / "SpatialLM"),
)
