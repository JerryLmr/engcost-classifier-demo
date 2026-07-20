from __future__ import annotations

from datetime import datetime
from pathlib import Path


ESTIMATOR_DIR = Path(__file__).resolve().parent
REPO_ROOT = ESTIMATOR_DIR.parent
CLASSIFIER_BACKEND_DIR = REPO_ROOT / "classifier" / "backend"


def resolve_repo_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def default_query_output_path() -> Path:
    return REPO_ROOT / "query" / f"{datetime.now().strftime('%Y%m%d%H%M')}.xlsx"
