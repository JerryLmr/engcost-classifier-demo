#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from estimator.ingestion import ocr_filter as _implementation  # noqa: E402

globals().update(
    {name: getattr(_implementation, name) for name in dir(_implementation) if not name.startswith("__")}
)

if __name__ == "__main__":
    raise SystemExit(_implementation.main())
