from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from estimator.output.formatting import apply_workbook_style


def write_estimate_workbook(
    output_path: Path,
    frames: Mapping[str, pd.DataFrame],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, frame in frames.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
    apply_workbook_style(output_path)
