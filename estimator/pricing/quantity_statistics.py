from __future__ import annotations

from typing import Any

import pandas as pd

from estimator.candidates.signatures import cell_text, join_non_empty, numeric_or_none, normalized_unit, truncate_text, json_text, append_warning, trace_row

def build_quantity_statistics(
    retrieved_samples: pd.DataFrame,
    target_unit: str,
) -> dict[str, Any]:
    normalized_target_unit = normalized_unit(target_unit)
    valid_quantities: list[float] = []
    seen_stable_ids: set[str] = set()
    for row_index, row in retrieved_samples.iterrows():
        stable_sample_id = cell_text(row.get("stable_sample_id"))
        if stable_sample_id:
            if stable_sample_id in seen_stable_ids:
                continue
            seen_stable_ids.add(stable_sample_id)
        sample_unit = normalized_unit(cell_text(row.get("unit_normalized")) or row.get("unit"))
        if not normalized_target_unit or sample_unit != normalized_target_unit:
            continue
        quantity = numeric_or_none(row.get("quantity"))
        if quantity is None or quantity <= 0:
            continue
        valid_quantities.append(float(quantity))
    if not valid_quantities:
        return {
            "sample_count": 0,
            "minimum": None,
            "median": None,
            "maximum": None,
            "fallback_used": False,
        }
    series = pd.Series(valid_quantities, dtype=float)
    return {
        "sample_count": int(len(series)),
        "minimum": float(series.min()),
        "median": float(series.median()),
        "maximum": float(series.max()),
        "fallback_used": False,
    }
