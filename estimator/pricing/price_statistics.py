from __future__ import annotations

from typing import Any

import pandas as pd

from estimator.candidates.signatures import cell_text, join_non_empty, numeric_or_none, normalized_unit, truncate_text, json_text, append_warning, trace_row
from estimator.candidates.families import numeric_values, ordered_refs
from estimator.pricing.evidence_expansion import expand_samples_for_option

def p10_median_p90(frame: pd.DataFrame, column: str) -> tuple[float | None, float | None, float | None]:
    values = numeric_values(frame, column)
    if values.empty:
        return None, None, None
    return (
        float(values.quantile(0.10, interpolation="nearest")),
        float(values.median()),
        float(values.quantile(0.90, interpolation="nearest")),
    )

def price_stats_for_option(
    option: dict[str, Any],
    display: pd.Series,
    candidate_families: pd.DataFrame,
    samples: pd.DataFrame,
) -> dict[str, Any]:
    expanded = expand_samples_for_option(option, display, candidate_families, samples)

    unit_price = p10_median_p90(expanded, "unit_price")
    labor_price = p10_median_p90(expanded, "labor_unit_price")
    machinery_price = p10_median_p90(expanded, "machinery_unit_price")
    return {
        "unit_price_p10": unit_price[0],
        "unit_price_median": unit_price[1],
        "unit_price_p90": unit_price[2],
        "labor_unit_price_p10": labor_price[0],
        "labor_unit_price_median": labor_price[1],
        "labor_unit_price_p90": labor_price[2],
        "machinery_unit_price_p10": machinery_price[0],
        "machinery_unit_price_median": machinery_price[1],
        "machinery_unit_price_p90": machinery_price[2],
        "source_refs": ordered_refs(expanded["source_ref"]),
        "evidence_count": int(len(expanded)),
        "expanded_evidence": expanded,
    }

def validate_price_stats(price_stats: dict[str, Any], stable_sample_id: str, practice_option_id: str) -> None:
    required = ["unit_price_p10", "unit_price_median", "unit_price_p90"]
    missing = [key for key in required if numeric_or_none(price_stats.get(key)) is None]
    if missing or int(numeric_or_none(price_stats.get("evidence_count")) or 0) <= 0:
        raise ValueError(
            f"价格回查失败: stable_sample_id={stable_sample_id}, "
            f"practice_option_id={practice_option_id}, missing={missing}"
        )
