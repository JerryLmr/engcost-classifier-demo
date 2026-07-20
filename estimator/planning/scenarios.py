from __future__ import annotations

from typing import Any

import pandas as pd

from estimator.candidates.signatures import cell_text
from estimator.query_models import EstimateScenario, ScenarioItem

def build_scenario_from_plan_items(
    project_package_id: str,
    plan_items: pd.DataFrame,
    sample_lookup: dict[str, dict[str, Any]],
    quantities: dict[int, dict[str, Any]],
) -> EstimateScenario:
    items: list[ScenarioItem] = []
    for _index, row in plan_items.iterrows():
        position = int(row["item_position"])
        stable_sample_id = cell_text(row.get("stable_sample_id"))
        sample = sample_lookup.get(stable_sample_id)
        if sample is None:
            raise ValueError(f"最终清单无法回查证据: item_position={position}, stable_sample_id={stable_sample_id}")
        if cell_text(sample.get("project_package_id")) != project_package_id:
            raise ValueError(f"最终清单工程包映射不一致: item_position={position}")
        quantity_result = quantities[position]
        items.append(
            ScenarioItem(
                project_package_id=project_package_id,
                stable_sample_id=stable_sample_id,
                source_ref=cell_text(sample.get("source_ref")),
                display_id=cell_text(row.get("display_id")) or cell_text(sample.get("display_id")),
                practice_option_id=cell_text(row.get("selected_option_id")) or cell_text(sample.get("practice_option_id")),
                original_option_id=cell_text(row.get("original_option_id")) or cell_text(sample.get("practice_option_id")),
                original_family_id=cell_text(row.get("original_family_id")) or cell_text(sample.get("family_id")),
                representative_family_id=cell_text(row.get("representative_family_id")) or cell_text(sample.get("family_id")),
                selection_reason="",
                quantity=quantity_result["quantity"],
                quantity_reason=quantity_result["quantity_reason"],
                item_position=position,
                quantity_source=quantity_result["quantity_source"],
                quantity_explanation=quantity_result["quantity_explanation"],
                quantity_sample_count=quantity_result["quantity_sample_count"],
                quantity_minimum=quantity_result["quantity_minimum"],
                quantity_median=quantity_result["quantity_median"],
                quantity_maximum=quantity_result["quantity_maximum"],
                quantity_fallback_used=quantity_result["quantity_fallback_used"],
                quantity_fallback_reason=quantity_result["quantity_fallback_reason"],
            )
        )
    return EstimateScenario("S001", 1, "", "", items)
