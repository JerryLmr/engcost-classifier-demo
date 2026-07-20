from __future__ import annotations

from typing import Any

import pandas as pd

from estimator.candidates.signatures import cell_text, join_non_empty, numeric_or_none, normalized_unit, truncate_text, json_text, append_warning, trace_row

def validate_quantity(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("quantity 必须为 object")
    quantity_type = cell_text(value.get("type"))
    if quantity_type == "exact":
        if set(value) != {"type", "value"}:
            raise ValueError("exact quantity 只能包含 type 和 value")
        number = numeric_or_none(value.get("value"))
        if number is None or number <= 0:
            raise ValueError("exact quantity value 必须大于 0")
        return {"type": "exact", "value": number}
    raise ValueError("quantity.type 只能是 exact")

def calc_amount(quantity: Any, unit_price: Any) -> float | None:
    quantity_number = numeric_or_none(quantity)
    price_number = numeric_or_none(unit_price)
    if quantity_number is None or price_number is None:
        return None
    return round(quantity_number * price_number, 2)

def format_number_cell(value: Any) -> str:
    number = numeric_or_none(value)
    if number is None:
        return ""
    if float(number).is_integer():
        return str(int(number))
    return str(number)

def calculate_quantities(
    plan_items: pd.DataFrame,
    determinations: dict[int, dict[str, Any]],
    quantity_statistics: dict[int, dict[str, Any]],
    sample_lookup: dict[str, dict[str, Any]],
    warnings: list[str] | None = None,
) -> dict[int, dict[str, Any]]:
    quantities: dict[int, dict[str, Any]] = {}
    for _index, row in plan_items.iterrows():
        position = int(row["item_position"])
        determination = determinations[position]
        source = cell_text(determination.get("quantity_source"))
        explanation = cell_text(determination.get("explanation"))
        stats = quantity_statistics.get(position, {})
        fallback_used = False
        fallback_reason = ""
        if source == "user_explicit":
            value = numeric_or_none(determination.get("quantity"))
            reason = explanation
            output_stats = {}
        else:
            value = numeric_or_none(stats.get("median"))
            output_stats = stats
            if value is None:
                append_warning(warnings, f"quantity_median_unavailable:{position}")
                stable_sample_id = cell_text(row.get("stable_sample_id"))
                best_sample = sample_lookup.get(stable_sample_id, {})
                best_quantity = numeric_or_none(best_sample.get("quantity"))
                target_unit = normalized_unit(cell_text(row.get("unit_normalized")) or row.get("unit"))
                best_unit = normalized_unit(cell_text(best_sample.get("unit_normalized")) or best_sample.get("unit"))
                if best_quantity is None or best_quantity <= 0 or not target_unit or best_unit != target_unit:
                    raise ValueError(f"无有效工程量中位数或最佳召回样本工程量: item_position={position}")
                value = best_quantity
                fallback_used = True
                fallback_reason = "best_sample_quantity"
                append_warning(warnings, f"quantity_best_sample_fallback:{position}")
                reason = f"{explanation}；全库同类样本无有效中位数，暂采用最佳召回样本工程量。"
            else:
                unit = cell_text(row.get("unit_normalized")) or cell_text(row.get("unit"))
                reason = (
                    f"{explanation}；采用全库召回的{int(stats.get('sample_count') or 0)}条同类历史样本"
                    f"工程量中位数{round(float(value), 4):g}{unit}暂估。"
                )
        rounded = None if value is None else round(float(value), 4)
        if rounded is None or rounded <= 0:
            raise ValueError(f"最终工程量必须大于 0: item_position={position}")
        quantities[position] = {
            "quantity": {"type": "exact", "value": rounded},
            "quantity_source": source,
            "quantity_explanation": explanation,
            "quantity_reason": reason,
            "quantity_sample_count": output_stats.get("sample_count"),
            "quantity_minimum": output_stats.get("minimum"),
            "quantity_median": output_stats.get("median"),
            "quantity_maximum": output_stats.get("maximum"),
            "quantity_fallback_used": fallback_used,
            "quantity_fallback_reason": fallback_reason,
        }
    return quantities

def quantity_display(quantity: dict[str, Any]) -> Any:
    return validate_quantity(quantity)["value"]

def quantity_values(quantity: dict[str, Any]) -> tuple[Any, Any, Any]:
    value = validate_quantity(quantity)["value"]
    return value, value, value

def quantity_amounts(quantity: dict[str, Any], price_stats: dict[str, Any]) -> tuple[Any, Any, Any]:
    value = validate_quantity(quantity)["value"]
    return (
        calc_amount(value, price_stats.get("unit_price_p10")),
        calc_amount(value, price_stats.get("unit_price_median")),
        calc_amount(value, price_stats.get("unit_price_p90")),
    )
