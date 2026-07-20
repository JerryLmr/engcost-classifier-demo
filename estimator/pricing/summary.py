from __future__ import annotations

from typing import Any

import pandas as pd

from estimator.candidates.signatures import cell_text, join_non_empty
from estimator.candidates.families import numeric_values
from estimator.pricing.estimate_calculator import calc_amount
from estimator.query_models import EstimateScenario

MIN_DISPLAY_PRICE_EVIDENCE_COUNT = 3

ESTIMATE_SUMMARY_COLUMNS = [
    "用户问题",
    "方案名称",
    "方案说明",
    "主要施工内容",
    "计价项目数",
    "参考项目数",
    "参考样本数",
    "合价P10",
    "合价中位数",
    "合价P90",
    "其中包含人工费P10",
    "其中包含人工费中位数",
    "其中包含人工费P90",
    "其中包含机械费P10",
    "其中包含机械费中位数",
    "其中包含机械费P90",
    "待现场确认事项",
]

def sum_component_amount(
    scenario_rows: pd.DataFrame,
    unit_price_column: str,
) -> float | None:
    amounts: list[float] = []
    for _index, row in scenario_rows.iterrows():
        amount = calc_amount(row.get("工程量"), row.get(unit_price_column))
        if amount is not None:
            amounts.append(amount)
    if not amounts:
        return None
    return round(sum(amounts), 2)

def count_reference_projects(price_evidence_items: pd.DataFrame) -> int:
    for column in ["project_key", "project_package_id"]:
        if column not in price_evidence_items.columns:
            continue
        values = price_evidence_items[column].map(cell_text)
        non_empty = values[values.ne("")]
        if not non_empty.empty:
            return int(non_empty.nunique())
    return 0

def filter_customer_display_outputs(
    scenario: EstimateScenario,
    scenario_rows_with_positions: pd.DataFrame,
    all_price_evidence_items: pd.DataFrame,
) -> tuple[EstimateScenario, pd.DataFrame, pd.DataFrame]:
    if "final_item_position" not in scenario_rows_with_positions.columns:
        raise ValueError("最终展示过滤缺少 final_item_position")
    evidence_counts = pd.to_numeric(
        scenario_rows_with_positions.get("价格证据样本数"), errors="coerce"
    ).fillna(0)
    retained_rows = scenario_rows_with_positions[
        evidence_counts.ge(MIN_DISPLAY_PRICE_EVIDENCE_COUNT)
    ].copy()
    retained_positions = {
        int(value)
        for value in pd.to_numeric(retained_rows["final_item_position"], errors="coerce").dropna().tolist()
    }
    filtered_scenario = EstimateScenario(
        scenario.scenario_id,
        scenario.scenario_order,
        scenario.scenario_name,
        scenario.scenario_summary,
        [item for item in scenario.items if item.item_position in retained_positions],
        scenario.site_confirmation,
    )
    display_evidence = all_price_evidence_items[
        pd.to_numeric(
            all_price_evidence_items.get("final_item_position", pd.Series(dtype=float)),
            errors="coerce",
        ).isin(retained_positions)
    ].copy()
    return (
        filtered_scenario,
        retained_rows.drop(columns=["final_item_position"]).reset_index(drop=True),
        display_evidence.reset_index(drop=True),
    )

def build_estimate_summary(
    raw_query: str,
    scenarios: list[EstimateScenario],
    estimate_scenarios: pd.DataFrame,
    display_price_evidence_items: pd.DataFrame,
) -> pd.DataFrame:
    scenario = scenarios[0] if scenarios else None
    row = {
        "用户问题": raw_query,
        "方案名称": scenario.scenario_name if scenario is not None else "",
        "方案说明": scenario.scenario_summary if scenario is not None else "",
        "主要施工内容": join_non_empty([
            item.get("清单名称") for _index, item in estimate_scenarios.iterrows()
        ]),
        "计价项目数": int(len(estimate_scenarios)),
        "参考项目数": count_reference_projects(display_price_evidence_items),
        "参考样本数": int(len(display_price_evidence_items)),
        "合价P10": sum_component_amount(estimate_scenarios, "综合单价P10"),
        "合价中位数": sum_component_amount(estimate_scenarios, "综合单价中位数"),
        "合价P90": sum_component_amount(estimate_scenarios, "综合单价P90"),
        "其中包含人工费P10": sum_component_amount(estimate_scenarios, "其中包含人工费单价P10"),
        "其中包含人工费中位数": sum_component_amount(estimate_scenarios, "其中包含人工费单价中位数"),
        "其中包含人工费P90": sum_component_amount(estimate_scenarios, "其中包含人工费单价P90"),
        "其中包含机械费P10": sum_component_amount(estimate_scenarios, "其中包含机械费单价P10"),
        "其中包含机械费中位数": sum_component_amount(estimate_scenarios, "其中包含机械费单价中位数"),
        "其中包含机械费P90": sum_component_amount(estimate_scenarios, "其中包含机械费单价P90"),
        "待现场确认事项": scenario.site_confirmation if scenario is not None else "",
    }
    return pd.DataFrame([row], columns=ESTIMATE_SUMMARY_COLUMNS).fillna("")
