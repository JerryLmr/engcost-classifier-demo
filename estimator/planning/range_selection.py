from __future__ import annotations

import re
import sys
from typing import Any

import pandas as pd

from estimator.paths import CLASSIFIER_BACKEND_DIR
from estimator.candidates.signatures import cell_text, join_non_empty, numeric_or_none, normalized_unit, truncate_text, json_text, append_warning, trace_row
if str(CLASSIFIER_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSIFIER_BACKEND_DIR))
from classifier.llm_client import LLMServiceError, request_llm_json_with_usage  # noqa: E402

def item_row_numeric_order(value: Any) -> float | None:
    text = cell_text(value)
    if not text:
        return None
    if re.fullmatch(r"\d+(?:\.0+)?", text):
        return float(text)
    match = re.search(r"(?:^|[-_])(\d+)$", text)
    return float(match.group(1)) if match else None

def ordered_project_items(samples: pd.DataFrame, project_package_id: str) -> pd.DataFrame:
    rows = samples[samples["project_package_id"].map(cell_text).eq(project_package_id)].copy()
    if rows.empty:
        return rows
    rows["_source_order"] = range(len(rows))
    rows["_item_row_order"] = rows.get("item_row_id", pd.Series(index=rows.index, dtype=object)).map(item_row_numeric_order)
    rows["_seq_order"] = pd.to_numeric(rows.get("seq", pd.Series(index=rows.index, dtype=object)), errors="coerce")
    rows["_preferred_order"] = rows["_item_row_order"].where(rows["_item_row_order"].notna(), rows["_seq_order"])
    rows["_missing_order"] = rows["_preferred_order"].isna()
    return rows.sort_values(
        ["_missing_order", "_preferred_order", "_source_order"],
        ascending=[True, True, True],
        kind="stable",
    ).drop(columns=["_source_order", "_item_row_order", "_seq_order", "_preferred_order", "_missing_order"])

def expand_selected_project_items(
    samples: pd.DataFrame,
    selected_package: pd.Series,
) -> pd.DataFrame:
    project_package_id = cell_text(selected_package.get("project_package_id"))
    selected_items = ordered_project_items(samples, project_package_id)
    if selected_items.empty:
        raise ValueError(f"所选工程包没有清单: {project_package_id}")
    expected_count = numeric_or_none(selected_package.get("item_count"))
    if expected_count is None or not float(expected_count).is_integer():
        raise ValueError(f"所选工程包 item_count 非法: {selected_package.get('item_count')!r}")
    if len(selected_items) != int(expected_count):
        raise ValueError(
            f"所选工程包完整清单数量与 item_count 不一致: "
            f"project_package_id={project_package_id}, expected={int(expected_count)}, "
            f"actual={len(selected_items)}"
        )
    selected_items = selected_items.reset_index(drop=True)
    selected_items["item_position"] = range(len(selected_items))
    return selected_items

def range_selection_item_records(selected_items: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "item_position": int(row["item_position"]),
            "cost_item_name": cell_text(row.get("cost_item_name")),
            "project_description": cell_text(row.get("project_description")),
        }
        for _index, row in selected_items.iterrows()
    ]

def build_contiguous_item_range_prompt(
    raw_query: str,
    selected_project_name: str,
    selected_items: pd.DataFrame,
) -> str:
    return f"""
你会收到一个已经由程序选定的真实历史工程，以及该工程按原始顺序排列的全部清单。

请根据用户原始需求，在该工程中选择一个最适合作为当前方案骨架的连续清单区间。

规则：
1. start_item_position 和 end_item_position 均为 0-based，并且包含边界。
2. 必须选择一个连续区间，不得返回多个区间。
3. 区间应尽量保留完整的相关施工内容，避免只选择用户直接提到的单个主体项。
4. 工程名称和清单原始顺序是重要参考。
5. 无法明确缩小时，选择完整工程。
6. 本阶段不判断工程量、不修改工艺、不生成价格。

只输出：
{{
  "start_item_position": 0,
  "end_item_position": 5
}}

输入：
{json_text({"user_query": raw_query, "selected_project_name": selected_project_name, "items": range_selection_item_records(selected_items)})}
""".strip()

def validate_contiguous_range(result: Any, item_count: int) -> tuple[int, int]:
    if not isinstance(result, dict):
        raise ValueError("区间结果必须是 object")
    if set(result) != {"start_item_position", "end_item_position"}:
        raise ValueError("区间结果必须且只能包含 start_item_position 和 end_item_position")
    start = result["start_item_position"]
    end = result["end_item_position"]
    if isinstance(start, bool) or not isinstance(start, int):
        raise ValueError("start_item_position 必须是整数")
    if isinstance(end, bool) or not isinstance(end, int):
        raise ValueError("end_item_position 必须是整数")
    if start < 0 or end >= item_count:
        raise ValueError("区间位置越界")
    if start > end:
        raise ValueError("区间起点不得大于终点")
    return start, end

def select_contiguous_item_range(
    raw_query: str,
    selected_project_name: str,
    selected_items: pd.DataFrame,
    *, request_fn=request_llm_json_with_usage,
) -> tuple[int, int, dict[str, Any]]:
    request_llm_json_with_usage = request_fn
    if selected_items.empty:
        raise ValueError("所选工程包没有清单，不执行连续区间选择")
    prompt = build_contiguous_item_range_prompt(raw_query, selected_project_name, selected_items)
    fallback_start, fallback_end = 0, len(selected_items) - 1
    raw_response = ""
    usage: dict[str, Any] = {}
    try:
        response = request_llm_json_with_usage(
            prompt,
            max_tokens=256,
            system_prompt="你只输出一个 JSON object，不输出解释、Markdown 或思考过程。",
        )
        raw_response = cell_text(getattr(response, "raw_content", ""))
        usage = getattr(response, "usage", {}) or {}
        start, end = validate_contiguous_range(response.content, len(selected_items))
        return start, end, {
            "range_selection_status": "ok",
            "fallback": False,
            "error_message": "",
            "prompt": prompt,
            "raw_response": raw_response,
            "usage": usage,
        }
    except (LLMServiceError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        return fallback_start, fallback_end, {
            "range_selection_status": "fallback_full_project",
            "fallback": True,
            "error_message": str(exc),
            "prompt": prompt,
            "raw_response": raw_response,
            "usage": usage,
        }


def build_range_selection_trace(
    selected_project_package_id: str,
    selected_items: pd.DataFrame,
    plan_items: pd.DataFrame,
    start: int,
    end: int,
    range_meta: dict[str, Any],
    *,
    trace_factory=trace_row,
) -> dict[str, Any]:
    return trace_factory(
        "range_selection",
        "在确定性选中的完整历史工程内选择连续清单区间",
        not range_meta["fallback"],
        error=range_meta["error_message"],
        prompt=range_meta["prompt"],
        max_tokens=256,
        input_summary=json_text({
            "selected_project_package_id": selected_project_package_id,
            "project_item_count": len(selected_items),
            "start_item_position": start,
            "end_item_position": end,
            "selected_item_count": len(plan_items),
            "fallback": range_meta["fallback"],
        }),
        usage=range_meta["usage"],
        raw_response=range_meta["raw_response"],
        scenario_count=1,
        scenario_item_count=len(plan_items),
    )
