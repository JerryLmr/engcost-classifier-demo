from __future__ import annotations

import sys
from typing import Any

import pandas as pd

from estimator.paths import CLASSIFIER_BACKEND_DIR
from estimator.candidates.signatures import cell_text, join_non_empty, numeric_or_none, normalized_unit, truncate_text, json_text, append_warning, trace_row
if str(CLASSIFIER_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSIFIER_BACKEND_DIR))
from classifier.llm_client import LLMServiceError, request_llm_json_with_usage  # noqa: E402
from estimator.candidates.practice_options import display_option_maps
from estimator.pricing.evidence_expansion import expand_samples_for_option
from estimator.pricing.quantity_statistics import build_quantity_statistics
from estimator.pricing.estimate_calculator import calculate_quantities
from estimator.planning.scenarios import build_scenario_from_plan_items
from estimator.query_models import EstimateScenario

def quantity_rule_payload(raw_text: str, plan_items: pd.DataFrame) -> dict[str, Any]:
    items = [
        {
            "item_position": int(row["item_position"]),
            "cost_item_name": cell_text(row.get("cost_item_name")),
            "project_description": cell_text(row.get("project_description")),
            "unit": cell_text(row.get("unit")),
        }
        for _index, row in plan_items.iterrows()
    ]
    if not items:
        raise ValueError("最终清单为空，无法确定工程量")
    return {"user_query": raw_text, "items": items}

def build_quantity_determination_prompt(
    raw_text: str, plan_items: pd.DataFrame,
) -> str:
    payload = quantity_rule_payload(raw_text, plan_items)
    return f"""
任务：判断每条清单的工程量来源。

工程量来源只能是：

- user_explicit
  用户明确给出数量，且该数量能根据部位、项目名称、材料、规格和单位直接对应当前清单。
  quantity 填用户给出的数量。

- historical_median
  当前清单没有可直接采用的用户数量。
  可能是用户未提供数量，也可能是用户提供的数量对应其他清单。
  quantity 必须为 null，具体工程量由程序使用全库同类历史样本中位数计算。

规则：

1. 只把用户数量绑定到直接匹配的清单。
2. 不得根据不同清单之间的历史工程量关系推导数量。
3. 材料、规格、部位和单位的匹配优先于清单顺序。
4. 必须完整覆盖全部 item_position，不得遗漏或重复。
5. explanation 必须结合当前清单说明原因，不得所有项目重复同一句模板。
6. 只输出合法 JSON，不得增加其他字段。

输出：

{{
  "items": [
    {{
      "item_position": 0,
      "quantity_source": "user_explicit",
      "quantity": 100,
      "explanation": "用户给出的数量与当前清单的部位、材料和规格直接对应"
    }},
    {{
      "item_position": 1,
      "quantity_source": "historical_median",
      "quantity": null,
      "explanation": "用户给出的数量对应其他施工内容，当前清单缺少可直接采用的数量依据"
    }}
  ]
}}

输入：
{json_text(payload)}
""".strip()

def parse_quantity_determination_result(
    result: Any,
    plan_items: pd.DataFrame,
) -> dict[int, dict[str, Any]]:
    return validate_quantity_determination_result(result, plan_items)

def validate_quantity_determination_result(
    result: Any, plan_items: pd.DataFrame,
) -> dict[int, dict[str, Any]]:
    if plan_items.empty:
        raise ValueError("最终清单为空，无法确定工程量")
    if not isinstance(result, dict) or set(result) != {"items"}:
        raise ValueError("quantity determination 顶层字段非法")
    raw_entries = result.get("items")
    if not isinstance(raw_entries, list):
        raise ValueError("quantity determination items 必须是数组")
    expected_positions = {int(value) for value in plan_items["item_position"].tolist()}
    parsed: dict[int, dict[str, Any]] = {}
    for entry in raw_entries:
        if not isinstance(entry, dict) or set(entry) != {
            "item_position", "quantity_source", "quantity", "explanation",
        }:
            raise ValueError("quantity determination item 字段非法")
        position = entry.get("item_position")
        if isinstance(position, bool) or not isinstance(position, int):
            raise ValueError("quantity item_position 必须是整数")
        if position in parsed:
            raise ValueError(f"quantity item_position 重复: {position}")
        if position not in expected_positions:
            raise ValueError(f"quantity item_position 非法: {position}")
        source = cell_text(entry.get("quantity_source"))
        if source not in {"user_explicit", "historical_median"}:
            raise ValueError(f"quantity_source 非法: item_position={position}")
        explanation = cell_text(entry.get("explanation"))
        if not explanation:
            raise ValueError(f"quantity explanation 不得为空: item_position={position}")
        raw_quantity = entry.get("quantity")
        quantity = None
        if source == "user_explicit":
            if isinstance(raw_quantity, bool) or not isinstance(raw_quantity, (int, float)):
                raise ValueError(f"user_explicit quantity 必须是正数: item_position={position}")
            quantity = numeric_or_none(raw_quantity)
            if quantity is None or quantity <= 0:
                raise ValueError(f"user_explicit quantity 必须是正数: item_position={position}")
        elif raw_quantity is not None:
            raise ValueError(f"historical_median quantity 必须为 null: item_position={position}")
        parsed[position] = {
            "quantity_source": source,
            "quantity": quantity,
            "explanation": explanation,
        }
    if set(parsed) != expected_positions:
        raise ValueError("quantity determination 必须完整覆盖全部最终清单")
    return parsed

def quantity_determination_fallback(plan_items: pd.DataFrame) -> dict[int, dict[str, Any]]:
    return {
        int(row["item_position"]): {
            "quantity_source": "historical_median",
            "quantity": None,
            "explanation": "未能从用户描述中获得可直接采用的当前清单工程量",
        }
        for _index, row in plan_items.iterrows()
    }

def generate_quantity_determination(
    raw_text: str,
    project_package_id: str,
    plan_items: pd.DataFrame,
    sample_lookup: dict[str, dict[str, Any]],
    displays_with_options: pd.DataFrame,
    candidate_families: pd.DataFrame,
    samples: pd.DataFrame,
    warnings: list[str] | None = None,
    *, request_fn=request_llm_json_with_usage, trace_factory=trace_row,
    warning_fn=append_warning,
) -> tuple[EstimateScenario, str, dict[str, Any]]:
    request_llm_json_with_usage = request_fn
    trace_row = trace_factory
    append_warning = warning_fn
    if plan_items.empty:
        raise ValueError("最终清单为空，无法确定工程量")
    prompt = build_quantity_determination_prompt(raw_text, plan_items)
    max_tokens = 4096
    response = None
    error_message = ""
    try:
        response = request_llm_json_with_usage(
            prompt,
            max_tokens=max_tokens,
            system_prompt="你只输出一个 JSON object，不输出解释、Markdown 或思考过程。",
        )
        determinations = validate_quantity_determination_result(response.content, plan_items)
    except (LLMServiceError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        error_message = str(exc)
        determinations = quantity_determination_fallback(plan_items)
        append_warning(warnings, "quantity_determination_fallback_historical_median")
    display_map, option_map = display_option_maps(displays_with_options)
    quantity_statistics: dict[int, dict[str, Any]] = {}
    for _index, row in plan_items.iterrows():
        position = int(row["item_position"])
        if determinations[position]["quantity_source"] == "user_explicit":
            quantity_statistics[position] = {}
            continue
        display_id = cell_text(row.get("display_id"))
        option_id = cell_text(row.get("selected_option_id")) or cell_text(row.get("practice_option_id"))
        display = display_map.get(display_id)
        option = option_map.get((display_id, option_id))
        if display is None or option is None:
            raise ValueError(f"工程量统计 display/option 回查失败: item_position={position}")
        expanded = expand_samples_for_option(option, display, candidate_families, samples)
        target_unit = cell_text(row.get("unit_normalized")) or cell_text(row.get("unit"))
        quantity_statistics[position] = build_quantity_statistics(expanded, target_unit)
    quantities = calculate_quantities(
        plan_items, determinations, quantity_statistics, sample_lookup, warnings
    )
    scenario = build_scenario_from_plan_items(
        project_package_id, plan_items, sample_lookup, quantities
    )
    trace = trace_row(
        "quantity_determination",
        "确定最终连续区间内全部清单的工程量",
        not error_message,
        error=error_message,
        prompt=prompt,
        max_tokens=max_tokens,
        input_summary=json_text({
            "selected_project_package_id": project_package_id,
            "expected_item_positions": plan_items["item_position"].tolist(),
        }),
        usage=response.usage if response is not None else None,
        raw_response=getattr(response, "raw_content", "") if response is not None else "",
        scenario_count=1,
        scenario_item_count=len(scenario.items),
    )
    trace["fallback"] = bool(error_message)
    trace["quantity_items"] = json_text([
        {
            "item_position": position,
            "quantity_source": result["quantity_source"],
            "llm_quantity": determinations[position]["quantity"],
            "final_quantity": result["quantity"]["value"],
            "quantity_sample_count": result["quantity_sample_count"],
            "quantity_minimum": result["quantity_minimum"],
            "quantity_median": result["quantity_median"],
            "quantity_maximum": result["quantity_maximum"],
            "median_available": result["quantity_median"] is not None,
            "quantity_fallback_used": result["quantity_fallback_used"],
            "quantity_fallback_reason": result["quantity_fallback_reason"],
            "error_message": error_message,
        }
        for position, result in quantities.items()
    ])
    return scenario, prompt, trace
