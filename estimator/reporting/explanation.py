from __future__ import annotations

import json
import re
import sys
from typing import Any, Callable

import pandas as pd

from estimator.candidates.signatures import append_warning, cell_text, json_text, numeric_or_none, trace_row
from estimator.paths import CLASSIFIER_BACKEND_DIR
from estimator.pricing.estimate_calculator import calc_amount
from estimator.pricing.summary import sum_component_amount
from estimator.query_models import EstimateScenario

if str(CLASSIFIER_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSIFIER_BACKEND_DIR))
from classifier.llm_client import LLMServiceError, request_llm_json_with_usage  # noqa: E402


def build_final_explanation_prompt(
    raw_query: str,
    final_items: list[dict[str, Any]],
    total_p10: float | None,
    total_median: float | None,
    total_p90: float | None,
) -> str:
    payload = {
        "user_query": raw_query,
        "final_items": final_items,
        "total_price": {"p10": total_p10, "median": total_median, "p90": total_p90},
    }
    return f"""
任务：根据已经确定的维修清单和估价结果，生成面向客户的组合维修参考方案说明。

你只负责解释输入中已经存在的最终结果，不得修改或补充计价内容。

方案定位：

当前结果是一套根据相似历史维修清单形成的常见组合参考，不代表已经完成现场勘察，也不代表唯一或最终施工方案。

要求：

1. 方案名称应简短、客观，概括维修对象和主要处理方式。
2. 方案说明使用2至3句话，清楚说明：
   - 当前方案针对什么维修需求；
   - 输入中的清单组合成什么常见维修做法；
   - 组合中主要项目分别承担什么作用；
   - 估价区间根据当前组合内各清单的历史价格证据汇总形成。
3. 可以整理输入清单之间明显且合理的先后关系。
4. 可以解释输入中已有项目的作用。
5. 不得增加输入中不存在的工序、材料、设备或计价项目。
6. 不得推断损坏原因、现场状态或故障结论。
7. 不得把当前参考组合写成唯一方案、最优方案或已经确定的施工方案。
8. 不介绍模型、检索、Family、Option、样本数或内部计算过程。
9. 待现场确认事项只写可能明显影响最终维修范围、工程量或价格的关键缺失信息。
10. 待现场确认事项应具体，不重复“仅供参考”“以现场实际为准”等空泛表达。
11. 不得编造用户未提供、清单中也无法确认的型号、规格、数量或施工条件。
12. 语言简洁、自然，适合直接展示给客户。

只输出合法 JSON，顶层只能包含以下字段：

{{
  "scenario_name": "",
  "scenario_summary": "",
  "site_confirmation": ""
}}

输入：
{json.dumps(payload, ensure_ascii=False)}
""".strip()


def parse_final_explanation_result(result: dict[str, Any], scenario: EstimateScenario) -> EstimateScenario:
    required_fields = {"scenario_name", "scenario_summary", "site_confirmation"}
    if not isinstance(result, dict) or set(result) != required_fields:
        raise ValueError("final_explanation 顶层字段非法")
    values: dict[str, str] = {}
    for field in required_fields:
        value = result.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} 必须为非空字符串")
        if re.search(r"(?:```|\*\*|__|(?:^|\n)\s*(?:#{1,6}\s|[-*+]\s|\d+\.\s|>\s))", value):
            raise ValueError(f"{field} 不得包含 Markdown")
        values[field] = value.strip()
    return EstimateScenario(
        scenario.scenario_id, scenario.scenario_order, values["scenario_name"],
        values["scenario_summary"], scenario.items, values["site_confirmation"],
    )


def final_explanation_items(estimate_scenarios: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "cost_item_name": cell_text(row.get("清单名称")),
            "project_description": cell_text(row.get("项目特征")),
            "unit": cell_text(row.get("单位")),
            "quantity": numeric_or_none(row.get("工程量")),
            "unit_price": numeric_or_none(row.get("综合单价中位数")),
            "estimated_amount": calc_amount(row.get("工程量"), row.get("综合单价中位数")),
        }
        for _index, row in estimate_scenarios.iterrows()
    ]


def short_description(value: Any, limit: int = 28) -> str:
    text = re.sub(r"\s+", " ", cell_text(value)).strip()
    return text if len(text) <= limit else f"{text[:limit]}..."


def fallback_scenario(scenario: EstimateScenario, estimate_scenarios: pd.DataFrame) -> EstimateScenario:
    first_name = next(
        (cell_text(value) for value in estimate_scenarios.get("清单名称", pd.Series(dtype=object)).tolist() if cell_text(value)),
        "",
    )
    scenario_name = f"{short_description(first_name, 18)}维修组合参考方案" if first_name else "维修组合初步估价"
    return EstimateScenario(
        scenario.scenario_id,
        scenario.scenario_order,
        scenario_name,
        "当前结果根据保留清单形成一套历史常见维修组合，估价区间由组合内各清单的历史价格证据汇总形成，具体清单和价格见明细表。",
        scenario.items,
        "需确认实际维修部位、范围、规格、数量及现场施工条件。",
    )


def insufficient_evidence_scenario(scenario: EstimateScenario) -> EstimateScenario:
    return EstimateScenario(
        scenario.scenario_id,
        scenario.scenario_order,
        "历史价格证据不足",
        "当前检索结果中没有清单达到最低历史价格证据要求，暂不形成可展示的估价组合。",
        [],
        "建议补充具体维修对象、部位、规格、数量及现场检测信息后重新估价。",
    )


def generate_final_explanation(
    raw_text: str,
    scenario: EstimateScenario,
    estimate_scenarios: pd.DataFrame,
    warnings: list[str] | None = None,
    *,
    request_fn: Callable[..., Any] | None = None,
    trace_factory: Callable[..., dict[str, Any]] | None = None,
    warning_fn: Callable[[list[str] | None, str], None] | None = None,
) -> tuple[EstimateScenario, bool, str, str, dict[str, Any]]:
    request_fn = request_fn or request_llm_json_with_usage
    trace_factory = trace_factory or trace_row
    warning_fn = warning_fn or append_warning
    total_p10 = sum_component_amount(estimate_scenarios, "综合单价P10")
    total_median = sum_component_amount(estimate_scenarios, "综合单价中位数")
    total_p90 = sum_component_amount(estimate_scenarios, "综合单价P90")
    prompt = build_final_explanation_prompt(
        raw_text, final_explanation_items(estimate_scenarios), total_p10, total_median, total_p90,
    )
    max_tokens = 4096
    try:
        response = request_fn(
            prompt, max_tokens=max_tokens,
            system_prompt="你只输出一个 JSON object，不输出解释、Markdown 或思考过程。",
        )
        explained = parse_final_explanation_result(response.content, scenario)
        trace = trace_factory(
            "final_explanation", "生成历史清单驱动的组合维修参考方案说明", True,
            prompt=prompt, max_tokens=max_tokens,
            input_summary=json_text({"item_count": len(scenario.items)}),
            usage=response.usage, raw_response=getattr(response, "raw_content", ""),
            scenario_count=1, scenario_item_count=len(scenario.items),
        )
        return explained, True, "", prompt, trace
    except (LLMServiceError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        warning_fn(warnings, "final_explanation_failed")
        trace = trace_factory(
            "final_explanation", "生成历史清单驱动的组合维修参考方案说明", False,
            error=str(exc), prompt=prompt, max_tokens=max_tokens,
            input_summary=json_text({
                "selected_project_package_id": scenario.items[0].project_package_id,
                "selected_stable_sample_ids": [item.stable_sample_id for item in scenario.items],
                "item_count": len(scenario.items),
            }),
            scenario_count=1, scenario_item_count=len(scenario.items),
        )
        return fallback_scenario(scenario, estimate_scenarios), False, str(exc), prompt, trace


def generate_optional_final_explanation(
    with_explanations: bool,
    raw_text: str,
    scenario: EstimateScenario,
    estimate_scenarios: pd.DataFrame,
    warnings: list[str] | None = None,
    **dependencies: Any,
) -> tuple[EstimateScenario, bool, str, str, dict[str, Any]]:
    if with_explanations:
        return generate_final_explanation(
            raw_text, scenario, estimate_scenarios, warnings=warnings, **dependencies
        )
    trace_factory = dependencies.get("trace_factory") or trace_row
    trace = trace_factory(
        "final_explanation", "客户方案说明已按运行配置使用程序 fallback", True,
        input_summary=json_text({"with_explanations": False}),
        scenario_count=1, scenario_item_count=len(scenario.items),
    )
    trace["fallback"] = True
    return fallback_scenario(scenario, estimate_scenarios), True, "", "", trace


def generate_customer_explanation(
    with_explanations: bool,
    raw_query: str,
    scenario: EstimateScenario,
    estimate_scenarios: pd.DataFrame,
    warnings: list[str] | None = None,
    **dependencies: Any,
) -> tuple[EstimateScenario, bool, str, str, dict[str, Any]]:
    if not estimate_scenarios.empty:
        return generate_optional_final_explanation(
            with_explanations, raw_query, scenario, estimate_scenarios,
            warnings=warnings, **dependencies,
        )
    trace_factory = dependencies.get("trace_factory") or trace_row
    trace = trace_factory(
        "final_explanation", "全部最终项因历史价格证据不足而跳过客户方案说明", True,
        input_summary=json_text({"skip_reason": "no_display_items"}),
        scenario_count=1, scenario_item_count=0,
    )
    trace["fallback"] = True
    trace["parsed_status"] = "skipped"
    return insufficient_evidence_scenario(scenario), True, "", "", trace
