#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_REPO_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(_REPO_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOTSTRAP))

from estimator.indexing.embedding_model import (
    encode_query,
    encode_texts,
    load_embedding_model as _load_embedding_model,
    normalize_embeddings,
    release_embedding_model,
)
from estimator.indexing.index_loader import load_index
from estimator.paths import (
    CLASSIFIER_BACKEND_DIR,
    REPO_ROOT,
    default_query_output_path,
    resolve_repo_path as resolve_path,
)
from estimator.query_models import EstimateScenario, QueryResult, QueryRewrite, ScenarioItem
from estimator.retrieval.constraints import (
    build_constraint_mask,
    ensure_project_package_candidates,
    filter_rows_and_embeddings,
    normalize_location,
    parse_consultation_dates,
    validate_query_constraints,
)
from estimator.retrieval.evidence_pool import (
    attach_source_refs,
    build_retrieved_evidence_items,
    matched_package_maps,
    source_identity_for_row,
)
from estimator.retrieval.item_retrieval import score_direct_items
from estimator.retrieval.package_retrieval import (
    package_dedupe_key,
    project_package_similarity_map,
    score_project_packages,
    top_score_indices,
)
from estimator.retrieval.query_rewrite import (
    build_query_rewrite_prompt,
    fallback_query_rewrite,
    query_rewrite_for_embedding as _query_rewrite_for_embedding,
    shift_months,
)
from estimator.retrieval.weights import (
    DEFAULT_PACKAGE_WEIGHT_TEMPERATURE,
    PACKAGE_EVIDENCE_WEIGHT_COLUMNS,
    build_package_evidence_weights,
    evidence_package_universe,
)

SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = CLASSIFIER_BACKEND_DIR
if str(CLASSIFIER_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSIFIER_BACKEND_DIR))

from classifier.llm_client import LLMServiceError, check_lmstudio_service, request_llm_json, request_llm_json_with_usage  # noqa: E402












ESTIMATE_SCENARIO_COLUMNS = [
    "清单名称",
    "项目特征",
    "单位",
    "工程量",
    "价格证据样本数",
    "综合单价P10",
    "综合单价",
    "综合单价P90",
    "暂估合价",
    "其中包含人工费单价P10",
    "其中包含人工费单价中位数",
    "其中包含人工费单价P90",
    "其中包含机械费单价P10",
    "其中包含机械费单价中位数",
    "其中包含机械费单价P90",
    "工程量最低值",
    "工程量中位数",
    "工程量最高值",
    "来源样本",
    "合价P10",
    "合价中位数",
    "合价P90",
    "综合单价中位数",
    "practice_option_id",
    "original_option_id",
    "original_family_id",
    "representative_family_id",
    "价格证据family",
    "display_id",
]


PRICE_EVIDENCE_ITEM_COLUMNS = [
    "final_item_position", "清单名称", "display_id", "practice_option_id",
    "family_id", "normalized_signature", "stable_sample_id", "project_key",
    "project_package_id", "source_ref", "工程名称", "location", "consultation_time", "cost_item_name",
    "project_description", "unit", "quantity", "unit_price", "labor_unit_price",
    "machinery_unit_price",
]


LLM_TRACE_COLUMNS = [
    "stage",
    "purpose",
    "prompt",
    "raw_response",
    "parsed_status",
    "error_message",
    "scenario_count",
    "scenario_item_count",
    "prompt_chars",
    "estimated_tokens",
    "max_tokens",
    "input_summary",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "fallback",
    "quantity_items",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="自然语言维修工程造价离线查询入口")
    parser.add_argument("--index-dir", default="embeddings", help="索引目录，默认 embeddings")
    parser.add_argument("--text", required=True, help="口语化维修需求")
    parser.add_argument("--top-packages", type=int, default=20, help="召回相似历史工程包数量")
    parser.add_argument("--top-items", type=int, default=300, help="召回直接相关历史清单行数量")
    parser.add_argument("--output", default=None, help="xlsx 输出路径，默认 query/YYYYMMDDHHMM.xlsx")
    parser.add_argument("--overwrite", action="store_true", help="若输出文件已存在则覆盖")
    parser.add_argument("--include-debug-text", action="store_true", help="在 parse_info 中保留 LLM 调试文本摘要")
    parser.add_argument("--display", action="store_true", help="输出时将部分数值格式化为易读文本")
    parser.add_argument(
        "--with-explanations",
        action="store_true",
        help="生成项目级和清单级 LLM 解释，默认关闭",
    )
    parser.add_argument(
        "--llm-check-timeout",
        type=float,
        default=3.0,
        help="启动前检查 LLM 服务可用性的超时时间，默认 3 秒",
    )
    parser.add_argument(
        "--max-packages-per-cache-subject",
        type=int,
        default=1,
        help="同一 cache_subject 最多保留的相似历史工程包数量，默认 1；设为 0 表示不限制",
    )
    parser.add_argument(
        "--package-weight-temperature",
        type=float,
        default=DEFAULT_PACKAGE_WEIGHT_TEMPERATURE,
        help=f"工程包证据权重 softmax temperature，必须大于 0，默认 {DEFAULT_PACKAGE_WEIGHT_TEMPERATURE}",
    )
    return parser.parse_args()


def validate_output_path(output_path: Path | None, overwrite: bool) -> None:
    if output_path is None:
        return
    if output_path.exists() and output_path.is_dir():
        raise ValueError(f"输出路径是目录，不是文件: {output_path}")
    if output_path.exists() and not overwrite:
        raise ValueError(f"输出已存在，请加 --overwrite 或更换输出路径: {output_path}")






def normalize_dedupe_text(value: Any) -> str:
    return re.sub(r"\s+", " ", cell_text(value).lower()).strip()










def normalize_source_row_id(value: Any) -> str:
    text = cell_text(value)
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return text






def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]






def load_embedding_model(model_name: str) -> Any:
    return _load_embedding_model(model_name, device="cpu")


def query_rewrite_for_embedding(
    query: str,
    current_date: date | None = None,
) -> tuple[QueryRewrite, dict[str, Any]]:
    return _query_rewrite_for_embedding(
        query,
        current_date=current_date,
        request_json=request_llm_json,
        trace_factory=trace_row,
    )


































def first_value(group: pd.DataFrame, column: str) -> str:
    if column not in group.columns:
        return ""
    for value in group[column].tolist():
        text = cell_text(value)
        if text:
            return text
    return ""




















def replace_nan_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return json.loads(frame.replace({np.nan: None}).to_json(orient="records", force_ascii=False))


def split_refs(value: Any, limit: int | None = None) -> list[str]:
    if isinstance(value, list):
        refs = [cell_text(item) for item in value if cell_text(item)]
    else:
        refs = [part.strip() for part in cell_text(value).replace("；", ",").split(",") if part.strip()]
    return refs[:limit] if limit is not None else refs


def trace_id_summary(ids: list[str], limit: int = 20) -> dict[str, Any]:
    if len(ids) <= limit:
        return {"ids": ids}
    return {"ids": ids[:limit], "total": len(ids)}


def append_trace_warnings(trace: dict[str, Any], warnings: list[str]) -> None:
    summary = cell_text(trace.get("input_summary"))
    if not summary:
        trace["input_summary"] = json_text({"warnings": warnings})
        return
    try:
        payload = json.loads(summary)
    except (TypeError, ValueError):
        trace["input_summary"] = f"{summary}; warnings={';'.join(warnings)}"
        return
    if isinstance(payload, dict):
        payload["warnings"] = warnings
        trace["input_summary"] = json_text(payload)
    else:
        trace["input_summary"] = f"{summary}; warnings={';'.join(warnings)}"






































































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
        "total_price": {
            "p10": total_p10,
            "median": total_median,
            "p90": total_p90,
        },
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
        scenario.scenario_id,
        scenario.scenario_order,
        values["scenario_name"],
        values["scenario_summary"],
        scenario.items,
        values["site_confirmation"],
    )


def final_explanation_items(estimate_scenarios: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _index, row in estimate_scenarios.iterrows():
        rows.append({
            "cost_item_name": cell_text(row.get("清单名称")),
            "project_description": cell_text(row.get("项目特征")),
            "unit": cell_text(row.get("单位")),
            "quantity": numeric_or_none(row.get("工程量")),
            "unit_price": numeric_or_none(row.get("综合单价中位数")),
            "estimated_amount": calc_amount(row.get("工程量"), row.get("综合单价中位数")),
        })
    return rows


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


def generate_final_explanation(
    raw_text: str,
    scenario: EstimateScenario,
    estimate_scenarios: pd.DataFrame,
    warnings: list[str] | None = None,
) -> tuple[EstimateScenario, bool, str, str, dict[str, Any]]:
    total_p10 = sum_component_amount(estimate_scenarios, "综合单价P10")
    total_median = sum_component_amount(estimate_scenarios, "综合单价中位数")
    total_p90 = sum_component_amount(estimate_scenarios, "综合单价P90")
    prompt = build_final_explanation_prompt(
        raw_text,
        final_explanation_items(estimate_scenarios),
        total_p10,
        total_median,
        total_p90,
    )
    max_tokens = 4096
    try:
        response = request_llm_json_with_usage(
            prompt, max_tokens=max_tokens,
            system_prompt="你只输出一个 JSON object，不输出解释、Markdown 或思考过程。",
        )
        explained = parse_final_explanation_result(response.content, scenario)
        trace = trace_row(
            "final_explanation", "生成历史清单驱动的组合维修参考方案说明", True,
            prompt=prompt, max_tokens=max_tokens,
            input_summary=json_text({
                "item_count": len(scenario.items),
            }),
            usage=response.usage, raw_response=getattr(response, "raw_content", ""),
            scenario_count=1, scenario_item_count=len(scenario.items),
        )
        return explained, True, "", prompt, trace
    except (LLMServiceError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        append_warning(warnings, "final_explanation_failed")
        trace = trace_row(
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
) -> tuple[EstimateScenario, bool, str, str, dict[str, Any]]:
    if with_explanations:
        return generate_final_explanation(
            raw_text, scenario, estimate_scenarios, warnings=warnings
        )
    trace = trace_row(
        "final_explanation",
        "客户方案说明已按运行配置使用程序 fallback",
        True,
        input_summary=json_text({"with_explanations": False}),
        scenario_count=1,
        scenario_item_count=len(scenario.items),
    )
    trace["fallback"] = True
    return fallback_scenario(scenario, estimate_scenarios), True, "", "", trace


def generate_customer_explanation(
    with_explanations: bool,
    raw_query: str,
    scenario: EstimateScenario,
    estimate_scenarios: pd.DataFrame,
    warnings: list[str] | None = None,
) -> tuple[EstimateScenario, bool, str, str, dict[str, Any]]:
    if not estimate_scenarios.empty:
        return generate_optional_final_explanation(
            with_explanations,
            raw_query,
            scenario,
            estimate_scenarios,
            warnings=warnings,
        )
    trace = trace_row(
        "final_explanation",
        "全部最终项因历史价格证据不足而跳过客户方案说明",
        True,
        input_summary=json_text({"skip_reason": "no_display_items"}),
        scenario_count=1,
        scenario_item_count=0,
    )
    trace["fallback"] = True
    trace["parsed_status"] = "skipped"
    return insufficient_evidence_scenario(scenario), True, "", "", trace


























def build_scenario_outputs(
    scenarios: list[EstimateScenario],
    displays_with_options: pd.DataFrame,
    candidate_families: pd.DataFrame,
    samples: pd.DataFrame,
    expanded_family_ids_by_position: dict[int, list[str]] | None = None,
    include_internal_positions: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    display_map, option_map = display_option_maps(displays_with_options)
    family_map = {cell_text(row.get("family_id")): row for _index, row in candidate_families.iterrows()}
    scenario_rows: list[dict[str, Any]] = []
    price_evidence_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for item in scenario.items:
            display_row = display_map.get(item.display_id)
            option = option_map.get((item.display_id, item.practice_option_id))
            if display_row is None or option is None:
                raise ValueError(f"最终清单 display/option 回查失败: {item.stable_sample_id}")
            representative = family_map.get(item.representative_family_id)
            if representative is None:
                raise ValueError(f"代表 Family 回查失败: {item.representative_family_id}")
            representative_unit = cell_text(representative.get("unit"))
            representative_unit_normalized = cell_text(representative.get("unit_normalized"))
            if not representative_unit and not representative_unit_normalized:
                continue
            family_ids = [cell_text(value) for value in option.get("family_ids", []) if cell_text(value)]
            if expanded_family_ids_by_position is not None:
                family_ids = list(expanded_family_ids_by_position.get(item.item_position, family_ids))
            price_option = {**option, "family_ids": family_ids}
            price_stats = price_stats_for_option(
                price_option, display_row, candidate_families, samples
            )
            validate_price_stats(price_stats, item.stable_sample_id, item.practice_option_id)
            amount_p10, amount_mid, amount_p90 = quantity_amounts(item.quantity, price_stats)
            quantity_value = quantity_display(item.quantity)
            scenario_unit = cell_text(representative.get("unit_normalized")) or cell_text(
                representative.get("unit")
            )
            price_evidence_count = int(numeric_or_none(price_stats.get("evidence_count")) or 0)
            quantity_reason = item.quantity_reason
            if item.quantity_source == "historical_median" and not item.quantity_fallback_used:
                quantity_reason = (
                    f"采用全库召回的{price_evidence_count}条同类历史样本"
                    f"工程量中位数{format_number_cell(quantity_value)}{scenario_unit}暂估。"
                )
            expanded_evidence = price_stats["expanded_evidence"]
            for _evidence_index, evidence in expanded_evidence.iterrows():
                price_evidence_rows.append({
                    "final_item_position": item.item_position,
                    "清单名称": cell_text(representative.get("representative_cost_item_name")),
                    "display_id": item.display_id,
                    "practice_option_id": item.practice_option_id,
                    "family_id": cell_text(evidence.get("family_id")),
                    "normalized_signature": cell_text(evidence.get("normalized_signature")),
                    "stable_sample_id": cell_text(evidence.get("stable_sample_id")),
                    "project_key": cell_text(evidence.get("project_key")),
                    "project_package_id": cell_text(evidence.get("project_package_id")),
                    "source_ref": cell_text(evidence.get("source_ref")),
                    "工程名称": cell_text(evidence.get("工程名称")) or cell_text(evidence.get("来源工程名称")) or cell_text(evidence.get("project_name_text")),
                    "location": cell_text(evidence.get("location")),
                    "consultation_time": cell_text(evidence.get("consultation_time")),
                    "cost_item_name": cell_text(evidence.get("cost_item_name")),
                    "project_description": cell_text(evidence.get("project_description")),
                    "unit": cell_text(evidence.get("unit")),
                    "quantity": evidence.get("quantity", ""),
                    "unit_price": evidence.get("unit_price", ""),
                    "labor_unit_price": evidence.get("labor_unit_price", ""),
                    "machinery_unit_price": evidence.get("machinery_unit_price", ""),
                })
            scenario_rows.append(
                {
                    "final_item_position": item.item_position,
                    "清单名称": cell_text(representative.get("representative_cost_item_name")),
                    "项目特征": cell_text(representative.get("representative_project_description")),
                    "单位": scenario_unit,
                    "工程量": quantity_value,
                    "工程量最低值": item.quantity_minimum,
                    "工程量中位数": item.quantity_median,
                    "工程量最高值": item.quantity_maximum,
                    "综合单价": price_stats.get("unit_price_median"),
                    "暂估合价": amount_mid,
                    "合价P10": amount_p10,
                    "合价中位数": amount_mid,
                    "合价P90": amount_p90,
                    "综合单价P10": price_stats.get("unit_price_p10"),
                    "综合单价中位数": price_stats.get("unit_price_median"),
                    "综合单价P90": price_stats.get("unit_price_p90"),
                    "其中包含人工费单价P10": price_stats.get("labor_unit_price_p10"),
                    "其中包含人工费单价中位数": price_stats.get("labor_unit_price_median"),
                    "其中包含人工费单价P90": price_stats.get("labor_unit_price_p90"),
                    "其中包含机械费单价P10": price_stats.get("machinery_unit_price_p10"),
                    "其中包含机械费单价中位数": price_stats.get("machinery_unit_price_median"),
                    "其中包含机械费单价P90": price_stats.get("machinery_unit_price_p90"),
                    "价格证据样本数": price_evidence_count,
                    "来源样本": cell_text(price_stats.get("source_refs")),
                    "practice_option_id": item.practice_option_id,
                    "original_option_id": item.original_option_id,
                    "original_family_id": item.original_family_id,
                    "representative_family_id": item.representative_family_id,
                    "价格证据family": ",".join(family_ids),
                    "display_id": item.display_id,
                }
            )
    scenario_columns = ["final_item_position", *ESTIMATE_SCENARIO_COLUMNS] if include_internal_positions else ESTIMATE_SCENARIO_COLUMNS
    return (
        pd.DataFrame(scenario_rows, columns=scenario_columns).fillna(""),
        pd.DataFrame(price_evidence_rows, columns=PRICE_EVIDENCE_ITEM_COLUMNS).fillna(""),
    )


def display_frame(frame: pd.DataFrame, display: bool) -> pd.DataFrame:
    output = frame.copy()
    _ = display  # 保留 CLI 参数兼容；显示精度由 Excel number_format 控制。
    for column in output.columns:
        if is_text_identifier_column(column):
            output[column] = output[column].map(cell_text)
    return output.rename(columns=EXCEL_DISPLAY_COLUMN_LABELS)


EXCEL_DISPLAY_COLUMN_LABELS = {
    "retrieval_package_support_ratio": "本次召回工程包支持比例",
    "retrieval_item_count": "本次召回清单行数",
    "retrieval_package_count": "本次召回工程包数",
    "support_rank": "本次召回支持度排序",
}


TEXT_IDENTIFIER_COLUMNS = {
    "scenario_id",
    "display_id",
    "family_id",
    "project_package_id",
    "project_key",
    "item_key",
    "stable_sample_id",
    "source_ref",
    "catalog_id",
    "batch_id",
    "source_row_id",
    "item_row_id",
    "project_code",
}

TEXT_VALUE_COLUMNS = {
    "方案说明",
    "主要施工内容",
    "待现场确认事项",
    "工程量来源",
    "工程量说明",
    "来源样本",
    "价格证据family",
}

INTEGER_COLUMNS = {
    "序号",
    "final_item_position",
    "rank",
    "selection_rank",
    "support_rank",
    "family_count",
    "本次召回family数量",
    "默认family本次召回样本数",
    "默认family本次召回工程包数",
    "本次召回样本数",
    "本次召回工程包数",
    "retrieval_item_count",
    "retrieval_package_count",
    "本次召回清单行数",
    "本次召回支持度排序",
    "item_count",
    "page_no",
    "prompt_chars",
    "estimated_tokens",
    "max_tokens",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
}

DECIMAL_VALUE_COLUMNS = {
    "quantity",
    "unit_price",
    "labor_unit_price",
    "machinery_unit_price",
}

AMOUNT_VALUE_COLUMNS = {
    "total_price",
    "其中包含人工费P10",
    "其中包含人工费中位数",
    "其中包含人工费P90",
    "其中包含机械费P10",
    "其中包含机械费中位数",
    "其中包含机械费P90",
}

AMOUNT_WIDTH_COLUMNS = {
    "total_price",
    "合价P10",
    "合价中位数",
    "合价P90",
    *AMOUNT_VALUE_COLUMNS,
}


def is_text_identifier_column(column: Any) -> bool:
    name = cell_text(column)
    return (
        name in TEXT_IDENTIFIER_COLUMNS
        or name.endswith("_id")
        or name.endswith("_ids")
    )


def excel_number_format(column: Any) -> str | None:
    name = cell_text(column)
    lower_name = name.lower()
    if is_text_identifier_column(name):
        return "@"
    if name in TEXT_VALUE_COLUMNS:
        return None
    if name == "package_evidence_weight":
        return "0.000000"
    if "similarity" in lower_name or "相似度" in name or lower_name.endswith("_ratio") or "比例" in name:
        return "0.0000"
    if (
        name in INTEGER_COLUMNS
        or lower_name.endswith("_count")
        or lower_name.endswith("_rank")
        or "样本数" in name
        or "工程包数" in name
        or ("数量" in name and "工程量" not in name)
    ):
        return "0"
    if name in AMOUNT_VALUE_COLUMNS or "金额" in name or "合价" in name:
        return "#,##0.00"
    if name in DECIMAL_VALUE_COLUMNS or "工程量" in name or "单价" in name:
        return "0.00"
    return None


def excel_min_column_width(column: Any) -> float | None:
    name = cell_text(column)
    if name in AMOUNT_WIDTH_COLUMNS:
        return 15.0
    if name in DECIMAL_VALUE_COLUMNS or "工程量" in name or "单价" in name:
        return 12.0
    return None




def amount_sum(frame: pd.DataFrame, column: str) -> float | None:
    values = numeric_values(frame, column)
    if values.empty:
        return None
    return round(float(values.sum()), 2)








def insufficient_evidence_scenario(scenario: EstimateScenario) -> EstimateScenario:
    return EstimateScenario(
        scenario.scenario_id,
        scenario.scenario_order,
        "历史价格证据不足",
        "当前检索结果中没有清单达到最低历史价格证据要求，暂不形成可展示的估价组合。",
        [],
        "建议补充具体维修对象、部位、规格、数量及现场检测信息后重新估价。",
    )


def short_description(value: Any, limit: int = 28) -> str:
    text = re.sub(r"\s+", " ", cell_text(value)).strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def display_item_label(row: pd.Series) -> str:
    name = cell_text(row.get("清单名称")) or cell_text(row.get("项目名称")) or cell_text(row.get("清单项名称"))
    description = short_description(row.get("项目特征"))
    if name and description:
        return f"{name}（{description}）"
    return name or description




def quantity_determination_status(quantity_trace: dict[str, Any]) -> str:
    if bool(quantity_trace.get("fallback")):
        return "fallback"
    if cell_text(quantity_trace.get("error_message")):
        return "failed"
    return "success"


def build_parse_info(
    rewrite: QueryRewrite,
    top_packages: int,
    top_items: int,
    max_packages_per_cache_subject: int,
    package_weight_temperature: float,
    evidence_package_universe_count: int,
    package_evidence_weight_count: int,
    package_evidence_weight_sum: float,
    meta: dict[str, Any],
    sample_count: int,
    package_count: int,
    project_packages_before_constraint: int,
    project_packages_after_constraint: int,
    samples_before_constraint: int,
    samples_after_constraint: int,
    invalid_sample_consultation_time_count: int,
    invalid_project_package_consultation_time_count: int,
    retrieved_evidence_item_row_count: int,
    evidence_item_row_count: int,
    candidate_family_count: int,
    candidate_display_group_count: int,
    matched_project_example_count: int,
    matched_project_example_item_count: int,
    display_option_grouping_display_count: int,
    display_option_grouping_trace: dict[str, Any],
    display_option_grouping_fallback: bool,
    display_option_grouping_error: str,
    display_option_grouping_meta: dict[str, Any],
    selected_project_package_id: str,
    selected_item_count: int,
    selected_exact_quantity_count: int,
    selected_range_quantity_count: int,
    range_selection_trace: dict[str, Any],
    range_selection_error: str,
    quantity_trace: dict[str, Any],
    final_explanation_trace: dict[str, Any],
    final_explanation_error: str,
    output_path: Path | None,
    started_at: datetime,
    index_dir: Path,
    include_debug_text: bool,
    display_option_grouping_prompt: str,
    range_selection_prompt: str,
    quantity_prompt: str,
    final_explanation_prompt: str,
    with_explanations: bool,
    warnings: list[str] | None = None,
) -> pd.DataFrame:
    rows = [
        ("原始用户需求", rewrite.raw_query),
        ("project_package_query_text", rewrite.project_package_query_text),
        ("item_query_text", rewrite.item_query_text),
        ("query_location", rewrite.location),
        ("query_start_date", rewrite.start_date),
        ("query_end_date", rewrite.end_date),
        ("query_constraint_notes", "；".join(rewrite.notes)),
        ("item_retrieval_text_fields", "cost_item_name + project_description + unit_normalized"),
        ("package_retrieval_text_fields", "工程名称 + project_name_text + cost_item_names_summary"),
        ("top_packages", top_packages),
        ("top_items", top_items),
        ("max_packages_per_cache_subject", max_packages_per_cache_subject),
        ("package_weight_temperature", package_weight_temperature),
        ("evidence_package_universe_count", evidence_package_universe_count),
        ("package_evidence_weight_count", package_evidence_weight_count),
        ("package_evidence_weight_sum", f"{package_evidence_weight_sum:.12f}"),
        ("embedding_model", meta.get("model", "")),
        ("sample_count", sample_count),
        ("package_count", package_count),
        ("project_packages_before_constraint", project_packages_before_constraint),
        ("project_packages_after_constraint", project_packages_after_constraint),
        ("samples_before_constraint", samples_before_constraint),
        ("samples_after_constraint", samples_after_constraint),
        ("invalid_sample_consultation_time_count", invalid_sample_consultation_time_count),
        ("invalid_project_package_consultation_time_count", invalid_project_package_consultation_time_count),
        ("LLM query rewrite 是否成功", "是" if rewrite.success else "否"),
        ("retrieved_evidence_item_row_count", retrieved_evidence_item_row_count),
        ("evidence_item_row_count", evidence_item_row_count),
        ("candidate_family_count", candidate_family_count),
        ("candidate_display_group_count", candidate_display_group_count),
        ("matched_project_example_count", matched_project_example_count),
        ("matched_project_example_item_count", matched_project_example_item_count),
        ("display_option_grouping_display_count", display_option_grouping_display_count),
        ("display_option_grouping_display_ids", json_text(display_option_grouping_meta.get("display_ids") or [])),
        ("display_option_grouping_llm_display_count", display_option_grouping_meta.get("llm_display_count", "")),
        ("display_option_grouping_programmatic_single_family_display_count", display_option_grouping_meta.get("programmatic_single_family_display_count", "")),
        ("display_option_grouping_practice_option_count", display_option_grouping_meta.get("practice_option_count", "")),
        ("display_option_grouping_families_grouped_count", display_option_grouping_meta.get("families_grouped_count", "")),
        ("display_option_grouping_option_count_by_display", json_text(display_option_grouping_meta.get("option_count_by_display") or {})),
        ("display_option_grouping_max_options_per_display", display_option_grouping_meta.get("max_options_per_display", "")),
        ("display_option_grouping_prompt_chars", display_option_grouping_trace.get("prompt_chars", "")),
        ("display_option_grouping_prompt_tokens", display_option_grouping_trace.get("prompt_tokens") or display_option_grouping_trace.get("estimated_tokens", "")),
        ("display_option_grouping_completion_tokens", display_option_grouping_trace.get("completion_tokens", "")),
        ("scenario_count", 1 if selected_item_count else 0),
        ("scenario_item_count", selected_item_count),
        ("scenario_exact_quantity_count", selected_exact_quantity_count),
        ("scenario_range_quantity_count", selected_range_quantity_count),
        ("selected_project_package_id", selected_project_package_id),
        ("range_selection_status", "fallback_full_project" if range_selection_error else "ok"),
        ("range_selection_prompt_chars", range_selection_trace.get("prompt_chars", "")),
        ("range_selection_prompt_tokens", range_selection_trace.get("prompt_tokens") or range_selection_trace.get("estimated_tokens", "")),
        ("range_selection_completion_tokens", range_selection_trace.get("completion_tokens", "")),
        (
            "quantity_determination_status",
            quantity_determination_status(quantity_trace),
        ),
        ("quantity_determination_prompt_chars", quantity_trace.get("prompt_chars", "")),
        ("quantity_determination_prompt_tokens", quantity_trace.get("prompt_tokens") or quantity_trace.get("estimated_tokens", "")),
        ("quantity_determination_completion_tokens", quantity_trace.get("completion_tokens", "")),
        ("with_explanations", with_explanations),
        (
            "final_explanation_status",
            "skipped" if not with_explanations
            else ("skipped_insufficient_evidence" if final_explanation_trace.get("parsed_status") == "skipped"
                  else ("fallback" if final_explanation_error else "success")),
        ),
        ("final_explanation_prompt_chars", final_explanation_trace.get("prompt_chars", "")),
        ("final_explanation_prompt_tokens", final_explanation_trace.get("prompt_tokens") or final_explanation_trace.get("estimated_tokens", "")),
        ("final_explanation_completion_tokens", final_explanation_trace.get("completion_tokens", "")),
        ("是否 display_option_grouping fallback", "是" if display_option_grouping_fallback else "否"),
        ("display_option_grouping LLM error", display_option_grouping_error),
        ("range_selection LLM error", range_selection_error),
        ("final_explanation LLM error", final_explanation_error),
        ("output_path", str(output_path or "")),
        ("运行时间", f"{(datetime.now() - started_at).total_seconds():.2f}s"),
        ("index_dir", str(index_dir)),
        ("主要文件路径", json_text((meta.get("files") or {}))),
        ("rewrite_notes", "；".join(rewrite.notes)),
        ("warnings", "；".join(warnings or [])),
    ]
    if include_debug_text:
        rows.append(("display_option_grouping_prompt_preview", display_option_grouping_prompt[:3000]))
        rows.append(("range_selection_prompt_preview", range_selection_prompt[:3000]))
        rows.append(("quantity_determination_prompt_preview", quantity_prompt[:3000]))
        rows.append(("final_explanation_prompt_preview", final_explanation_prompt[:3000]))
    return pd.DataFrame(rows, columns=["字段", "值"])


def write_query_result_workbook(output_path: Path, result: QueryResult, display: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        display_frame(result.estimate_summary, display).to_excel(writer, sheet_name="estimate_summary", index=False)
        display_frame(result.estimate_scenarios, display).to_excel(writer, sheet_name="estimate_scenarios", index=False)
        display_frame(result.option_evidence_expansion, display).to_excel(
            writer, sheet_name="option_evidence_expansion", index=False
        )
        display_frame(result.price_evidence_items, display).to_excel(
            writer, sheet_name="price_evidence_items", index=False
        )
        display_frame(result.candidate_display_groups, display).to_excel(
            writer,
            sheet_name="candidate_display_groups",
            index=False,
        )
        display_frame(result.candidate_families, display).to_excel(
            writer,
            sheet_name="candidate_families",
            index=False,
        )
        display_frame(result.display_option_grouping_trace, display).to_excel(
            writer,
            sheet_name="display_option_grouping_trace",
            index=False,
        )
        display_frame(result.option_selection_trace, display).to_excel(
            writer, sheet_name="option_selection_trace", index=False,
        )
        display_frame(result.matched_project_packages, display).to_excel(
            writer,
            sheet_name="matched_project_packages",
            index=False,
        )
        display_frame(result.matched_project_examples, display).to_excel(
            writer,
            sheet_name="matched_project_examples",
            index=False,
        )
        display_frame(result.package_evidence_weights, display).to_excel(
            writer,
            sheet_name="package_evidence_weights",
            index=False,
        )
        display_frame(result.evidence_items, display).to_excel(writer, sheet_name="evidence_items", index=False)
        result.parse_info.to_excel(writer, sheet_name="parse_info", index=False)
        result.llm_trace.to_excel(writer, sheet_name="llm_trace", index=False)
    apply_workbook_style(output_path)


def apply_workbook_style(path: Path) -> None:
    try:
        import openpyxl
        from openpyxl.styles import Alignment, Font
    except ImportError:
        return

    workbook = openpyxl.load_workbook(path)
    for worksheet in workbook.worksheets:
        worksheet.freeze_panes = None
        column_formats = {
            cell.column: excel_number_format(cell.value)
            for cell in worksheet[1]
        }
        for cell in worksheet[1]:
            min_width = excel_min_column_width(cell.value)
            if min_width is None:
                continue
            column_letter = cell.column_letter
            current_width = worksheet.column_dimensions[column_letter].width or 0
            worksheet.column_dimensions[column_letter].width = max(current_width, min_width)
        for cell in worksheet[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(wrap_text=False, vertical="top")
        for row in worksheet.iter_rows():
            for cell in row:
                cell.alignment = Alignment(wrap_text=False, vertical="top")
                number_format = column_formats.get(cell.column)
                if number_format:
                    cell.number_format = number_format
    workbook.save(path)
    workbook.close()


from estimator.candidates import display_groups as _candidate_display_groups
from estimator.candidates import families as _candidate_families
from estimator.candidates import practice_options as _practice_options
from estimator.candidates import signatures as _candidate_signatures
from estimator.planning import option_selection as _option_selection
from estimator.planning import package_selection as _package_selection
from estimator.planning import quantity_determination as _quantity_determination
from estimator.planning import range_selection as _range_selection
from estimator.planning import scenarios as _planning_scenarios
from estimator.pricing import estimate_calculator as _estimate_calculator
from estimator.pricing import evidence_expansion as _evidence_expansion
from estimator.pricing import price_statistics as _price_statistics
from estimator.pricing import quantity_statistics as _quantity_statistics
from estimator.pricing import summary as _pricing_summary


def _reexport_stage_two(module: Any) -> None:
    for name, value in vars(module).items():
        if name.startswith("_"):
            continue
        if name.isupper() or getattr(value, "__module__", None) == module.__name__:
            globals()[name] = value


for _stage_two_module in [
    _candidate_signatures, _candidate_families, _candidate_display_groups,
    _practice_options, _range_selection, _package_selection, _option_selection,
    _planning_scenarios, _evidence_expansion, _quantity_statistics,
    _price_statistics, _estimate_calculator, _pricing_summary,
    _quantity_determination,
]:
    _reexport_stage_two(_stage_two_module)


def generate_display_option_grouping(*args: Any, **kwargs: Any):
    return _practice_options.generate_display_option_grouping(
        *args, **kwargs, request_fn=request_llm_json_with_usage,
        trace_factory=trace_row, warning_fn=append_warning,
    )


def select_contiguous_item_range(*args: Any, **kwargs: Any):
    return _range_selection.select_contiguous_item_range(
        *args, **kwargs, request_fn=request_llm_json_with_usage,
    )


def select_final_options(*args: Any, **kwargs: Any):
    return _option_selection.select_final_options(
        *args, **kwargs, request_fn=request_llm_json_with_usage,
        trace_factory=trace_row, warning_fn=append_warning,
    )


def generate_quantity_determination(*args: Any, **kwargs: Any):
    return _quantity_determination.generate_quantity_determination(
        *args, **kwargs, request_fn=request_llm_json_with_usage,
        trace_factory=trace_row, warning_fn=append_warning,
    )


def expand_option_price_evidence_families(*args: Any, **kwargs: Any):
    return _evidence_expansion.expand_option_price_evidence_families(
        *args, **kwargs, request_fn=request_llm_json_with_usage,
        trace_factory=trace_row, warning_fn=append_warning,
    )


def run_query(
    index_dir: Path,
    raw_text: str,
    top_packages: int,
    top_items: int,
    output: Path | None,
    max_packages_per_cache_subject: int = 1,
    package_weight_temperature: float = DEFAULT_PACKAGE_WEIGHT_TEMPERATURE,
    include_debug_text: bool = False,
    display: bool = False,
    with_explanations: bool = False,
) -> QueryResult:
    if package_weight_temperature <= 0:
        raise ValueError("package weight temperature 必须大于 0")
    started_at = datetime.now()
    warnings: list[str] = []
    samples, project_packages, project_package_embeddings, item_embeddings, meta = load_index(index_dir)
    rewrite, rewrite_trace = query_rewrite_for_embedding(raw_text)

    package_mask = build_constraint_mask(
        project_packages, rewrite.location, rewrite.start_date, rewrite.end_date
    )
    item_mask = build_constraint_mask(
        samples, rewrite.location, rewrite.start_date, rewrite.end_date
    )
    candidate_project_packages, candidate_project_package_embeddings = filter_rows_and_embeddings(
        project_packages, project_package_embeddings, package_mask, "工程包"
    )
    candidate_samples, candidate_item_embeddings = filter_rows_and_embeddings(
        samples, item_embeddings, item_mask, "清单样本"
    )
    ensure_project_package_candidates(candidate_project_packages, rewrite)

    model = load_embedding_model(str(meta.get("model") or "BAAI/bge-m3"))
    try:
        package_query_embedding = encode_query(model, rewrite.project_package_query_text)
        item_query_embedding = encode_query(model, rewrite.item_query_text)
    except Exception:
        release_embedding_model(model)
        del model
        gc.collect()
        raise

    if package_query_embedding.shape[0] != candidate_project_package_embeddings.shape[1]:
        raise ValueError("package query embedding 维度与索引 embedding 维度不一致")
    if item_query_embedding.shape[0] != candidate_item_embeddings.shape[1]:
        raise ValueError("item query embedding 维度与索引 embedding 维度不一致")

    package_query_similarities = candidate_project_package_embeddings @ package_query_embedding
    package_query_similarity_by_id = project_package_similarity_map(candidate_project_packages, package_query_similarities)
    matched_raw = score_project_packages(
        candidate_project_packages,
        candidate_project_package_embeddings,
        package_query_embedding,
        top_packages,
        max_packages_per_cache_subject=max_packages_per_cache_subject,
    )
    candidate_item_query_similarities = candidate_item_embeddings @ item_query_embedding
    direct_item_hits = score_direct_items(candidate_samples, candidate_item_query_similarities, top_items)
    item_query_similarities = np.zeros(len(samples), dtype=np.float32)
    constrained_sample_indices = pd.to_numeric(candidate_samples["sample_index"], errors="raise").astype(int).to_numpy()
    item_query_similarities[constrained_sample_indices] = candidate_item_query_similarities
    if direct_item_hits.empty:
        append_warning(warnings, "direct_item_hits_empty_after_constraints")
    evidence_package_ids = evidence_package_universe(matched_raw, direct_item_hits)
    package_evidence_weights = build_package_evidence_weights(
        evidence_package_ids,
        package_query_similarity_by_id,
        package_weight_temperature,
    )
    retrieved_evidence_items = build_retrieved_evidence_items(
        samples,
        matched_raw,
        direct_item_hits,
        item_query_similarities,
        package_query_similarity_by_id=package_query_similarity_by_id,
        warnings=warnings,
    )
    candidate_families = build_candidate_families(retrieved_evidence_items)
    evidence_items = attach_family_ids_to_evidence_items(retrieved_evidence_items, candidate_families)
    candidate_display_groups, display_group_families = build_candidate_display_groups(candidate_families, evidence_items)
    candidate_display_groups = attach_display_support_ratios(
        candidate_display_groups,
        display_group_families,
        evidence_items,
        package_evidence_weights,
    )
    release_embedding_model(model)
    del model
    gc.collect()

    selected_package, ranked_packages = select_representative_project_package(matched_raw)
    matched_project_packages = matched_project_packages_for_output(ranked_packages)
    selected_project_package_id = cell_text(selected_package.get("project_package_id"))
    selected_project_name = cell_text(selected_package.get("工程名称")) or cell_text(
        selected_package.get("project_name_text")
    )
    matched_project_examples = build_matched_project_examples(
        matched_project_packages, samples, limit=5
    )
    matched_project_examples_output = matched_project_examples_frame(matched_project_examples)
    selected_items = attach_family_and_display_ids_to_selected_items(
        expand_selected_project_items(samples, selected_package),
        evidence_items,
        display_group_families,
    )
    start, end, range_meta = select_contiguous_item_range(
        raw_text, selected_project_name, selected_items
    )
    if range_meta["fallback"]:
        append_warning(warnings, "range_selection_fallback_full_project")
    plan_items = selected_items.iloc[start : end + 1].copy()
    required_display_ids, candidate_display_groups, display_group_families = filter_required_display_groups(
        plan_items, candidate_display_groups, display_group_families
    )
    (
        displays_with_options,
        _display_option_grouping_success,
        display_option_grouping_fallback,
        display_option_grouping_error,
        display_option_grouping_prompt,
        display_option_grouping_trace,
        display_option_grouping_meta,
        display_option_grouping_trace_frame,
        display_option_grouping_llm_traces,
    ) = generate_display_option_grouping(
        candidate_display_groups,
        display_group_families,
        candidate_families,
        warnings=warnings,
    )
    displays_with_options = attach_option_support_counts(displays_with_options, evidence_items)
    display_option_grouping_trace_frame = build_display_option_grouping_trace_frame(
        displays_with_options, display_group_families, candidate_families
    )
    plan_items = attach_original_practice_options(plan_items, displays_with_options)
    required_family_ids = set(display_group_families["family_id"].map(cell_text).tolist())
    lookup_evidence_items = evidence_items[
        evidence_items["family_id"].map(cell_text).isin(required_family_ids)
    ].copy()
    sample_lookup = build_stable_sample_lookup(
        samples, lookup_evidence_items, display_group_families, displays_with_options
    )
    range_selection_trace = _range_selection.build_range_selection_trace(
        selected_project_package_id, selected_items, plan_items, start, end, range_meta,
        trace_factory=trace_row,
    )
    plan_items, option_selection_trace_frame, option_selection_llm_traces = select_final_options(
        raw_text, plan_items, sample_lookup, displays_with_options, candidate_families, warnings, evidence_items
    )
    display_option_grouping_trace_frame = apply_option_selection_to_grouping_trace(
        display_option_grouping_trace_frame, option_selection_trace_frame
    )
    scenario, quantity_prompt, quantity_trace = generate_quantity_determination(
        raw_text,
        selected_project_package_id,
        plan_items,
        sample_lookup,
        displays_with_options,
        candidate_families,
        candidate_samples,
        warnings,
    )
    scenarios = [scenario]
    (
        expanded_family_ids_by_position,
        option_evidence_expansion,
        option_evidence_expansion_traces,
    ) = expand_option_price_evidence_families(
        scenario.items,
        displays_with_options,
        candidate_families,
        candidate_samples,
        warnings,
    )
    scenario_rows_with_positions, all_price_evidence_items = build_scenario_outputs(
        scenarios,
        displays_with_options,
        candidate_families,
        candidate_samples,
        expanded_family_ids_by_position,
        include_internal_positions=True,
    )
    scenario, estimate_scenarios, display_price_evidence_items = filter_customer_display_outputs(
        scenario,
        scenario_rows_with_positions,
        all_price_evidence_items,
    )
    (
        scenario,
        final_explanation_success,
        final_explanation_error,
        final_explanation_prompt,
        final_explanation_trace,
    ) = generate_customer_explanation(
        with_explanations,
        rewrite.raw_query,
        scenario,
        estimate_scenarios,
        warnings=warnings,
    )
    scenarios = [scenario]
    estimate_summary = build_estimate_summary(
        rewrite.raw_query,
        scenarios,
        estimate_scenarios,
        display_price_evidence_items,
    )
    if warnings:
        append_trace_warnings(display_option_grouping_trace, warnings)
        append_trace_warnings(range_selection_trace, warnings)
        append_trace_warnings(quantity_trace, warnings)
        append_trace_warnings(final_explanation_trace, warnings)
    scenario_item_count = sum(len(scenario.items) for scenario in scenarios)
    scenario_exact_quantity_count = sum(1 for scenario in scenarios for item in scenario.items if cell_text(item.quantity.get("type")) == "exact")
    scenario_range_quantity_count = 0
    parse_info = build_parse_info(
        rewrite=rewrite,
        top_packages=top_packages,
        top_items=top_items,
        max_packages_per_cache_subject=max_packages_per_cache_subject,
        package_weight_temperature=package_weight_temperature,
        evidence_package_universe_count=len(evidence_package_ids),
        package_evidence_weight_count=len(package_evidence_weights),
        package_evidence_weight_sum=float(package_evidence_weights.get("package_evidence_weight", pd.Series(dtype=float)).sum()),
        meta=meta,
        sample_count=len(samples),
        package_count=len(project_packages),
        project_packages_before_constraint=len(project_packages),
        project_packages_after_constraint=len(candidate_project_packages),
        samples_before_constraint=len(samples),
        samples_after_constraint=len(candidate_samples),
        invalid_sample_consultation_time_count=int(parse_consultation_dates(samples["consultation_time"]).isna().sum()),
        invalid_project_package_consultation_time_count=int(parse_consultation_dates(project_packages["consultation_time"]).isna().sum()),
        retrieved_evidence_item_row_count=len(retrieved_evidence_items),
        evidence_item_row_count=len(evidence_items),
        candidate_family_count=len(candidate_families),
        candidate_display_group_count=len(candidate_display_groups),
        matched_project_example_count=len(matched_project_examples),
        matched_project_example_item_count=len(matched_project_examples_output),
        display_option_grouping_display_count=len(displays_with_options),
        display_option_grouping_trace=display_option_grouping_trace,
        display_option_grouping_fallback=display_option_grouping_fallback,
        display_option_grouping_error=display_option_grouping_error,
        display_option_grouping_meta=display_option_grouping_meta,
        selected_project_package_id=selected_project_package_id,
        selected_item_count=scenario_item_count,
        selected_exact_quantity_count=scenario_exact_quantity_count,
        selected_range_quantity_count=scenario_range_quantity_count,
        range_selection_trace=range_selection_trace,
        range_selection_error=range_meta["error_message"],
        quantity_trace=quantity_trace,
        final_explanation_trace=final_explanation_trace,
        final_explanation_error=final_explanation_error,
        output_path=output,
        started_at=started_at,
        index_dir=index_dir,
        include_debug_text=include_debug_text,
        display_option_grouping_prompt=display_option_grouping_prompt,
        range_selection_prompt=range_meta["prompt"],
        quantity_prompt=quantity_prompt,
        final_explanation_prompt=final_explanation_prompt,
        with_explanations=with_explanations,
        warnings=warnings,
    )
    llm_trace = pd.DataFrame(
        [
            rewrite_trace,
            *display_option_grouping_llm_traces,
            range_selection_trace,
            *option_selection_llm_traces,
            quantity_trace,
            *option_evidence_expansion_traces,
            final_explanation_trace,
        ],
        columns=LLM_TRACE_COLUMNS,
    )

    result = QueryResult(
        rewrite=rewrite,
        estimate_summary=estimate_summary,
        estimate_scenarios=estimate_scenarios,
        matched_project_packages=matched_project_packages,
        candidate_families=candidate_families,
        candidate_display_groups=candidate_display_groups,
        package_evidence_weights=package_evidence_weights,
        display_group_families=display_group_families,
        display_option_grouping_trace=display_option_grouping_trace_frame,
        option_selection_trace=option_selection_trace_frame,
        matched_project_examples=matched_project_examples_output,
        evidence_items=evidence_items,
        option_evidence_expansion=option_evidence_expansion,
        price_evidence_items=all_price_evidence_items,
        parse_info=parse_info,
        llm_trace=llm_trace,
        success=True,
        error_message=final_explanation_error,
    )
    if output:
        write_query_result_workbook(output, result, display=display)
    return result


def print_terminal_summary(result: QueryResult, output_path: Path | None) -> None:
    print(f"[DONE] package query: {result.rewrite.project_package_query_text}")
    print(f"[DONE] item query: {result.rewrite.item_query_text}")
    print(f"[DONE] matched project packages: {len(result.matched_project_packages)}")
    print(f"[DONE] candidate families: {len(result.candidate_families)}")
    print(f"[DONE] candidate display groups: {len(result.candidate_display_groups)}")
    print(f"[DONE] matched project example items: {len(result.matched_project_examples)}")
    print(f"[DONE] evidence items: {len(result.evidence_items)}")
    print(f"[DONE] estimate scenarios: {len(result.estimate_scenarios)}")
    if result.rewrite.notes:
        print(f"rewrite notes: {'；'.join(result.rewrite.notes)}")
    if output_path:
        print(f"输出文件: {output_path}")


def main() -> int:
    args = parse_args()
    index_dir = resolve_path(args.index_dir)
    output_path = resolve_path(args.output) if args.output is not None else default_query_output_path()

    try:
        validate_output_path(output_path, args.overwrite)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 1

    try:
        check_lmstudio_service(timeout_seconds=args.llm_check_timeout)
    except RuntimeError as exc:
        detail = str(exc)
        marker = "LMSTUDIO_BASE_URL="
        suffix = detail[detail.find(marker) :] if marker in detail else detail
        print(f"[ERROR] LLM 服务不可用，请先启动 LM Studio Server，并检查 {suffix}")
        return 1

    try:
        result = run_query(
            index_dir=index_dir,
            raw_text=args.text,
            top_packages=args.top_packages,
            top_items=args.top_items,
            output=output_path,
            max_packages_per_cache_subject=args.max_packages_per_cache_subject,
            package_weight_temperature=args.package_weight_temperature,
            include_debug_text=args.include_debug_text,
            display=args.display,
            with_explanations=args.with_explanations,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    print_terminal_summary(result, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
