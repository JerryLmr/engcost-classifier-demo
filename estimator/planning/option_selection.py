from __future__ import annotations

import sys
from typing import Any

import pandas as pd

from estimator.paths import CLASSIFIER_BACKEND_DIR
from estimator.candidates.signatures import cell_text, join_non_empty, numeric_or_none, normalized_unit, truncate_text, json_text, append_warning, trace_row
if str(CLASSIFIER_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSIFIER_BACKEND_DIR))
from classifier.llm_client import request_llm_json_with_usage  # noqa: E402
from estimator.candidates.practice_options import (
    DISPLAY_OPTION_GROUPING_TRACE_COLUMNS,
    attach_option_support_counts,
    display_option_maps,
)

OPTION_SELECTION_TRACE_COLUMNS = [
    "final_item_position", "display_id", "original_option_id", "selected_option_id",
    "original_family_id", "representative_family_id", "candidate_option_count",
    "whether_replaced", "selection_reason", "option_selection_decision",
    "representative_selection_reason",
]

def attach_family_and_display_ids_to_selected_items(
    selected_items: pd.DataFrame,
    evidence_items: pd.DataFrame,
    display_group_families: pd.DataFrame,
) -> pd.DataFrame:
    selected = selected_items.copy()
    selected_ids = selected.get("stable_sample_id", pd.Series(index=selected.index, dtype=object)).map(cell_text)
    if selected_ids.eq("").any():
        raise ValueError("selected_items 中 stable_sample_id 不得为空")
    duplicate_selected_ids = selected_ids[selected_ids.duplicated(keep=False)]
    if not duplicate_selected_ids.empty:
        raise ValueError(
            f"selected_items 中 stable_sample_id 必须唯一: "
            f"{join_non_empty(duplicate_selected_ids.tolist(), limit=10)}"
        )

    evidence_mapping = evidence_items[["stable_sample_id", "family_id"]].copy()
    evidence_mapping["stable_sample_id"] = evidence_mapping["stable_sample_id"].map(cell_text)
    evidence_mapping["family_id"] = evidence_mapping["family_id"].map(cell_text)
    if evidence_mapping["stable_sample_id"].eq("").any() or evidence_mapping["family_id"].eq("").any():
        raise ValueError("evidence_items 中 stable_sample_id/family_id 不得为空")
    duplicate_evidence_ids = evidence_mapping.loc[
        evidence_mapping["stable_sample_id"].duplicated(keep=False), "stable_sample_id"
    ]
    if not duplicate_evidence_ids.empty:
        raise ValueError(
            f"evidence_items 中 stable_sample_id 必须唯一: "
            f"{join_non_empty(duplicate_evidence_ids.tolist(), limit=10)}"
        )

    display_mapping = display_group_families[["family_id", "display_id"]].copy()
    display_mapping["family_id"] = display_mapping["family_id"].map(cell_text)
    display_mapping["display_id"] = display_mapping["display_id"].map(cell_text)
    if display_mapping["family_id"].eq("").any() or display_mapping["display_id"].eq("").any():
        raise ValueError("display_group_families 中 family_id/display_id 不得为空")
    duplicate_family_ids = display_mapping.loc[
        display_mapping["family_id"].duplicated(keep=False), "family_id"
    ]
    if not duplicate_family_ids.empty:
        raise ValueError(
            f"family_id 必须唯一映射到一个 display_id: "
            f"{join_non_empty(duplicate_family_ids.tolist(), limit=10)}"
        )

    selected["stable_sample_id"] = selected_ids
    original_count = len(selected)
    selected = selected.merge(evidence_mapping, on="stable_sample_id", how="left", validate="one_to_one")
    selected = selected.merge(display_mapping, on="family_id", how="left", validate="many_to_one")
    if len(selected) != original_count:
        raise ValueError("selected_items 合并 family/display 后行数发生变化")
    missing_family = selected["family_id"].map(cell_text).eq("")
    missing_display = selected["display_id"].map(cell_text).eq("")
    if missing_family.any() or missing_display.any():
        missing_ids = selected.loc[missing_family | missing_display, "stable_sample_id"].tolist()
        raise ValueError(
            f"selected_items 每行必须唯一映射到 family_id 和 display_id: "
            f"{join_non_empty(missing_ids, limit=10)}"
        )
    return selected

def attach_original_practice_options(
    plan_items: pd.DataFrame,
    displays_with_options: pd.DataFrame,
) -> pd.DataFrame:
    family_to_option: dict[str, str] = {}
    for _index, display in displays_with_options.iterrows():
        options = display.get("practice_options") if isinstance(display.get("practice_options"), list) else []
        for option in options:
            if not isinstance(option, dict):
                continue
            option_id = cell_text(option.get("practice_option_id"))
            if not option_id:
                raise ValueError("practice_option_id 不得为空")
            for family_value in option.get("family_ids") or []:
                family_id = cell_text(family_value)
                if not family_id:
                    raise ValueError("practice option 中 family_id 不得为空")
                if family_id in family_to_option:
                    raise ValueError(f"family_id 必须唯一映射到一个 practice_option_id: {family_id}")
                family_to_option[family_id] = option_id

    output = plan_items.copy()
    output["practice_option_id"] = output["family_id"].map(
        lambda value: family_to_option.get(cell_text(value), "")
    )
    missing = output["practice_option_id"].map(cell_text).eq("")
    if missing.any():
        raise ValueError(
            f"plan_items family_id 无法映射到 practice_option_id: "
            f"{join_non_empty(output.loc[missing, 'family_id'].map(cell_text).tolist(), limit=10)}"
        )
    return output

def apply_option_selection_to_grouping_trace(
    trace: pd.DataFrame,
    option_selection_trace: pd.DataFrame,
) -> pd.DataFrame:
    output = trace.copy()
    if output.empty or option_selection_trace.empty:
        return output
    selection_by_display = {
        cell_text(row.get("display_id")): row
        for _index, row in option_selection_trace.iterrows()
    }
    for index, row in output.iterrows():
        selection = selection_by_display.get(cell_text(row.get("display_id")))
        if selection is None:
            continue
        option_id = cell_text(row.get("practice_option_id"))
        family_id = cell_text(row.get("family_id"))
        output.at[index, "is_original_option"] = option_id == cell_text(selection.get("original_option_id"))
        output.at[index, "is_selected_option"] = option_id == cell_text(selection.get("selected_option_id"))
        output.at[index, "option_selection_decision"] = cell_text(selection.get("option_selection_decision"))
        output.at[index, "option_selection_reason"] = cell_text(selection.get("selection_reason"))
        output.at[index, "is_original_family"] = family_id == cell_text(selection.get("original_family_id"))
        output.at[index, "is_representative_family"] = family_id == cell_text(selection.get("representative_family_id"))
        output.at[index, "representative_selection_reason"] = cell_text(selection.get("representative_selection_reason"))
    return output[DISPLAY_OPTION_GROUPING_TRACE_COLUMNS]

def family_payload(family_id: str, family_map: dict[str, pd.Series]) -> dict[str, Any]:
    row = family_map.get(family_id, pd.Series(dtype=object))
    return {
        "family_id": family_id,
        "cost_item_name": cell_text(row.get("representative_cost_item_name")),
        "project_description": cell_text(row.get("representative_project_description")),
        "unit": cell_text(row.get("unit_normalized")) or cell_text(row.get("unit")),
        "normalized_signature": cell_text(row.get("normalized_signature")),
    }

def option_selection_family_payload(family_id: str, family_map: dict[str, pd.Series]) -> dict[str, Any]:
    payload = family_payload(family_id, family_map)
    return {
        "cost_item_name": payload["cost_item_name"],
        "project_description": payload["project_description"],
        "unit": payload["unit"],
    }

def choose_representative_family(
    option: dict[str, Any], original_family_id: str, family_map: dict[str, pd.Series]
) -> tuple[str, str]:
    family_ids = [cell_text(value) for value in option.get("family_ids", []) if cell_text(value)]
    ranked = sorted(
        family_ids,
        key=lambda family_id: (
            -int(numeric_or_none(family_map.get(family_id, pd.Series(dtype=object)).get("本次召回样本数")) or 0),
            -int(numeric_or_none(family_map.get(family_id, pd.Series(dtype=object)).get("本次召回工程包数")) or 0),
            0 if family_id == original_family_id else 1,
            family_id,
        ),
    )
    if not ranked:
        raise ValueError("selected option 不得缺少 family")
    selected = ranked[0]
    rows = {family_id: family_map.get(family_id, pd.Series(dtype=object)) for family_id in family_ids}
    max_samples = max(int(numeric_or_none(row.get("本次召回样本数")) or 0) for row in rows.values())
    sample_tied = [fid for fid, row in rows.items() if int(numeric_or_none(row.get("本次召回样本数")) or 0) == max_samples]
    if len(sample_tied) == 1:
        reason = "selected_option_family_sample_count_max"
    else:
        max_packages = max(int(numeric_or_none(rows[fid].get("本次召回工程包数")) or 0) for fid in sample_tied)
        package_tied = [fid for fid in sample_tied if int(numeric_or_none(rows[fid].get("本次召回工程包数")) or 0) == max_packages]
        if len(package_tied) == 1:
            reason = "selected_option_family_package_count_max"
        elif original_family_id in package_tied:
            reason = "support_tie_original_family"
        else:
            reason = "stable_family_id_tiebreak"
    return selected, reason

def choose_most_supported_option(
    options: list[dict[str, Any]], original_option_id: str,
) -> tuple[str, str]:
    if not options:
        raise ValueError("Display 不得缺少 Option")
    ranked = sorted(options, key=lambda option: (
        -int(option.get("option_sample_count") or 0),
        -int(option.get("option_package_count") or 0),
        0 if cell_text(option.get("practice_option_id")) == original_option_id else 1,
        cell_text(option.get("practice_option_id")),
    ))
    selected = ranked[0]
    max_samples = int(selected.get("option_sample_count") or 0)
    sample_tied = [option for option in options if int(option.get("option_sample_count") or 0) == max_samples]
    if len(sample_tied) == 1:
        reason = "no_explicit_match_selected_by_sample_count"
    else:
        max_packages = int(selected.get("option_package_count") or 0)
        package_tied = [option for option in sample_tied if int(option.get("option_package_count") or 0) == max_packages]
        if len(package_tied) == 1:
            reason = "no_explicit_match_selected_by_package_count"
        elif any(cell_text(option.get("practice_option_id")) == original_option_id for option in package_tied):
            reason = "support_tie_original_option"
        else:
            reason = "stable_option_id_tiebreak"
    return cell_text(selected.get("practice_option_id")), reason

def select_final_options(
    raw_text: str,
    plan_items: pd.DataFrame,
    sample_lookup: dict[str, dict[str, Any]],
    displays_with_options: pd.DataFrame,
    candidate_families: pd.DataFrame,
    warnings: list[str] | None = None,
    evidence_items: pd.DataFrame | None = None,
    *, request_fn=request_llm_json_with_usage, trace_factory=trace_row,
    warning_fn=append_warning,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    request_llm_json_with_usage = request_fn
    trace_row = trace_factory
    append_warning = warning_fn
    displays_with_options = attach_option_support_counts(
        displays_with_options, evidence_items if evidence_items is not None else pd.DataFrame()
    )
    display_map, option_map = display_option_maps(displays_with_options)
    family_map = {cell_text(row.get("family_id")): row for _index, row in candidate_families.iterrows()}
    selected_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    llm_traces: list[dict[str, Any]] = []
    for _index, item in plan_items.iterrows():
        position = int(item["item_position"])
        sample = sample_lookup.get(cell_text(item.get("stable_sample_id")))
        if sample is None:
            raise ValueError(f"最终清单无法回查证据: item_position={position}")
        display_id = cell_text(sample.get("display_id"))
        original_family_id = cell_text(sample.get("family_id"))
        original_option_id = cell_text(sample.get("practice_option_id"))
        display = display_map.get(display_id)
        options = display.get("practice_options") if display is not None and isinstance(display.get("practice_options"), list) else []
        selected_option_id = original_option_id
        decision = "no_explicit_match"
        selection_reason = "single_option"
        prompt = ""
        response = None
        error_message = ""
        if len(options) > 1:
            payload = {
                "raw_query": raw_text,
                "display_id": display_id,
                "original_option_id": original_option_id,
                "options": [{
                    "option_id": cell_text(option.get("practice_option_id")),
                    "families": [
                        option_selection_family_payload(cell_text(fid), family_map)
                        for fid in option.get("family_ids", [])
                    ],
                } for option in options],
            }
            prompt = f"""
你只负责判断用户是否明确提出了足以区分当前 Options 的条件。

raw_query 是用户原始需求。
original_option_id 是参考历史工程中的原 Option。
options 是当前同一 Display 下可选的所有 Option。

判断规则：

1. 只有当 raw_query 明确提到能够区分 Options 的材料、工艺、规格、厚度、楼层、层数、部位或其他限定条件时，才返回 explicit_match。

2. 用户只是提到当前清单对象本身，不算明确区分条件。
   例如用户只说“需要垂直运输”“需要脚手架”，而不同 Options 的差异在楼层或具体部位时，不得据此选择某个带限定的 Option。

3. 用户没有提到楼层、层数或高度时，不得根据参考历史工程中的楼层限定推断用户需要该限定。
   例如：
   - “1层楼垂直运输费”
   - “垂直运输费”
   用户未说明楼层时，返回 no_explicit_match。

4. 用户没有提到厚度、材料、规格、部位或施工方法时，不得根据 original_option_id、original_family 或历史工程内容补充这些条件。

5. original_option_id 只是历史参考，不是用户要求。

6. 不得根据样本数、价格、相似度、Option 顺序、Option 编号、Family 数量或历史出现频率判断 explicit_match。

7. 只有当用户明确条件能够唯一对应某个现有 Option 时，才能返回 explicit_match。

8. 如果用户表达模糊、没有明确区分条件，或者多个 Options 都可能符合，必须返回 no_explicit_match。

9. selected_option_id 只能是 options 中已有的 option_id。

10. 不得新增 Option，不得修改 Option，不得解释，不得输出思考过程。

输出格式只能是以下两种之一：

{{"decision":"explicit_match","selected_option_id":"..."}}

或：

{{"decision":"no_explicit_match","selected_option_id":""}}

输入：
{json_text(payload)}
""".strip()
            try:
                response = request_llm_json_with_usage(
                    prompt, max_tokens=128,
                    system_prompt="只输出 decision 和 selected_option_id 两字段 JSON object。",
                )
                result = response.content
                allowed_ids = {cell_text(option.get("practice_option_id")) for option in options}
                if not isinstance(result, dict) or set(result) != {"decision", "selected_option_id"}:
                    raise ValueError("option selection 只允许 decision 和 selected_option_id")
                decision = cell_text(result.get("decision"))
                candidate = cell_text(result.get("selected_option_id"))
                if decision == "explicit_match":
                    if not candidate or candidate not in allowed_ids:
                        raise ValueError("explicit_match 必须返回现有非空 Option ID")
                    selected_option_id = candidate
                    selection_reason = "user_explicit_match"
                elif decision == "no_explicit_match":
                    if candidate:
                        raise ValueError("no_explicit_match 的 selected_option_id 必须为空")
                    selected_option_id, selection_reason = choose_most_supported_option(options, original_option_id)
                else:
                    raise ValueError("option selection decision 非法")
            except (RuntimeError, ValueError) as exc:
                error_message = str(exc)
                decision = "no_explicit_match"
                selected_option_id, _support_reason = choose_most_supported_option(options, original_option_id)
                selection_reason = "llm_failed_selected_by_support"
                append_warning(warnings, f"option_selection_llm_failed_selected_by_support:{position}")
            llm_traces.append(trace_row(
                "option_selection", f"为最终清单 {position} 选择现有 Option", not error_message,
                error=error_message, prompt=prompt, max_tokens=128,
                usage=response.usage if response is not None else None,
                raw_response=getattr(response, "raw_content", "") if response is not None else "",
            ))
        selected_option = option_map.get((display_id, selected_option_id))
        if selected_option is None:
            raise ValueError(f"最终 Option 回查失败: {display_id}/{selected_option_id}")
        representative_family_id, representative_selection_reason = choose_representative_family(selected_option, original_family_id, family_map)
        representative = family_payload(representative_family_id, family_map)
        selected_rows.append({
            **item.to_dict(), "display_id": display_id,
            "original_option_id": original_option_id, "selected_option_id": selected_option_id,
            "original_family_id": original_family_id, "representative_family_id": representative_family_id,
            "cost_item_name": representative["cost_item_name"],
            "project_description": representative["project_description"], "unit": representative["unit"],
            "normalized_signature": representative["normalized_signature"],
        })
        trace_rows.append({
            "final_item_position": position, "display_id": display_id,
            "original_option_id": original_option_id, "selected_option_id": selected_option_id,
            "original_family_id": original_family_id, "representative_family_id": representative_family_id,
            "candidate_option_count": len(options), "whether_replaced": selected_option_id != original_option_id,
            "selection_reason": selection_reason,
            "option_selection_decision": decision,
            "representative_selection_reason": representative_selection_reason,
        })
    return pd.DataFrame(selected_rows), pd.DataFrame(trace_rows, columns=OPTION_SELECTION_TRACE_COLUMNS), llm_traces

def build_stable_sample_lookup(
    samples: pd.DataFrame,
    evidence_items: pd.DataFrame,
    display_group_families: pd.DataFrame,
    displays_with_options: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    stable_ids = samples.get("stable_sample_id", pd.Series(dtype=object)).map(cell_text)
    if len(stable_ids) != len(samples) or stable_ids.eq("").any():
        raise ValueError("samples 中 stable_sample_id 不得为空")
    duplicate_ids = stable_ids[stable_ids.duplicated(keep=False)]
    if not duplicate_ids.empty:
        raise ValueError(f"stable_sample_id 必须全局唯一: {join_non_empty(duplicate_ids.tolist(), limit=10)}")

    family_to_display: dict[str, str] = {}
    for _index, row in display_group_families.iterrows():
        family_id = cell_text(row.get("family_id"))
        display_id = cell_text(row.get("display_id"))
        if not family_id or not display_id:
            raise ValueError("family/display 映射不得为空")
        if family_id in family_to_display and family_to_display[family_id] != display_id:
            raise ValueError(f"family_id 映射到多个 display: {family_id}")
        family_to_display[family_id] = display_id

    family_to_option: dict[str, str] = {}
    for _index, display in displays_with_options.iterrows():
        options = display.get("practice_options") if isinstance(display.get("practice_options"), list) else []
        for option in options:
            if not isinstance(option, dict):
                continue
            option_id = cell_text(option.get("practice_option_id"))
            for family_value in option.get("family_ids") or []:
                family_id = cell_text(family_value)
                if family_id in family_to_option and family_to_option[family_id] != option_id:
                    raise ValueError(f"family_id 映射到多个 practice option: {family_id}")
                family_to_option[family_id] = option_id

    lookup: dict[str, dict[str, Any]] = {}
    for _index, row in evidence_items.iterrows():
        stable_sample_id = cell_text(row.get("stable_sample_id"))
        if not stable_sample_id:
            raise ValueError("evidence item 缺少 stable_sample_id")
        if stable_sample_id in lookup:
            raise ValueError(f"evidence pool 中 stable_sample_id 重复: {stable_sample_id}")
        family_id = cell_text(row.get("family_id"))
        display_id = family_to_display.get(family_id, "")
        practice_option_id = family_to_option.get(family_id, "")
        if not display_id or not practice_option_id:
            raise ValueError(f"stable_sample_id 无法映射到 display/option: {stable_sample_id}")
        lookup[stable_sample_id] = {
            **row.to_dict(),
            "stable_sample_id": stable_sample_id,
            "source_ref": cell_text(row.get("source_ref")),
            "family_id": family_id,
            "display_id": display_id,
            "practice_option_id": practice_option_id,
            "project_package_id": cell_text(row.get("project_package_id")),
        }
    return lookup
