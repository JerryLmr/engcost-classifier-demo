from __future__ import annotations

import sys
from typing import Any

import pandas as pd

from estimator.paths import CLASSIFIER_BACKEND_DIR
from estimator.candidates.signatures import cell_text, join_non_empty, numeric_or_none, normalized_unit, truncate_text, json_text, append_warning, trace_row
if str(CLASSIFIER_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSIFIER_BACKEND_DIR))
from classifier.llm_client import request_llm_json_with_usage  # noqa: E402
from estimator.candidates.display_groups import normalize_display_description

DISPLAY_OPTION_GROUPING_TRACE_COLUMNS = [
    "display_id",
    "display_name",
    "practice_option_id",
    "family_id",
    "representative_cost_item_name",
    "representative_project_description",
    "unit",
    "本次召回样本数",
    "本次召回工程包数",
    "item_query_similarity最大值",
    "unit_price_min",
    "unit_price_median",
    "unit_price_max",
    "option_sample_count",
    "option_package_count",
    "is_original_option",
    "is_selected_option",
    "option_selection_decision",
    "option_selection_reason",
    "is_original_family",
    "is_representative_family",
    "representative_selection_reason",
]

OPTION_GROUPING_TAG_FIELDS = ("thickness", "material", "level")

def option_grouping_payload_for_display(
    display_id: str,
    display_group_families: pd.DataFrame,
    candidate_families: pd.DataFrame,
) -> list[dict[str, Any]]:
    family_map = {cell_text(row.get("family_id")): row for _index, row in candidate_families.iterrows()}
    rows = display_group_families[display_group_families["display_id"].map(cell_text).eq(display_id)].copy()
    rows = rows.sort_values(["item_query_similarity最大值", "本次召回样本数", "本次召回工程包数"], ascending=[False, False, False])
    payload: list[dict[str, Any]] = []
    for _index, row in rows.iterrows():
        family_id = cell_text(row.get("family_id"))
        family = family_map.get(family_id)
        if family is None:
            continue
        payload.append(
            {
                "family_id": family_id,
                "name": truncate_text(row.get("representative_cost_item_name"), 40),
                "spec": truncate_text(normalize_display_description(row.get("representative_project_description")), 120),
                "unit": cell_text(family.get("unit_normalized")) or cell_text(row.get("unit")) or cell_text(family.get("unit")),
            }
        )
    return payload

def build_display_option_grouping_prompt(
    candidate_display_groups: pd.DataFrame,
    display_group_families: pd.DataFrame,
    candidate_families: pd.DataFrame,
) -> tuple[str, dict[str, Any]]:
    if len(candidate_display_groups) != 1:
        raise ValueError("display_option_grouping prompt 每次必须且只能包含一个 Display")
    display = candidate_display_groups.iloc[0]
    display_id = cell_text(display.get("display_id"))
    record = {
        "display_name": truncate_text(display.get("display_name"), 40),
        "candidate_families": option_grouping_payload_for_display(
            display_id, display_group_families, candidate_families
        ),
    }
    payload = {"candidate_display": record}
    prompt = f"""
你的任务是：先逐个提取当前唯一 Display 下每个 Family 的关键标签，再将 Family 划分为若干 Option。

Option 的定义是：

同一 Option 内的所有 Family，后续必须能够作为一个整体被选中，并作为一个整体展开回全库查询价格证据。

如果用户明确某个参数后，需要保留同组中的一部分 Family、排除另一部分 Family，那么这些 Family 就不能属于同一个 Option。

第一步必须为每个 Family 提取：

- family_id：必须来自输入，不得遗漏、重复或新增；
- thickness：只填写项目特征中明确出现的厚度或核心尺寸规格，没有时为 ""；3mm、3.0mm、3厚统一为 3mm，其他如 1.2mm、1.5mm、2mm、4mm；
- material：只填写明确材料类别并标准化同义表达，没有时为 ""；
- level：只填写明确楼层、层数或高度条件，例如 1层、2层、5层、高度20m以内，没有时为 ""。

material 至少区分：聚氨酯、聚合物水泥基、水泥基、水泥基渗透结晶、JS、沥青防水涂料、非固化防水涂料、SBS改性沥青、自粘卷材、高分子卷材。
“单组分”和“单组份”不影响材料类别；“SBS改性沥青”和“弹性体改性沥青”统一为 SBS改性沥青。
自粘卷材不得与普通 SBS改性沥青统一；聚氨酯不得与聚合物水泥基、水泥基、JS统一；非固化防水涂料不得与普通沥青防水涂料统一。
不要把材料、层数或施工方法写入 thickness；不要把厚度、层数、基层清理或垃圾清运写入 material。
不要增加 is_generic 或其他标签。

第二步再根据上述标签输出 groups。groups 是二维数组：
- 每个内层数组代表一个独立 Option；
- 同一内层数组中的 Family 会在后续被整体选择、整体展开；
- 不同内层数组表示后续可以分别选择的不同候选做法。

不得只按照 Display 名称、共同用途、共同关键词或文本相似度粗略聚合。
相同厚度不代表材料相同，相同材料也不代表厚度相同。
不同明确 level 不得合并，明确 level 与空 level 也不得合并。

例如：

- 3mm 与 4mm 必须拆分；
- 2mm 水泥基渗透结晶与 2mm 聚氨酯必须拆分；
- 3mm SBS改性沥青与 3mm 聚氨酯必须拆分；
- 3mm 自粘卷材与 3mm SBS改性沥青卷材必须拆分；
- 1层、2层、5层、未注明层数必须分别拆分；
- 1.5mm单组份聚氨酯与1.5mm厚单组分聚氨酯可以合并；
- 1.2mm聚合物水泥基，含基层清理，与1.2mm聚合物水泥基可以合并。

以下差异当前可以忽略，不必单独拆分：

- OCR、标点、空格、换行、编号；
- 普通文字详略；
- 基层清理；
- 垃圾清运；
- 普通修补；
- 一般性的附带施工描述；
- 不影响用户后续选择的文字详略差异。

每个输入 family_id 必须且只能出现一次，不得遗漏、重复或新增。

每个输入 family_id 必须且只能在 families 中出现一次，也必须且只能在 groups 中出现一次。

输出格式（顶层只允许 families 和 groups）：

{{
  "families": [
    {{"family_id": "F004", "thickness": "3mm", "material": "SBS改性沥青", "level": ""}},
    {{"family_id": "F007", "thickness": "4mm", "material": "SBS改性沥青", "level": ""}}
  ],
  "groups": [
    ["F004"],
    ["F007"]
  ]
}}

每个内层数组就是一个 Option。

只输出 JSON，不输出解释或 Markdown。

【输入数据】
{json_text(payload)}
""".strip()
    return prompt, record

def display_family_unit(row: pd.Series) -> str:
    return cell_text(row.get("unit_normalized")) or normalized_unit(row.get("unit"))

def empty_display_option_grouping_meta() -> dict[str, Any]:
    return {
        "display_ids": [],
        "practice_option_count": 0,
        "families_grouped_count": 0,
        "option_count_by_display": {},
        "max_options_per_display": 0,
        "llm_display_count": 0,
        "programmatic_single_family_display_count": 0,
    }

def validate_option_grouping_family_tags(raw_families: Any, allowed_family_ids: set[str]) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    if not isinstance(raw_families, list):
        raise ValueError("families 必须为 list")
    parsed: list[dict[str, str]] = []
    tag_map: dict[str, dict[str, str]] = {}
    for item in raw_families:
        if not isinstance(item, dict) or set(item) != {"family_id", *OPTION_GROUPING_TAG_FIELDS}:
            raise ValueError("families 中每项必须且只能包含 family_id、thickness、material、level")
        family_id = cell_text(item.get("family_id"))
        if not family_id or family_id in tag_map:
            raise ValueError(f"families 中 family_id 为空或重复: {family_id}")
        if family_id not in allowed_family_ids:
            raise ValueError(f"families 包含不属于当前 display 的 family: {family_id}")
        if any(not isinstance(item.get(field), str) for field in OPTION_GROUPING_TAG_FIELDS):
            raise ValueError(f"families 标签必须为字符串: {family_id}")
        tags = {field: cell_text(item.get(field)) for field in OPTION_GROUPING_TAG_FIELDS}
        parsed_item = {"family_id": family_id, **tags}
        parsed.append(parsed_item)
        tag_map[family_id] = tags
    if set(tag_map) != allowed_family_ids:
        raise ValueError("families 中的 family_id 集合与当前 Display 输入 Family 集合不一致")
    return parsed, tag_map

def option_group_conflict_fields(family_ids: list[str], tag_map: dict[str, dict[str, str]]) -> list[str]:
    conflicts: list[str] = []
    for field in ("thickness", "material"):
        non_empty_values = {tag_map[family_id][field] for family_id in family_ids if tag_map[family_id][field]}
        if len(non_empty_values) > 1:
            conflicts.append(field)
    if len({tag_map[family_id]["level"] for family_id in family_ids}) > 1:
        conflicts.append("level")
    return conflicts

def split_conflicting_option_groups(
    groups: list[list[str]], tag_map: dict[str, dict[str, str]],
) -> tuple[list[list[str]], bool, list[str]]:
    final_groups: list[list[str]] = []
    reasons: list[str] = []
    for group in groups:
        conflict_fields = option_group_conflict_fields(group, tag_map)
        if not conflict_fields:
            final_groups.append(list(group))
            continue
        reason = f"{conflict_fields[0]}_conflict" if len(conflict_fields) == 1 else "multiple_conflicts"
        if reason not in reasons:
            reasons.append(reason)
        subgroups: dict[tuple[str, ...], list[str]] = {}
        for family_id in group:
            key = tuple(tag_map[family_id][field] for field in conflict_fields)
            subgroups.setdefault(key, []).append(family_id)
        final_groups.extend(subgroups.values())
    return final_groups, bool(reasons), reasons

def parse_display_option_grouping_result(
    result: dict[str, Any],
    candidate_display_groups: pd.DataFrame,
    display_group_families: pd.DataFrame,
    warnings: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if len(candidate_display_groups) != 1:
        raise ValueError("display_option_grouping 每次必须且只能校验一个 Display")
    if not isinstance(result, dict) or set(result) != {"families", "groups"}:
        actual_keys = sorted(result) if isinstance(result, dict) else [type(result).__name__]
        raise ValueError(f"display_option_grouping 顶层只允许包含 families 和 groups，实际为: {actual_keys}")
    bound_display_id = cell_text(candidate_display_groups.iloc[0].get("display_id"))
    raw_rows = [{"display_id": bound_display_id, "families": result.get("families"), "groups": result.get("groups")}]

    display_ids = [cell_text(value) for value in candidate_display_groups.get("display_id", pd.Series(dtype=object)).tolist()]
    allowed_order_by_display = {
        display_id: [
            family_id
            for family_id in display_group_families[
                display_group_families["display_id"].map(cell_text).eq(display_id)
            ]["family_id"].map(cell_text).tolist()
            if family_id
        ]
        for display_id in display_ids
    }
    allowed_by_display = {
        display_id: set(family_ids)
        for display_id, family_ids in allowed_order_by_display.items()
    }
    display_map = {cell_text(row.get("display_id")): row for _index, row in candidate_display_groups.iterrows()}
    family_row_map = {
        (cell_text(row.get("display_id")), cell_text(row.get("family_id"))): row
        for _index, row in display_group_families.iterrows()
    }
    result_by_display: dict[str, dict[str, Any]] = {}
    meta = empty_display_option_grouping_meta()

    for item in raw_rows:
        if not isinstance(item, dict):
            raise ValueError("display_option_grouping display_result 必须为 object")
        if set(item) != {"display_id", "families", "groups"}:
            raise ValueError("display_result 只允许包含 display_id、families 和 groups")
        display_id = cell_text(item.get("display_id"))
        if display_id not in allowed_by_display:
            raise ValueError(f"display_option_grouping 返回未知 display: {display_id}")
        if display_id in result_by_display:
            raise ValueError(f"display_option_grouping 重复返回 display: {display_id}")
        allowed_family_ids = allowed_by_display[display_id]
        if not allowed_family_ids:
            raise ValueError(f"display 缺少 candidate families: {display_id}")
        parsed_family_tags, family_tag_map = validate_option_grouping_family_tags(
            item.get("families"), allowed_family_ids
        )
        raw_options = item.get("groups")
        if not isinstance(raw_options, list) or not raw_options:
            raise ValueError(f"groups 必须为非空 list: {display_id}")

        original_groups: list[list[str]] = []
        seen_family_ids: set[str] = set()
        seen_family_groups: set[frozenset[str]] = set()
        for option_index, raw_option in enumerate(raw_options, start=1):
            raw_family_ids = raw_option
            if not isinstance(raw_family_ids, list) or not raw_family_ids:
                raise ValueError(f"group 必须为非空 family_id list: {display_id}/O{option_index:02d}")
            family_ids = [cell_text(family_id) for family_id in raw_family_ids]
            if any(not family_id for family_id in family_ids):
                raise ValueError(f"family_ids 不得为空: {display_id}/O{option_index:02d}")
            duplicate_in_option = [family_id for family_id in family_ids if family_ids.count(family_id) > 1]
            if duplicate_in_option:
                raise ValueError(f"同一 practice_option 内 family 重复: {display_id}/{join_non_empty(duplicate_in_option)}")
            unknown_family_ids = [family_id for family_id in family_ids if family_id not in allowed_family_ids]
            if unknown_family_ids:
                raise ValueError(f"practice_option 包含不属于当前 display 的 family: {display_id}/{join_non_empty(unknown_family_ids)}")
            family_group = frozenset(family_ids)
            if family_group in seen_family_groups:
                raise ValueError(f"同一 display 内不得存在完全相同的 family_ids 分组: {display_id}")
            duplicate_across_options = [family_id for family_id in family_ids if family_id in seen_family_ids]
            if duplicate_across_options:
                raise ValueError(f"family 不得出现在多个 practice_options: {display_id}/{join_non_empty(duplicate_across_options)}")
            option_units = [
                display_family_unit(family_row_map.get((display_id, family_id), pd.Series(dtype=object)))
                for family_id in family_ids
            ]
            if len(set(option_units)) > 1:
                raise ValueError(f"同一 practice_option 中 family 单位必须一致: {display_id}/O{option_index:02d}")
            seen_family_groups.add(family_group)
            seen_family_ids.update(family_ids)
            original_groups.append(family_ids)

        missing_family_ids = [family_id for family_id in allowed_order_by_display[display_id] if family_id not in seen_family_ids]
        if missing_family_ids:
            raise ValueError(f"practice_options 遗漏 candidate family: {display_id}/{join_non_empty(missing_family_ids)}")
        extra_family_ids = [family_id for family_id in seen_family_ids if family_id not in allowed_family_ids]
        if extra_family_ids:
            raise ValueError(f"practice_options 新增非法 family: {display_id}/{join_non_empty(extra_family_ids)}")
        final_groups, auto_split_applied, auto_split_reasons = split_conflicting_option_groups(
            original_groups, family_tag_map
        )
        practice_options: list[dict[str, Any]] = []
        for stable_index, family_ids in enumerate(final_groups, start=1):
            option_units = [
                display_family_unit(family_row_map.get((display_id, family_id), pd.Series(dtype=object)))
                for family_id in family_ids
            ]
            if len(set(option_units)) > 1:
                raise ValueError(f"同一 practice_option 中 family 单位必须一致: {display_id}/O{stable_index:02d}")
            sample_count = sum(
                int(numeric_or_none(family_row_map.get((display_id, family_id), pd.Series(dtype=object)).get("本次召回样本数")) or 0)
                for family_id in family_ids
            )
            practice_options.append({
                "practice_option_id": f"{display_id}-O{stable_index:02d}",
                "sample_count": sample_count,
                "family_ids": family_ids,
            })
        display_row = display_map.get(display_id, pd.Series(dtype=object))
        result_by_display[display_id] = {
            "display_id": display_id,
            "display_name": cell_text(display_row.get("display_name")),
            "unit": cell_text(display_row.get("unit")),
            "family_count": display_row.get("family_count", ""),
            "practice_options": practice_options,
        }
        meta.setdefault("grouping_details_by_display", {})[display_id] = {
            "parsed_family_tags": parsed_family_tags,
            "original_groups": original_groups,
            "final_groups": final_groups,
            "auto_split_applied": auto_split_applied,
            "auto_split_reasons": auto_split_reasons,
        }
        meta["display_ids"].append(display_id)
        meta["practice_option_count"] += len(practice_options)
        meta["families_grouped_count"] += len(seen_family_ids)
        meta["option_count_by_display"][display_id] = len(practice_options)
        meta["max_options_per_display"] = max(meta["max_options_per_display"], len(practice_options))

    missing = [display_id for display_id in display_ids if display_id not in result_by_display]
    if missing:
        raise ValueError(f"每个 candidate display 必须返回 practice_options: {join_non_empty(missing)}")
    return pd.DataFrame([result_by_display[display_id] for display_id in display_ids]), meta

def build_display_option_grouping_trace_frame(
    displays_with_options: pd.DataFrame,
    display_group_families: pd.DataFrame,
    candidate_families: pd.DataFrame,
) -> pd.DataFrame:
    family_map = {cell_text(row.get("family_id")): row for _index, row in candidate_families.iterrows()}
    display_family_map = {
        (cell_text(row.get("display_id")), cell_text(row.get("family_id"))): row
        for _index, row in display_group_families.iterrows()
    }
    display_map = {cell_text(row.get("display_id")): row for _index, row in displays_with_options.iterrows()}
    rows: list[dict[str, Any]] = []
    for display_id, display in display_map.items():
        practice_options = display.get("practice_options") if isinstance(display.get("practice_options"), list) else []
        for option in practice_options:
            if not isinstance(option, dict):
                continue
            practice_option_id = cell_text(option.get("practice_option_id"))
            family_ids = option.get("family_ids") if isinstance(option.get("family_ids"), list) else []
            for family_id_value in family_ids:
                family_id = cell_text(family_id_value)
                row = display_family_map.get((display_id, family_id), pd.Series(dtype=object))
                candidate = family_map.get(family_id, pd.Series(dtype=object))
                rows.append(
                    {
                        "display_id": display_id,
                        "display_name": cell_text(row.get("display_name")) or cell_text(display.get("display_name")),
                        "practice_option_id": practice_option_id,
                        "family_id": family_id,
                        "representative_cost_item_name": cell_text(row.get("representative_cost_item_name")),
                        "representative_project_description": cell_text(row.get("representative_project_description")),
                        "unit": cell_text(candidate.get("unit_normalized")) or cell_text(row.get("unit")) or cell_text(candidate.get("unit")),
                        "本次召回样本数": row.get("本次召回样本数", ""),
                        "本次召回工程包数": row.get("本次召回工程包数", ""),
                        "item_query_similarity最大值": row.get("item_query_similarity最大值", ""),
                        "unit_price_min": candidate.get("本次召回综合单价最低值", ""),
                        "unit_price_median": candidate.get("本次召回综合单价中位数", ""),
                        "unit_price_max": candidate.get("本次召回综合单价最高值", ""),
                        "option_sample_count": int(option.get("option_sample_count") or 0),
                        "option_package_count": int(option.get("option_package_count") or 0),
                        "is_original_option": False,
                        "is_selected_option": False,
                        "option_selection_decision": "",
                        "option_selection_reason": "",
                        "is_original_family": False,
                        "is_representative_family": False,
                        "representative_selection_reason": "",
                    }
                )
    frame = pd.DataFrame(rows, columns=DISPLAY_OPTION_GROUPING_TRACE_COLUMNS)
    if frame.empty:
        return frame
    frame["_item_similarity_sort"] = pd.to_numeric(frame["item_query_similarity最大值"], errors="coerce").fillna(-1)
    frame = frame.sort_values(
        ["display_id", "practice_option_id", "family_id", "_item_similarity_sort"],
        ascending=[True, True, True, False],
    ).drop(columns=["_item_similarity_sort"])
    return frame[DISPLAY_OPTION_GROUPING_TRACE_COLUMNS]

def generate_display_option_grouping(
    candidate_display_groups: pd.DataFrame,
    display_group_families: pd.DataFrame,
    candidate_families: pd.DataFrame,
    warnings: list[str] | None = None,
    *, request_fn=request_llm_json_with_usage, trace_factory=trace_row,
    warning_fn=append_warning,
) -> tuple[pd.DataFrame, bool, bool, str, str, dict[str, Any], dict[str, Any], pd.DataFrame, list[dict[str, Any]]]:
    request_llm_json_with_usage = request_fn
    trace_row = trace_factory
    append_warning = warning_fn
    family_ids_by_display = {
        display_id: [family_id for family_id in group["family_id"].map(cell_text).tolist() if family_id]
        for display_id, group in display_group_families.groupby("display_id", sort=False, dropna=False)
    }
    multi_family_mask = candidate_display_groups["display_id"].map(
        lambda value: len(family_ids_by_display.get(cell_text(value), [])) > 1
    )
    multi_family_displays = candidate_display_groups[multi_family_mask].copy().reset_index(drop=True)
    family_count = sum(
        len(family_ids_by_display.get(cell_text(row.get("display_id")), []))
        for _index, row in multi_family_displays.iterrows()
    )
    prompts: list[str] = []
    responses: list[Any] = []
    errors_by_display: dict[str, str] = {}
    per_display_traces: list[dict[str, Any]] = []
    fallback = False
    grouped_frames: list[pd.DataFrame] = []
    for _index, display in multi_family_displays.iterrows():
        display_id = cell_text(display.get("display_id"))
        current_display = pd.DataFrame([display.to_dict()])
        prompt, record = build_display_option_grouping_prompt(
            current_display, display_group_families, candidate_families
        )
        prompts.append(prompt)
        current_family_count = len(record.get("candidate_families") or [])
        max_tokens = min(4096, max(512, 256 + current_family_count * 96))
        response = None
        current_error = ""
        grouping_details = {
            "parsed_family_tags": [], "original_groups": [], "final_groups": [],
            "auto_split_applied": False, "auto_split_reasons": [],
        }
        try:
            response = request_llm_json_with_usage(
                prompt, max_tokens=max_tokens,
                system_prompt="只输出顶层仅含 families 和 groups 的 JSON object，不输出 display_id 或解释。",
            )
            grouped_display, grouped_meta = parse_display_option_grouping_result(
                response.content, current_display, display_group_families, warnings,
            )
            grouping_details = grouped_meta.get("grouping_details_by_display", {}).get(display_id, grouping_details)
        except (RuntimeError, ValueError) as exc:
            fallback = True
            current_error = f"fallback_single_family_options: {exc}"
            errors_by_display[display_id] = current_error
            append_warning(warnings, f"display_option_grouping_fallback_single_family_options:{display_id}")
            fallback_family_ids = family_ids_by_display[display_id]
            fallback_result = {
                "families": [
                    {"family_id": family_id, "thickness": "", "material": "", "level": ""}
                    for family_id in fallback_family_ids
                ],
                "groups": [[family_id] for family_id in fallback_family_ids],
            }
            grouped_display, grouped_meta = parse_display_option_grouping_result(
                fallback_result, current_display, display_group_families, warnings,
            )
            grouping_details = grouped_meta.get("grouping_details_by_display", {}).get(display_id, grouping_details)
        display_name = cell_text(display.get("display_name"))
        per_display_trace = trace_row(
            "display_option_grouping",
            f"为 Display {display_id} {display_name} 划分 Option",
            not current_error,
            error=current_error,
            prompt=prompt,
            max_tokens=max_tokens,
            input_summary=json_text({
                "display_id": display_id,
                "display_name": display_name,
                "candidate_family_count": current_family_count,
            }),
            usage=response.usage if response is not None else None,
            raw_response=getattr(response, "raw_content", "") if response is not None else "",
        )
        per_display_trace.update({
            "display_id": display_id,
            "display_name": display_name,
            "candidate_family_count": current_family_count,
            "parsed_family_tags": grouping_details["parsed_family_tags"],
            "original_groups": grouping_details["original_groups"],
            "final_groups": grouping_details["final_groups"],
            "auto_split_applied": grouping_details["auto_split_applied"],
            "auto_split_reasons": grouping_details["auto_split_reasons"],
            "status": "fallback" if current_error else "success",
            "error": current_error,
        })
        per_display_traces.append(per_display_trace)
        grouped_frames.append(grouped_display)
        if response is not None:
            responses.append(response)

    grouped_displays = pd.concat(grouped_frames, ignore_index=True) if grouped_frames else pd.DataFrame()
    error_message = "；".join(f"{display_id}: {message}" for display_id, message in errors_by_display.items())
    combined_prompt = "\n\n".join(prompts)
    combined_usage = {
        key: sum(int((getattr(response, "usage", {}) or {}).get(key) or 0) for response in responses)
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"]
    }
    combined_raw_response = "\n".join(
        cell_text(getattr(response, "raw_content", "")) for response in responses
    )

    family_map = {cell_text(row.get("family_id")): row for _index, row in candidate_families.iterrows()}
    grouped_map = {cell_text(row.get("display_id")): row for _index, row in grouped_displays.iterrows()}
    display_rows: list[dict[str, Any]] = []
    meta = empty_display_option_grouping_meta()
    for _index, display in candidate_display_groups.iterrows():
        display_id = cell_text(display.get("display_id"))
        family_ids = family_ids_by_display.get(display_id, [])
        if not family_ids:
            raise ValueError(f"display 缺少 candidate families: {display_id}")
        if len(family_ids) == 1:
            family_id = family_ids[0]
            family = family_map.get(family_id, pd.Series(dtype=object))
            row = {
                "display_id": display_id,
                "display_name": cell_text(display.get("display_name")),
                "unit": cell_text(display.get("unit")),
                "family_count": display.get("family_count", 1),
                "practice_options": [
                    {
                        "practice_option_id": f"{display_id}-O01",
                        "sample_count": int(numeric_or_none(family.get("本次召回样本数")) or 0),
                        "family_ids": [family_id],
                    }
                ],
            }
            meta["programmatic_single_family_display_count"] += 1
        else:
            grouped = grouped_map.get(display_id)
            if grouped is None:
                raise ValueError(f"display_option_grouping 缺少 display: {display_id}")
            row = grouped.to_dict()
        options = row["practice_options"]
        display_rows.append(row)
        meta["display_ids"].append(display_id)
        meta["practice_option_count"] += len(options)
        meta["families_grouped_count"] += len(family_ids)
        meta["option_count_by_display"][display_id] = len(options)
        meta["max_options_per_display"] = max(meta["max_options_per_display"], len(options))
    meta["llm_display_count"] = len(multi_family_displays)
    displays_with_options = pd.DataFrame(display_rows)
    trace_frame = build_display_option_grouping_trace_frame(
        displays_with_options, display_group_families, candidate_families
    )
    trace = trace_row(
        "display_option_grouping",
        "将全部 candidate display 内的 family 划分为独立价格统计口径的 practice options",
        not fallback,
        error=error_message,
        prompt=combined_prompt,
        max_tokens=4096 if len(multi_family_displays) else 0,
        input_summary=json_text(
            {
                "display_count": len(candidate_display_groups),
                "llm_display_count": len(multi_family_displays),
                "programmatic_single_family_display_count": meta["programmatic_single_family_display_count"],
                "family_count": family_count,
                "practice_option_count": meta.get("practice_option_count", 0),
            }
        ),
        usage=combined_usage if responses else None,
        raw_response=combined_raw_response,
    )
    return (
        displays_with_options, not fallback, fallback, error_message, combined_prompt,
        trace, meta, trace_frame, per_display_traces,
    )

def display_option_maps(
    displays_with_options: pd.DataFrame,
) -> tuple[dict[str, pd.Series], dict[tuple[str, str], dict[str, Any]]]:
    display_map = {
        cell_text(row.get("display_id")): row
        for _index, row in displays_with_options.iterrows()
    }
    option_map: dict[tuple[str, str], dict[str, Any]] = {}
    for display_id, row in display_map.items():
        options = row.get("practice_options") if isinstance(row.get("practice_options"), list) else []
        for option in options:
            if isinstance(option, dict):
                option_map[(display_id, cell_text(option.get("practice_option_id")))] = option
    return display_map, option_map

def attach_option_support_counts(
    displays_with_options: pd.DataFrame,
    evidence_items: pd.DataFrame,
) -> pd.DataFrame:
    output = displays_with_options.copy(deep=True)
    evidence_family_ids = evidence_items.get("family_id", pd.Series(dtype=object)).map(cell_text)
    for index, display in output.iterrows():
        options = display.get("practice_options") if isinstance(display.get("practice_options"), list) else []
        supported_options: list[dict[str, Any]] = []
        for raw_option in options:
            option = dict(raw_option)
            family_ids = {cell_text(value) for value in option.get("family_ids", []) if cell_text(value)}
            evidence = evidence_items[evidence_family_ids.isin(family_ids)]
            package_ids = evidence.get("project_package_id", pd.Series(dtype=object)).map(cell_text)
            option["option_sample_count"] = int(len(evidence))
            option["option_package_count"] = int(package_ids[package_ids.ne("")].nunique())
            supported_options.append(option)
        output.at[index, "practice_options"] = supported_options
    return output
