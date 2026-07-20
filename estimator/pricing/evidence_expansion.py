from __future__ import annotations

import json
import re
import sys
import unicodedata
from typing import Any

import pandas as pd

from estimator.paths import CLASSIFIER_BACKEND_DIR
from estimator.candidates.signatures import cell_text, join_non_empty, numeric_or_none, normalized_unit, truncate_text, json_text, append_warning, trace_row
if str(CLASSIFIER_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSIFIER_BACKEND_DIR))
from classifier.llm_client import LLMServiceError, request_llm_json_with_usage  # noqa: E402
from estimator.query_models import ScenarioItem
from estimator.retrieval.evidence_pool import source_identity_for_row
from estimator.candidates.practice_options import display_option_maps

EVIDENCE_EXPANSION_SCOPE_PATTERN = re.compile(
    r"^(?:含)?(?:人工费?|安装|拆除及安装(?:人工费?)?|拆机及安装(?:人工费?)?|运输|运费|起吊费)$"
)

OPTION_EVIDENCE_EXPANSION_COLUMNS = [
    "final_item_position",
    "清单名称",
    "practice_option_id",
    "原family数",
    "候选family数",
    "新增family数",
    "扩展后family数",
    "原价格证据样本数",
    "扩展后价格证据样本数",
    "新增family_ids",
]

def normalize_evidence_expansion_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", cell_text(value))
    text = text.replace("（", "(").replace("）", ")")

    def remove_scope_parentheses(match: re.Match[str]) -> str:
        content = re.sub(r"\s+", "", match.group(1))
        if EVIDENCE_EXPANSION_SCOPE_PATTERN.fullmatch(content):
            return ""
        return match.group(0)

    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\(([^()]*)\)", remove_scope_parentheses, text)
    return re.sub(r"\s+", " ", text).strip()

def build_option_evidence_expansion_prompt(
    representative_family: dict[str, Any],
    candidate_families: list[dict[str, Any]],
) -> str:
    payload = {
        "representative_family": representative_family,
        "candidate_families": candidate_families,
    }

    return f"""
任务：判断哪些候选 Family 可以与代表 Family 使用同一个价格统计口径。

你需要先为代表 Family 和每个候选 Family 提取辅助标签，再结合清单名称、项目特征、单位和辅助标签，判断候选是否可以加入代表 Family 的价格证据。

判断目标：

只有候选 Family 与代表 Family 表示同一种维修对象、同一种主要施工动作，并且能够作为同一类清单直接替换计价时，才可以接受。

辅助标签：

- object_action：维修对象与主要施工动作的标准化表达。
- thickness：明确出现的厚度或核心尺寸规格，没有时为 ""。
- material：明确出现的主要材料类别，没有时为 ""。
- level：明确出现的楼层、层数或高度条件，没有时为 ""。

判断原则：

1. 标签仅用于辅助判断，不得只机械比较标签字符串。
2. 名称表达、语序、标点、OCR、空格或普通文字详略不同，不影响合并。
3. “含人工”“含安装”“含拆除及安装人工”“含运输”“基层清理”“垃圾清运”等一般附带说明不同，可以忽略。
4. 如果候选可以直接替换代表 Family，且不会改变用户所选择的维修对象、工艺或计价范围，可以接受。
5. 如果用户明确某个参数后，需要保留其中一个、排除另一个，则不能接受。
6. 不得仅因为属于同一专业、同一系统、同一工程、单位相同或关键词相似而接受。
7. 主体设备与部件、附件、配套设备不得互相替代。
8. 维修对象不同，不得接受。
9. 主要施工动作不同，不得接受，例如更换、维修、拆除、新做、安装、调试、改造必须区分。
10. 明确材料不同，不得接受。
11. 明确厚度、核心尺寸或规格冲突，不得接受。
12. 明确楼层、层数或高度条件不同，不得接受；明确值与空值也应谨慎，不得因信息缺失直接视为一致。
13. 普通品牌、设备编号、图纸编号或内部型号，如不改变维修对象和主要计价口径，可以忽略。
14. 直径、长度、根数、结构类型、材料类别、厚度等会明显影响计价口径的核心规格，不得忽略。
15. 信息不足、对象关系不清楚或是否可直接替换无法确认时，不接受。

典型不得接受的关系包括：

- 消防报警主机、消防广播主机、消防电话主机之间；
- 主机与多线盘、回路板、电话盘、广播盘之间；
- 主体设备与配套附件之间；
- 曳引钢丝绳、限速器钢丝绳、钢带、曳引轮、绳轮组件之间；
- 瓦屋面、卷材防水、涂膜防水、涂料面层之间；
- 设备更换与设备维修、系统调试、线路改造之间。

每个输入 family_id 必须且只能在 families 中出现一次，不得遗漏、重复或新增。

accepted_family_ids：

- 只能包含 candidate_families 中存在的 family_id；
- 不得包含代表 Family；
- 不得重复；
- 只填写可以与代表 Family 使用同一价格统计口径的候选；
- 宁可少选，不要扩大到相关但不可直接替换的清单。

只输出合法 JSON，顶层只能包含 families 和 accepted_family_ids：

{{
  "families": [
    {{
      "family_id": "F001",
      "object_action": "曳引钢丝绳更换",
      "thickness": "",
      "material": "",
      "level": ""
    }}
  ],
  "accepted_family_ids": ["F002"]
}}

输入：
{json.dumps(payload, ensure_ascii=False)}
""".strip()

def validate_option_evidence_expansion_result(
    result: Any,
    representative_family_id: str,
    candidate_family_ids: list[str],
) -> tuple[dict[str, dict[str, str]], list[str]]:
    if not isinstance(result, dict) or set(result) != {"families", "accepted_family_ids"}:
        raise ValueError("option evidence expansion 顶层字段非法")
    families = result.get("families")
    if not isinstance(families, list):
        raise ValueError("option evidence expansion families 必须是数组")
    expected_fields = {"family_id", "object_action", "thickness", "material", "level"}
    parsed: dict[str, dict[str, str]] = {}
    seen: set[str] = set()
    for family in families:
        if not isinstance(family, dict) or set(family) != expected_fields:
            raise ValueError("option evidence expansion family 字段非法")
        if any(not isinstance(family[field], str) for field in expected_fields):
            raise ValueError("option evidence expansion family 字段必须为字符串")
        family_id = family["family_id"]
        if not family_id:
            raise ValueError("option evidence expansion family_id 必须是非空字符串")
        if family_id in seen:
            raise ValueError(f"option evidence expansion family_id 重复: {family_id}")
        seen.add(family_id)
        parsed[family_id] = {
            "object_action": family["object_action"],
            "thickness": family["thickness"],
            "material": family["material"],
            "level": family["level"],
        }
    expected_family_ids = [representative_family_id, *candidate_family_ids]
    if seen != set(expected_family_ids) or len(families) != len(expected_family_ids):
        raise ValueError("option evidence expansion family_id 集合与输入不一致")
    accepted = result.get("accepted_family_ids")
    if not isinstance(accepted, list):
        raise ValueError("option evidence expansion accepted_family_ids 必须是数组")
    candidate_set = set(candidate_family_ids)
    accepted_family_ids: list[str] = []
    accepted_seen: set[str] = set()
    for family_id in accepted:
        if not isinstance(family_id, str) or not family_id:
            raise ValueError("option evidence expansion accepted family_id 必须是非空字符串")
        if family_id in accepted_seen:
            raise ValueError(f"option evidence expansion accepted family_id 重复: {family_id}")
        if family_id not in candidate_set:
            raise ValueError(f"option evidence expansion accepted family_id 非法: {family_id}")
        accepted_seen.add(family_id)
        accepted_family_ids.append(family_id)
    return parsed, accepted_family_ids

def filter_option_evidence_hard_conflicts(
    accepted_family_ids: list[str],
    representative_family_id: str,
    tags_by_family_id: dict[str, dict[str, str]],
    representative_unit: str,
    candidate_unit_map: dict[str, str],
) -> list[str]:
    representative_tags = tags_by_family_id[representative_family_id]
    filtered: list[str] = []
    for family_id in accepted_family_ids:
        candidate_tags = tags_by_family_id[family_id]
        if candidate_unit_map[family_id] != representative_unit:
            continue
        if (
            representative_tags["thickness"]
            and candidate_tags["thickness"]
            and representative_tags["thickness"] != candidate_tags["thickness"]
        ):
            continue
        if (
            representative_tags["material"]
            and candidate_tags["material"]
            and representative_tags["material"] != candidate_tags["material"]
        ):
            continue
        if (
            representative_tags["level"] != candidate_tags["level"]
            and (representative_tags["level"] or candidate_tags["level"])
        ):
            continue
        filtered.append(family_id)
    return filtered

def option_evidence_expansion_candidates(
    item: ScenarioItem,
    option: dict[str, Any],
    candidate_families: pd.DataFrame,
) -> tuple[list[str], dict[str, Any], list[dict[str, Any]]]:
    original_family_ids = list(dict.fromkeys(
        cell_text(value) for value in option.get("family_ids", []) if cell_text(value)
    ))
    if not original_family_ids:
        raise ValueError(f"Option 缺少 family_ids: option_id={item.practice_option_id}")
    if not item.representative_family_id or item.representative_family_id not in original_family_ids:
        raise ValueError(
            f"代表 Family 不属于原 Option: family_id={item.representative_family_id}"
        )

    family_ids = candidate_families.get("family_id", pd.Series(dtype=object)).map(cell_text)
    representative_rows = candidate_families[family_ids.eq(item.representative_family_id)]
    if len(representative_rows) != 1:
        raise ValueError(
            f"代表 Family 无法唯一回查: family_id={item.representative_family_id}, "
            f"matched_rows={len(representative_rows)}"
        )
    representative = representative_rows.iloc[0]
    target_unit = cell_text(representative.get("unit_normalized"))
    representative_family = {
        "family_id": item.representative_family_id,
        "cost_item_name": normalize_evidence_expansion_name(
            representative.get("representative_cost_item_name")
        ),
        "project_description": cell_text(
            representative.get("representative_project_description")
        ),
        "unit": target_unit,
    }

    candidates: list[dict[str, Any]] = []
    seen = set(original_family_ids)
    for _index, family in candidate_families.iterrows():
        family_id = cell_text(family.get("family_id"))
        if not family_id or family_id in seen:
            continue
        seen.add(family_id)
        if cell_text(family.get("unit_normalized")) != target_unit:
            continue
        candidates.append({
            "family_id": family_id,
            "cost_item_name": normalize_evidence_expansion_name(
                family.get("representative_cost_item_name")
            ),
            "project_description": cell_text(
                family.get("representative_project_description")
            ),
            "unit": cell_text(family.get("unit_normalized")),
        })
        if len(candidates) >= 20:
            break
    return original_family_ids, representative_family, candidates

def expand_option_price_evidence_families(
    final_items: list[ScenarioItem],
    displays_with_options: pd.DataFrame,
    candidate_families: pd.DataFrame,
    samples: pd.DataFrame,
    warnings: list[str] | None = None,
    *, request_fn=request_llm_json_with_usage, trace_factory=trace_row,
    warning_fn=append_warning,
) -> tuple[dict[int, list[str]], pd.DataFrame, list[dict[str, Any]]]:
    request_llm_json_with_usage = request_fn
    trace_row = trace_factory
    append_warning = warning_fn
    display_map, option_map = display_option_maps(displays_with_options)
    family_map = {
        cell_text(row.get("family_id")): row
        for _index, row in candidate_families.iterrows()
        if cell_text(row.get("family_id"))
    }
    expanded_by_position: dict[int, list[str]] = {}
    sheet_rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []

    for item in final_items:
        display = display_map.get(item.display_id)
        option = option_map.get((item.display_id, item.practice_option_id))
        if display is None or option is None:
            raise ValueError(
                f"价格证据扩充 display/option 回查失败: item_position={item.item_position}"
            )
        original_family_ids = list(dict.fromkeys(
            cell_text(value) for value in option.get("family_ids", []) if cell_text(value)
        ))
        if not original_family_ids:
            raise ValueError(f"Option 缺少 family_ids: option_id={item.practice_option_id}")
        original_option = {**option, "family_ids": original_family_ids}
        original_evidence = expand_samples_for_option(
            original_option, display, candidate_families, samples
        )
        original_evidence_count = len(original_evidence)

        representative = family_map.get(item.representative_family_id)
        representative_family: dict[str, Any] = {}
        candidates: list[dict[str, Any]] = []
        candidate_ids: list[str] = []
        prompt = ""
        response = None
        accepted_family_ids: list[str] = []
        llm_accepted_count = 0
        error_message = ""
        max_tokens = 0
        skipped_reason = ""

        if original_evidence_count >= 10:
            skipped_reason = "threshold"
        else:
            try:
                (
                    checked_original_family_ids,
                    representative_family,
                    candidates,
                ) = option_evidence_expansion_candidates(item, option, candidate_families)
                if checked_original_family_ids != original_family_ids:
                    raise ValueError("option evidence expansion 原 family_ids 不一致")
                candidate_ids = [candidate["family_id"] for candidate in candidates]
                if not candidates:
                    skipped_reason = "no_candidates"
                else:
                    prompt = build_option_evidence_expansion_prompt(
                        representative_family, candidates
                    )
                    max_tokens = 2048
                    response = request_llm_json_with_usage(
                        prompt,
                        max_tokens=max_tokens,
                        system_prompt="你只输出一个 JSON object，不输出解释、Markdown 或思考过程。",
                    )
                    tags_by_family_id, llm_accepted_family_ids = (
                        validate_option_evidence_expansion_result(
                            response.content,
                            item.representative_family_id,
                            candidate_ids,
                        )
                    )
                    llm_accepted_count = len(llm_accepted_family_ids)
                    filtered_accepted_family_ids = filter_option_evidence_hard_conflicts(
                        llm_accepted_family_ids,
                        item.representative_family_id,
                        tags_by_family_id,
                        representative_family["unit"],
                        {candidate["family_id"]: candidate["unit"] for candidate in candidates},
                    )
                    accepted_set = set(filtered_accepted_family_ids)
                    accepted_family_ids = [
                        family_id for family_id in candidate_ids if family_id in accepted_set
                    ]
            except (LLMServiceError, RuntimeError, ValueError, TypeError, KeyError) as exc:
                error_message = str(exc)
                accepted_family_ids = []
                candidates = candidates if representative_family else []
                candidate_ids = [candidate["family_id"] for candidate in candidates]
                append_warning(
                    warnings,
                    f"option_evidence_expansion_failed:item_position={item.item_position}",
                )

        expanded_family_ids = list(dict.fromkeys([
            *original_family_ids,
            *accepted_family_ids,
        ]))
        expanded_by_position[item.item_position] = expanded_family_ids
        expanded_option = {**option, "family_ids": expanded_family_ids}
        expanded_evidence = (
            original_evidence
            if not accepted_family_ids
            else expand_samples_for_option(
                expanded_option, display, candidate_families, samples
            )
        )
        sheet_name = ""
        if representative is not None and item.representative_family_id in original_family_ids:
            sheet_name = cell_text(representative.get("representative_cost_item_name"))
        if not sheet_name:
            sheet_name = cell_text(display.get("display_name"))
        sheet_rows.append({
            "final_item_position": item.item_position,
            "清单名称": sheet_name,
            "practice_option_id": item.practice_option_id,
            "原family数": len(original_family_ids),
            "候选family数": len(candidates) if prompt else 0,
            "新增family数": len(accepted_family_ids),
            "扩展后family数": len(expanded_family_ids),
            "原价格证据样本数": original_evidence_count,
            "扩展后价格证据样本数": len(expanded_evidence),
            "新增family_ids": ",".join(accepted_family_ids),
        })
        raw_response = ""
        if response is not None:
            raw_response = cell_text(getattr(response, "raw_content", ""))
            if not raw_response:
                raw_response = json_text(response.content)

        if skipped_reason == "threshold":
            input_summary = (
                f"item_position={item.item_position}; "
                f"original_evidence={original_evidence_count}; skipped=threshold"
            )
        elif skipped_reason == "no_candidates":
            input_summary = (
                f"item_position={item.item_position}; "
                f"original_evidence={original_evidence_count}; candidates=0; "
                "skipped=no_candidates"
            )
        else:
            input_summary = (
                f"item_position={item.item_position}; "
                f"original_evidence={original_evidence_count}; "
                f"candidates={len(candidate_ids)}; llm_accepted={llm_accepted_count}; "
                f"accepted_after_guard={len(accepted_family_ids)}"
            )
        trace = trace_row(
            "option_evidence_expansion",
            "由 LLM 结合辅助标签判断同一价格统计口径并执行硬冲突兜底",
            not error_message,
            error=error_message,
            prompt=prompt,
            max_tokens=max_tokens,
            input_summary=input_summary,
            usage=response.usage if response is not None else None,
            raw_response=raw_response,
            scenario_count=1,
            scenario_item_count=1,
        )
        if skipped_reason:
            trace["parsed_status"] = "skipped"
        trace["fallback"] = bool(error_message)
        traces.append(trace)

    return (
        expanded_by_position,
        pd.DataFrame(sheet_rows, columns=OPTION_EVIDENCE_EXPANSION_COLUMNS).fillna(""),
        traces,
    )

def expand_samples_for_option(
    option: dict[str, Any],
    display: pd.Series,
    candidate_families: pd.DataFrame,
    samples: pd.DataFrame,
) -> pd.DataFrame:
    option_id = cell_text(option.get("practice_option_id")) or cell_text(option.get("option_id"))
    family_ids = [cell_text(value) for value in option.get("family_ids", []) if cell_text(value)]
    if not family_ids:
        raise ValueError(f"Option 缺少 family_ids: option_id={option_id}")
    if "normalized_signature" not in samples.columns:
        raise ValueError(f"samples 缺少 normalized_signature: option_id={option_id}")

    candidate_family_ids = candidate_families.get("family_id", pd.Series(dtype=object)).map(cell_text)
    family_signatures: list[tuple[str, str]] = []
    expected_units_by_signature: dict[str, set[str]] = {}
    for family_id in family_ids:
        family_rows = candidate_families[candidate_family_ids.eq(family_id)]
        if len(family_rows) != 1:
            raise ValueError(
                f"Family 无法唯一映射 normalized_signature: option_id={option_id}, "
                f"family_id={family_id}, matched_rows={len(family_rows)}"
            )
        family = family_rows.iloc[0]
        signature = cell_text(family.get("normalized_signature"))
        if not signature:
            raise ValueError(
                f"Family normalized_signature 为空: option_id={option_id}, family_id={family_id}"
            )
        family_signatures.append((family_id, signature))
        expected_units_by_signature.setdefault(signature, set()).add(
            normalized_unit(cell_text(family.get("unit_normalized")) or family.get("unit"))
        )

    selected_signatures = list(dict.fromkeys(signature for _family_id, signature in family_signatures))
    sample_signatures = samples["normalized_signature"].map(cell_text)
    expanded = samples[sample_signatures.isin(selected_signatures)].copy()
    for family_id, signature in family_signatures:
        if not sample_signatures.eq(signature).any():
            raise ValueError(
                f"Family normalized_signature 在全量 samples 中无匹配: option_id={option_id}, "
                f"family_id={family_id}, normalized_signature={signature}"
            )

    display_unit = normalized_unit(display.get("unit"))
    for row_index, row in expanded.iterrows():
        signature = cell_text(row.get("normalized_signature"))
        sample_unit = normalized_unit(cell_text(row.get("unit_normalized")) or row.get("unit"))
        family_units = expected_units_by_signature.get(signature, set())
        if not sample_unit or "" in family_units or sample_unit not in family_units or sample_unit != display_unit:
            raise ValueError(
                f"全库价格证据单位不兼容: option_id={option_id}, "
                f"stable_sample_id={cell_text(row.get('stable_sample_id'))}, "
                f"normalized_signature={signature}, sample_unit={sample_unit}, "
                f"family_units={join_non_empty(sorted(family_units))}, display_unit={display_unit}, "
                f"row_index={row_index}"
            )

    source_refs: list[str] = []
    for _row_index, row in expanded.iterrows():
        source_refs.append(cell_text(row.get("source_ref")) or source_identity_for_row(row)[2])
    expanded["source_ref"] = source_refs

    stable_ids = expanded.get("stable_sample_id", pd.Series("", index=expanded.index)).map(cell_text)
    dedupe_keys = [
        f"stable:{stable_sample_id}" if stable_sample_id else f"source:{source_ref}"
        for stable_sample_id, source_ref in zip(stable_ids.tolist(), source_refs)
    ]
    expanded = expanded.loc[~pd.Series(dedupe_keys, index=expanded.index).duplicated(keep="first")].copy()

    signature_family = {}
    for family_id, signature in family_signatures:
        signature_family.setdefault(signature, family_id)
    expanded["family_id"] = expanded["normalized_signature"].map(
        lambda value: signature_family[cell_text(value)]
    )
    return expanded
