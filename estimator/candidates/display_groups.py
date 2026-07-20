from __future__ import annotations

import re
import unicodedata
from typing import Any

import pandas as pd

from estimator.candidates.signatures import cell_text, join_non_empty, normalized_unit, truncate_text, json_text
from estimator.candidates.families import max_numeric_or_zero

CANDIDATE_DISPLAY_GROUP_COLUMNS = [
    "display_id",
    "display_key",
    "display_name",
    "unit",
    "family_count",
    "family_ids",
    "retrieval_package_support_ratio",
    "support_rank",
    "retrieval_item_count",
    "retrieval_package_count",
    "top_family_examples",
    "direct_item_similarity_max",
]

DISPLAY_GROUP_FAMILY_COLUMNS = [
    "display_id",
    "display_key",
    "display_name",
    "family_id",
    "normalized_signature",
    "representative_cost_item_name",
    "representative_project_description",
    "unit",
    "本次召回样本数",
    "本次召回工程包数",
    "item_query_similarity最大值",
]

def normalize_display_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", cell_text(value)).lower()
    replacements = {
        "，": ",",
        "。": ".",
        "；": ";",
        "：": ":",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "、": ",",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" \t\r\n,.;:!?，。；：、()[]【】")
    return text

def normalize_display_description(value: Any) -> str:
    text = cell_text(value)
    if not text:
        return ""
    return re.sub(r"^1\.(\d+\.\d+\s*mm)", r"\1", text, count=1, flags=re.IGNORECASE)

def display_unit_for_family(row: pd.Series) -> str:
    return cell_text(row.get("unit_normalized")) or normalized_unit(row.get("unit"))

def top_family_examples(group: pd.DataFrame, limit: int = 3) -> list[dict[str, Any]]:
    if group.empty:
        return []
    ordered = group.sort_values(
        ["item_query_similarity最大值", "本次召回样本数", "本次召回工程包数"],
        ascending=[False, False, False],
    )
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for _index, row in ordered.iterrows():
        key = (
            normalize_display_description(row.get("representative_project_description")),
            cell_text(row.get("family_id")),
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "family_id": cell_text(row.get("family_id")),
                "项目特征简述": truncate_text(normalize_display_description(row.get("representative_project_description")), 60),
                "samples": int(row.get("本次召回样本数") or 0),
                "packages": int(row.get("本次召回工程包数") or 0),
            }
        )
        if len(rows) >= limit:
            break
    return rows

def build_candidate_display_groups(
    candidate_families: pd.DataFrame,
    evidence_items: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidate_families.empty:
        return (
            pd.DataFrame(columns=CANDIDATE_DISPLAY_GROUP_COLUMNS),
            pd.DataFrame(columns=DISPLAY_GROUP_FAMILY_COLUMNS),
        )

    families = candidate_families.copy()
    families["_display_name_normalized"] = families["representative_cost_item_name"].map(normalize_display_name)
    families["_display_unit_normalized"] = families.apply(display_unit_for_family, axis=1).map(cell_text)
    families["_display_key"] = families["_display_name_normalized"] + "|" + families["_display_unit_normalized"]

    group_rows: list[dict[str, Any]] = []
    grouped_family_rows: list[dict[str, Any]] = []
    display_id_by_key: dict[str, str] = {}
    raw_groups: list[tuple[dict[str, Any], pd.DataFrame]] = []
    for display_key, group in families.groupby("_display_key", sort=False, dropna=False):
        group = group.sort_values(["item_query_similarity最大值", "本次召回样本数", "本次召回工程包数"], ascending=[False, False, False])
        representative = group.iloc[0]
        family_ids = [cell_text(value) for value in group["family_id"].tolist() if cell_text(value)]
        evidence = evidence_items[evidence_items.get("family_id", pd.Series(dtype=object)).map(cell_text).isin(family_ids)].copy()
        package_values = evidence.get("project_package_id", pd.Series(dtype=object)).map(cell_text)
        package_values = package_values[package_values.ne("")]
        if package_values.empty:
            package_values = evidence.get("project_key", pd.Series(dtype=object)).map(cell_text)
            package_values = package_values[package_values.ne("")]
        raw_groups.append(
            (
                {
                    "display_key": cell_text(display_key),
                    "display_name": cell_text(representative.get("representative_cost_item_name")),
                    "unit": display_unit_for_family(representative),
                    "family_count": int(len(group)),
                    "family_ids": ",".join(family_ids),
                    "retrieval_package_support_ratio": 0.0,
                    "retrieval_item_count": 0,
                    "retrieval_package_count": int(package_values.nunique()),
                    "top_family_examples": json_text(top_family_examples(group)),
                    "direct_item_similarity_max": max_numeric_or_zero(group, "item_query_similarity最大值"),
                },
                group,
            )
        )

    raw_groups.sort(
        key=lambda item: (
            -float(item[0].get("direct_item_similarity_max") or 0.0),
            -int(item[0].get("retrieval_package_count") or 0),
            cell_text(item[0].get("display_name")),
        )
    )
    for index, (row, _group) in enumerate(raw_groups, start=1):
        display_id = f"D{index:03d}"
        row["display_id"] = display_id
        display_id_by_key[cell_text(row.get("display_key"))] = display_id
        group_rows.append(row)

    group_by_key = {cell_text(row.get("display_key")): group for row, group in raw_groups}
    for display_key, display_id in display_id_by_key.items():
        group = group_by_key.get(display_key, pd.DataFrame())
        display_row = next(row for row in group_rows if cell_text(row.get("display_key")) == display_key)
        for _index, family in group.iterrows():
            grouped_family_rows.append(
                {
                    "display_id": display_id,
                    "display_key": display_key,
                    "display_name": cell_text(display_row.get("display_name")),
                    "family_id": cell_text(family.get("family_id")),
                    "normalized_signature": cell_text(family.get("normalized_signature")),
                    "representative_cost_item_name": cell_text(family.get("representative_cost_item_name")),
                    "representative_project_description": cell_text(family.get("representative_project_description")),
                    "unit": display_unit_for_family(family),
                    "本次召回样本数": family.get("本次召回样本数", ""),
                    "本次召回工程包数": family.get("本次召回工程包数", ""),
                    "item_query_similarity最大值": family.get("item_query_similarity最大值", ""),
                }
            )

    display_groups = pd.DataFrame(group_rows)
    display_families = pd.DataFrame(grouped_family_rows)
    for column in CANDIDATE_DISPLAY_GROUP_COLUMNS:
        if column not in display_groups.columns:
            display_groups[column] = ""
    for column in DISPLAY_GROUP_FAMILY_COLUMNS:
        if column not in display_families.columns:
            display_families[column] = ""
    missing_display = set(candidate_families["family_id"].map(cell_text)) - set(display_families["family_id"].map(cell_text))
    if missing_display:
        raise ValueError(f"存在 candidate family 未映射到 display_id: {join_non_empty(sorted(missing_display))}")
    return (
        display_groups[CANDIDATE_DISPLAY_GROUP_COLUMNS].reset_index(drop=True),
        display_families[DISPLAY_GROUP_FAMILY_COLUMNS].reset_index(drop=True),
    )

def attach_display_support_ratios(
    candidate_display_groups: pd.DataFrame,
    display_group_families: pd.DataFrame,
    evidence_items: pd.DataFrame,
    package_evidence_weights: pd.DataFrame,
) -> pd.DataFrame:
    if candidate_display_groups.empty:
        return candidate_display_groups.copy()

    weight_map = {
        cell_text(row.get("project_package_id")): float(row.get("package_evidence_weight") or 0.0)
        for _index, row in package_evidence_weights.iterrows()
        if cell_text(row.get("project_package_id"))
    }
    family_ids_by_display = {
        display_id: set(group["family_id"].map(cell_text).tolist())
        for display_id, group in display_group_families.groupby("display_id", sort=False, dropna=False)
    }
    evidence_family_ids = evidence_items.get("family_id", pd.Series(dtype=object)).map(cell_text)
    evidence_package_ids = evidence_items.get("project_package_id", pd.Series(dtype=object)).map(cell_text)

    output = candidate_display_groups.copy()
    support_values: list[float] = []
    item_counts: list[int] = []
    package_counts: list[int] = []
    direct_item_similarity_values: list[float] = []
    for _index, row in output.iterrows():
        display_id = cell_text(row.get("display_id"))
        family_ids = family_ids_by_display.get(display_id, set())
        if not family_ids:
            support_values.append(0.0)
            item_counts.append(0)
            package_counts.append(0)
            direct_item_similarity_values.append(0.0)
            continue
        evidence = evidence_items[evidence_family_ids.isin(family_ids)].copy()
        package_ids = {
            package_id
            for package_id in evidence_package_ids.loc[evidence.index].tolist()
            if package_id
        }
        support_values.append(float(sum(weight_map.get(package_id, 0.0) for package_id in package_ids)))
        item_counts.append(int(len(evidence)))
        package_counts.append(int(len(package_ids)))
        direct_item_similarity_values.append(max_numeric_or_zero(evidence, "item_query_similarity"))

    output["retrieval_package_support_ratio"] = support_values
    output["retrieval_item_count"] = item_counts
    output["retrieval_package_count"] = package_counts
    output["direct_item_similarity_max"] = direct_item_similarity_values
    ranked = output.sort_values(
        ["retrieval_package_support_ratio", "retrieval_package_count", "retrieval_item_count", "display_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    ranks = pd.Series(range(1, len(ranked) + 1), index=ranked.index)
    output["support_rank"] = ranks.reindex(output.index).astype(int)
    return output[CANDIDATE_DISPLAY_GROUP_COLUMNS].reset_index(drop=True)

def filter_required_display_groups(
    plan_items: pd.DataFrame,
    candidate_display_groups: pd.DataFrame,
    display_group_families: pd.DataFrame,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    required_display_ids = list(dict.fromkeys(plan_items["display_id"].map(cell_text).tolist()))
    if not required_display_ids or any(not display_id for display_id in required_display_ids):
        raise ValueError("plan_items 每行必须包含 display_id")

    display_order = {display_id: index for index, display_id in enumerate(required_display_ids)}
    available_display_ids = set(candidate_display_groups["display_id"].map(cell_text).tolist())
    missing_display_ids = [display_id for display_id in required_display_ids if display_id not in available_display_ids]
    if missing_display_ids:
        raise ValueError(f"required display 不存在于 candidate_display_groups: {join_non_empty(missing_display_ids)}")

    filtered_groups = candidate_display_groups[
        candidate_display_groups["display_id"].map(cell_text).isin(required_display_ids)
    ].copy()
    filtered_groups["_required_order"] = filtered_groups["display_id"].map(cell_text).map(display_order)
    filtered_groups = filtered_groups.sort_values("_required_order", kind="stable").drop(columns="_required_order")

    filtered_families = display_group_families[
        display_group_families["display_id"].map(cell_text).isin(required_display_ids)
    ].copy()
    present_family_displays = set(filtered_families["display_id"].map(cell_text).tolist())
    missing_family_displays = [display_id for display_id in required_display_ids if display_id not in present_family_displays]
    if missing_family_displays:
        raise ValueError(f"required display 缺少 family 映射: {join_non_empty(missing_family_displays)}")
    filtered_families["_required_order"] = filtered_families["display_id"].map(cell_text).map(display_order)
    filtered_families = filtered_families.sort_values("_required_order", kind="stable").drop(columns="_required_order")
    return (
        required_display_ids,
        filtered_groups.reset_index(drop=True),
        filtered_families.reset_index(drop=True),
    )
