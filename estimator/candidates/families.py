from __future__ import annotations

from typing import Any

import pandas as pd

from estimator.candidates.signatures import cell_text, join_non_empty, numeric_or_none, normalized_unit, truncate_text, json_text, append_warning, trace_row

CANDIDATE_FAMILY_COLUMNS = [
    "family_id",
    "normalized_signature",
    "representative_cost_item_name",
    "representative_project_description",
    "unit",
    "unit_normalized",
    "本次召回样本数",
    "本次召回工程包数",
    "本次召回工程量最低值",
    "本次召回工程量中位数",
    "本次召回工程量最高值",
    "本次召回综合单价最低值",
    "本次召回综合单价中位数",
    "本次召回综合单价最高值",
    "本次召回合价最低值",
    "本次召回合价中位数",
    "本次召回合价最高值",
    "本次召回人工费单价最低值",
    "本次召回人工费单价中位数",
    "本次召回人工费单价最高值",
    "本次召回机械费单价最低值",
    "本次召回机械费单价中位数",
    "本次召回机械费单价最高值",
    "package_query_similarity最大值",
    "item_query_similarity最大值",
    "source_refs",
]

EVIDENCE_ITEM_COLUMNS = [
    "source_ref",
    "family_id",
    "normalized_signature",
    "project_key",
    "item_row_id",
    "stable_sample_id",
    "batch_id",
    "source_row_id",
    "来源工程名称",
    "project_package_id",
    "consultation_time",
    "location",
    "catalog_id",
    "一级分类",
    "二级分类",
    "维修状态",
    "标准对象",
    "cost_item_name",
    "project_description",
    "unit",
    "unit_normalized",
    "quantity",
    "unit_price",
    "total_price",
    "labor_unit_price",
    "machinery_unit_price",
    "package_rank",
    "package_query_similarity",
    "item_query_similarity",
]

def numeric_values(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").dropna()

def min_median_max(frame: pd.DataFrame, column: str) -> tuple[float | None, float | None, float | None]:
    values = numeric_values(frame, column)
    if values.empty:
        return None, None, None
    return float(values.min()), float(values.median()), float(values.max())

def max_numeric_or_zero(frame: pd.DataFrame, column: str) -> float:
    values = numeric_values(frame, column)
    if values.empty:
        return 0.0
    return float(values.max())

def ordered_refs(values: pd.Series, limit: int = 10) -> str:
    seen: set[str] = set()
    refs: list[str] = []
    for value in values.tolist():
        text = cell_text(value)
        if text and text not in seen:
            refs.append(text)
            seen.add(text)
        if len(refs) >= limit:
            break
    return ", ".join(refs)

def source_package_count(group: pd.DataFrame) -> int:
    for column in ["project_key", "project_package_id"]:
        if column not in group.columns:
            continue
        values = group[column].map(cell_text)
        non_empty = values[values.ne("")]
        if not non_empty.empty:
            return int(non_empty.nunique())
    if "source_ref" not in group.columns:
        return 0
    package_keys = []
    for value in group["source_ref"].tolist():
        text = cell_text(value)
        if "::" in text:
            package_keys.append(text.rsplit("::", 1)[0])
    return len(set(package_keys))

def build_candidate_families(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=CANDIDATE_FAMILY_COLUMNS)

    rows: list[dict[str, Any]] = []
    for normalized_signature, group in candidates.groupby("normalized_signature", sort=False, dropna=False):
        representative = group.iloc[0]
        quantity_min, quantity_median, quantity_max = min_median_max(group, "quantity")
        unit_price_min, unit_price_median, unit_price_max = min_median_max(group, "unit_price")
        total_price_min, total_price_median, total_price_max = min_median_max(group, "total_price")
        labor_min, labor_median, labor_max = min_median_max(group, "labor_unit_price")
        machinery_min, machinery_median, machinery_max = min_median_max(group, "machinery_unit_price")
        rows.append(
            {
                "normalized_signature": cell_text(normalized_signature),
                "representative_cost_item_name": cell_text(representative.get("cost_item_name")),
                "representative_project_description": cell_text(representative.get("project_description")),
                "unit": cell_text(representative.get("unit")),
                "unit_normalized": cell_text(representative.get("unit_normalized")) or cell_text(representative.get("unit")),
                "本次召回样本数": int(len(group)),
                "本次召回工程包数": source_package_count(group),
                "本次召回工程量最低值": quantity_min,
                "本次召回工程量中位数": quantity_median,
                "本次召回工程量最高值": quantity_max,
                "本次召回综合单价最低值": unit_price_min,
                "本次召回综合单价中位数": unit_price_median,
                "本次召回综合单价最高值": unit_price_max,
                "本次召回合价最低值": total_price_min,
                "本次召回合价中位数": total_price_median,
                "本次召回合价最高值": total_price_max,
                "本次召回人工费单价最低值": labor_min,
                "本次召回人工费单价中位数": labor_median,
                "本次召回人工费单价最高值": labor_max,
                "本次召回机械费单价最低值": machinery_min,
                "本次召回机械费单价中位数": machinery_median,
                "本次召回机械费单价最高值": machinery_max,
                "package_query_similarity最大值": max_numeric_or_zero(group, "package_query_similarity"),
                "item_query_similarity最大值": max_numeric_or_zero(group, "item_query_similarity"),
                "source_refs": ordered_refs(group.get("source_ref", pd.Series(dtype=object)), limit=10),
            }
        )

    output = pd.DataFrame(rows)
    output.insert(0, "family_id", [f"F{index:03d}" for index in range(1, len(output) + 1)])
    for column in CANDIDATE_FAMILY_COLUMNS:
        if column not in output.columns:
            output[column] = None
    output = output[CANDIDATE_FAMILY_COLUMNS].reset_index(drop=True)
    return output

def attach_family_ids_to_evidence_items(candidates: pd.DataFrame, candidate_families: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=EVIDENCE_ITEM_COLUMNS)
    signature_to_family_id = {
        cell_text(row.get("normalized_signature")): cell_text(row.get("family_id"))
        for _index, row in candidate_families.iterrows()
        if cell_text(row.get("normalized_signature"))
    }
    normalized_signatures = candidates.get("normalized_signature", pd.Series([""] * len(candidates), index=candidates.index))
    output = pd.DataFrame(
        {
            "source_ref": candidates.get("source_ref", ""),
            "family_id": normalized_signatures.map(lambda value: signature_to_family_id.get(cell_text(value), "")),
            "normalized_signature": normalized_signatures,
            "project_key": candidates.get("project_key", ""),
            "item_row_id": candidates.get("item_row_id", ""),
            "stable_sample_id": candidates.get("stable_sample_id", ""),
            "batch_id": candidates.get("batch_id", ""),
            "source_row_id": candidates.get("source_row_id", ""),
            "来源工程名称": candidates.get("工程名称", ""),
            "project_package_id": candidates.get("project_package_id", ""),
            "consultation_time": candidates.get("consultation_time", ""),
            "location": candidates.get("location", ""),
            "catalog_id": candidates.get("catalog_id", ""),
            "一级分类": candidates.get("一级分类", ""),
            "二级分类": candidates.get("二级分类", ""),
            "维修状态": candidates.get("维修状态", ""),
            "标准对象": candidates.get("标准对象", ""),
            "cost_item_name": candidates.get("cost_item_name", ""),
            "project_description": candidates.get("project_description", ""),
            "unit": candidates.get("unit", ""),
            "unit_normalized": candidates.get("unit_normalized", candidates.get("unit", "")),
            "quantity": candidates.get("quantity", ""),
            "unit_price": candidates.get("unit_price", ""),
            "total_price": candidates.get("total_price", ""),
            "labor_unit_price": candidates.get("labor_unit_price", ""),
            "machinery_unit_price": candidates.get("machinery_unit_price", ""),
            "package_rank": candidates.get("package_rank", ""),
            "package_query_similarity": candidates.get("package_query_similarity", ""),
            "item_query_similarity": candidates.get("item_query_similarity", ""),
        }
    )
    for column in EVIDENCE_ITEM_COLUMNS:
        if column not in output.columns:
            output[column] = ""
    output = output[EVIDENCE_ITEM_COLUMNS].where(pd.notna(output), "")
    if len(candidates) != len(output):
        raise ValueError("evidence_items 行数与 candidate pool 不一致")
    missing_family_ids = output["family_id"].map(cell_text).eq("")
    if missing_family_ids.any():
        raise ValueError(f"存在 {int(missing_family_ids.sum())} 条 evidence item 未映射到 family_id")
    return output
