from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _normalize_source_row_id(value: Any) -> str:
    text = _cell_text(value)
    if re.fullmatch(r"\d+\.0+", text):
        return text.split(".", 1)[0]
    return text


def _append_warning(warnings: list[str] | None, code: str) -> None:
    if warnings is not None and code not in warnings:
        warnings.append(code)


def source_identity_for_row(
    row: pd.Series,
    warnings: list[str] | None = None,
) -> tuple[str, str, str]:
    project_key = _cell_text(row.get("project_key"))
    if not project_key:
        batch_id = _cell_text(row.get("batch_id"))
        source_row_id = _normalize_source_row_id(row.get("source_row_id"))
        if batch_id and source_row_id:
            project_key = f"{batch_id}::{source_row_id}"
            _append_warning(warnings, "source_ref_recovered_from_batch_source_row")

    item_row_id = _cell_text(row.get("item_row_id"))
    if not item_row_id:
        source_row_id = _normalize_source_row_id(row.get("source_row_id"))
        seq = _cell_text(row.get("seq"))
        if source_row_id and seq:
            item_row_id = f"{source_row_id}-{seq}"
            _append_warning(warnings, "source_ref_recovered_from_source_row_seq")

    source_ref = f"{project_key}::{item_row_id}" if project_key and item_row_id else ""
    if not source_ref:
        _append_warning(warnings, "source_ref_missing")
    return project_key, item_row_id, source_ref


def attach_source_refs(
    rows: pd.DataFrame,
    warnings: list[str] | None = None,
) -> pd.DataFrame:
    if rows.empty:
        return rows
    output = rows.copy()
    identities = output.apply(lambda row: source_identity_for_row(row, warnings), axis=1)
    output["project_key"] = [item[0] for item in identities]
    output["item_row_id"] = [item[1] for item in identities]
    output["source_ref"] = [item[2] for item in identities]
    return output


def matched_package_maps(
    matched_project_packages: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, int]]:
    score_map: dict[str, float] = {}
    rank_map: dict[str, int] = {}
    for _index, row in matched_project_packages.iterrows():
        package_id = _cell_text(row.get("project_package_id"))
        if not package_id:
            continue
        score_map[package_id] = float(row.get("package_query_similarity") or 0.0)
        rank_map[package_id] = int(row.get("rank") or 0)
    return score_map, rank_map


def build_retrieved_evidence_items(
    samples: pd.DataFrame,
    matched_project_packages: pd.DataFrame,
    direct_item_hits: pd.DataFrame,
    item_query_similarities: np.ndarray,
    package_query_similarity_by_id: dict[str, float] | None = None,
    warnings: list[str] | None = None,
) -> pd.DataFrame:
    package_query_similarity_map, package_rank_map = matched_package_maps(matched_project_packages)
    package_query_similarity_by_id = package_query_similarity_by_id or package_query_similarity_map
    matched_package_ids = list(package_query_similarity_map.keys())
    direct_indices = {
        int(index)
        for index in pd.to_numeric(direct_item_hits.get("sample_index", pd.Series(dtype=int)), errors="coerce").dropna()
    }

    package_rows = samples[samples["project_package_id"].astype(str).isin(matched_package_ids)].copy()
    candidate_indices = set(pd.to_numeric(package_rows["sample_index"], errors="coerce").dropna().astype(int).tolist())
    candidate_indices.update(direct_indices)
    if not candidate_indices:
        return samples.head(0).copy()

    sample_index_series = pd.to_numeric(samples["sample_index"], errors="coerce").astype("Int64")
    rows = samples[sample_index_series.isin(candidate_indices)].copy()
    rows["sample_index"] = pd.to_numeric(rows["sample_index"], errors="raise").astype(int)
    if rows["sample_index"].min() < 0 or rows["sample_index"].max() >= len(item_query_similarities):
        raise ValueError("sample_index 超出 item_embeddings 范围")

    rows["package_query_similarity"] = rows["project_package_id"].map(package_query_similarity_by_id).fillna(0.0).astype(float)
    rows["package_rank"] = rows["project_package_id"].map(package_rank_map)
    rows["item_query_similarity"] = rows["sample_index"].map(lambda sample_index: float(item_query_similarities[int(sample_index)]))
    rows["direct_hit"] = rows["sample_index"].isin(direct_indices)
    rows = attach_source_refs(rows, warnings)
    sort_columns = ["item_query_similarity", "package_query_similarity"]
    return rows.sort_values(sort_columns, ascending=[False, False]).reset_index(drop=True)
