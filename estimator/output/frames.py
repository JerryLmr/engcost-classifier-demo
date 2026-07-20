from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from estimator.candidates.signatures import cell_text, json_text
from estimator.output.columns import (
    ESTIMATE_SCENARIO_COLUMNS,
    LLM_TRACE_COLUMNS,
    PRICE_EVIDENCE_ITEM_COLUMNS,
    WORKBOOK_SHEET_FIELDS,
)
from estimator.query_models import QueryResult, QueryRewrite


EXCEL_DISPLAY_COLUMN_LABELS = {
    "retrieval_package_support_ratio": "本次召回工程包支持比例",
    "retrieval_item_count": "本次召回清单行数",
    "retrieval_package_count": "本次召回工程包数",
    "support_rank": "本次召回支持度排序",
}


TEXT_IDENTIFIER_COLUMNS = {
    "scenario_id", "display_id", "family_id", "project_package_id", "project_key",
    "item_key", "stable_sample_id", "source_ref", "catalog_id", "batch_id",
    "source_row_id", "item_row_id", "project_code",
}


def is_text_identifier_column(column: Any) -> bool:
    name = cell_text(column)
    return name in TEXT_IDENTIFIER_COLUMNS or name.endswith("_id") or name.endswith("_ids")


def display_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        if is_text_identifier_column(column):
            output[column] = output[column].map(cell_text)
    return output.rename(columns=EXCEL_DISPLAY_COLUMN_LABELS)


def build_scenario_output_frames(
    scenario_rows: list[dict[str, Any]],
    price_evidence_rows: list[dict[str, Any]],
    *,
    include_internal_positions: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario_columns = (
        ["final_item_position", *ESTIMATE_SCENARIO_COLUMNS]
        if include_internal_positions else ESTIMATE_SCENARIO_COLUMNS
    )
    return (
        pd.DataFrame(scenario_rows, columns=scenario_columns).fillna(""),
        pd.DataFrame(price_evidence_rows, columns=PRICE_EVIDENCE_ITEM_COLUMNS).fillna(""),
    )


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
        ("quantity_determination_status", quantity_determination_status(quantity_trace)),
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


def build_llm_trace_frame(traces: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(traces, columns=LLM_TRACE_COLUMNS)


def build_workbook_frames(result: QueryResult) -> dict[str, pd.DataFrame]:
    return {
        sheet_name: display_frame(getattr(result, field_name)) if use_display_frame
        else getattr(result, field_name)
        for sheet_name, field_name, use_display_frame in WORKBOOK_SHEET_FIELDS
    }
