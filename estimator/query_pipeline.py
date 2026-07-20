from __future__ import annotations

import gc
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from estimator.paths import CLASSIFIER_BACKEND_DIR
from estimator.query_models import EstimateScenario, QueryResult
from estimator.indexing import embedding_model as _embedding_model
from estimator.indexing.index_loader import load_index
from estimator.retrieval import query_rewrite as _query_rewrite
from estimator.retrieval.constraints import build_constraint_mask, ensure_project_package_candidates, filter_rows_and_embeddings, parse_consultation_dates
from estimator.retrieval.evidence_pool import build_retrieved_evidence_items
from estimator.retrieval.item_retrieval import score_direct_items
from estimator.retrieval.package_retrieval import project_package_similarity_map, score_project_packages
from estimator.retrieval.weights import DEFAULT_PACKAGE_WEIGHT_TEMPERATURE, build_package_evidence_weights, evidence_package_universe
from estimator.candidates import practice_options as _practice_options
from estimator.candidates.display_groups import attach_display_support_ratios, build_candidate_display_groups, filter_required_display_groups
from estimator.candidates.families import attach_family_ids_to_evidence_items, build_candidate_families
from estimator.candidates.practice_options import attach_option_support_counts, build_display_option_grouping_trace_frame, display_option_maps
from estimator.candidates.signatures import append_warning, cell_text, json_text, numeric_or_none, trace_row
from estimator.planning import option_selection as _option_selection
from estimator.planning import quantity_determination as _quantity_determination
from estimator.planning import range_selection as _range_selection
from estimator.planning.option_selection import apply_option_selection_to_grouping_trace, attach_family_and_display_ids_to_selected_items, attach_original_practice_options, build_stable_sample_lookup
from estimator.planning.package_selection import build_matched_project_examples, matched_project_examples_frame, matched_project_packages_for_output, select_representative_project_package
from estimator.planning.quantity_determination import attach_quantity_results_to_trace
from estimator.planning.range_selection import expand_selected_project_items
from estimator.planning.scenarios import build_scenario_from_plan_items
from estimator.pricing import evidence_expansion as _evidence_expansion
from estimator.pricing.estimate_calculator import calculate_quantities, format_number_cell, quantity_amounts, quantity_display
from estimator.pricing.evidence_expansion import expand_samples_for_option
from estimator.pricing.price_statistics import price_stats_for_option, validate_price_stats
from estimator.pricing.quantity_statistics import build_quantity_statistics
from estimator.pricing.summary import build_estimate_summary, filter_customer_display_outputs
from estimator.reporting import explanation as _explanation
from estimator.reporting.frames import build_llm_trace_frame, build_parse_info, build_scenario_output_frames

if str(CLASSIFIER_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSIFIER_BACKEND_DIR))
from classifier.llm_client import request_llm_json, request_llm_json_with_usage  # noqa: E402

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


def build_scenario_output_records(
    scenarios: list[EstimateScenario],
    displays_with_options: pd.DataFrame,
    candidate_families: pd.DataFrame,
    samples: pd.DataFrame,
    expanded_family_ids_by_position: dict[int, list[str]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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
    return scenario_rows, price_evidence_rows


def build_scenario_outputs(
    scenarios: list[EstimateScenario],
    displays_with_options: pd.DataFrame,
    candidate_families: pd.DataFrame,
    samples: pd.DataFrame,
    expanded_family_ids_by_position: dict[int, list[str]] | None = None,
    include_internal_positions: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenario_records, price_evidence_records = build_scenario_output_records(
        scenarios,
        displays_with_options,
        candidate_families,
        samples,
        expanded_family_ids_by_position,
    )
    return build_scenario_output_frames(
        scenario_records,
        price_evidence_records,
        include_internal_positions=include_internal_positions,
    )


def run_estimate_query(
    index_dir: Path,
    raw_text: str,
    top_packages: int,
    top_items: int,
    reported_output_path: Path | None,
    max_packages_per_cache_subject: int = 1,
    package_weight_temperature: float = DEFAULT_PACKAGE_WEIGHT_TEMPERATURE,
    include_debug_text: bool = False,
    with_explanations: bool = False,
    *,
    request_json_fn=request_llm_json,
    request_with_usage_fn=request_llm_json_with_usage,
    trace_factory=trace_row,
    warning_fn=append_warning,
) -> QueryResult:
    if package_weight_temperature <= 0:
        raise ValueError("package weight temperature 必须大于 0")
    started_at = datetime.now()
    warnings: list[str] = []
    samples, project_packages, project_package_embeddings, item_embeddings, meta = load_index(index_dir)
    rewrite, rewrite_trace = _query_rewrite.query_rewrite_for_embedding(
        raw_text, request_json=request_json_fn, trace_factory=trace_factory,
    )

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

    model = _embedding_model.load_embedding_model(
        str(meta.get("model") or "BAAI/bge-m3"), device="cpu"
    )
    try:
        package_query_embedding = _embedding_model.encode_query(model, rewrite.project_package_query_text)
        item_query_embedding = _embedding_model.encode_query(model, rewrite.item_query_text)
    except Exception:
        _embedding_model.release_embedding_model(model)
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
        warning_fn(warnings, "direct_item_hits_empty_after_constraints")
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
    _embedding_model.release_embedding_model(model)
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
    start, end, range_meta = _range_selection.select_contiguous_item_range(
        raw_text, selected_project_name, selected_items,
        request_fn=request_with_usage_fn,
    )
    if range_meta["fallback"]:
        warning_fn(warnings, "range_selection_fallback_full_project")
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
    ) = _practice_options.generate_display_option_grouping(
        candidate_display_groups,
        display_group_families,
        candidate_families,
        warnings=warnings,
        request_fn=request_with_usage_fn,
        trace_factory=trace_factory,
        warning_fn=warning_fn,
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
        trace_factory=trace_factory,
    )
    plan_items, option_selection_trace_frame, option_selection_llm_traces = _option_selection.select_final_options(
        raw_text, plan_items, sample_lookup, displays_with_options, candidate_families, warnings, evidence_items,
        request_fn=request_with_usage_fn,
        trace_factory=trace_factory,
        warning_fn=warning_fn,
    )
    display_option_grouping_trace_frame = apply_option_selection_to_grouping_trace(
        display_option_grouping_trace_frame, option_selection_trace_frame
    )
    determinations, quantity_prompt, quantity_trace = _quantity_determination.generate_quantity_determination(
        raw_text,
        selected_project_package_id,
        plan_items,
        warnings,
        request_fn=request_with_usage_fn,
        trace_factory=trace_factory,
        warning_fn=warning_fn,
    )
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
        expanded = expand_samples_for_option(option, display, candidate_families, candidate_samples)
        target_unit = cell_text(row.get("unit_normalized")) or cell_text(row.get("unit"))
        quantity_statistics[position] = build_quantity_statistics(expanded, target_unit)
    quantities = calculate_quantities(
        plan_items, determinations, quantity_statistics, sample_lookup, warnings
    )
    scenario = build_scenario_from_plan_items(
        selected_project_package_id, plan_items, sample_lookup, quantities
    )
    quantity_trace = attach_quantity_results_to_trace(
        quantity_trace, determinations, quantities
    )
    scenarios = [scenario]
    (
        expanded_family_ids_by_position,
        option_evidence_expansion,
        option_evidence_expansion_traces,
    ) = _evidence_expansion.expand_option_price_evidence_families(
        scenario.items,
        displays_with_options,
        candidate_families,
        candidate_samples,
        warnings,
        request_fn=request_with_usage_fn,
        trace_factory=trace_factory,
        warning_fn=warning_fn,
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
    ) = _explanation.generate_customer_explanation(
        with_explanations,
        rewrite.raw_query,
        scenario,
        estimate_scenarios,
        warnings=warnings,
        request_fn=request_with_usage_fn,
        trace_factory=trace_factory,
        warning_fn=warning_fn,
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
        output_path=reported_output_path,
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
    llm_trace = build_llm_trace_frame(
        [
            rewrite_trace,
            *display_option_grouping_llm_traces,
            range_selection_trace,
            *option_selection_llm_traces,
            quantity_trace,
            *option_evidence_expansion_traces,
            final_explanation_trace,
        ]
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
    return result
