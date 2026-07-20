from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class QueryRewrite:
    raw_query: str
    project_package_query_text: str
    item_query_text: str
    location: str
    start_date: str
    end_date: str
    notes: list[str]
    success: bool


@dataclass(frozen=True)
class ScenarioItem:
    project_package_id: str
    stable_sample_id: str
    source_ref: str
    display_id: str
    practice_option_id: str
    original_option_id: str
    original_family_id: str
    representative_family_id: str
    selection_reason: str
    quantity: dict[str, Any]
    quantity_reason: str
    item_position: int = 0
    quantity_source: str = ""
    quantity_explanation: str = ""
    quantity_sample_count: int | None = None
    quantity_minimum: float | None = None
    quantity_median: float | None = None
    quantity_maximum: float | None = None
    quantity_fallback_used: bool = False
    quantity_fallback_reason: str = ""


@dataclass(frozen=True)
class EstimateScenario:
    scenario_id: str
    scenario_order: int
    scenario_name: str
    scenario_summary: str
    items: list[ScenarioItem]
    site_confirmation: str = ""


@dataclass(frozen=True)
class QueryResult:
    rewrite: QueryRewrite
    estimate_summary: pd.DataFrame
    estimate_scenarios: pd.DataFrame
    matched_project_packages: pd.DataFrame
    candidate_families: pd.DataFrame
    candidate_display_groups: pd.DataFrame
    package_evidence_weights: pd.DataFrame
    display_group_families: pd.DataFrame
    display_option_grouping_trace: pd.DataFrame
    option_selection_trace: pd.DataFrame
    matched_project_examples: pd.DataFrame
    evidence_items: pd.DataFrame
    option_evidence_expansion: pd.DataFrame
    price_evidence_items: pd.DataFrame
    parse_info: pd.DataFrame
    llm_trace: pd.DataFrame
    success: bool = True
    error_message: str = ""
