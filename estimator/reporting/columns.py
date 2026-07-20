from __future__ import annotations


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


# Sheet order, QueryResult field mapping, and display conversion policy are
# intentionally defined together as the workbook's single output contract.
WORKBOOK_SHEET_FIELDS = [
    ("estimate_summary", "estimate_summary", True),
    ("estimate_scenarios", "estimate_scenarios", True),
    ("option_evidence_expansion", "option_evidence_expansion", True),
    ("price_evidence_items", "price_evidence_items", True),
    ("candidate_display_groups", "candidate_display_groups", True),
    ("candidate_families", "candidate_families", True),
    ("display_option_grouping_trace", "display_option_grouping_trace", True),
    ("option_selection_trace", "option_selection_trace", True),
    ("matched_project_packages", "matched_project_packages", True),
    ("matched_project_examples", "matched_project_examples", True),
    ("package_evidence_weights", "package_evidence_weights", True),
    ("evidence_items", "evidence_items", True),
    ("parse_info", "parse_info", False),
    ("llm_trace", "llm_trace", False),
]
