from typing import Any, Dict, List


DISPLAY_MAPPING = {
    "compliant": "初步符合",
    "non_compliant": "疑似违规",
    "need_supplement": "需补充材料",
    "manual_review": "建议人工复核",
}

SUB_AUDIT_KEYS = (
    "scope_audit",
    "process_audit",
    "document_completeness_audit",
    "timeline_audit",
    "amount_audit",
    "emergency_audit",
)


def _empty_sub_audit() -> Dict[str, Any]:
    return {
        "applicable": False,
        "result": None,
        "display_result": None,
        "reason_codes": [],
        "reasons": [],
        "missing_items": [],
        "basis_documents": [],
        "audit_path": [],
        "facts_used": [],
    }


def _build_default_sub_audits() -> Dict[str, Dict[str, Any]]:
    return {key: _empty_sub_audit() for key in SUB_AUDIT_KEYS}


def build_high_freq_result(
    project_name: str,
    high_freq_result: Dict[str, Any],
    action: str,
) -> Dict[str, Any]:
    overall_result = high_freq_result.get("default_result")
    if action == "route_to_manual_review":
        overall_result = "manual_review"
    elif action == "route_to_need_supplement":
        overall_result = "need_supplement"
    elif action == "direct_reject":
        overall_result = "non_compliant"

    display_result = DISPLAY_MAPPING.get(overall_result, DISPLAY_MAPPING["manual_review"])
    reason_codes = list(high_freq_result.get("reason_codes", []))
    reasons: List[str] = [high_freq_result.get("business_statement", "")] if high_freq_result.get("business_statement") else []
    summary_message = reasons[0] if reasons else display_result
    summary_conclusion = {
        "type": "high_freq_routed",
        "scope_prelim_pass": False,
        "conflict_detected": False,
        "gap_categories": [],
        "primary_message": summary_message,
        "display_summary": summary_message,
    }

    return {
        "project_name": project_name,
        "mapped_objects": [],
        "matched_object_ids": [],
        "normalized_tags": list(high_freq_result.get("tags", [])),
        "overall_result": overall_result,
        "display_result": display_result,
        "reason_codes": reason_codes,
        "reasons": reasons,
        "basis_documents": [],
        "missing_items": [],
        "audit_path": ["input_normalization", "high_freq_mapping", action],
        "manual_review_required": overall_result == "manual_review",
        "sub_audits": _build_default_sub_audits(),
        "document_extraction_targets": {},
        "summary_conclusion": summary_conclusion,
        "display_summary": summary_message,
    }
