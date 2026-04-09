from typing import Any, Dict

from services.audit_service import audit_project
from services.high_freq_mapping_service import match_high_freq_category
from services.mapping_service import map_project_name
from services.result_builder import build_high_freq_result


def run_audit_pipeline(payload: Dict[str, Any]) -> Dict[str, Any]:
    project_name = payload.get("project_name", "")
    high_freq = match_high_freq_category(project_name)
    action = high_freq.get("action", "no_match")

    if high_freq.get("matched") and action in {
        "direct_reject",
        "route_to_manual_review",
        "route_to_need_supplement",
    }:
        return build_high_freq_result(project_name=project_name, high_freq_result=high_freq, action=action)

    mapping_result = map_project_name(project_name)
    result = audit_project(payload, mapping_result)
    if high_freq.get("matched"):
        result["audit_path"] = ["input_normalization", "high_freq_mapping", action, *result.get("audit_path", [])]
    else:
        result["audit_path"] = ["input_normalization", "high_freq_mapping", "no_match", *result.get("audit_path", [])]
    return result
