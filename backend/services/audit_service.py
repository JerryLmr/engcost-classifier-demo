import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from core.config import RULE_CONFIG_DIR


FLOW_BY_STAGE = {
    "INPUT_CHECK": "input_check_flow",
    "CATALOG_CHECK": "input_check_flow",
    "EXCLUSION_CHECK": "exclusion_flow",
    "GRAY_CASE_ROUTING": "manual_review_flow",
    "GRAY_CASE_REVIEW_CHECK": "gray_case_review_flow",
    "EMERGENCY_CHECK": "emergency_flow",
    "NORMAL_SCOPE_CHECK": "normal_flow",
    "PROCESS_CHECK": "normal_flow",
}
CONFIDENCE_ORDER = {"low": 1, "medium": 2, "high": 3}
STRUCTURED_INPUT_GROUPS = (
    "scope_facts",
    "process_facts",
    "document_facts",
    "timeline_facts",
    "amount_facts",
    "emergency_facts",
    "gray_case_facts",
)
NON_FACT_FIELDS = {
    "project_name",
    "project_desc",
    "matched_object_ids",
    "normalized_tags",
    "mapping_confidence",
    "gray_case_type",
    "split_projects",
    "catalog_domains",
    "source_documents",
    "extracted_document_fields",
    "field_confidence_map",
}


def normalize_text(text: str) -> str:
    return "".join((text or "").split())


def simplify_text(text: str) -> str:
    simplified = normalize_text(text)
    for token in ("系统", "工程", "项目", "事项", "对象", "设施", "设备"):
        simplified = simplified.replace(token, "")
    return simplified


@lru_cache(maxsize=1)
def load_rule_mapping() -> Dict[str, Any]:
    path = Path(RULE_CONFIG_DIR) / "rule_mapping.json"
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


@lru_cache(maxsize=1)
def load_rule_engine() -> Dict[str, Any]:
    path = Path(RULE_CONFIG_DIR) / "rule_engine.json"
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


@lru_cache(maxsize=1)
def load_output_schema() -> Dict[str, Any]:
    path = Path(RULE_CONFIG_DIR) / "output_schema.json"
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = normalize_text(text)
    normalized_phrase = normalize_text(phrase)
    if normalized_phrase in normalized_text:
        return True
    return simplify_text(normalized_phrase) in simplify_text(normalized_text)


def _score_mapping_entry(text: str, entry: Dict[str, Any]) -> Tuple[int, List[str]]:
    hits: List[str] = []
    score = 0
    for phrase in entry.get("match_text", []):
        if _contains_phrase(text, phrase):
            hits.append(phrase)
            score += max(2, len(simplify_text(phrase)))
    return score, hits


def _select_mapping_entries(project_name: str, split_projects: Sequence[str]) -> List[Dict[str, Any]]:
    mapping_config = load_rule_mapping()
    parts = list(split_projects) or [project_name]
    selected: List[Dict[str, Any]] = []
    seen_ids: Set[str] = set()

    for part in parts:
        best_entry: Optional[Dict[str, Any]] = None
        best_score = 0
        best_hits: List[str] = []
        for entry in mapping_config["mappings"]:
            score, hits = _score_mapping_entry(part, entry)
            if score > best_score or (score == best_score and len(hits) > len(best_hits)):
                best_entry = entry
                best_score = score
                best_hits = hits
        if best_entry is not None and best_score > 0 and best_entry["mapping_id"] not in seen_ids:
            selected.append({**best_entry, "_hits": best_hits})
            seen_ids.add(best_entry["mapping_id"])

    if selected:
        return selected

    all_candidates: List[Tuple[int, int, Dict[str, Any], List[str]]] = []
    for entry in mapping_config["mappings"]:
        score, hits = _score_mapping_entry(project_name, entry)
        if score <= 0:
            continue
        all_candidates.append((score, len(hits), entry, hits))
    all_candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    for score, _hit_count, entry, hits in all_candidates[:2]:
        if score <= 0 or entry["mapping_id"] in seen_ids:
            continue
        selected.append({**entry, "_hits": hits})
        seen_ids.add(entry["mapping_id"])
    return selected


def _infer_tags_from_catalog(mapped_objects: Sequence[Dict[str, Any]], project_name: str) -> Tuple[List[str], Optional[str], str]:
    tags: List[str] = []
    gray_case_type: Optional[str] = None
    confidence = "low"
    normalized_text = normalize_text(project_name)

    def add_tags(values: Iterable[str]) -> None:
        for value in values:
            if value not in tags:
                tags.append(value)

    for item in mapped_objects:
        path = item["full_path"]
        if path.startswith("电梯/"):
            add_tags(["repairable_object", "shared_facility"])
            confidence = "medium"
        elif path.startswith("消防系统/") or path.startswith("消防泵/"):
            add_tags(["repairable_object", "shared_facility"])
            confidence = "medium"
        elif path.startswith("排水、排污设施/") or path.startswith("供水系统/"):
            add_tags(["repairable_object", "shared_facility"])
            if any(keyword in normalized_text for keyword in ("爆裂", "堵塞", "故障", "抢修")):
                add_tags(["emergency_scope"])
            confidence = "medium"
        elif path.startswith("楼栋外立面/") or "屋面" in path or "屋顶" in path:
            add_tags(["repairable_object", "shared_part"])
            confidence = "medium"
        elif "公共窗户" in path and "玻璃" in path:
            add_tags(["gray_case"])
            gray_case_type = "weak"
            confidence = "medium"

    return tags, gray_case_type, confidence


def build_normalized_tags(project_name: str, mapping_result: Dict[str, Any]) -> Dict[str, Any]:
    selected_entries = _select_mapping_entries(project_name, mapping_result.get("split_projects", []))
    normalized_tags: List[str] = []
    gray_case_type: Optional[str] = None
    mapping_confidence = "low"
    matched_mapping_ids: List[str] = []

    for entry in selected_entries:
        matched_mapping_ids.append(entry["mapping_id"])
        for tag in entry.get("normalized_tags", []):
            if tag not in normalized_tags:
                normalized_tags.append(tag)
        confidence = entry.get("mapping_confidence", "low")
        if CONFIDENCE_ORDER[confidence] > CONFIDENCE_ORDER[mapping_confidence]:
            mapping_confidence = confidence
        entry_gray_case_type = entry.get("gray_case_type")
        if entry_gray_case_type == "strong":
            gray_case_type = "strong"
        elif entry_gray_case_type == "weak" and gray_case_type is None:
            gray_case_type = "weak"

    if not normalized_tags and mapping_result.get("mapped_objects"):
        inferred_tags, inferred_gray_case_type, inferred_confidence = _infer_tags_from_catalog(
            mapping_result["mapped_objects"],
            project_name,
        )
        for tag in inferred_tags:
            if tag not in normalized_tags:
                normalized_tags.append(tag)
        gray_case_type = gray_case_type or inferred_gray_case_type
        mapping_confidence = inferred_confidence

    catalog_domains = mapping_result.get("catalog_domains", [])
    if len(mapping_result.get("split_projects", [])) > 1 or len(catalog_domains) > 1:
        if "multi_project" not in normalized_tags:
            normalized_tags.insert(0, "multi_project")

    if not normalized_tags and not mapping_result.get("matched_object_ids"):
        normalized_tags.append("unknown")

    return {
        "normalized_tags": normalized_tags,
        "gray_case_type": gray_case_type,
        "mapping_confidence": mapping_confidence,
        "matched_mapping_ids": matched_mapping_ids,
    }


def _merge_structured_request(request_payload: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(request_payload)
    for group_name in STRUCTURED_INPUT_GROUPS:
        group_value = request_payload.get(group_name)
        if not isinstance(group_value, dict):
            continue
        for field, value in group_value.items():
            if merged.get(field) is None:
                merged[field] = value

    parse_context = request_payload.get("document_parse_context")
    if isinstance(parse_context, dict):
        for field in ("source_documents", "extracted_document_fields", "field_confidence_map"):
            if merged.get(field) is None:
                merged[field] = parse_context.get(field)
    return merged


def _build_rule_context(
    request_payload: Dict[str, Any],
    mapping_result: Dict[str, Any],
    tag_result: Dict[str, Any],
) -> Dict[str, Any]:
    normalized_payload = _merge_structured_request(request_payload)
    context = {
        "project_name": normalized_payload.get("project_name"),
        "project_desc": normalized_payload.get("project_desc"),
        "matched_object_ids": mapping_result.get("matched_object_ids", []),
        "normalized_tags": tag_result.get("normalized_tags", []),
        "mapping_confidence": tag_result.get("mapping_confidence"),
        "gray_case_type": tag_result.get("gray_case_type"),
        "split_projects": mapping_result.get("split_projects", []),
        "catalog_domains": mapping_result.get("catalog_domains", []),
        "source_documents": normalized_payload.get("source_documents", []),
        "extracted_document_fields": normalized_payload.get("extracted_document_fields", {}),
        "field_confidence_map": normalized_payload.get("field_confidence_map", {}),
    }

    for field in load_rule_engine()["input_schema"]["optional_fields"]:
        if field not in context:
            context[field] = normalized_payload.get(field)

    return context


def _evaluate_condition(condition: Dict[str, Any], context: Dict[str, Any]) -> bool:
    if "all" in condition:
        return all(_evaluate_condition(item, context) for item in condition["all"])
    if "any" in condition:
        return any(_evaluate_condition(item, context) for item in condition["any"])

    field = condition["field"]
    op = condition["op"]
    value = condition.get("value")
    current = context.get(field)

    if op == "contains":
        return isinstance(current, Sequence) and not isinstance(current, (str, bytes)) and value in current
    if op == "length_gt":
        return isinstance(current, Sequence) and not isinstance(current, (str, bytes)) and len(current) > value
    if op == "empty_array":
        return isinstance(current, Sequence) and not isinstance(current, (str, bytes)) and len(current) == 0
    if op == "empty_or_null":
        return current is None or current == ""
    if op == "not_empty_or_null":
        return current is not None and current != ""
    if op == "not_eq":
        return current != value
    if op == "is_false":
        return current is False
    if op == "eq":
        return current == value
    if op == "gte":
        return current is not None and current >= value
    if op == "empty_after_strip_terms":
        if current is None:
            return True
        reduced = normalize_text(str(current))
        for term in condition.get("terms", []):
            reduced = reduced.replace(normalize_text(str(term)), "")
        reduced = reduced.strip(condition.get("strip_chars", "-_/"))
        return len(reduced) == 0
    raise ValueError(f"不支持的规则操作: {op}")


def _collect_missing_items(condition: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
    missing_items: List[str] = []
    if "all" in condition or "any" in condition:
        key = "all" if "all" in condition else "any"
        for item in condition[key]:
            missing_items.extend(_collect_missing_items(item, context))
        return missing_items

    if condition["op"] == "eq" and condition.get("value") is False and context.get(condition["field"]) is False:
        return [condition["field"]]
    if condition["op"] == "empty_or_null" and (context.get(condition["field"]) is None or context.get(condition["field"]) == ""):
        return [condition["field"]]
    if condition["op"] == "empty_after_strip_terms" and _evaluate_condition(condition, context):
        return [condition["field"]]
    return []


def _append_unique(items: List[str], values: Iterable[str]) -> None:
    for value in values:
        if value not in items:
            items.append(value)


def _build_basis_documents(triggered_rules: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    basis_documents: List[Dict[str, Any]] = []
    seen: Set[Tuple[Any, ...]] = set()
    for rule in triggered_rules:
        source = rule.get("source")
        if not source:
            continue
        key = (
            source.get("display_name"),
            source.get("title"),
            source.get("article"),
            source.get("section"),
        )
        if key in seen:
            continue
        seen.add(key)
        basis_documents.append(
            {
                "display_name": source.get("display_name"),
                "source_type": source.get("source_type"),
                "title": source.get("title"),
                "issuer": source.get("issuer"),
                "document_no": source.get("document_no"),
                "article": source.get("article"),
                "section": source.get("section"),
            }
        )
    return basis_documents


def _build_display_result(overall_result: str) -> str:
    return load_output_schema()["display_mapping"][overall_result]


def _get_ordered_active_rules(engine: Dict[str, Any]) -> List[Dict[str, Any]]:
    stage_order = {stage: index for index, stage in enumerate(engine["decision_flow"])}
    active_rules = [rule for rule in engine["rules"] if rule.get("is_active", True)]
    return sorted(active_rules, key=lambda rule: (stage_order.get(rule["stage"], 999), rule["priority"]))


def _get_rule_dimensions(rule: Dict[str, Any]) -> List[str]:
    dimensions = rule.get("audit_dimensions", [])
    if isinstance(dimensions, str):
        return [dimensions]
    return list(dimensions)


def _is_business_fact_field(field_name: str) -> bool:
    return field_name not in NON_FACT_FIELDS


def _collect_present_facts(context: Dict[str, Any], field_names: Sequence[str]) -> List[str]:
    present: List[str] = []
    for field_name in field_names:
        if not _is_business_fact_field(field_name):
            continue
        value = context.get(field_name)
        if value is None or value == "":
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 0:
            continue
        present.append(field_name)
    return present


def _collect_condition_fields(condition: Dict[str, Any]) -> List[str]:
    fields: List[str] = []
    if "all" in condition or "any" in condition:
        key = "all" if "all" in condition else "any"
        for item in condition[key]:
            _append_unique(fields, _collect_condition_fields(item))
        return fields

    field_name = condition.get("field")
    if field_name and _is_business_fact_field(field_name):
        fields.append(field_name)
    return fields


def _collect_required_missing_fields(context: Dict[str, Any], field_names: Sequence[str]) -> List[str]:
    missing: List[str] = []
    for field_name in field_names:
        value = context.get(field_name)
        if value is None or value == "":
            missing.append(field_name)
    return missing


def _build_sub_audit_result(
    sub_audit_key: str,
    definition: Dict[str, Any],
    context: Dict[str, Any],
    ordered_rules: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    applicable_when = definition.get("applicable_when")
    applicable = True if not applicable_when else _evaluate_condition(applicable_when, context)
    facts_used = _collect_present_facts(context, definition.get("facts_used_fields", []))

    if not applicable:
        return {
            "applicable": False,
            "result": None,
            "display_result": None,
            "reason_codes": [],
            "reasons": [],
            "missing_items": [],
            "basis_documents": [],
            "audit_path": [],
            "facts_used": facts_used,
        }

    for rule in ordered_rules:
        if sub_audit_key not in _get_rule_dimensions(rule):
            continue
        if not _evaluate_condition(rule["when"], context):
            continue
        then = rule["then"]
        stage_path = FLOW_BY_STAGE.get(rule["stage"], rule["stage"].lower())
        rule_facts_used = list(facts_used)
        _append_unique(rule_facts_used, _collect_condition_fields(rule["when"]))
        if then["result"] == "continue":
            sub_audit_result = then.get("sub_audit_result")
            if not sub_audit_result:
                continue
            return {
                "applicable": True,
                "result": sub_audit_result,
                "display_result": _build_display_result(sub_audit_result),
                "reason_codes": list(then.get("reason_codes", [])),
                "reasons": [then["message"]] if then.get("message") else [],
                "missing_items": [],
                "basis_documents": _build_basis_documents([rule]),
                "audit_path": [stage_path, sub_audit_key],
                "facts_used": rule_facts_used,
            }
        return {
            "applicable": True,
            "result": then["result"],
            "display_result": _build_display_result(then["result"]),
            "reason_codes": list(then.get("reason_codes", [])),
            "reasons": [then["message"]],
            "missing_items": _collect_missing_items(rule["when"], context),
            "basis_documents": _build_basis_documents([rule]),
            "audit_path": [stage_path, sub_audit_key],
            "facts_used": rule_facts_used,
        }

    default_result = definition.get("default_result")
    return {
        "applicable": True,
        "result": default_result,
        "display_result": _build_display_result(default_result) if default_result else None,
        "reason_codes": list(definition.get("default_reason_codes", [])),
        "reasons": [definition["default_message"]] if definition.get("default_message") else [],
        "missing_items": _collect_required_missing_fields(context, definition.get("required_fields", [])),
        "basis_documents": [],
        "audit_path": list(definition.get("default_audit_path", [])),
        "facts_used": facts_used,
    }


def _build_sub_audits(context: Dict[str, Any], ordered_rules: Sequence[Dict[str, Any]], engine: Dict[str, Any]) -> Dict[str, Any]:
    sub_audits: Dict[str, Any] = {}
    for sub_audit_key, definition in engine.get("sub_audit_definitions", {}).items():
        sub_audits[sub_audit_key] = _build_sub_audit_result(sub_audit_key, definition, context, ordered_rules)
    return sub_audits


def audit_project(request_payload: Dict[str, Any], mapping_result: Dict[str, Any]) -> Dict[str, Any]:
    normalized_payload = _merge_structured_request(request_payload)
    project_name = normalized_payload.get("project_name", "")
    tag_result = build_normalized_tags(project_name, mapping_result)
    context = _build_rule_context(normalized_payload, mapping_result, tag_result)

    engine = load_rule_engine()
    ordered_rules = _get_ordered_active_rules(engine)
    sub_audits = _build_sub_audits(context, ordered_rules, engine)

    audit_path: List[str] = ["mapping", "tag_mapping"]
    triggered_rules: List[Dict[str, Any]] = []

    for rule in ordered_rules:
        if not _evaluate_condition(rule["when"], context):
            continue
        triggered_rules.append(rule)
        _append_unique(audit_path, [FLOW_BY_STAGE.get(rule["stage"], rule["stage"].lower())])
        then = rule["then"]
        if then["result"] == "continue":
            route_to = then.get("route_to")
            if route_to == "NORMAL_SCOPE_CHECK":
                _append_unique(audit_path, ["normal_flow"])
            continue

        result = then["result"]
        reason_codes = list(then.get("reason_codes", []))
        reasons = [then["message"]]
        missing_items = _collect_missing_items(rule["when"], context)
        return {
            "project_name": project_name,
            "mapped_objects": mapping_result.get("mapped_objects", []),
            "matched_object_ids": mapping_result.get("matched_object_ids", []),
            "normalized_tags": context["normalized_tags"],
            "overall_result": result,
            "display_result": _build_display_result(result),
            "reason_codes": reason_codes,
            "reasons": reasons,
            "basis_documents": _build_basis_documents(triggered_rules),
            "missing_items": missing_items,
            "audit_path": audit_path,
            "manual_review_required": result == "manual_review",
            "sub_audits": sub_audits,
            "document_extraction_targets": engine.get("document_extraction_targets", {}),
        }

    raise ValueError("rule_engine 未命中任何终态规则，请检查规则配置")
