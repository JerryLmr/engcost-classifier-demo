from typing import Any

from classifier import llm_client
from classifier.candidate_retriever import candidate_label, retrieve_candidates
from classifier.standard_catalog_loader import (
    OUT_OF_SCOPE_ID,
    StandardCatalogItem,
    get_standard_catalog_by_id,
    load_emergency_triggers,
    load_fallback_config,
)


UNCERTAIN_STATUS = "不确定"
TERMITE_TERMS = ("白蚁", "蚁害", "灭治", "防治")
EMERGENCY_TERMS = (
    "脱落危险",
    "严重渗漏",
    "停水",
    "停电",
    "漏电",
    "冲顶",
    "蹲底",
    "坍塌",
    "堵塞",
    "爆裂",
    "功能故障",
    "安全隐患",
    "必须消除",
)
LLM_SERVICE_ERROR_TERMS = (
    "Connection refused",
    "RemoteDisconnected",
    "Max retries exceeded",
    "Connection aborted",
    "HTTPConnectionPool",
)


def _contains_any(text: str, terms: list[str] | tuple[str, ...]) -> bool:
    compact_text = "".join(str(text or "").split())
    return any("".join(term.split()) in compact_text for term in terms if term)


def is_emergency_project(project_name: str) -> bool:
    return _contains_any(project_name, EMERGENCY_TERMS) or _contains_any(project_name, load_emergency_triggers())


def is_termite_related(project_name: str, catalog_id: str) -> bool:
    return catalog_id == "TERMITE-001" or _contains_any(project_name, TERMITE_TERMS)


def is_llm_service_error(reason: str) -> bool:
    text = str(reason or "")
    return any(term in text for term in LLM_SERVICE_ERROR_TERMS)


def _fallback_result(
    project_name: str,
    reason: str,
    candidate_labels: list[str] | None = None,
) -> dict[str, Any]:
    fallback = load_fallback_config()
    catalog_id = str(fallback.get("id") or OUT_OF_SCOPE_ID)
    pipeline_status = "llm_service_error" if is_llm_service_error(reason) else "fallback"
    return {
        "project_name": project_name,
        "catalog_id": catalog_id,
        "standard_group": "",
        "category": str(fallback.get("category") or "体系外/不确定"),
        "item": str(fallback.get("item") or "未能匹配标准目录"),
        "repair_status": UNCERTAIN_STATUS,
        "is_composite": False,
        "secondary_catalog_ids": [],
        "secondary_catalog_labels": [],
        "is_emergency": is_emergency_project(project_name),
        "termite_related": is_termite_related(project_name, catalog_id),
        "needs_review": True,
        "candidate_labels": candidate_labels or [],
        "reason": reason,
        "pipeline_status": pipeline_status,
    }


def _selected_result(
    project_name: str,
    selected_item: StandardCatalogItem,
    item_selection,
    status_selection,
    candidate_labels: list[str],
) -> dict[str, Any]:
    catalog_by_id = get_standard_catalog_by_id()
    secondary_ids = list(item_selection.secondary_catalog_ids)
    secondary_labels = [
        candidate_label(catalog_by_id[item_id])
        for item_id in secondary_ids
        if item_id in catalog_by_id
    ]
    repair_status = status_selection.repair_status
    invalid_after_retry = item_selection.invalid_after_retry or status_selection.invalid_after_retry
    needs_review = (
        item_selection.needs_review
        or status_selection.needs_review
        or repair_status == UNCERTAIN_STATUS
        or item_selection.is_composite
        or bool(secondary_ids)
        or invalid_after_retry
    )
    return {
        "project_name": project_name,
        "catalog_id": selected_item.id,
        "standard_group": selected_item.standard_group,
        "category": selected_item.category,
        "item": selected_item.item,
        "repair_status": repair_status,
        "is_composite": item_selection.is_composite,
        "secondary_catalog_ids": secondary_ids,
        "secondary_catalog_labels": secondary_labels,
        "is_emergency": is_emergency_project(project_name),
        "termite_related": is_termite_related(project_name, selected_item.id),
        "needs_review": needs_review,
        "candidate_labels": candidate_labels,
        "reason": f"{item_selection.reason}；{status_selection.reason}",
        "pipeline_status": "ok",
    }


def classify_project_standard(project_name: str) -> dict[str, Any]:
    candidates = []
    candidate_labels: list[str] = []
    try:
        candidates = retrieve_candidates(project_name)
        candidate_items = [candidate.item for candidate in candidates]
        candidate_labels = [candidate_label(item) for item in candidate_items]
        if not candidate_items:
            return _fallback_result(project_name, "未召回候选目录", candidate_labels)

        item_selection = llm_client.llm_select_catalog_item(project_name, candidate_items)
        if item_selection.catalog_id == OUT_OF_SCOPE_ID:
            result = _fallback_result(project_name, item_selection.reason, candidate_labels)
            result["is_composite"] = item_selection.is_composite
            result["secondary_catalog_ids"] = list(item_selection.secondary_catalog_ids)
            result["needs_review"] = True
            return result

        catalog_by_id = get_standard_catalog_by_id()
        selected_item = catalog_by_id[item_selection.catalog_id]
        status_selection = llm_client.llm_select_repair_status(project_name, selected_item)
        return _selected_result(project_name, selected_item, item_selection, status_selection, candidate_labels)
    except Exception as exc:  # noqa: BLE001
        return _fallback_result(project_name, f"分类失败：{exc}", candidate_labels)
