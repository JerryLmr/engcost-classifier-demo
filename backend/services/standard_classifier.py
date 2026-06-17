from typing import Any

from classifier import llm_client
from classifier.alias_matcher import AliasMatchResult, match_aliases
from classifier.candidate_retriever import candidate_label, retrieve_candidates
from classifier.review_policy import decide_review, fallback_result, selected_result
from classifier.standard_catalog_loader import (
    OUT_OF_SCOPE_ID,
    get_standard_catalog_by_id,
    load_emergency_triggers,
)
from classifier.standard_normalizer import NormalizedProjectText, normalize_project_text


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


def _context_hints(
    normalized: NormalizedProjectText,
    alias_result: AliasMatchResult,
    candidate_reasons: list[str],
) -> list[str]:
    hints: list[str] = []
    if normalized.action_hints:
        hints.append(f"动作/状态词：{'、'.join(normalized.action_hints)}")
    if normalized.review_hints:
        hints.append(f"复核提示词：{'、'.join(normalized.review_hints)}")
    for hit in alias_result.catalog_hits:
        hints.append(f"{hit.reason} -> {hit.catalog_id}")
    if alias_result.negative_hints:
        hints.append(f"负向提示词：{'、'.join(alias_result.negative_hints)}")
    hints.extend(reason for reason in candidate_reasons if reason)
    return hints


def classify_project_standard(project_name: str) -> dict[str, Any]:
    candidates = []
    candidate_labels: list[str] = []
    try:
        normalized = normalize_project_text(project_name)
        alias_result = match_aliases(normalized)
        candidates = retrieve_candidates(project_name)
        candidate_items = [candidate.item for candidate in candidates]
        candidate_labels = [candidate_label(item) for item in candidate_items]
        candidate_reasons = [candidate.reason for candidate in candidates if candidate.reason]
        context_hints = _context_hints(normalized, alias_result, candidate_reasons)
        if not candidate_items:
            return fallback_result(
                project_name,
                "未召回候选目录",
                candidate_labels,
                is_composite=False,
                is_emergency=is_emergency_project(project_name),
                termite_related=is_termite_related(project_name, OUT_OF_SCOPE_ID),
            )

        item_selection = llm_client.llm_select_catalog_item(project_name, candidate_items, context_hints)
        if item_selection.catalog_id == OUT_OF_SCOPE_ID:
            decision = decide_review(normalized, alias_result, item_selection, None)
            return fallback_result(
                project_name,
                "；".join([item_selection.reason, *decision.reason_suffixes]),
                candidate_labels,
                is_composite=decision.is_composite,
                secondary_catalog_ids=list(decision.secondary_catalog_ids),
                is_emergency=is_emergency_project(project_name),
                termite_related=is_termite_related(project_name, OUT_OF_SCOPE_ID),
            )

        catalog_by_id = get_standard_catalog_by_id()
        selected_item = catalog_by_id[item_selection.catalog_id]
        status_selection = llm_client.llm_select_repair_status(project_name, selected_item)
        decision = decide_review(normalized, alias_result, item_selection, status_selection)
        return selected_result(
            project_name,
            selected_item,
            item_selection,
            status_selection,
            candidate_labels,
            decision,
            is_emergency=is_emergency_project(project_name),
            termite_related=is_termite_related(project_name, selected_item.id),
        )
    except Exception as exc:  # noqa: BLE001
        reason = f"分类失败：{exc}"
        return fallback_result(
            project_name,
            reason,
            candidate_labels,
            is_emergency=is_emergency_project(project_name),
            termite_related=is_termite_related(project_name, OUT_OF_SCOPE_ID),
            pipeline_status="llm_service_error" if is_llm_service_error(reason) else "fallback",
        )
