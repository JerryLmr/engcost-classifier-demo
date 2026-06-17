from typing import Any

from classifier import llm_client
from classifier.alias_matcher import TextAliasResult, match_aliases
from classifier.candidate_retriever import (
    FamilyMatch,
    candidate_label,
    detect_family_matches,
    retrieve_candidates,
)
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
FAMILY_FALLBACK_REASON = (
    "LLM returned OUT_OF_SCOPE, but text contains strong in-scope family term; "
    "using family fallback."
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
    alias_result: TextAliasResult,
) -> list[str]:
    hints: list[str] = []
    if normalized.action_hints:
        hints.append(f"动作/状态词：{'、'.join(normalized.action_hints)}")
    if normalized.review_hints:
        hints.append(f"复核提示词：{'、'.join(normalized.review_hints)}")
    return hints


def _family_fallback_item(family_matches: tuple[FamilyMatch, ...]):
    if not family_matches:
        return None
    catalog_by_id = get_standard_catalog_by_id()
    return catalog_by_id.get(family_matches[0].catalog_id)


def classify_project_standard(project_name: str) -> dict[str, Any]:
    candidates = []
    candidate_labels: list[str] = []
    try:
        normalized = normalize_project_text(project_name)
        alias_result = match_aliases(normalized)
        candidates = retrieve_candidates(project_name)
        candidate_items = [candidate.item for candidate in candidates]
        candidate_labels = [candidate_label(item) for item in candidate_items]
        context_hints = _context_hints(normalized, alias_result)
        family_matches = detect_family_matches(normalized)
        if not candidate_items:
            fallback_item = _family_fallback_item(family_matches)
            if fallback_item is not None:
                item_selection = llm_client.ItemSelection(
                    catalog_id=fallback_item.id,
                    secondary_catalog_ids=(),
                    is_composite=False,
                    needs_review=True,
                    reason=FAMILY_FALLBACK_REASON,
                )
                status_selection = llm_client.StatusSelection(
                    repair_status=UNCERTAIN_STATUS,
                    needs_review=True,
                    reason="一级明确但二级未明确，维修状态待复核",
                )
                decision = decide_review(normalized, alias_result, item_selection, status_selection)
                return selected_result(
                    project_name,
                    fallback_item,
                    item_selection,
                    status_selection,
                    candidate_labels,
                    decision,
                    is_emergency=is_emergency_project(project_name),
                    termite_related=is_termite_related(project_name, fallback_item.id),
                )
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
            fallback_item = _family_fallback_item(family_matches)
            if fallback_item is not None:
                item_selection = llm_client.ItemSelection(
                    catalog_id=fallback_item.id,
                    secondary_catalog_ids=(),
                    is_composite=item_selection.is_composite,
                    needs_review=True,
                    reason="；".join([item_selection.reason, FAMILY_FALLBACK_REASON]),
                    invalid_after_retry=item_selection.invalid_after_retry,
                )
                status_selection = llm_client.StatusSelection(
                    repair_status=UNCERTAIN_STATUS,
                    needs_review=True,
                    reason="一级明确但二级未明确，维修状态待复核",
                )
                decision = decide_review(normalized, alias_result, item_selection, status_selection)
                return selected_result(
                    project_name,
                    fallback_item,
                    item_selection,
                    status_selection,
                    candidate_labels,
                    decision,
                    is_emergency=is_emergency_project(project_name),
                    termite_related=is_termite_related(project_name, fallback_item.id),
                )
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
        candidate_ids = {item.id for item in candidate_items}
        selected_item = catalog_by_id[item_selection.catalog_id]
        if item_selection.catalog_id not in candidate_ids:
            return fallback_result(
                project_name,
                f"LLM returned catalog_id outside retrieved candidates: {item_selection.catalog_id}",
                candidate_labels,
                is_composite=item_selection.is_composite,
                secondary_catalog_ids=list(item_selection.secondary_catalog_ids),
                is_emergency=is_emergency_project(project_name),
                termite_related=is_termite_related(project_name, OUT_OF_SCOPE_ID),
            )
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
