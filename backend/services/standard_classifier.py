from typing import Any

from classifier import llm_client
from classifier.alias_matcher import TextAliasResult, match_aliases
from classifier.catalog_postprocess import postprocess_item_selection
from classifier.review_policy import decide_review, fallback_result, selected_result
from classifier.standard_catalog_loader import (
    OUT_OF_SCOPE_ID,
    StandardCatalogItem,
    load_emergency_triggers,
    load_standard_catalog,
)
from classifier.standard_normalizer import NormalizedProjectText, normalize_project_text


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
    alias_result: TextAliasResult,
) -> list[str]:
    hints: list[str] = []
    if alias_result.expanded_terms:
        hints.append(f"alias辅助扩展词（不能直接决定分类）：{'、'.join(alias_result.expanded_terms)}")
    if normalized.action_hints:
        hints.append(f"动作/状态词：{'、'.join(normalized.action_hints)}")
    if normalized.review_hints:
        hints.append(f"复核提示词：{'、'.join(normalized.review_hints)}")
    return hints


def _selected_catalog_result(
    project_name: str,
    normalized: NormalizedProjectText,
    alias_result: TextAliasResult,
    item_selection: llm_client.ItemSelection,
    selected_item: StandardCatalogItem,
) -> dict[str, Any]:
    status_selection = llm_client.llm_select_repair_status(project_name, selected_item)
    decision = decide_review(normalized, alias_result, item_selection, status_selection)
    return selected_result(
        project_name,
        selected_item,
        item_selection,
        status_selection,
        decision,
        is_emergency=is_emergency_project(project_name),
        termite_related=is_termite_related(project_name, selected_item.id),
    )


def _out_of_scope_result(
    project_name: str,
    normalized: NormalizedProjectText,
    alias_result: TextAliasResult,
    item_selection: llm_client.ItemSelection,
) -> dict[str, Any]:
    decision = decide_review(normalized, alias_result, item_selection, None)
    return fallback_result(
        project_name,
        "；".join([item_selection.reason, *decision.reason_suffixes]),
        is_composite=decision.is_composite,
        secondary_catalog_ids=list(decision.secondary_catalog_ids),
        is_emergency=is_emergency_project(project_name),
        termite_related=is_termite_related(project_name, OUT_OF_SCOPE_ID),
    )


def _classify_project_full_catalog(
    project_name: str,
    normalized: NormalizedProjectText,
    alias_result: TextAliasResult,
    context_hints: list[str],
) -> dict[str, Any]:
    catalog = load_standard_catalog()
    catalog_by_id = {item.id: item for item in catalog}
    item_selection = llm_client.llm_select_catalog_item_from_full_catalog(
        project_name,
        catalog,
        context_hints,
    )
    item_selection = postprocess_item_selection(project_name, item_selection, catalog_by_id)
    if item_selection.catalog_id == OUT_OF_SCOPE_ID:
        return _out_of_scope_result(project_name, normalized, alias_result, item_selection)

    selected_item = catalog_by_id.get(item_selection.catalog_id)
    if selected_item is None:
        invalid_selection = llm_client.ItemSelection(
            catalog_id=OUT_OF_SCOPE_ID,
            secondary_catalog_ids=(),
            is_composite=item_selection.is_composite,
            needs_review=True,
            reason=f"LLM returned invalid catalog_id: {item_selection.catalog_id}",
            invalid_after_retry=item_selection.invalid_after_retry,
        )
        return _out_of_scope_result(project_name, normalized, alias_result, invalid_selection)

    return _selected_catalog_result(
        project_name,
        normalized,
        alias_result,
        item_selection,
        selected_item,
    )


def classify_project_standard(project_name: str) -> dict[str, Any]:
    try:
        normalized = normalize_project_text(project_name)
        alias_result = match_aliases(normalized)
        context_hints = _context_hints(normalized, alias_result)
        return _classify_project_full_catalog(project_name, normalized, alias_result, context_hints)
    except Exception as exc:  # noqa: BLE001
        reason = f"分类失败：{exc}"
        return fallback_result(
            project_name,
            reason,
            is_emergency=is_emergency_project(project_name),
            termite_related=is_termite_related(project_name, OUT_OF_SCOPE_ID),
            pipeline_status="llm_service_error" if is_llm_service_error(reason) else "fallback",
        )
