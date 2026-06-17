from dataclasses import dataclass
from typing import Sequence

from classifier.alias_matcher import AliasMatchResult
from classifier.candidate_retriever import candidate_label
from classifier.llm_client import ItemSelection, StatusSelection
from classifier.standard_catalog_loader import (
    OUT_OF_SCOPE_ID,
    StandardCatalogItem,
    get_standard_catalog_by_id,
    load_fallback_config,
)
from classifier.standard_normalizer import NormalizedProjectText


UNCERTAIN_STATUS = "不确定"

COMPOSITE_GROUPS = (
    ("屋面", "外墙"),
    ("屋顶", "外墙"),
    ("大堂", "外墙"),
    ("楼道", "外墙"),
    ("监控", "线路"),
    ("水池", "喷泉", "电气"),
    ("道路", "基础"),
    ("面层", "基础"),
)


@dataclass(frozen=True)
class ReviewDecision:
    needs_review: bool
    is_composite: bool
    secondary_catalog_ids: tuple[str, ...]
    reason_suffixes: tuple[str, ...]


def infer_is_composite(text: str) -> bool:
    compact = "".join(str(text or "").split())
    return any(all(term in compact for term in group) for group in COMPOSITE_GROUPS)


def decide_review(
    normalized: NormalizedProjectText,
    alias_result: AliasMatchResult,
    item_selection: ItemSelection,
    status_selection: StatusSelection | None,
) -> ReviewDecision:
    """Decide whether a row truly needs manual review.

    The previous policy treated every secondary candidate and every alias review_hint as
    a hard review trigger. That made normal weak-current / waterproofing rows flood the
    review queue even when the selected catalog item was already clear. Secondary IDs are
    now kept as useful context only; soft alias hints are written only when another hard
    signal already requires review.
    """
    secondary_ids = list(item_selection.secondary_catalog_ids)
    is_composite = item_selection.is_composite or infer_is_composite(normalized.original_text)

    item_needs_review = (
        item_selection.needs_review
        or item_selection.catalog_id == OUT_OF_SCOPE_ID
        or item_selection.invalid_after_retry
    )
    status_needs_review = False
    if status_selection is not None:
        status_needs_review = (
            status_selection.needs_review
            or status_selection.repair_status == UNCERTAIN_STATUS
            or status_selection.invalid_after_retry
        )

    needs_review = (
        item_needs_review
        or status_needs_review
        or is_composite
        or bool(normalized.review_hints)
        or bool(alias_result.negative_hints)
    )

    reason_suffixes: list[str] = []
    if alias_result.negative_hints:
        reason_suffixes.append(f"负向提示：{'、'.join(alias_result.negative_hints)}")
    if normalized.review_hints:
        reason_suffixes.append(f"复核提示：{'、'.join(dict.fromkeys(normalized.review_hints))}")
    if needs_review and alias_result.review_hints and (item_needs_review or status_needs_review):
        reason_suffixes.append(f"复核提示：{'、'.join(dict.fromkeys(alias_result.review_hints))}")
    if is_composite:
        reason_suffixes.append("疑似复合工程")

    return ReviewDecision(
        needs_review=needs_review,
        is_composite=is_composite,
        secondary_catalog_ids=tuple(secondary_ids),
        reason_suffixes=tuple(reason_suffixes),
    )


def _dedupe_reason_parts(parts: Sequence[str]) -> list[str]:
    deduped: list[str] = []
    for part in parts:
        text = str(part or "").strip("； ")
        if not text:
            continue
        if text not in deduped:
            deduped.append(text)
    return deduped


def selected_result(
    project_name: str,
    selected_item: StandardCatalogItem,
    item_selection: ItemSelection,
    status_selection: StatusSelection,
    candidate_labels: Sequence[str],
    decision: ReviewDecision,
    *,
    is_emergency: bool,
    termite_related: bool,
) -> dict[str, object]:
    catalog_by_id = get_standard_catalog_by_id()
    secondary_labels = [
        candidate_label(catalog_by_id[item_id])
        for item_id in decision.secondary_catalog_ids
        if item_id in catalog_by_id
    ]
    reason_parts = _dedupe_reason_parts([item_selection.reason, status_selection.reason, *decision.reason_suffixes])
    return {
        "project_name": project_name,
        "catalog_id": selected_item.id,
        "standard_group": selected_item.standard_group,
        "category": selected_item.category,
        "item": selected_item.item,
        "repair_status": status_selection.repair_status,
        "is_composite": decision.is_composite,
        "secondary_catalog_ids": list(decision.secondary_catalog_ids),
        "secondary_catalog_labels": secondary_labels,
        "is_emergency": is_emergency,
        "termite_related": termite_related,
        "needs_review": decision.needs_review,
        "candidate_labels": list(candidate_labels),
        "reason": "；".join(part for part in reason_parts if part),
        "pipeline_status": "ok",
    }


def fallback_result(
    project_name: str,
    reason: str,
    candidate_labels: Sequence[str] | None = None,
    *,
    is_composite: bool = False,
    secondary_catalog_ids: Sequence[str] | None = None,
    is_emergency: bool = False,
    termite_related: bool = False,
    pipeline_status: str = "fallback",
) -> dict[str, object]:
    fallback = load_fallback_config()
    catalog_id = str(fallback.get("id") or OUT_OF_SCOPE_ID)
    catalog_by_id = get_standard_catalog_by_id()
    secondary_ids = list(secondary_catalog_ids or [])
    secondary_labels = [
        candidate_label(catalog_by_id[item_id])
        for item_id in secondary_ids
        if item_id in catalog_by_id
    ]
    return {
        "project_name": project_name,
        "catalog_id": catalog_id,
        "standard_group": "",
        "category": str(fallback.get("category") or "体系外/不确定"),
        "item": str(fallback.get("item") or "未能匹配标准目录"),
        "repair_status": UNCERTAIN_STATUS,
        "is_composite": is_composite,
        "secondary_catalog_ids": secondary_ids,
        "secondary_catalog_labels": secondary_labels,
        "is_emergency": is_emergency,
        "termite_related": termite_related,
        "needs_review": True,
        "candidate_labels": list(candidate_labels or []),
        "reason": reason,
        "pipeline_status": pipeline_status,
    }
