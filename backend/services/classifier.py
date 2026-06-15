from typing import Dict, List, Sequence

from classifier.catalog_loader import CatalogItem
from classifier.llm_client import llm_select_item
from classifier.rule_engine import (
    HIGH_CONFIDENCE_SCORE,
    MEDIUM_CONFIDENCE_SCORE,
    ScoredCandidate,
    build_reason,
    candidate_ids,
    candidate_labels_by_id,
    confidence_for_score,
    match_type_for_candidates,
    needs_review_for_match,
    score_catalog,
    select_candidate_window,
)
from classifier.settings import (
    DEFAULT_FALLBACK_LEVEL1,
    DEFAULT_FALLBACK_LEVEL2,
)


UNCLEAR_LEVEL3_ITEM = "未明确具体细项"


def _result_from_item(
    text: str,
    item: CatalogItem,
    method: str,
    confidence: str,
    match_type: str,
    needs_review: bool,
    ids: Sequence[str],
    reason: str,
    level3_item: str = "",
    matched_level3_items: Sequence[str] | None = None,
    candidate_level3_items: Sequence[str] | None = None,
) -> Dict[str, object]:
    unique_ids: List[str] = []
    for item_id in ids:
        if item_id not in unique_ids:
            unique_ids.append(item_id)
    if item.id not in unique_ids:
        unique_ids.insert(0, item.id)

    labels = candidate_labels_by_id(unique_ids)
    matched_items = list(matched_level3_items or [])
    primary_level3_item = level3_item or (matched_items[0] if matched_items else UNCLEAR_LEVEL3_ITEM)
    final_confidence = confidence
    final_needs_review = needs_review
    final_reason = reason
    if not matched_items:
        final_needs_review = True
        if final_confidence == "高":
            final_confidence = "中"
        if "未命中具体三级细项" not in final_reason:
            if method == "规则优先":
                final_reason = f"{final_reason}；仅命中二级/对象词，未命中具体三级细项"
            else:
                final_reason = f"{final_reason}；未明确具体三级细项"
    return {
        "project_name": text,
        "level1": item.level1,
        "level2": item.level2,
        "level3_item": primary_level3_item,
        "matched_level3_items": matched_items,
        "method": method,
        "confidence": final_confidence,
        "match_type": match_type,
        "needs_review": final_needs_review,
        "candidate_ids": unique_ids,
        "candidate_labels": [labels[item_id] for item_id in unique_ids if item_id in labels],
        "candidate_level3_items": list(candidate_level3_items or matched_items),
        "reason": final_reason,
    }


def fallback_classify(text: str, reason: str, candidate_ids_value: Sequence[str] | None = None) -> Dict[str, object]:
    ids = list(candidate_ids_value or [])
    labels = candidate_labels_by_id(ids)
    return {
        "project_name": text,
        "level1": DEFAULT_FALLBACK_LEVEL1,
        "level2": DEFAULT_FALLBACK_LEVEL2,
        "level3_item": UNCLEAR_LEVEL3_ITEM,
        "matched_level3_items": [],
        "method": "默认兜底",
        "confidence": "低",
        "match_type": "fallback",
        "needs_review": True,
        "candidate_ids": ids,
        "candidate_labels": [labels[item_id] for item_id in ids if item_id in labels],
        "candidate_level3_items": [],
        "reason": reason,
    }


def _llm_candidates(rule_candidates: Sequence[ScoredCandidate]) -> List[CatalogItem] | None:
    if not rule_candidates:
        return None
    window = select_candidate_window(rule_candidates)
    if window:
        return [candidate.item for candidate in window]
    return [candidate.item for candidate in rule_candidates[:10]]


def _classify_with_llm(text: str, rule_candidates: Sequence[ScoredCandidate]) -> Dict[str, object]:
    ids = candidate_ids(rule_candidates)
    try:
        item, level3_item, reason, _llm_needs_review = llm_select_item(text, _llm_candidates(rule_candidates))
    except Exception as exc:  # noqa: BLE001
        return fallback_classify(text, f"LLM 不可用或返回无效目录，返回默认分类：{exc}", ids)

    return _result_from_item(
        text=text,
        item=item,
        method="LLM兜底",
        confidence="中",
        match_type="llm_fallback",
        needs_review=True,
        ids=[item.id, *ids],
        reason=reason,
        level3_item=level3_item,
        matched_level3_items=[level3_item] if level3_item else [],
        candidate_level3_items=[level3_item] if level3_item else [],
    )


def rule_classify(text: str) -> Dict[str, object] | None:
    candidates = score_catalog(text)
    if not candidates:
        return None

    top = candidates[0]
    if top.score < MEDIUM_CONFIDENCE_SCORE:
        return None

    selected = select_candidate_window(candidates)
    ids = candidate_ids(selected or candidates)
    selected_candidates = selected or candidates[:1]
    candidate_level3_items: List[str] = []
    for candidate in selected_candidates:
        for level3_item in candidate.matched_level3_items:
            if level3_item not in candidate_level3_items:
                candidate_level3_items.append(level3_item)
    confidence = confidence_for_score(top.score)
    match_type = match_type_for_candidates(candidates)
    needs_review = needs_review_for_match(match_type, confidence)
    return _result_from_item(
        text=text,
        item=top.item,
        method="规则优先",
        confidence=confidence,
        match_type=match_type,
        needs_review=needs_review,
        ids=ids,
        reason=build_reason(top),
        matched_level3_items=top.matched_level3_items,
        candidate_level3_items=candidate_level3_items,
    )


def classify_text(text: str) -> Dict[str, object]:
    candidates = score_catalog(text)
    if candidates and candidates[0].score >= HIGH_CONFIDENCE_SCORE:
        result = rule_classify(text)
        if result is not None:
            return result

    if candidates and candidates[0].score >= MEDIUM_CONFIDENCE_SCORE:
        result = rule_classify(text)
        if result is not None and result["match_type"] in {"single", "same_domain_multi_item", "cross_domain"}:
            return result

    return _classify_with_llm(text, candidates)
