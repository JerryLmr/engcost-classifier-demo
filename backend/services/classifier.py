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
    DEFAULT_FALLBACK_LEVEL3,
)


def _result_from_item(
    text: str,
    item: CatalogItem,
    method: str,
    confidence: str,
    match_type: str,
    needs_review: bool,
    ids: Sequence[str],
    reason: str,
) -> Dict[str, object]:
    unique_ids: List[str] = []
    for item_id in ids:
        if item_id not in unique_ids:
            unique_ids.append(item_id)
    if item.id not in unique_ids:
        unique_ids.insert(0, item.id)

    labels = candidate_labels_by_id(unique_ids)
    return {
        "project_name": text,
        "level1": item.level1,
        "level2": item.level2,
        "level3": item.level3,
        "method": method,
        "confidence": confidence,
        "match_type": match_type,
        "needs_review": needs_review,
        "candidate_ids": unique_ids,
        "candidate_labels": [labels[item_id] for item_id in unique_ids if item_id in labels],
        "reason": reason,
    }


def fallback_classify(text: str, reason: str, candidate_ids_value: Sequence[str] | None = None) -> Dict[str, object]:
    ids = list(candidate_ids_value or [])
    labels = candidate_labels_by_id(ids)
    return {
        "project_name": text,
        "level1": DEFAULT_FALLBACK_LEVEL1,
        "level2": DEFAULT_FALLBACK_LEVEL2,
        "level3": DEFAULT_FALLBACK_LEVEL3,
        "method": "默认兜底",
        "confidence": "低",
        "match_type": "fallback",
        "needs_review": True,
        "candidate_ids": ids,
        "candidate_labels": [labels[item_id] for item_id in ids if item_id in labels],
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
        item, reason, _llm_needs_review = llm_select_item(text, _llm_candidates(rule_candidates))
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
