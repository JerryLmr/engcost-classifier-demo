from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from classifier.catalog_loader import CatalogItem, load_catalog
from classifier.text_normalizer import normalize_text


HIGH_CONFIDENCE_SCORE = 6
MEDIUM_CONFIDENCE_SCORE = 3
CLOSE_SCORE_GAP = 2
MAX_CANDIDATES = 5
MIN_SPLIT_PHRASE_LENGTH = 2


@dataclass(frozen=True)
class ScoredCandidate:
    item: CatalogItem
    score: int
    level2_hit: bool
    level3_hits: tuple[str, ...]
    object_hits: tuple[str, ...]
    action_hits: tuple[str, ...]
    weak_hits: tuple[str, ...]

    @property
    def id(self) -> str:
        return self.item.id


def _split_phrases(value: str) -> List[str]:
    separators = [",", "，", "、", ";", "；"]
    phrases = [value]
    for separator in separators:
        next_phrases: List[str] = []
        for phrase in phrases:
            next_phrases.extend(phrase.split(separator))
        phrases = next_phrases
    return [phrase.strip() for phrase in phrases if len(phrase.strip()) >= MIN_SPLIT_PHRASE_LENGTH]


def _hits(text: str, keywords: Iterable[str]) -> tuple[str, ...]:
    seen: List[str] = []
    for keyword in keywords:
        phrases = [keyword, *_split_phrases(keyword)]
        matched = any(normalize_text(phrase) and normalize_text(phrase) in text for phrase in phrases)
        if matched and keyword not in seen:
            seen.append(keyword)
    return tuple(seen)


def score_item(text: str, item: CatalogItem) -> ScoredCandidate:
    level2_hit = any(
        normalize_text(phrase) and normalize_text(phrase) in text
        for phrase in [item.level2, *_split_phrases(item.level2)]
    )
    level3_hits = _hits(text, _split_phrases(item.level3))
    object_hits = _hits(text, item.object_keywords)
    action_hits = _hits(text, item.action_keywords)
    weak_hits = _hits(text, item.weak_keywords)

    score = 0
    if level2_hit:
        score += 4
    if level3_hits:
        score += 4
    score += len(object_hits) * 3
    score += len(action_hits) * 2
    score += len(weak_hits)

    return ScoredCandidate(
        item=item,
        score=score,
        level2_hit=level2_hit,
        level3_hits=level3_hits,
        object_hits=object_hits,
        action_hits=action_hits,
        weak_hits=weak_hits,
    )


def score_catalog(text: str) -> List[ScoredCandidate]:
    normalized = normalize_text(text)
    candidates = [score_item(normalized, item) for item in load_catalog()]
    candidates = [candidate for candidate in candidates if candidate.score > 0]
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.score,
            bool(candidate.level3_hits),
            bool(candidate.object_hits),
            len(candidate.level3_hits) + len(candidate.object_hits) + len(candidate.action_hits),
        ),
        reverse=True,
    )


def select_candidate_window(candidates: Sequence[ScoredCandidate]) -> List[ScoredCandidate]:
    if not candidates:
        return []
    top_score = candidates[0].score
    selected = [
        candidate
        for candidate in candidates
        if candidate.score >= MEDIUM_CONFIDENCE_SCORE and top_score - candidate.score <= CLOSE_SCORE_GAP
    ]
    return list(selected[:MAX_CANDIDATES])


def confidence_for_score(score: int) -> str:
    if score >= HIGH_CONFIDENCE_SCORE:
        return "高"
    if score >= MEDIUM_CONFIDENCE_SCORE:
        return "中"
    return "低"


def match_type_for_candidates(candidates: Sequence[ScoredCandidate], used_llm: bool = False) -> str:
    if used_llm:
        return "llm_fallback"
    if not candidates:
        return "low_confidence"
    if candidates[0].score < MEDIUM_CONFIDENCE_SCORE:
        return "low_confidence"
    close_candidates = select_candidate_window(candidates)
    if len(close_candidates) <= 1:
        return "single"
    level1_values = {candidate.item.level1 for candidate in close_candidates}
    item_ids = {candidate.item.id for candidate in close_candidates}
    if len(level1_values) > 1:
        return "cross_domain"
    if len(item_ids) > 1:
        return "same_domain_multi_item"
    return "single"


def needs_review_for_match(match_type: str, confidence: str) -> bool:
    return confidence != "高" or match_type in {
        "cross_domain",
        "same_domain_multi_item",
        "low_confidence",
        "llm_fallback",
        "fallback",
    }


def build_reason(candidate: ScoredCandidate) -> str:
    parts: List[str] = []
    if candidate.level2_hit:
        parts.append(f"命中二级目录：{candidate.item.level2}")
    if candidate.level3_hits:
        parts.append(f"命中三级短语：{'、'.join(candidate.level3_hits)}")
    if candidate.object_hits:
        parts.append(f"命中对象词：{'、'.join(candidate.object_hits)}")
    if candidate.action_hits:
        parts.append(f"命中动作词：{'、'.join(candidate.action_hits)}")
    if candidate.weak_hits:
        parts.append(f"命中弱词：{'、'.join(candidate.weak_hits)}")
    parts.append(f"得分：{candidate.score}")
    return "；".join(parts)


def candidate_ids(candidates: Sequence[ScoredCandidate]) -> List[str]:
    return [candidate.item.id for candidate in candidates[:MAX_CANDIDATES]]


def candidate_labels_by_id(ids: Sequence[str]) -> Dict[str, str]:
    by_id = {item.id: item for item in load_catalog()}
    return {item_id: by_id[item_id].label for item_id in ids if item_id in by_id}
