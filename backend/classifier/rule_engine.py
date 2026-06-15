from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from classifier.catalog_loader import CatalogItem, load_catalog
from classifier.text_normalizer import normalize_text


HIGH_CONFIDENCE_SCORE = 6
MEDIUM_CONFIDENCE_SCORE = 3
CLOSE_SCORE_GAP = 2
MAX_CANDIDATES = 5
MIN_SPLIT_PHRASE_LENGTH = 2
GENERIC_ACTION_TERMS = {
    "维修",
    "修理",
    "修补",
    "更换",
    "改造",
    "更新",
    "安装",
    "整修",
    "翻修",
    "翻新",
    "新做",
    "拆装",
    "粉刷",
    "调换",
    "疏通",
    "修换",
    "补种",
    "修剪",
    "包扎",
    "检修",
    "拆除",
    "铲除",
    "斩粉",
    "添配",
    "修复",
    "新砌",
}
NORMALIZED_GENERIC_ACTION_TERMS = {normalize_text(term) for term in GENERIC_ACTION_TERMS}
TERM_ALIASES = {
    "消防栓": ("消火栓",),
    "消火栓": ("消防栓",),
}
OBJECT_MARKERS = (
    "外墙",
    "屋面",
    "坡屋面",
    "平屋面",
    "水泵",
    "污水泵",
    "排水泵",
    "生活水泵",
    "生活用泵",
    "消防栓",
    "消火栓",
    "防盗门",
    "视频监控",
    "电视监控控制台",
    "垃圾房",
)


@dataclass(frozen=True)
class ScoredCandidate:
    item: CatalogItem
    score: int
    level2_hit: bool
    exact_level3_item_hits: tuple[str, ...]
    derived_level3_item_hits: tuple[str, ...]
    matched_level3_items: tuple[str, ...]
    object_hits: tuple[str, ...]
    action_hits: tuple[str, ...]
    weak_hits: tuple[str, ...]

    @property
    def id(self) -> str:
        return self.item.id


def split_comma_separated_items(value: str) -> List[str]:
    separators = [",", "，", ";", "；"]
    phrases = [value]
    for separator in separators:
        next_phrases: List[str] = []
        for phrase in phrases:
            next_phrases.extend(phrase.split(separator))
        phrases = next_phrases
    return [phrase.strip() for phrase in phrases if len(phrase.strip()) >= MIN_SPLIT_PHRASE_LENGTH]


def is_generic_action_phrase(value: str) -> bool:
    return normalize_text(value) in NORMALIZED_GENERIC_ACTION_TERMS


def _phrase_matches(text: str, phrase: str) -> bool:
    normalized = normalize_text(phrase)
    return bool(normalized and normalized in text)


def _term_variants(value: str) -> List[str]:
    variants: List[str] = []

    def add(term: str) -> None:
        term = term.strip()
        if len(term) >= MIN_SPLIT_PHRASE_LENGTH and term not in variants:
            variants.append(term)
        for source, aliases in TERM_ALIASES.items():
            if source in term:
                for alias in aliases:
                    alias_term = term.replace(source, alias)
                    if len(alias_term) >= MIN_SPLIT_PHRASE_LENGTH and alias_term not in variants:
                        variants.append(alias_term)

    add(value)
    for phrase in split_comma_separated_items(value):
        add(phrase)
    if value.endswith("工程"):
        add(value.removesuffix("工程"))
    return variants


def _strip_leading_action(value: str) -> str:
    normalized_value = normalize_text(value)
    for action in sorted(GENERIC_ACTION_TERMS, key=len, reverse=True):
        normalized_action = normalize_text(action)
        if normalized_value.startswith(normalized_action):
            return value[len(action):].strip(" ,，、;；")
    return value


def extract_level3_core_terms(level3_item: str) -> tuple[str, ...]:
    terms: List[str] = []

    def add(term: str) -> None:
        for variant in _term_variants(term):
            if not is_generic_action_phrase(variant) and variant not in terms:
                terms.append(variant)

    stripped = _strip_leading_action(level3_item)
    add(stripped)
    for phrase in split_comma_separated_items(stripped):
        add(_strip_leading_action(phrase))
    for marker in OBJECT_MARKERS:
        if marker in stripped:
            add(marker)
    return tuple(terms)


def _derived_object_terms(item: CatalogItem) -> tuple[str, ...]:
    terms: List[str] = []

    def add(term: str) -> None:
        for variant in _term_variants(term):
            if not is_generic_action_phrase(variant) and variant not in terms:
                terms.append(variant)
        for marker in OBJECT_MARKERS:
            if marker in term and marker not in terms:
                terms.append(marker)

    add(item.level1)
    add(item.level2)
    for keyword in item.object_keywords:
        add(keyword)
    for level3_item in item.level3_items:
        for term in extract_level3_core_terms(level3_item):
            add(term)
    return tuple(terms)


def _keyword_matches(text: str, keyword: str, split_keyword: bool = True) -> bool:
    phrases = _term_variants(keyword) if split_keyword else [keyword]
    return any(_phrase_matches(text, phrase) for phrase in phrases)


def _hits(
    text: str,
    keywords: Iterable[str],
    *,
    include_generic_actions: bool = True,
    split_keyword: bool = True,
) -> tuple[str, ...]:
    seen: List[str] = []
    for keyword in keywords:
        if not include_generic_actions and is_generic_action_phrase(keyword):
            continue
        matched = _keyword_matches(text, keyword, split_keyword=split_keyword)
        if matched and keyword not in seen:
            seen.append(keyword)
    return tuple(seen)


def score_item(text: str, item: CatalogItem) -> ScoredCandidate:
    level2_hit = any(
        _phrase_matches(text, phrase)
        for phrase in [item.level2, *split_comma_separated_items(item.level2)]
    )
    exact_level3_item_hits = _hits(
        text,
        item.level3_items,
        include_generic_actions=False,
        split_keyword=False,
    )
    derived_level3_item_hits: List[str] = []
    for level3_item in item.level3_items:
        if level3_item in exact_level3_item_hits:
            continue
        if any(_phrase_matches(text, term) for term in extract_level3_core_terms(level3_item)):
            derived_level3_item_hits.append(level3_item)
    matched_level3_items = tuple([*exact_level3_item_hits, *derived_level3_item_hits])
    object_hits = _hits(
        text,
        _derived_object_terms(item),
        include_generic_actions=False,
        split_keyword=False,
    )
    action_hits = _hits(text, item.action_keywords)
    weak_hits = _hits(text, item.weak_keywords)

    has_object_signal = bool(level2_hit or matched_level3_items or object_hits)
    score = 0
    if level2_hit:
        score += 4
    if matched_level3_items:
        score += 4
    score += len(object_hits) * 3
    if has_object_signal:
        score += len(action_hits) * 2
    score += len(weak_hits)

    return ScoredCandidate(
        item=item,
        score=score,
        level2_hit=level2_hit,
        exact_level3_item_hits=exact_level3_item_hits,
        derived_level3_item_hits=tuple(derived_level3_item_hits),
        matched_level3_items=matched_level3_items,
        object_hits=object_hits,
        action_hits=action_hits if has_object_signal else (),
        weak_hits=weak_hits,
    )


def score_catalog(text: str) -> List[ScoredCandidate]:
    normalized = normalize_text(text)
    candidates = [score_item(normalized, item) for item in load_catalog()]
    candidates = [
        candidate
        for candidate in candidates
        if candidate.score > 0 and candidate.score >= candidate.item.min_score
    ]
    return sorted(
        candidates,
        key=lambda candidate: (
            candidate.score,
            bool(candidate.exact_level3_item_hits),
            bool(candidate.matched_level3_items),
            bool(candidate.object_hits),
            len(candidate.matched_level3_items) + len(candidate.object_hits) + len(candidate.action_hits),
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
    if candidate.exact_level3_item_hits:
        parts.append(f"完整命中三级细项：{'、'.join(candidate.exact_level3_item_hits)}")
    if candidate.derived_level3_item_hits:
        parts.append(f"推断命中三级细项：{'、'.join(candidate.derived_level3_item_hits)}")
    if candidate.object_hits:
        parts.append(f"命中对象词：{'、'.join(candidate.object_hits)}")
    if candidate.action_hits:
        parts.append(f"命中动作词：{'、'.join(candidate.action_hits)}")
    if candidate.weak_hits:
        parts.append(f"命中弱词：{'、'.join(candidate.weak_hits)}")
    if (candidate.level2_hit or candidate.object_hits) and not candidate.matched_level3_items:
        parts.append("仅命中二级/对象词，未命中具体三级细项")
    parts.append(f"得分：{candidate.score}")
    return "；".join(parts)


def candidate_ids(candidates: Sequence[ScoredCandidate]) -> List[str]:
    return [candidate.item.id for candidate in candidates[:MAX_CANDIDATES]]


def candidate_labels_by_id(ids: Sequence[str]) -> Dict[str, str]:
    by_id = {item.id: item for item in load_catalog()}
    return {item_id: by_id[item_id].label for item_id in ids if item_id in by_id}
