import os
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from classifier.alias_matcher import match_aliases
from classifier.standard_catalog_loader import StandardCatalogItem, load_standard_catalog
from classifier.standard_normalizer import NormalizedProjectText, normalize_project_text


TOP_K = min(max(int(os.getenv("CLASSIFIER_TOP_K", "5")), 3), 8)
INTERNAL_TOP_K = min(max(int(os.getenv("CLASSIFIER_INTERNAL_TOP_K", "20")), 15), 50)
FAMILY_FALLBACK_IDS = {"CF-017-00", "CF-018-00", "CF-028-00"}
TERMITE_TERMS = (
    "白蚁",
    "蚁害",
    "灭蚁",
    "防蚁",
    "白蚁防治",
    "白蚁预防",
    "白蚁灭治",
)
GENERIC_ELEVATOR_TERMS = (
    "电梯",
    "客梯",
    "乘客电梯",
    "住宅电梯",
    "垂直电梯",
    "老旧电梯",
)
SPECIFIC_ELEVATOR_TERMS = (
    "限速器钢丝绳",
    "曳引机",
    "制动器",
    "电磁铁",
    "电动机",
    "导向轮",
    "曳引轮",
    "钢丝绳",
    "限速器",
    "限速系统",
    "控制柜",
    "励磁柜",
    "层门",
    "轿门",
    "轿厢门",
    "门机板",
    "门挂轮",
    "门挂板",
    "门锁",
    "导靴",
    "吊门轮",
    "缓冲器",
    "紧急报警",
    "呼叫电话",
    "呼梯",
    "按钮",
    "液压梯",
    "液压泵站",
    "自动扶梯",
    "自动人行道",
    "扶手带",
    "梯级",
    "踏板",
    "梯级链",
    "滚轮",
    "钢带",
    "曳引带",
    "控制面板",
    "主板",
    "主控板",
    "电路板",
    "三方通话",
    "五方通话",
    "紧急通话",
    "轿厢对讲",
    "电梯对讲",
)
NON_ELEVATOR_OBJECT_CONTEXT_TERMS = (
    "电梯厅",
    "电梯间",
    "电梯前室",
    "电梯监控",
    "轿厢监控",
    "梯控",
    "电梯门禁",
)
FAMILY_FORCED_CANDIDATES = (
    {
        "catalog_id": "CF-017-00",
        "family": "电梯",
        "terms": ("电梯", "货梯", "客梯", "乘客电梯", "住宅电梯", "老旧电梯", "垂直电梯"),
        "exclude_terms": NON_ELEVATOR_OBJECT_CONTEXT_TERMS,
    },
    {
        "catalog_id": "CF-018-00",
        "family": "弱电系统",
        "terms": ("弱电", "智能化", "弱电智能化", "安防系统", "安保系统"),
        "exclude_terms": (),
    },
    {
        "catalog_id": "CF-028-00",
        "family": "消防系统",
        "terms": ("消防", "消防系统", "消防设施", "消防设备", "消防改造", "消防维修", "消防设施维修"),
        "exclude_terms": ("消防技术咨询",),
    },
)

_ASCII_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


@dataclass(frozen=True)
class RetrievedCandidate:
    item: StandardCatalogItem
    rank: int
    retrieval_score: float
    source: str = "ngram"
    reason: str = ""


@dataclass(frozen=True)
class FamilyMatch:
    catalog_id: str
    family: str
    reason: str


def normalize_retrieval_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", str(text or "")).lower()
    return re.sub(r"\s+", " ", normalized).strip()


def _ngrams(value: str, sizes: Iterable[int]) -> set[str]:
    tokens: set[str] = set()
    for size in sizes:
        if len(value) >= size:
            tokens.update(value[index : index + size] for index in range(0, len(value) - size + 1))
    return tokens


def tokenize_for_retrieval(text: str) -> set[str]:
    normalized = normalize_retrieval_text(text)
    tokens = set(_ASCII_TOKEN_RE.findall(normalized))
    for chunk in _CJK_RE.findall(normalized):
        tokens.update(_ngrams(chunk, (2, 3)))
    return {token for token in tokens if token}


def _item_search_text(item: StandardCatalogItem) -> str:
    parts = [
        item.id,
        item.standard_group,
        item.category,
        item.item,
        item.item_group or "",
        *item.status_basis.keys(),
        *item.status_basis.values(),
    ]
    return " ".join(part for part in parts if part)


def _allows_termite_candidate(normalized_text: str) -> bool:
    return any(term in normalized_text for term in TERMITE_TERMS)


def _is_termite_catalog_item(item: StandardCatalogItem) -> bool:
    return item.id == "TERMITE-001" or item.category == "白蚁防治" or "白蚁" in item.item


def _allows_generic_elevator_candidate(normalized_text: str) -> bool:
    if any(term in normalized_text for term in NON_ELEVATOR_OBJECT_CONTEXT_TERMS):
        return False
    has_generic = any(term in normalized_text for term in GENERIC_ELEVATOR_TERMS)
    has_specific = any(term in normalized_text for term in SPECIFIC_ELEVATOR_TERMS)
    return has_generic and not has_specific


def detect_family_matches(project_name: str | NormalizedProjectText) -> tuple[FamilyMatch, ...]:
    normalized = (
        project_name
        if isinstance(project_name, NormalizedProjectText)
        else normalize_project_text(str(project_name or ""))
    )
    matches: list[FamilyMatch] = []
    for family_candidate in FAMILY_FORCED_CANDIDATES:
        catalog_id = str(family_candidate["catalog_id"])
        if catalog_id == "CF-017-00" and not _allows_generic_elevator_candidate(normalized.normalized_text):
            continue
        exclude_terms = family_candidate["exclude_terms"]
        if any(term in normalized.normalized_text for term in exclude_terms):
            continue
        terms = family_candidate["terms"]
        if not any(term in normalized.normalized_text for term in terms):
            continue
        matches.append(
            FamilyMatch(
                catalog_id=catalog_id,
                family=str(family_candidate["family"]),
                reason=f"family_forced: {family_candidate['family']}，一级明确但二级未明确",
            )
        )
    return tuple(matches)


def candidate_label(item: StandardCatalogItem) -> str:
    return f"{item.id} | {item.category} | {item.item}"


def candidate_prompt_label(item: StandardCatalogItem) -> str:
    return f"{candidate_label(item)} | 可选状态：{'/'.join(item.allowed_statuses)}"


@lru_cache(maxsize=1)
def _indexed_catalog() -> tuple[tuple[StandardCatalogItem, str, frozenset[str]], ...]:
    indexed = []
    for item in load_standard_catalog():
        search_text = _item_search_text(item)
        indexed.append((item, normalize_retrieval_text(search_text), frozenset(tokenize_for_retrieval(search_text))))
    return tuple(indexed)


def _phrase_bonus(query_text: str, item: StandardCatalogItem, search_text: str) -> float:
    score = 0.0
    for field, weight in (
        (item.id, 12.0),
        (item.item, 10.0),
        (item.category, 7.0),
        (item.item_group or "", 5.0),
        (item.standard_group, 2.0),
    ):
        normalized = normalize_retrieval_text(field)
        if normalized and normalized in query_text:
            score += weight
        if query_text and len(query_text) >= 2 and query_text in normalized:
            score += weight / 2
    if query_text and query_text in search_text:
        score += 6.0
    return score


def _catalog_by_id() -> dict[str, StandardCatalogItem]:
    return {item.id: item for item in load_standard_catalog()}


def _append_recall_candidate(
    selected: list[tuple[float, StandardCatalogItem, str, str]],
    selected_ids: set[str],
    entry: tuple[float, StandardCatalogItem, str, str],
    limit: int,
) -> None:
    item_id = entry[1].id
    if item_id in selected_ids:
        return
    if len(selected) < limit:
        selected.append(entry)
        selected_ids.add(item_id)
        return
    for index in range(len(selected) - 1, -1, -1):
        if selected[index][2] in {"ngram", "alias_hint+ngram"}:
            removed_id = selected.pop(index)[1].id
            selected_ids.discard(removed_id)
            selected.append(entry)
            selected_ids.add(item_id)
            return


def retrieve_candidates(project_name: str, top_k: int | None = None) -> list[RetrievedCandidate]:
    limit = min(max(top_k if top_k is not None else TOP_K, 3), 8)
    normalized = normalize_project_text(project_name)
    query_text = normalize_retrieval_text(normalized.retrieval_text)
    query_tokens = tokenize_for_retrieval(normalized.retrieval_text)
    alias_result = match_aliases(normalized)
    alias_hits_by_id = {hit.catalog_id: hit for hit in alias_result.catalog_hits}
    family_matches_by_id = {match.catalog_id: match for match in detect_family_matches(normalized)}
    elevator_as_context = any(term in normalized.normalized_text for term in NON_ELEVATOR_OBJECT_CONTEXT_TERMS)

    scored: list[tuple[float, StandardCatalogItem, str, str]] = []
    for item, search_text, item_tokens in _indexed_catalog():
        if _is_termite_catalog_item(item) and not _allows_termite_candidate(normalized.normalized_text):
            continue
        if elevator_as_context and item.category == "电梯":
            continue
        if item.id in FAMILY_FALLBACK_IDS:
            continue
        overlap = query_tokens & item_tokens
        score = float(len(overlap))
        if query_tokens:
            score += len(overlap) / max(len(query_tokens), 1)
        score += _phrase_bonus(query_text, item, search_text)
        source = "ngram"
        reason = ""
        alias_hit = alias_hits_by_id.get(item.id)
        if alias_hit:
            source = "alias_hint+ngram"
            reason = alias_hit.reason
        family_match = family_matches_by_id.get(item.id)
        if family_match:
            source = "family+alias_hint+ngram" if alias_hit else "family+ngram"
            reason = "；".join(part for part in (reason, family_match.reason) if part)
        if score > 0:
            scored.append((score, item, source, reason))

    scored.sort(key=lambda entry: (entry[0], entry[1].id), reverse=True)
    internal = scored[:INTERNAL_TOP_K]
    selected: list[tuple[float, StandardCatalogItem, str, str]] = []
    selected_ids: set[str] = set()
    for entry in internal:
        if len(selected) >= limit:
            break
        selected.append(entry)
        selected_ids.add(entry[1].id)

    catalog_by_id = _catalog_by_id()
    for hit in alias_result.catalog_hits:
        if hit.catalog_id in selected_ids or hit.catalog_id in FAMILY_FALLBACK_IDS:
            continue
        item = catalog_by_id.get(hit.catalog_id)
        if item is None:
            continue
        _append_recall_candidate(selected, selected_ids, (0.0, item, "alias_recall", hit.reason), limit)

    for match in family_matches_by_id.values():
        item = catalog_by_id.get(match.catalog_id)
        if item is None:
            continue
        _append_recall_candidate(selected, selected_ids, (0.0, item, "family_recall", match.reason), limit)

    return [
        RetrievedCandidate(item=item, rank=index + 1, retrieval_score=score, source=source, reason=reason)
        for index, (score, item, source, reason) in enumerate(selected)
    ]
