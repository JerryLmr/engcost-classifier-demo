import json
import os
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from classifier.alias_matcher import match_aliases
from classifier.standard_catalog_loader import StandardCatalogItem, load_standard_catalog
from classifier.standard_normalizer import NormalizedProjectText, normalize_project_text


TOP_K = min(max(int(os.getenv("CLASSIFIER_TOP_K", "5")), 3), 8)
INTERNAL_TOP_K = min(max(int(os.getenv("CLASSIFIER_INTERNAL_TOP_K", "20")), 15), 50)
FAMILY_FALLBACK_IDS = {"CF-017-00", "CF-018-00", "CF-028-00"}
FAMILY_FALLBACK_RULES_PATH = Path(__file__).resolve().parents[1] / "config" / "family_fallback_rules.json"
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
NON_ELEVATOR_OBJECT_CONTEXT_TERMS = ("电梯厅", "电梯间", "电梯前室", "电梯监控", "轿厢监控", "梯控", "电梯门禁")

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


@dataclass(frozen=True)
class FamilyFallbackRule:
    catalog_id: str
    category: str
    family_terms: tuple[str, ...]
    specific_terms: tuple[str, ...]
    negative_context_terms: tuple[str, ...]


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
    if _has_non_elevator_object_context(normalized_text):
        return False
    has_generic = any(term in normalized_text for term in GENERIC_ELEVATOR_TERMS)
    has_specific = any(term in normalized_text for term in SPECIFIC_ELEVATOR_TERMS)
    return has_generic and not has_specific


def _term_matches_text(term: str, normalized_text: str) -> bool:
    if term == "梯控":
        return re.search(r"梯控(?!制)", normalized_text) is not None
    return term in normalized_text


def _has_non_elevator_object_context(normalized_text: str) -> bool:
    return any(_term_matches_text(term, normalized_text) for term in NON_ELEVATOR_OBJECT_CONTEXT_TERMS)


def _clean_terms(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    terms: list[str] = []
    for item in value:
        term = str(item or "").strip()
        if term and term not in terms:
            terms.append(term)
    return tuple(terms)


@lru_cache(maxsize=1)
def load_family_fallback_rules() -> tuple[FamilyFallbackRule, ...]:
    with FAMILY_FALLBACK_RULES_PATH.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, list):
        raise ValueError("family_fallback_rules.json must contain a JSON array")

    allowed_ids = {"CF-017-00", "CF-018-00", "CF-028-00"}
    rules: list[FamilyFallbackRule] = []
    for raw in payload:
        if not isinstance(raw, dict):
            raise ValueError("family fallback rule must be a JSON object")
        catalog_id = str(raw.get("catalog_id") or "").strip()
        if catalog_id not in allowed_ids:
            raise ValueError(f"unsupported family fallback catalog_id: {catalog_id}")
        family_terms = _clean_terms(raw.get("family_terms"))
        if not family_terms:
            raise ValueError(f"family fallback {catalog_id} missing family_terms")
        rules.append(
            FamilyFallbackRule(
                catalog_id=catalog_id,
                category=str(raw.get("category") or "").strip(),
                family_terms=family_terms,
                specific_terms=_clean_terms(raw.get("specific_terms")),
                negative_context_terms=_clean_terms(raw.get("negative_context_terms")),
            )
        )
    return tuple(rules)


def detect_family_matches(project_name: str | NormalizedProjectText) -> tuple[FamilyMatch, ...]:
    normalized = (
        project_name
        if isinstance(project_name, NormalizedProjectText)
        else normalize_project_text(str(project_name or ""))
    )
    matches: list[FamilyMatch] = []
    for rule in load_family_fallback_rules():
        if not any(term in normalized.normalized_text for term in rule.family_terms):
            continue
        diagnostics: list[str] = []
        if any(_term_matches_text(term, normalized.normalized_text) for term in rule.specific_terms):
            diagnostics.append("含具体对象词")
        if any(_term_matches_text(term, normalized.normalized_text) for term in rule.negative_context_terms):
            diagnostics.append("含可能为修饰语的位置/服务词")
        diagnostic_suffix = f"（{'、'.join(diagnostics)}，仅作辅助提示）" if diagnostics else ""
        matches.append(
            FamilyMatch(
                catalog_id=rule.catalog_id,
                family=rule.category,
                reason=f"family_fallback: {rule.category}，一级相关{diagnostic_suffix}",
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
    query_parts = tuple(part for part in query_text.split(" ") if len(part) >= 2)
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
        for part in query_parts:
            if part != query_text and part in normalized:
                score += weight
    if query_text and query_text in search_text:
        score += 6.0
    for part in query_parts:
        if part != query_text and part in search_text:
            score += 3.0
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
    removed_item_id = selected[-1][1].id
    selected.pop()
    selected_ids.discard(removed_item_id)
    selected.append(entry)
    selected_ids.add(item_id)


def retrieve_candidates(project_name: str, top_k: int | None = None) -> list[RetrievedCandidate]:
    limit = min(max(top_k if top_k is not None else TOP_K, 3), 8)
    normalized = normalize_project_text(project_name)
    alias_result = match_aliases(normalized)
    retrieval_text = " ".join(part for part in (normalized.retrieval_text, *alias_result.expanded_terms) if part)
    query_text = normalize_retrieval_text(retrieval_text)
    query_tokens = tokenize_for_retrieval(retrieval_text)
    family_matches_by_id = {match.catalog_id: match for match in detect_family_matches(normalized)}
    elevator_as_context = _has_non_elevator_object_context(normalized.normalized_text)
    generic_elevator_only = _allows_generic_elevator_candidate(normalized.normalized_text)

    scored: list[tuple[float, StandardCatalogItem, str, str]] = []
    for item, search_text, item_tokens in _indexed_catalog():
        if _is_termite_catalog_item(item) and not _allows_termite_candidate(normalized.normalized_text):
            continue
        if elevator_as_context and item.category == "电梯":
            continue
        if generic_elevator_only and item.category == "电梯":
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
    for match in family_matches_by_id.values():
        item = catalog_by_id.get(match.catalog_id)
        if item is None:
            continue
        _append_recall_candidate(selected, selected_ids, (0.0, item, "family_recall", match.reason), limit)

    return [
        RetrievedCandidate(item=item, rank=index + 1, retrieval_score=score, source=source, reason=reason)
        for index, (score, item, source, reason) in enumerate(selected)
    ]
