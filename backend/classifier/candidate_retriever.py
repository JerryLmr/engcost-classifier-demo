import os
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from classifier.standard_catalog_loader import StandardCatalogItem, load_standard_catalog


TOP_K = min(max(int(os.getenv("CLASSIFIER_TOP_K", "5")), 3), 5)

_ASCII_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


@dataclass(frozen=True)
class RetrievedCandidate:
    item: StandardCatalogItem
    rank: int
    retrieval_score: float


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


def retrieve_candidates(project_name: str, top_k: int | None = None) -> list[RetrievedCandidate]:
    limit = min(max(top_k if top_k is not None else TOP_K, 3), 5)
    query_text = normalize_retrieval_text(project_name)
    query_tokens = tokenize_for_retrieval(project_name)

    scored: list[tuple[float, StandardCatalogItem]] = []
    for item, search_text, item_tokens in _indexed_catalog():
        overlap = query_tokens & item_tokens
        score = float(len(overlap))
        if query_tokens:
            score += len(overlap) / max(len(query_tokens), 1)
        score += _phrase_bonus(query_text, item, search_text)
        if score > 0:
            scored.append((score, item))

    scored.sort(key=lambda entry: (entry[0], entry[1].id), reverse=True)
    return [
        RetrievedCandidate(item=item, rank=index + 1, retrieval_score=score)
        for index, (score, item) in enumerate(scored[:limit])
    ]
