import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from classifier.standard_catalog_loader import OUT_OF_SCOPE_ID, get_standard_catalog_by_id
from classifier.standard_normalizer import NormalizedProjectText, normalize_project_text


ALIAS_DICTIONARY_PATH = Path(__file__).resolve().parents[1] / "config" / "alias_dictionary.json"


@dataclass(frozen=True)
class AliasEntry:
    alias_pattern: str
    target_catalog_id: str
    match_type: str
    priority: int
    action_hint: str = ""
    status_hint: str = ""
    review_hint: str = ""
    notes: str = ""


@dataclass(frozen=True)
class AliasCatalogHit:
    catalog_id: str
    matched_aliases: tuple[str, ...]
    priority: int
    weight: float
    reason: str


@dataclass(frozen=True)
class AliasMatchResult:
    catalog_hits: tuple[AliasCatalogHit, ...]
    negative_hints: tuple[str, ...]
    action_hints: tuple[str, ...]
    status_hints: tuple[str, ...]
    review_hints: tuple[str, ...]


def _clean_string(value: Any) -> str:
    return str(value or "").strip()


def _parse_entry(raw: dict[str, Any]) -> AliasEntry:
    alias_pattern = _clean_string(raw.get("alias_pattern"))
    target_catalog_id = _clean_string(raw.get("target_catalog_id"))
    match_type = _clean_string(raw.get("match_type") or "contains").lower()
    if not alias_pattern:
        raise ValueError("alias entry missing alias_pattern")
    if not target_catalog_id:
        raise ValueError(f"alias {alias_pattern} missing target_catalog_id")
    if match_type not in {"contains", "regex"}:
        raise ValueError(f"alias {alias_pattern} has unsupported match_type: {match_type}")
    return AliasEntry(
        alias_pattern=alias_pattern,
        target_catalog_id=target_catalog_id,
        match_type=match_type,
        priority=int(raw.get("priority") or 0),
        action_hint=_clean_string(raw.get("action_hint")),
        status_hint=_clean_string(raw.get("status_hint")),
        review_hint=_clean_string(raw.get("review_hint")),
        notes=_clean_string(raw.get("notes")),
    )


@lru_cache(maxsize=1)
def load_alias_dictionary() -> tuple[AliasEntry, ...]:
    with ALIAS_DICTIONARY_PATH.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, list):
        raise ValueError("alias_dictionary.json must contain a JSON array")

    catalog_by_id = get_standard_catalog_by_id()
    entries = tuple(_parse_entry(raw) for raw in payload)
    for entry in entries:
        if entry.target_catalog_id != OUT_OF_SCOPE_ID and entry.target_catalog_id not in catalog_by_id:
            raise ValueError(
                f"alias {entry.alias_pattern} targets unknown catalog id: {entry.target_catalog_id}"
            )
        if entry.match_type == "regex":
            re.compile(entry.alias_pattern)
    return entries


def _matches(entry: AliasEntry, text: str) -> bool:
    if entry.match_type == "contains":
        return entry.alias_pattern in text
    return re.search(entry.alias_pattern, text) is not None


def _add_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def match_aliases(project_name: str | NormalizedProjectText) -> AliasMatchResult:
    normalized = (
        project_name
        if isinstance(project_name, NormalizedProjectText)
        else normalize_project_text(str(project_name or ""))
    )
    grouped: dict[str, list[AliasEntry]] = {}
    negative_hints: list[str] = []
    action_hints: list[str] = []
    status_hints: list[str] = []
    review_hints: list[str] = []

    for entry in load_alias_dictionary():
        if not _matches(entry, normalized.normalized_text):
            continue
        if entry.target_catalog_id == OUT_OF_SCOPE_ID:
            _add_unique(negative_hints, entry.alias_pattern)
        else:
            grouped.setdefault(entry.target_catalog_id, []).append(entry)
        _add_unique(action_hints, entry.action_hint)
        _add_unique(status_hints, entry.status_hint)
        _add_unique(review_hints, entry.review_hint)

    hits: list[AliasCatalogHit] = []
    for catalog_id, entries in grouped.items():
        aliases = tuple(entry.alias_pattern for entry in entries)
        priority = max(entry.priority for entry in entries)
        weight = float(sum(max(entry.priority, 1) for entry in entries))
        hits.append(
            AliasCatalogHit(
                catalog_id=catalog_id,
                matched_aliases=aliases,
                priority=priority,
                weight=weight,
                reason=f"alias命中：{'、'.join(aliases)}",
            )
        )

    hits.sort(key=lambda hit: (hit.priority, hit.weight, hit.catalog_id), reverse=True)
    return AliasMatchResult(
        catalog_hits=tuple(hits),
        negative_hints=tuple(negative_hints),
        action_hints=tuple(action_hints),
        status_hints=tuple(status_hints),
        review_hints=tuple(review_hints),
    )
