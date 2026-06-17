import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from classifier.standard_normalizer import NormalizedProjectText, normalize_project_text


TEXT_ALIASES_PATH = Path(__file__).resolve().parents[1] / "config" / "text_aliases.json"


@dataclass(frozen=True)
class TextAliasEntry:
    canonical_term: str
    match_type: str
    patterns: tuple[str, ...]


@dataclass(frozen=True)
class TextAliasHit:
    canonical_term: str
    matched_patterns: tuple[str, ...]


@dataclass(frozen=True)
class TextAliasResult:
    expanded_terms: tuple[str, ...]
    hits: tuple[TextAliasHit, ...]


def _clean_string(value: Any) -> str:
    return str(value or "").strip()


def _parse_patterns(raw_patterns: Any, canonical_term: str) -> tuple[str, ...]:
    if not isinstance(raw_patterns, list):
        raise ValueError(f"text alias {canonical_term} has invalid patterns")
    patterns: list[str] = []
    for value in raw_patterns:
        pattern = _clean_string(value)
        if pattern and pattern not in patterns:
            patterns.append(pattern)
    if not patterns:
        raise ValueError(f"text alias {canonical_term} missing patterns")
    return tuple(patterns)


def _parse_entry(raw: dict[str, Any]) -> TextAliasEntry:
    canonical_term = _clean_string(raw.get("canonical_term"))
    match_type = _clean_string(raw.get("match_type") or "contains").lower()
    if not canonical_term:
        raise ValueError("text alias entry missing canonical_term")
    if match_type not in {"contains", "regex"}:
        raise ValueError(f"text alias {canonical_term} has unsupported match_type: {match_type}")
    patterns = _parse_patterns(raw.get("patterns"), canonical_term)
    return TextAliasEntry(canonical_term=canonical_term, match_type=match_type, patterns=patterns)


@lru_cache(maxsize=1)
def load_text_aliases() -> tuple[TextAliasEntry, ...]:
    with TEXT_ALIASES_PATH.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, list):
        raise ValueError("text_aliases.json must contain a JSON array")

    entries = tuple(_parse_entry(raw) for raw in payload)
    for entry in entries:
        if entry.match_type == "regex":
            for pattern in entry.patterns:
                re.compile(pattern)
    return entries


def _pattern_matches(pattern: str, match_type: str, text: str) -> bool:
    if match_type == "contains":
        return pattern in text
    return re.search(pattern, text) is not None


def _add_unique(values: list[str], value: str) -> None:
    if value and value not in values:
        values.append(value)


def match_aliases(project_name: str | NormalizedProjectText) -> TextAliasResult:
    normalized = (
        project_name
        if isinstance(project_name, NormalizedProjectText)
        else normalize_project_text(str(project_name or ""))
    )

    expanded_terms: list[str] = []
    hits: list[TextAliasHit] = []
    for entry in load_text_aliases():
        matched_patterns = tuple(
            pattern
            for pattern in entry.patterns
            if _pattern_matches(pattern, entry.match_type, normalized.normalized_text)
        )
        if not matched_patterns:
            continue
        _add_unique(expanded_terms, entry.canonical_term)
        hits.append(TextAliasHit(canonical_term=entry.canonical_term, matched_patterns=matched_patterns))

    return TextAliasResult(expanded_terms=tuple(expanded_terms), hits=tuple(hits))
