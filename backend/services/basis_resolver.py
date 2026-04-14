import json
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from core.config import resolve_rule_file


def _none_if_blank(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


@lru_cache(maxsize=1)
def load_reason_code_basis_registry() -> Dict[str, Any]:
    path = resolve_rule_file("reason_code_basis_registry.json")
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def _registry_sources_for_reason(reason_code: str) -> List[Dict[str, Any]]:
    registry = load_reason_code_basis_registry()
    entries = registry.get("entries", {})
    sources = entries.get(reason_code, [])
    return sources if isinstance(sources, list) else []


def _normalize_source(source: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "display_name": _none_if_blank(source.get("display_name")),
        "source_type": _none_if_blank(source.get("source_type")),
        "title": _none_if_blank(source.get("title")),
        "issuer": _none_if_blank(source.get("issuer")),
        "document_no": _none_if_blank(source.get("document_no")),
        "article": _none_if_blank(source.get("article")),
        "section": _none_if_blank(source.get("section")),
    }


def _dedupe_key(document: Dict[str, Any]) -> Tuple[str, str, str, str]:
    return (
        str(document.get("title") or ""),
        str(document.get("document_no") or ""),
        str(document.get("article") or ""),
        str(document.get("section") or ""),
    )


def _normalize_fallback_sources(fallback_sources: Optional[Iterable[Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    if not fallback_sources:
        return normalized
    for source in fallback_sources:
        if not isinstance(source, dict):
            continue
        normalized.append(_normalize_source(source))
    return normalized


def resolve_basis_documents(
    reason_codes: Sequence[str],
    fallback_sources: Optional[Iterable[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    collected: List[Tuple[int, int, Dict[str, Any]]] = []

    for reason_index, reason_code in enumerate(reason_codes or []):
        for source_index, source in enumerate(_registry_sources_for_reason(str(reason_code))):
            normalized = _normalize_source(source)
            collected.append(
                (
                    0 if source.get("primary") is True else 1,
                    reason_index * 100 + source_index,
                    normalized,
                )
            )

    if not collected:
        for fallback_index, source in enumerate(_normalize_fallback_sources(fallback_sources)):
            collected.append((1, 10_000 + fallback_index, source))

    deduped: List[Tuple[int, int, Dict[str, Any]]] = []
    seen: set[Tuple[str, str, str, str]] = set()
    for priority, order, document in sorted(collected, key=lambda item: (item[0], item[1])):
        key = _dedupe_key(document)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((priority, order, document))

    return [item[2] for item in deduped]
