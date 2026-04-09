import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

from core.config import RULE_CONFIG_DIR, RULE_CONFIG_FALLBACK_DIR


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def _load_json_from_candidates(filename: str) -> Dict[str, Any]:
    candidates = [RULE_CONFIG_DIR / filename, RULE_CONFIG_FALLBACK_DIR / filename]
    for path in candidates:
        if path.exists():
            with Path(path).open("r", encoding="utf-8") as fp:
                return json.load(fp)
    raise FileNotFoundError(f"未找到高频映射配置: {filename}")


@lru_cache(maxsize=1)
def load_high_freq_mapping() -> Dict[str, Any]:
    return _load_json_from_candidates("high_freq_mapping.json")


def _match_item(normalized_text: str, item: Dict[str, Any]) -> Tuple[int, List[str]]:
    matched_patterns: List[str] = []
    score = 0
    for pattern in item.get("patterns", []):
        normalized_pattern = _normalize_text(pattern)
        if normalized_pattern and normalized_pattern in normalized_text:
            matched_patterns.append(pattern)
            score += len(normalized_pattern)
    return score, matched_patterns


def match_high_freq_category(project_name: str) -> Dict[str, Any]:
    normalized_text = _normalize_text(project_name)
    if not normalized_text:
        return {
            "matched": False,
            "action": "no_match",
        }

    best: Dict[str, Any] | None = None
    best_score = 0
    best_hits: List[str] = []
    for item in load_high_freq_mapping().get("items", []):
        score, hits = _match_item(normalized_text, item)
        if score <= 0:
            continue
        priority = int(item.get("priority", 0))
        if best is None:
            best, best_score, best_hits = item, score, hits
            continue
        best_priority = int(best.get("priority", 0))
        if (score, priority) > (best_score, best_priority):
            best, best_score, best_hits = item, score, hits

    if best is None:
        return {
            "matched": False,
            "action": "no_match",
        }

    return {
        "matched": True,
        "category_id": best.get("category_id"),
        "category_name": best.get("category_name"),
        "standard_object": best.get("standard_object"),
        "tags": list(best.get("tags", [])),
        "default_flow": best.get("default_flow"),
        "default_result": best.get("default_result"),
        "action": best.get("action", "no_match"),
        "confidence": best.get("confidence", "medium"),
        "matched_patterns": best_hits,
        "reason_codes": list(best.get("reason_codes", [])),
        "business_statement": best.get("business_statement", ""),
    }
