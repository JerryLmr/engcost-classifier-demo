import re
from typing import Dict, List, Optional, Tuple

from core.config import DEFAULT_FALLBACK_LEVEL1, DEFAULT_FALLBACK_LEVEL2
from data.categories import CATEGORY_TREE
from data.rules import KeywordRule, LEVEL1_RULES, LEVEL2_RULES
from services.llm_client import llm_classify


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text.strip())


def score_keywords(text: str, rules: List[KeywordRule]) -> Tuple[int, List[str]]:
    score = 0
    hits: List[str] = []
    for keyword, weight in rules:
        if keyword in text:
            score += weight
            hits.append(keyword)
    return score, hits


def match_best_rule(rule_map: Dict[str, List[KeywordRule]], text: str) -> Tuple[Optional[str], List[str], int]:
    best_name: Optional[str] = None
    best_hits: List[str] = []
    best_score = 0

    for name, rules in rule_map.items():
        score, hits = score_keywords(text, rules)
        if score == 0:
            continue
        if (
            score > best_score
            or (score == best_score and len(hits) > len(best_hits))
            or (score == best_score and len(hits) == len(best_hits) and best_name is None)
        ):
            best_name = name
            best_hits = hits
            best_score = score

    return best_name, best_hits, best_score


def fallback_classify(text: str, reason: str):
    return {
        "project_name": text,
        "level1": DEFAULT_FALLBACK_LEVEL1,
        "level2": DEFAULT_FALLBACK_LEVEL2,
        "method": "降级兜底",
        "reason": reason,
    }


def rule_classify(text: str):
    normalized = normalize_text(text)
    level1, level1_hits, _ = match_best_rule(LEVEL1_RULES, normalized)
    if not level1:
        return None

    level2, level2_hits, _ = match_best_rule(LEVEL2_RULES[level1], normalized)
    if not level2:
        level2 = CATEGORY_TREE[level1][0]

    hits = level1_hits + [keyword for keyword in level2_hits if keyword not in level1_hits]
    return {
        "project_name": text,
        "level1": level1,
        "level2": level2,
        "method": "规则优先",
        "reason": f"一级命中：{level1}；关键词：{'、'.join(hits)}",
    }


def classify_text(text: str):
    return rule_classify(text) or llm_classify(text)
