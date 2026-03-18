import re
from typing import Dict, List, Optional, Sequence, Tuple

from data.boundaries import find_boundary_decision
from core.config import DEFAULT_FALLBACK_LEVEL1, DEFAULT_FALLBACK_LEVEL2
from data.categories import CATEGORY_TREE
from data.rules import DETAILED_LEVEL2_RULES, DetailedLevel2Rule, KeywordRule, LEVEL1_RULES, LEVEL2_RULES
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


def filter_rule_map(
    rule_map: Dict[str, List[KeywordRule]],
    allowed_names: Optional[Sequence[str]],
) -> Dict[str, List[KeywordRule]]:
    if not allowed_names:
        return rule_map
    allowed = set(allowed_names)
    return {name: rules for name, rules in rule_map.items() if name in allowed}


def score_detailed_rule(text: str, rule: DetailedLevel2Rule) -> Tuple[int, int, int, List[str]]:
    object_score, object_hits = score_keywords(text, rule.get("object_keywords", []))
    action_score, action_hits = score_keywords(text, rule.get("action_keywords", []))
    weak_score, weak_hits = score_keywords(text, rule.get("weak_keywords", []))

    total_score = object_score + action_score + weak_score
    hits = object_hits + [keyword for keyword in action_hits if keyword not in object_hits]
    hits += [keyword for keyword in weak_hits if keyword not in hits]
    return total_score, object_score, action_score, hits


def match_detailed_level2(
    text: str,
    level1: str,
    allowed_level2: Optional[Sequence[str]] = None,
) -> Tuple[Optional[str], List[str], int]:
    rules = DETAILED_LEVEL2_RULES.get(level1)
    if not rules:
        return None, [], 0

    allowed = set(allowed_level2) if allowed_level2 else None
    best_name: Optional[str] = None
    best_hits: List[str] = []
    best_score = 0
    best_object_score = 0
    best_action_score = 0

    for name, rule in rules.items():
        if allowed is not None and name not in allowed:
            continue

        total_score, object_score, action_score, hits = score_detailed_rule(text, rule)
        min_score = rule.get("min_score", 1)
        if object_score == 0 and not rule.get("default_on_object", False):
            continue
        if object_score > 0 and action_score == 0 and not rule.get("default_on_object", False):
            continue
        if object_score == 0 and action_score > 0:
            continue
        if object_score > 0 and action_score == 0 and rule.get("default_on_object", False):
            total_score = max(total_score, min_score)
        if total_score < min_score:
            continue

        if (
            object_score > best_object_score
            or (object_score == best_object_score and action_score > best_action_score)
            or (
                object_score == best_object_score
                and action_score == best_action_score
                and total_score > best_score
            )
        ):
            best_name = name
            best_hits = hits
            best_score = total_score
            best_object_score = object_score
            best_action_score = action_score

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
    boundary = find_boundary_decision(normalized)
    allowed_level1 = [boundary["level1"]] if boundary else None
    level1_map = filter_rule_map(LEVEL1_RULES, allowed_level1)
    level1, level1_hits, _ = match_best_rule(level1_map, normalized)
    if boundary and boundary.get("level1") and not level1:
        level1 = boundary["level1"]
        level1_hits = []
    if not level1:
        return None

    allowed_level2 = boundary.get("allowed_level2") if boundary else None
    level2, level2_hits, _ = match_detailed_level2(normalized, level1, allowed_level2)
    if not level2:
        level2_map = filter_rule_map(LEVEL2_RULES[level1], allowed_level2)
        level2, level2_hits, _ = match_best_rule(level2_map, normalized)
    if not level2:
        return None

    hits = level1_hits + [keyword for keyword in level2_hits if keyword not in level1_hits]
    if not hits and boundary:
        hits = [boundary["reason"]]

    reason_parts = [f"一级命中：{level1}"]
    if boundary:
        reason_parts.append(f"边界判定：{boundary['reason']}")
    if hits:
        reason_parts.append(f"关键词：{'、'.join(hits)}")
    return {
        "project_name": text,
        "level1": level1,
        "level2": level2,
        "method": "规则优先",
        "reason": "；".join(reason_parts),
    }


def classify_text(text: str):
    return rule_classify(text) or llm_classify(text)
