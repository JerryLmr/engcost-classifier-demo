import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

from data.boundaries import find_boundary_decision
from core.config import DEFAULT_FALLBACK_LEVEL1, DEFAULT_FALLBACK_LEVEL2
from data.categories import CATEGORY_TREE
from data.rules import DETAILED_LEVEL2_RULES, DetailedLevel2Rule, KeywordRule, LEVEL1_RULES, LEVEL2_RULES
from services.llm_client import llm_classify

COMPOSITE_CONNECTORS = ["及", "和", "以及", "并", "同时", "+", "兼", "并且"]

DOMAIN_STRONG_KEYWORDS: Dict[str, List[str]] = {
    "电梯": ["电梯", "扶梯", "钢丝绳", "主机", "抱闸", "层门"],
    "消防": ["消火栓", "消防栓", "喷淋", "报警", "灭火器", "防火门", "稳压泵", "报警阀"],
    "监控": ["监控", "摄像头", "球机", "录像", "存储"],
    "防水工程": ["防水", "渗漏", "漏水", "渗水", "防水层", "屋面", "屋顶", "地下室"],
    "外立面修缮": ["粉刷", "空鼓", "脱落", "裂缝", "翻新", "涂料", "外立面"],
    "给排水": ["给水", "排水", "水泵", "二次供水", "水管"],
    "污水": ["污水", "化粪池", "污水井", "污水管"],
    "绿化景观": ["绿化", "补种", "景观", "树木", "草坪", "园路"],
    "停车交通": ["车位", "停车", "道闸", "标线", "交通设施"],
    "公共设施": ["公共区域", "无障碍通道", "入口通道", "通道"],
    "弱电系统": ["对讲", "网络", "智能化", "布线", "楼宇对讲"],
    "门禁设施": ["门禁", "刷卡", "人脸", "门控"],
    "道路工程": ["道路", "路面", "人行道", "拓宽", "路面积水"],
}

SAME_DOMAIN_COMPONENTS: Dict[str, Dict[str, List[str]]] = {
    "消防": {
        "消火栓": ["消火栓", "消防栓"],
        "喷淋": ["喷淋", "稳压泵", "报警阀"],
        "报警": ["报警", "报警系统", "火灾自动报警"],
        "设备": ["灭火器", "防火门"],
    },
    "电梯": {
        "整梯": ["电梯", "扶梯"],
        "部件": ["钢丝绳", "主机", "抱闸", "层门", "平层感应器"],
    },
    "门禁设施": {
        "门禁": ["门禁", "门控"],
        "对讲": ["对讲", "楼宇对讲"],
        "识别": ["刷卡", "人脸"],
    },
}


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


def collect_level1_candidates(text: str) -> List[Tuple[str, int, List[str]]]:
    candidates: List[Tuple[str, int, List[str]]] = []
    for name, rules in LEVEL1_RULES.items():
        score, hits = score_keywords(text, rules)
        if score > 0:
            candidates.append((name, score, hits))
    return sorted(candidates, key=lambda item: (item[1], len(item[2])), reverse=True)


def collect_strong_domain_hits(text: str) -> Dict[str, List[str]]:
    domain_hits: Dict[str, List[str]] = {}
    for level1, keywords in DOMAIN_STRONG_KEYWORDS.items():
        hits = [keyword for keyword in keywords if keyword in text]
        if hits:
            domain_hits[level1] = hits
    return domain_hits


def collect_same_domain_components(text: str, primary_level1: str) -> Set[str]:
    components = SAME_DOMAIN_COMPONENTS.get(primary_level1, {})
    matched: Set[str] = set()
    for component_name, keywords in components.items():
        if any(keyword in text for keyword in keywords):
            matched.add(component_name)
    return matched


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


def build_candidate_labels(level1_names: Sequence[str]) -> List[str]:
    labels: List[str] = []
    for level1 in level1_names:
        if level1 not in labels:
            labels.append(level1)
        if len(labels) >= 3:
            break
    return labels


def detect_composite_metadata(
    text: str,
    primary_level1: str,
    method: str,
) -> Dict[str, object]:
    normalized = normalize_text(text)
    connectors = [connector for connector in COMPOSITE_CONNECTORS if connector in text]
    candidates = collect_level1_candidates(normalized)
    strong_domain_hits = collect_strong_domain_hits(normalized)
    same_domain_components = collect_same_domain_components(normalized, primary_level1)

    strong_candidate_names = [
        level1
        for level1, score, _hits in candidates
        if score >= 3 and level1 in strong_domain_hits
    ]
    cross_domain = [level1 for level1 in strong_candidate_names if level1 != primary_level1]

    is_composite = False
    needs_review = method == "降级兜底"
    composite_reason = None
    structure_type = "single_project"
    secondary_candidates: List[str] = []

    if cross_domain and connectors:
        is_composite = True
        structure_type = "composite_project"
        secondary_names = cross_domain[:2]
        connector_text = f"连接词：{'、'.join(connectors)}；" if connectors else ""
        composite_reason = (
            f"{connector_text}同时命中多个工程域：{primary_level1}"
            + (f"、{'、'.join(secondary_names)}" if secondary_names else "")
        )
        secondary_candidates = build_candidate_labels(secondary_names)
        if len(strong_candidate_names) >= 2:
            top_score = next(
                (score for level1, score, _hits in candidates if level1 == primary_level1),
                0,
            )
            second_score = next(
                (score for level1, score, _hits in candidates if level1 == secondary_names[0]),
                0,
            )
            if top_score - second_score <= 2:
                needs_review = True
    elif len(same_domain_components) >= 2:
        structure_type = "multi_system_same_domain"
        needs_review = True

    if method == "降级兜底":
        needs_review = True

    return {
        "is_composite": is_composite,
        "needs_review": needs_review,
        "composite_reason": composite_reason,
        "secondary_candidates": secondary_candidates,
        "structure_type": structure_type,
    }


def attach_result_metadata(text: str, result: Dict[str, str]) -> Dict[str, object]:
    metadata = detect_composite_metadata(text, result["level1"], result["method"])
    metadata.pop("structure_type", None)
    return {**result, **metadata}


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
    result = rule_classify(text) or llm_classify(text)
    return attach_result_metadata(text, result)
