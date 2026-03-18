import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

from data.boundaries import find_boundary_decision
from core.config import DEFAULT_FALLBACK_LEVEL1, DEFAULT_FALLBACK_LEVEL2
from data.categories import CATEGORY_TREE
from data.rules import DETAILED_LEVEL2_RULES, DetailedLevel2Rule, KeywordRule, LEVEL1_RULES, LEVEL2_RULES
from services.llm_client import llm_classify

DOMAIN_STRONG_KEYWORDS: Dict[str, List[str]] = {
    "电梯": ["电梯", "扶梯", "钢丝绳", "主机", "抱闸", "层门", "曳引机", "限速器", "主钢索", "大修", "修理"],
    "消防": ["消火栓", "消防栓", "喷淋", "报警", "灭火器", "防火门", "稳压泵", "报警阀", "消防水带"],
    "监控": ["监控", "摄像头", "球机", "录像", "存储"],
    "防水工程": ["防水工程", "防水层", "防水维修", "防水施工", "渗漏", "漏水", "渗水", "屋面", "屋顶"],
    "外立面修缮": ["粉刷", "空鼓", "脱落", "裂缝", "翻新", "涂料", "修补"],
    "给排水": ["给水", "排水", "水泵", "生活泵", "供水泵", "直供水", "二次供水", "水管"],
    "污水": ["污水", "化粪池", "污水井", "污水管", "污水泵", "排污泵", "集水井", "污水总管"],
    "绿化景观": ["绿化", "补种", "景观", "树木", "草坪", "园路"],
    "停车交通": ["车位", "停车", "停车场", "车辆", "出入口", "道闸", "标线", "交通设施"],
    "公共设施": ["公共区域", "无障碍通道", "入口通道", "防汛挡板", "车棚", "非机动车棚"],
    "弱电系统": ["弱电", "网络", "智能化", "布线", "可视对讲", "楼宇对讲"],
    "门禁设施": ["门禁", "门禁一体机", "刷卡门禁", "人脸门禁", "门控", "楼宇对讲门禁", "车牌识别", "防盗门", "自动门", "单元门"],
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
    has_monitor_context = any(keyword in text for keyword in ["监控", "摄像头", "球机", "录像"])
    has_sewage_context = any(keyword in text for keyword in ["污水泵", "排污泵", "集水井", "化粪池", "污水井", "污水总管", "排污"])
    has_elevator_decoration_context = any(keyword in text for keyword in ["墙面", "粉刷", "翻新", "涂料", "装修"])
    has_waterproof_context = any(keyword in text for keyword in ["屋顶", "屋面", "防水", "漏水", "渗漏", "渗水"])
    has_access_control_context = any(keyword in text for keyword in ["门禁", "对讲", "可视对讲", "智能化", "门禁系统", "梯控"])
    for level1, keywords in DOMAIN_STRONG_KEYWORDS.items():
        hits = [keyword for keyword in keywords if keyword in text]
        if level1 == "电梯" and ("电梯房" in text or has_monitor_context):
            explicit_elevator_work = any(
                keyword in text
                for keyword in [
                    "钢丝绳",
                    "主机",
                    "抱闸",
                    "层门",
                    "扶梯",
                    "电梯维修",
                    "电梯更换",
                    "电梯改造",
                    "电梯更新",
                    "电梯升级",
                    "电梯故障",
                    "电梯抢修",
                ]
            )
            if not explicit_elevator_work:
                hits = [keyword for keyword in hits if keyword != "电梯"]
        if level1 == "电梯" and any(keyword in text for keyword in ["电梯厅", "电梯间", "电梯门套"]) and has_elevator_decoration_context:
            hits = [keyword for keyword in hits if keyword not in {"电梯"}]
        if level1 == "电梯" and "电梯底坑" in text and has_waterproof_context:
            hits = [keyword for keyword in hits if keyword not in {"电梯"}]
        if level1 == "电梯" and has_access_control_context:
            explicit_elevator_work = any(
                keyword in text
                for keyword in [
                    "钢丝绳",
                    "主机",
                    "抱闸",
                    "层门",
                    "扶梯",
                    "曳引机",
                    "限速器",
                    "控制柜",
                    "主钢索",
                    "电梯维修",
                    "电梯更换",
                    "电梯改造",
                    "电梯更新",
                    "电梯升级",
                    "电梯故障",
                    "电梯抢修",
                ]
            )
            if not explicit_elevator_work:
                hits = [keyword for keyword in hits if keyword not in {"电梯"}]
        if level1 == "道路工程" and has_monitor_context:
            explicit_road_work = any(
                keyword in text
                for keyword in ["道路改造", "道路维修", "道路拓宽", "路面维修", "路面翻新", "沥青"]
            )
            if not explicit_road_work:
                hits = [keyword for keyword in hits if keyword not in {"道路", "路面", "人行道"}]
        if level1 == "给排水" and has_sewage_context:
            hits = [keyword for keyword in hits if keyword not in {"排水", "水泵", "水管"}]
        if level1 == "给排水" and "消防泵房" in text:
            hits = [keyword for keyword in hits if keyword != "水泵"]
        if level1 == "弱电系统" and "门禁" in text:
            hits = [keyword for keyword in hits if keyword not in {"楼宇对讲", "对讲", "可视对讲"}]
        if level1 == "防水工程" and "消防水带" in text:
            hits = [keyword for keyword in hits if keyword not in {"防水", "防水工程", "防水维修", "防水施工"}]
        if level1 == "污水" and "防汛挡板" in text:
            hits = []
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


def should_mark_multi_system(primary_level1: str, same_domain_components: Set[str]) -> bool:
    if len(same_domain_components) < 2:
        return False
    if primary_level1 == "电梯" and same_domain_components == {"整梯", "部件"}:
        return False
    if primary_level1 == "门禁设施" and same_domain_components.issubset({"门禁", "对讲"}):
        return False
    return True


def resolve_structure_type(
    primary_level1: str,
    strong_candidate_names: List[str],
    same_domain_components: Set[str],
) -> Tuple[str, List[str]]:
    if len(strong_candidate_names) <= 1:
        if should_mark_multi_system(primary_level1, same_domain_components):
            return "multi_system_same_domain", []
        return "single_project", []

    cross_domain = [level1 for level1 in strong_candidate_names if level1 != primary_level1]
    if cross_domain:
        return "composite_project", cross_domain

    if should_mark_multi_system(primary_level1, same_domain_components):
        return "multi_system_same_domain", []

    return "single_project", []


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
    candidates = collect_level1_candidates(normalized)
    strong_domain_hits = collect_strong_domain_hits(normalized)
    same_domain_components = collect_same_domain_components(normalized, primary_level1)

    strong_candidate_names = [
        level1
        for level1, score, _hits in candidates
        if score >= 3 and level1 in strong_domain_hits
    ]
    is_composite = False
    needs_review = method == "降级兜底"
    composite_reason = None
    structure_type, cross_domain = resolve_structure_type(
        primary_level1,
        strong_candidate_names,
        same_domain_components,
    )
    secondary_candidates: List[str] = []

    if structure_type == "composite_project":
        is_composite = True
        secondary_names = cross_domain[:2]
        reason_domains = strong_candidate_names[:3] if strong_candidate_names else [primary_level1]
        composite_reason = (
            f"同时命中多个工程域：{'、'.join(reason_domains)}"
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
    elif structure_type == "multi_system_same_domain":
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
