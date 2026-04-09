import json
import re
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from core.config import resolve_rule_file


GENERIC_TOKENS = {
    "工程",
    "项目",
    "维修",
    "维修工程",
    "维修项目",
    "修缮",
    "整修",
    "设施",
    "系统",
    "设备",
    "事项",
    "对象",
    "区域",
    "公共",
    "小区",
}
DERIVED_SUFFIXES = ("系统", "设施", "设备", "事项", "对象", "工程", "点")
SPLIT_PATTERN = re.compile(r"[，,、；;|\n]+")
DEFAULT_GENERIC_LEVEL3_TERMS = {"主机", "系统", "设备", "装置", "设施", "工程", "项目"}
DEFAULT_DOMAIN_INTENT_RULES = [
    {
        "intent_keyword": "电梯",
        "preferred_level1": "电梯",
        "preferred_bonus": 4,
        "non_preferred_penalty": 1,
    }
]
DEFAULT_SEMANTIC_BRIDGES = [
    {
        "all_keywords": ("电梯", "主机"),
        "target_level1": "电梯",
        "target_level3": "曳引机",
        "bonus": 6,
    }
]
DEFAULT_SINGLE_DOMAIN_FOCUS = {"enabled": True, "min_score_gap": 3}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "")


def simplify_text(text: str) -> str:
    simplified = normalize_text(text)
    for token in ("系统", "工程", "项目", "事项", "对象", "设施", "设备"):
        simplified = simplified.replace(token, "")
    return simplified


def _is_generic_phrase(text: str) -> bool:
    return text in GENERIC_TOKENS or len(text) < 2


def _split_keywords(value: str) -> Set[str]:
    fragments = re.split(r"[/、，,（）()]+", value)
    return {fragment for fragment in fragments if fragment}


def _derive_aliases(value: str) -> Set[str]:
    aliases: Set[str] = set()
    if "楼栋外立面" in value or "外墙饰面" in value or "外墙渗漏点" in value:
        aliases.add("外墙")
    if "平屋面" in value or "坡屋面" in value or "屋顶" in value:
        aliases.update({"屋面", "屋顶"})
    if "公共窗户" in value:
        aliases.update({"窗户", "玻璃"})
    if "防盗门禁" in value or "小区防盗门禁" in value:
        aliases.update({"门禁", "防盗门"})
    if "公共景观绿化" in value:
        aliases.add("绿化")
    if "监控系统" in value or "摄像头" in value:
        aliases.update({"监控", "摄像头"})
    if "排水、排污设施" in value:
        aliases.update({"排水", "排污"})
    if "消防系统" in value or "消防泵" in value:
        aliases.add("消防")
    if "电梯" in value:
        aliases.add("电梯")
    if "曳引机" in value:
        aliases.add("曳引机")
    return aliases


def _derive_trimmed_keywords(values: Iterable[str]) -> Set[str]:
    derived: Set[str] = set()
    for value in values:
        for suffix in DERIVED_SUFFIXES:
            if value.endswith(suffix) and len(value) > len(suffix) + 1:
                trimmed = value[: -len(suffix)]
                if not _is_generic_phrase(trimmed):
                    derived.add(trimmed)
    return derived


def _build_item_keywords(item: Dict[str, object]) -> Set[str]:
    raw_values = {
        str(item["level_1"]),
        str(item["level_2"]),
        str(item["level_3"]),
        str(item["full_path"]),
    }
    keywords: Set[str] = set(raw_values)
    for value in raw_values:
        keywords.update(_split_keywords(value))
        keywords.update(_derive_aliases(value))
    keywords.update(_derive_trimmed_keywords(keywords))
    return {
        keyword
        for keyword in keywords
        if len(keyword) >= 2 and not _is_generic_phrase(keyword)
    }


@lru_cache(maxsize=1)
def load_object_catalog() -> Dict[str, object]:
    path = resolve_rule_file("repairable_object_catalog.json", fallback_filenames=("object_catalog.json",))
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


@lru_cache(maxsize=1)
def load_mapping_scoring_config() -> Dict[str, Any]:
    try:
        path = resolve_rule_file("mapping_scoring.json")
        with path.open("r", encoding="utf-8") as fp:
            config = json.load(fp)
    except FileNotFoundError:
        config = {}
    return {
        "generic_level3_terms": set(config.get("generic_level3_terms", list(DEFAULT_GENERIC_LEVEL3_TERMS))),
        "domain_intent_rules": list(config.get("domain_intent_rules", DEFAULT_DOMAIN_INTENT_RULES)),
        "semantic_bridges": list(config.get("semantic_bridges", DEFAULT_SEMANTIC_BRIDGES)),
        "single_domain_focus": dict(config.get("single_domain_focus", DEFAULT_SINGLE_DOMAIN_FOCUS)),
    }


@lru_cache(maxsize=1)
def get_catalog_items() -> List[Dict[str, object]]:
    items = load_object_catalog()["items"]
    catalog_items: List[Dict[str, object]] = []
    for item in items:
        if item.get("status") != "active":
            continue
        catalog_items.append({**item, "_keywords": _build_item_keywords(item)})
    return catalog_items


def split_project_name(project_name: str) -> List[str]:
    parts = [part.strip() for part in SPLIT_PATTERN.split(project_name) if part.strip()]
    if len(parts) <= 1:
        return []
    return parts


def _score_item(text: str, item: Dict[str, object]) -> Tuple[int, Set[str]]:
    score = 0
    hits: Set[str] = set()
    simplified_text = simplify_text(text)
    scoring_config = load_mapping_scoring_config()
    level1 = str(item["level_1"])
    level2 = str(item["level_2"])
    level3 = str(item["level_3"])

    if str(item["full_path"]) in text:
        score += 8
        hits.add(str(item["full_path"]))
    if level3 in text:
        # Generic terminal nodes (e.g. "主机") are too broad for direct high weighting.
        score += 2 if level3 in scoring_config["generic_level3_terms"] else 6
        hits.add(level3)
    if level2 in text:
        score += 5
        hits.add(level2)
    if level1 in text:
        score += 3
        hits.add(level1)

    for keyword in item["_keywords"]:
        if keyword in text or simplify_text(keyword) in simplified_text:
            score += 2 if len(keyword) >= 4 else 1
            hits.add(keyword)

    for rule in scoring_config["domain_intent_rules"]:
        intent_keyword = str(rule.get("intent_keyword", ""))
        preferred_level1 = str(rule.get("preferred_level1", ""))
        preferred_bonus = int(rule.get("preferred_bonus", 0))
        non_preferred_penalty = int(rule.get("non_preferred_penalty", 0))
        if intent_keyword and intent_keyword in text:
            if level1 == preferred_level1:
                score += preferred_bonus
            else:
                score -= non_preferred_penalty

    for bridge in scoring_config["semantic_bridges"]:
        all_keywords = tuple(bridge.get("all_keywords", []))
        target_level1 = str(bridge.get("target_level1", ""))
        target_level3 = str(bridge.get("target_level3", ""))
        bonus = int(bridge.get("bonus", 0))
        if all(keyword in text for keyword in all_keywords) and level1 == target_level1 and level3 == target_level3:
            score += bonus

    return score, hits


def _to_mapped_object(item: Dict[str, object], score: int) -> Dict[str, object]:
    match_score = min(0.99, round(score / 15, 2))
    return {
        "id": int(item["id"]),
        "full_path": str(item["full_path"]),
        "match_score": match_score,
        "match_method": "keyword_path",
        "_level_1": str(item["level_1"]),
    }


def _focus_single_domain_if_needed(
    normalized_part: str,
    candidates: List[Tuple[int, int, Dict[str, object]]],
) -> List[Tuple[int, int, Dict[str, object]]]:
    config = load_mapping_scoring_config()["single_domain_focus"]
    if not config.get("enabled", True):
        return candidates
    if len(candidates) < 2:
        return candidates
    top_score, _top_hits, top_item = candidates[0]
    second_score = candidates[1][0]
    top_domain = str(top_item["level_1"])
    min_score_gap = int(config.get("min_score_gap", 3))
    # Only suppress pseudo cross-domain in single-segment input when one domain is clearly dominant.
    if top_score - second_score >= min_score_gap and top_domain in normalized_part:
        return [candidate for candidate in candidates if str(candidate[2]["level_1"]) == top_domain]
    return candidates


def map_project_name(project_name: str, limit: int = 5) -> Dict[str, object]:
    normalized = normalize_text(project_name)
    split_projects = split_project_name(project_name)
    parts: Sequence[str] = split_projects or [project_name]
    aggregated: Dict[int, Dict[str, object]] = {}

    for part in parts:
        normalized_part = normalize_text(part)
        candidates: List[Tuple[int, int, Dict[str, object]]] = []
        for item in get_catalog_items():
            score, hits = _score_item(normalized_part, item)
            if score < 3:
                continue
            candidates.append((score, len(hits), item))

        candidates.sort(key=lambda entry: (entry[0], entry[1], -int(entry[2]["id"])), reverse=True)
        if not split_projects:
            candidates = _focus_single_domain_if_needed(normalized_part, candidates)
        for score, _hit_count, item in candidates[:2]:
            mapped = _to_mapped_object(item, score)
            existing = aggregated.get(mapped["id"])
            if existing is None or mapped["match_score"] > existing["match_score"]:
                aggregated[mapped["id"]] = mapped

    mapped_objects = sorted(
        aggregated.values(),
        key=lambda item: (item["match_score"], item["id"]),
        reverse=True,
    )[:limit]

    return {
        "mapped_objects": [
            {
                "id": item["id"],
                "full_path": item["full_path"],
                "match_score": item["match_score"],
                "match_method": item["match_method"],
            }
            for item in mapped_objects
        ],
        "matched_object_ids": [item["id"] for item in mapped_objects],
        "split_projects": split_projects,
        "catalog_domains": sorted({item["_level_1"] for item in mapped_objects}),
    }
