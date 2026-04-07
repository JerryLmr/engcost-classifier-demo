import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from core.config import RULE_CONFIG_DIR


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
    path = Path(RULE_CONFIG_DIR) / "object_catalog.json"
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


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

    if str(item["full_path"]) in text:
        score += 8
        hits.add(str(item["full_path"]))
    if str(item["level_3"]) in text:
        score += 6
        hits.add(str(item["level_3"]))
    if str(item["level_2"]) in text:
        score += 5
        hits.add(str(item["level_2"]))
    if str(item["level_1"]) in text:
        score += 3
        hits.add(str(item["level_1"]))

    for keyword in item["_keywords"]:
        if keyword in text or simplify_text(keyword) in simplified_text:
            score += 2 if len(keyword) >= 4 else 1
            hits.add(keyword)

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
