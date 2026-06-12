import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List


CATALOG_PATH = Path(__file__).resolve().parents[1] / "config" / "catalog.json"


@dataclass(frozen=True)
class CatalogItem:
    id: str
    level1: str
    level2: str
    level3_items: tuple[str, ...]
    object_keywords: tuple[str, ...]
    action_keywords: tuple[str, ...]
    weak_keywords: tuple[str, ...]
    min_score: int

    @property
    def label(self) -> str:
        return f"{self.id} {self.level1} > {self.level2}"

    @property
    def item_label(self) -> str:
        return "、".join(self.level3_items)


def _as_string_list(value: Any, field_name: str, item_id: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"catalog item {item_id} has invalid {field_name}")
    return tuple(item.strip() for item in value if item.strip())


def _parse_item(raw: Dict[str, Any]) -> CatalogItem:
    item_id = str(raw.get("id", "")).strip()
    if not item_id:
        raise ValueError("catalog item missing id")
    for field_name in ("level1", "level2"):
        if not isinstance(raw.get(field_name), str) or not raw[field_name].strip():
            raise ValueError(f"catalog item {item_id} missing {field_name}")
    level3_items = _as_string_list(raw.get("level3_items"), "level3_items", item_id)
    if not level3_items:
        raise ValueError(f"catalog item {item_id} missing level3_items")

    rules = raw.get("rules") or {}
    if not isinstance(rules, dict):
        raise ValueError(f"catalog item {item_id} has invalid rules")

    min_score = rules.get("min_score", 1)
    if not isinstance(min_score, int):
        raise ValueError(f"catalog item {item_id} has invalid min_score")

    return CatalogItem(
        id=item_id,
        level1=raw["level1"].strip(),
        level2=raw["level2"].strip(),
        level3_items=level3_items,
        object_keywords=_as_string_list(rules.get("object_keywords"), "object_keywords", item_id),
        action_keywords=_as_string_list(rules.get("action_keywords"), "action_keywords", item_id),
        weak_keywords=_as_string_list(rules.get("weak_keywords"), "weak_keywords", item_id),
        min_score=min_score,
    )


@lru_cache(maxsize=1)
def load_catalog() -> List[CatalogItem]:
    with CATALOG_PATH.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)

    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("catalog.json must contain non-empty items list")

    catalog = [_parse_item(item) for item in items]
    ids = [item.id for item in catalog]
    duplicate_ids = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
    if duplicate_ids:
        raise ValueError(f"catalog contains duplicate ids: {', '.join(duplicate_ids)}")
    return catalog


def get_catalog_by_id() -> Dict[str, CatalogItem]:
    return {item.id: item for item in load_catalog()}
