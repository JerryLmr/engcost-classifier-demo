import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


STANDARD_CATALOG_PATH = Path(__file__).resolve().parents[1] / "config" / "standard_catalog.json"
OUT_OF_SCOPE_ID = "OUT_OF_SCOPE"


@dataclass(frozen=True)
class StandardCatalogItem:
    id: str
    standard_group: str
    category: str
    item: str
    status_basis: dict[str, str]
    source: dict[str, str]
    item_group: str | None = None

    @property
    def label(self) -> str:
        return catalog_label(self)

    @property
    def allowed_statuses(self) -> tuple[str, ...]:
        return tuple(self.status_basis.keys())


def _required_string(raw: dict[str, Any], field_name: str, item_id: str) -> str:
    value = raw.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"standard catalog item {item_id} missing {field_name}")
    return value.strip()


def _parse_status_basis(raw: dict[str, Any], item_id: str) -> dict[str, str]:
    value = raw.get("status_basis")
    if not isinstance(value, dict) or not value:
        raise ValueError(f"standard catalog item {item_id} has invalid status_basis")

    parsed: dict[str, str] = {}
    for key, basis in value.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"standard catalog item {item_id} has blank status_basis key")
        if not isinstance(basis, str) or not basis.strip():
            raise ValueError(f"standard catalog item {item_id} has blank status_basis value")
        parsed[key.strip()] = basis.strip()
    return parsed


def _parse_source(raw: Any, item_id: str) -> dict[str, str]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"standard catalog item {item_id} has invalid source")
    return {str(key): str(value) for key, value in raw.items()}


def _parse_item(raw: dict[str, Any]) -> StandardCatalogItem:
    item_id = str(raw.get("id", "")).strip()
    if not item_id:
        raise ValueError("standard catalog item missing id")
    if item_id == OUT_OF_SCOPE_ID:
        raise ValueError("OUT_OF_SCOPE must not appear in standard catalog items")

    item_group = raw.get("item_group")
    if item_group is not None and (not isinstance(item_group, str) or not item_group.strip()):
        item_group = None

    return StandardCatalogItem(
        id=item_id,
        standard_group=_required_string(raw, "standard_group", item_id),
        category=_required_string(raw, "category", item_id),
        item=_required_string(raw, "item", item_id),
        status_basis=_parse_status_basis(raw, item_id),
        source=_parse_source(raw.get("source"), item_id),
        item_group=item_group.strip() if isinstance(item_group, str) else None,
    )


@lru_cache(maxsize=1)
def _load_payload() -> dict[str, Any]:
    with STANDARD_CATALOG_PATH.open("r", encoding="utf-8") as fp:
        payload = json.load(fp)
    if not isinstance(payload, dict):
        raise ValueError("standard_catalog.json must contain a JSON object")
    return payload


@lru_cache(maxsize=1)
def load_standard_catalog() -> list[StandardCatalogItem]:
    payload = _load_payload()
    items = payload.get("items")
    if not isinstance(items, list) or not items:
        raise ValueError("standard_catalog.json must contain non-empty items list")

    catalog = [_parse_item(item) for item in items]
    ids = [item.id for item in catalog]
    duplicate_ids = sorted({item_id for item_id in ids if ids.count(item_id) > 1})
    if duplicate_ids:
        raise ValueError(f"standard catalog contains duplicate ids: {', '.join(duplicate_ids)}")
    return catalog


def get_standard_catalog_by_id() -> dict[str, StandardCatalogItem]:
    return {item.id: item for item in load_standard_catalog()}


def catalog_label(item: StandardCatalogItem) -> str:
    return f"{item.id} | {item.category} | {item.item}"


def load_emergency_triggers() -> list[str]:
    emergency_basis = _load_payload().get("emergency_basis") or {}
    if not isinstance(emergency_basis, dict):
        return []
    triggers = emergency_basis.get("triggers") or []
    if not isinstance(triggers, list):
        return []
    return [trigger.strip() for trigger in triggers if isinstance(trigger, str) and trigger.strip()]


def load_fallback_config() -> dict[str, Any]:
    fallback = _load_payload().get("fallback") or {}
    if not isinstance(fallback, dict):
        return {}
    return dict(fallback)
