from classifier.llm_client import ItemSelection
from classifier.standard_catalog_loader import StandardCatalogItem


PROMOTE_REASON = "已将具体子项提升为主分类"
ROOF_LEAK_REASON = "屋面渗漏水优先归入防水层"
ROOF_STRUCTURE_ITEM_ID = "CP-002-01"
ROOF_WATERPROOF_ITEM_ID = "CP-002-03"
EXTERIOR_WALL_ITEM_ID = "CP-003-01"
ROOF_LEAK_TERMS = ("漏水", "渗水", "渗漏", "渗漏水", "防水", "补漏", "筑漏")
ROOF_STRUCTURE_TERMS = ("檐口", "屋脊", "瓦面", "型材屋面", "构造层", "平改坡", "腐木", "脱落")
EXTERIOR_WALL_TERMS = ("外墙", "墙面", "墙体")


def _is_unspecified_internal_item(item: StandardCatalogItem | None) -> bool:
    if item is None:
        return False
    return item.item_group == "内部扩展项" or item.item == "未明确具体子项"


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    compact_text = "".join(str(text or "").split())
    return any(term in compact_text for term in terms if term)


def _append_reason(reason: str, suffix: str) -> str:
    parts = [part.strip() for part in str(reason or "").split("；") if part.strip()]
    if suffix not in parts:
        parts.append(suffix)
    return "；".join(parts) or suffix


def postprocess_item_selection(
    project_name: str,
    item_selection: ItemSelection,
    catalog_by_id: dict[str, StandardCatalogItem],
) -> ItemSelection:
    item_selection = roof_leak_postprocess(project_name, item_selection, catalog_by_id)

    selected_item = catalog_by_id.get(item_selection.catalog_id)
    if not _is_unspecified_internal_item(selected_item):
        return item_selection

    promoted_id = ""
    for item_id in item_selection.secondary_catalog_ids:
        secondary_item = catalog_by_id.get(item_id)
        if (
            secondary_item is not None
            and secondary_item.category == selected_item.category
            and not _is_unspecified_internal_item(secondary_item)
        ):
            promoted_id = item_id
            break

    if not promoted_id:
        return item_selection

    secondary_ids: list[str] = []
    for item_id in item_selection.secondary_catalog_ids:
        if item_id == promoted_id or item_id in secondary_ids:
            continue
        secondary_item = catalog_by_id.get(item_id)
        if (
            secondary_item is not None
            and secondary_item.category == selected_item.category
            and _is_unspecified_internal_item(secondary_item)
        ):
            continue
        secondary_ids.append(item_id)

    has_other_concrete_secondary = any(
        not _is_unspecified_internal_item(catalog_by_id.get(item_id))
        for item_id in secondary_ids
    )

    return ItemSelection(
        catalog_id=promoted_id,
        secondary_catalog_ids=tuple(secondary_ids),
        is_composite=True if has_other_concrete_secondary else item_selection.is_composite,
        needs_review=True,
        reason=_append_reason(item_selection.reason, PROMOTE_REASON),
        project_name_text=item_selection.project_name_text,
        invalid_after_retry=item_selection.invalid_after_retry,
    )


def roof_leak_postprocess(
    project_name: str,
    item_selection: ItemSelection,
    catalog_by_id: dict[str, StandardCatalogItem],
) -> ItemSelection:
    if item_selection.catalog_id != ROOF_STRUCTURE_ITEM_ID:
        return item_selection
    if ROOF_WATERPROOF_ITEM_ID not in catalog_by_id:
        return item_selection
    if not _contains_any(project_name, ROOF_LEAK_TERMS):
        return item_selection
    if _contains_any(project_name, ROOF_STRUCTURE_TERMS):
        return item_selection

    secondary_ids: list[str] = []
    for item_id in item_selection.secondary_catalog_ids:
        if item_id in (ROOF_STRUCTURE_ITEM_ID, ROOF_WATERPROOF_ITEM_ID) or item_id in secondary_ids:
            continue
        secondary_ids.append(item_id)

    if (
        EXTERIOR_WALL_ITEM_ID in catalog_by_id
        and _contains_any(project_name, EXTERIOR_WALL_TERMS)
        and EXTERIOR_WALL_ITEM_ID not in secondary_ids
    ):
        secondary_ids.append(EXTERIOR_WALL_ITEM_ID)

    return ItemSelection(
        catalog_id=ROOF_WATERPROOF_ITEM_ID,
        secondary_catalog_ids=tuple(secondary_ids),
        is_composite=True if secondary_ids else item_selection.is_composite,
        needs_review=True,
        reason=_append_reason(item_selection.reason, ROOF_LEAK_REASON),
        project_name_text=item_selection.project_name_text,
        invalid_after_retry=item_selection.invalid_after_retry,
    )
