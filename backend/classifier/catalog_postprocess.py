from classifier.llm_client import ItemSelection
from classifier.standard_catalog_loader import StandardCatalogItem


PROMOTE_REASON = "已将具体子项提升为主分类"


def _is_unspecified_internal_item(item: StandardCatalogItem | None) -> bool:
    if item is None:
        return False
    return item.item_group == "内部扩展项" or item.item == "未明确具体子项"


def _append_reason(reason: str) -> str:
    parts = [part.strip() for part in str(reason or "").split("；") if part.strip()]
    if PROMOTE_REASON not in parts:
        parts.append(PROMOTE_REASON)
    return "；".join(parts) or PROMOTE_REASON


def postprocess_item_selection(
    project_name: str,
    item_selection: ItemSelection,
    catalog_by_id: dict[str, StandardCatalogItem],
) -> ItemSelection:
    del project_name

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
        reason=_append_reason(item_selection.reason),
        invalid_after_retry=item_selection.invalid_after_retry,
    )
