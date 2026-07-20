from __future__ import annotations

from typing import Any

import pandas as pd

from estimator.candidates.signatures import cell_text, numeric_or_none
from estimator.planning.range_selection import ordered_project_items
from estimator.retrieval.evidence_pool import source_identity_for_row

MATCHED_PROJECT_PACKAGE_COLUMNS = [
    "rank",
    "package_query_similarity",
    "project_package_id",
    "工程名称",
    "project_name_text",
    "cost_item_names_summary",
    "consultation_time",
    "location",
    "cache_subject",
    "item_count",
    "item_count_average",
    "is_top5_similarity",
    "item_count_distance_to_average",
    "project_selection_rank",
    "is_selected_package",
]

MATCHED_PROJECT_EXAMPLE_COLUMNS = [
    "rank",
    "project_package_id",
    "工程名称",
    "project_name_text",
    "consultation_time",
    "location",
    "item_order",
    "stable_sample_id",
    "source_ref",
    "cost_item_name",
    "project_description",
    "unit",
    "quantity",
    "unit_price",
    "total_price",
]

def matched_project_packages_for_output(matched: pd.DataFrame) -> pd.DataFrame:
    output = matched.copy()
    for column in MATCHED_PROJECT_PACKAGE_COLUMNS:
        if column not in output.columns:
            output[column] = ""
    return output[MATCHED_PROJECT_PACKAGE_COLUMNS].fillna("")

def select_representative_project_package(
    matched_project_packages: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    if matched_project_packages.empty:
        raise ValueError("matched_project_packages 为空，无法选择历史工程包")
    required = {"package_query_similarity", "item_count", "project_package_id"}
    missing = sorted(required - set(matched_project_packages.columns))
    if missing:
        raise ValueError(f"matched_project_packages 缺少字段: {', '.join(missing)}")

    ranked = matched_project_packages.copy()
    ranked["_recall_order"] = range(len(ranked))
    ranked["package_query_similarity"] = pd.to_numeric(
        ranked["package_query_similarity"], errors="raise"
    )
    ranked["item_count"] = pd.to_numeric(ranked["item_count"], errors="raise")
    if ranked["item_count"].isna().any():
        raise ValueError("matched_project_packages.item_count 不得为空")
    ranked = ranked.sort_values(
        ["package_query_similarity", "_recall_order"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    average_item_count = float(ranked["item_count"].mean())
    ranked["item_count_average"] = average_item_count
    ranked["project_selection_rank"] = range(1, len(ranked) + 1)
    ranked["is_top5_similarity"] = ranked.index < 5
    ranked["item_count_distance_to_average"] = (
        ranked["item_count"].astype(float) - average_item_count
    ).abs()
    top_five = ranked.head(5)
    selected_index = min(
        range(len(top_five)),
        key=lambda index: (
            float(top_five.iloc[index]["item_count_distance_to_average"]),
            index,
        ),
    )
    ranked["is_selected_package"] = False
    ranked.loc[selected_index, "is_selected_package"] = True
    selected_package = ranked.iloc[selected_index].copy()
    return selected_package, ranked.drop(columns=["_recall_order"])

def build_matched_project_examples(
    matched_project_packages: pd.DataFrame,
    samples: pd.DataFrame,
    limit: int = 3,
) -> list[dict[str, Any]]:
    if matched_project_packages.empty or limit <= 0:
        return []
    ranked = matched_project_packages.copy()
    ranked["_rank_order"] = pd.to_numeric(ranked.get("rank"), errors="coerce")
    ranked["_source_order"] = range(len(ranked))
    ranked = ranked.sort_values(["_rank_order", "_source_order"], kind="stable").head(limit)
    examples: list[dict[str, Any]] = []
    for fallback_rank, (_index, project) in enumerate(ranked.iterrows(), start=1):
        project_package_id = cell_text(project.get("project_package_id"))
        project_items = ordered_project_items(samples, project_package_id)
        items: list[dict[str, Any]] = []
        for _item_index, item in project_items.iterrows():
            source_ref = cell_text(item.get("source_ref")) or source_identity_for_row(item)[2]
            items.append(
                {
                    "stable_sample_id": cell_text(item.get("stable_sample_id")),
                    "source_ref": source_ref,
                    "cost_item_name": cell_text(item.get("cost_item_name")),
                    "project_description": cell_text(item.get("project_description")),
                    "unit": cell_text(item.get("unit")) or cell_text(item.get("unit_normalized")),
                    "quantity": numeric_or_none(item.get("quantity")),
                    "unit_price": numeric_or_none(item.get("unit_price")),
                    "total_price": numeric_or_none(item.get("total_price")),
                }
            )
        examples.append(
            {
                "rank": int(numeric_or_none(project.get("rank")) or fallback_rank),
                "project_package_id": project_package_id,
                "package_query_similarity": numeric_or_none(project.get("package_query_similarity")),
                "project_name": cell_text(project.get("工程名称")),
                "project_name_text": cell_text(project.get("project_name_text")),
                "consultation_time": cell_text(project.get("consultation_time")),
                "location": cell_text(project.get("location")),
                "items": items,
            }
        )
    return examples

def matched_project_examples_frame(examples: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for example in examples:
        items = example.get("items") if isinstance(example.get("items"), list) else []
        for item_order, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "rank": example.get("rank", ""),
                    "project_package_id": cell_text(example.get("project_package_id")),
                    "工程名称": cell_text(example.get("project_name")),
                    "project_name_text": cell_text(example.get("project_name_text")),
                    "consultation_time": cell_text(example.get("consultation_time")),
                    "location": cell_text(example.get("location")),
                    "item_order": item_order,
                    "stable_sample_id": cell_text(item.get("stable_sample_id")),
                    "source_ref": cell_text(item.get("source_ref")),
                    "cost_item_name": cell_text(item.get("cost_item_name")),
                    "project_description": cell_text(item.get("project_description")),
                    "unit": cell_text(item.get("unit")),
                    "quantity": item.get("quantity"),
                    "unit_price": item.get("unit_price"),
                    "total_price": item.get("total_price"),
                }
            )
    return pd.DataFrame(rows, columns=MATCHED_PROJECT_EXAMPLE_COLUMNS)
