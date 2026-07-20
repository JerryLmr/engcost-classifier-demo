from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PACKAGE_WEIGHT_TEMPERATURE = 0.10
PACKAGE_EVIDENCE_WEIGHT_COLUMNS = [
    "project_package_id",
    "package_query_similarity",
    "package_evidence_weight",
]


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def evidence_package_universe(
    matched_project_packages: pd.DataFrame,
    direct_item_hits: pd.DataFrame,
) -> list[str]:
    package_ids: list[str] = []
    seen: set[str] = set()
    for frame in [matched_project_packages, direct_item_hits]:
        if frame.empty or "project_package_id" not in frame.columns:
            continue
        for value in frame["project_package_id"].tolist():
            package_id = _cell_text(value)
            if package_id and package_id not in seen:
                package_ids.append(package_id)
                seen.add(package_id)
    return package_ids


def build_package_evidence_weights(
    evidence_package_ids: list[str],
    package_query_similarity_by_id: dict[str, float],
    temperature: float,
) -> pd.DataFrame:
    if temperature <= 0:
        raise ValueError("package weight temperature 必须大于 0")
    package_ids = [package_id for package_id in evidence_package_ids if _cell_text(package_id)]
    if not package_ids:
        return pd.DataFrame(columns=PACKAGE_EVIDENCE_WEIGHT_COLUMNS)

    similarities = np.array(
        [float(package_query_similarity_by_id.get(package_id, 0.0)) for package_id in package_ids],
        dtype=np.float64,
    )
    max_similarity = float(np.max(similarities))
    raw_weights = np.exp((similarities - max_similarity) / float(temperature))
    denominator = float(raw_weights.sum())
    weights = raw_weights / denominator if denominator > 0 else np.zeros_like(raw_weights)
    return pd.DataFrame(
        {
            "project_package_id": package_ids,
            "package_query_similarity": similarities.astype(float),
            "package_evidence_weight": weights.astype(float),
        },
        columns=PACKAGE_EVIDENCE_WEIGHT_COLUMNS,
    )
