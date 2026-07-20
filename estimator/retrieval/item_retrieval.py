from __future__ import annotations

import numpy as np
import pandas as pd

from estimator.retrieval.package_retrieval import top_score_indices


def score_direct_items(
    samples: pd.DataFrame,
    item_query_similarities: np.ndarray,
    top_items: int,
) -> pd.DataFrame:
    indices = top_score_indices(item_query_similarities, top_items)
    rows = samples.iloc[indices].copy()
    rows["item_query_similarity"] = item_query_similarities[indices].astype(float)
    return rows
