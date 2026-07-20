from __future__ import annotations

import re
import unicodedata
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from estimator.query_models import QueryRewrite


MUNICIPALITIES = {"北京市", "上海市", "天津市", "重庆市"}
PREFECTURE_LOCATION_PATTERN = re.compile(
    r"^(?:[^,，/、]+省|[^,，/、]+自治区)[^,，/、省市]+市$"
)


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def normalize_location(value: Any) -> str:
    text = unicodedata.normalize("NFKC", _cell_text(value)).strip()
    return re.sub(r"\s+", " ", text)


def validate_query_constraints(
    location: Any,
    start_date: Any,
    end_date: Any,
) -> tuple[str, str, str, list[str]]:
    notes: list[str] = []
    normalized_location = normalize_location(location)
    if normalized_location and normalized_location not in MUNICIPALITIES and not PREFECTURE_LOCATION_PATTERN.fullmatch(normalized_location):
        notes.append(f"location 格式非法，已清空: {normalized_location}")
        normalized_location = ""

    parsed_dates: dict[str, str] = {}
    for field_name, value in (("start_date", start_date), ("end_date", end_date)):
        text = _cell_text(value)
        if not text:
            parsed_dates[field_name] = ""
            continue
        try:
            parsed_dates[field_name] = datetime.strptime(text, "%Y-%m-%d").date().isoformat()
        except ValueError:
            parsed_dates[field_name] = ""
            notes.append(f"{field_name} 格式非法，已清空: {text}")

    start_text = parsed_dates["start_date"]
    end_text = parsed_dates["end_date"]
    if start_text and end_text and start_text > end_text:
        notes.append("start_date 晚于 end_date，两个日期约束均已清空")
        start_text = ""
        end_text = ""
    return normalized_location, start_text, end_text, notes


def parse_consultation_dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, format="%Y-%m-%d", errors="coerce").dt.date


def build_constraint_mask(
    rows: pd.DataFrame,
    location: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    mask = pd.Series(True, index=rows.index, dtype=bool)
    if location:
        if "location" not in rows.columns:
            return pd.Series(False, index=rows.index, dtype=bool)
        mask &= rows["location"].map(normalize_location).eq(location)
    if start_date or end_date:
        if "consultation_time" not in rows.columns:
            return pd.Series(False, index=rows.index, dtype=bool)
        dates = parse_consultation_dates(rows["consultation_time"])
        if start_date:
            mask &= dates.notna() & dates.ge(datetime.strptime(start_date, "%Y-%m-%d").date())
        if end_date:
            mask &= dates.notna() & dates.le(datetime.strptime(end_date, "%Y-%m-%d").date())
    return mask


def filter_rows_and_embeddings(
    rows: pd.DataFrame,
    embeddings: np.ndarray,
    mask: pd.Series,
    label: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    if len(rows) != len(embeddings) or len(rows) != len(mask):
        raise ValueError(f"{label} 约束过滤前 DataFrame、embedding 与 mask 行数不一致")
    filtered_rows = rows.loc[mask].copy()
    filtered_embeddings = embeddings[mask.to_numpy()]
    if len(filtered_rows) != len(filtered_embeddings):
        raise ValueError(f"约束过滤后的{label}与 embedding 行数不一致")
    return filtered_rows, filtered_embeddings


def ensure_project_package_candidates(rows: pd.DataFrame, rewrite: QueryRewrite) -> None:
    if not rows.empty:
        return
    if rewrite.location or rewrite.start_date or rewrite.end_date:
        raise ValueError("没有找到同时满足地域和时间约束的历史工程包。")
    raise ValueError("没有找到历史工程包。")
