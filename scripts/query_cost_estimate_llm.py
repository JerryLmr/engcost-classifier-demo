#!/usr/bin/env python3
import argparse
import gc
import json
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from classifier.llm_client import LLMServiceError, request_llm_json  # noqa: E402
from classifier.semantic_prompt import SEMANTIC_PROJECT_TEXT_RULES  # noqa: E402


TIME_RANGE_DAYS = {
    "last_year": 365,
    "last_half_year": 182,
    "last_3_months": 90,
}

RECOMMENDED_ITEM_COLUMNS = [
    "序号",
    "一级分类",
    "二级分类",
    "维修状态",
    "清单项名称",
    "项目特征/施工工艺",
    "单位",
    "历史工程量最小值",
    "历史工程量中位数",
    "历史工程量最大值",
    "本次估算金额最小值",
    "本次估算金额中位数",
    "本次估算金额最大值",
    "历史综合单价最小值",
    "历史综合单价中位数",
    "历史综合单价最大值",
    "历史总价最小值",
    "历史总价中位数",
    "历史总价最大值",
    "其中包含人工单价最小值",
    "其中包含人工单价中位数",
    "其中包含人工单价最大值",
    "其中包含机械单价最小值",
    "其中包含机械单价中位数",
    "其中包含机械单价最大值",
    "来源清单行",
    "历史样本数",
]

MATCH_COLUMNS = [
    "project_rank",
    "project_score",
    "project_name_score",
    "project_detail_score",
    "project_key",
    "batch_id",
    "source_row_id",
    "item_row_id",
    "consultation_time",
    "location",
    "catalog_id",
    "一级分类",
    "二级分类",
    "维修状态",
    "标准对象",
    "是否复合工程",
    "复合目录",
    "seq",
    "cost_item_name",
    "unit",
    "unit_normalized",
    "quantity",
    "unit_price",
    "total_price",
    "labor_unit_price",
    "machinery_unit_price",
]

DEBUG_MATCH_COLUMNS = [
    "工程名称",
    "project_name_text",
    "project_detail_text",
]

@dataclass(frozen=True)
class ParsedQuery:
    raw_query: str
    semantic_query_text: str
    quantity: float | None
    unit: str
    location: str
    consultation_time_from: date | None
    consultation_time_to: date | None
    parse_notes: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="自然语言维修工程造价查询入口")
    parser.add_argument("--index-dir", default="embeddings", help="索引目录，默认 embeddings")
    parser.add_argument("--text", required=True, help="口语化维修需求")
    parser.add_argument("--top-k", type=int, default=20, help="召回的相似工程组数量")
    parser.add_argument("--output", default=None, help="xlsx 输出路径，默认 query/YYYYMMDDHHMM.xlsx")
    parser.add_argument("--overwrite", action="store_true", help="若输出文件已存在则覆盖")
    parser.add_argument("--include-debug-text", action="store_true", help="matches 输出历史工程明文字段")
    parser.add_argument("--display", action="store_true", help="recommend_items 数值单元格拼接单位和金额单位")
    parser.add_argument("--project-name-weight", type=float, default=0.85, help="工程名称 embedding 召回权重")
    parser.add_argument("--project-detail-weight", type=float, default=0.15, help="工程明细 embedding 召回权重")
    parser.add_argument("--parsed-quantity", type=float, default=None, help="高级覆盖：用户工程量")
    parser.add_argument("--parsed-unit", default="", help="高级覆盖：用户工程量单位")
    parser.add_argument("--parsed-location", default="", help="高级覆盖：样本库 location")
    parser.add_argument("--consultation-time-from", default="", help="高级覆盖：咨询时间起点 YYYY-MM-DD")
    parser.add_argument("--consultation-time-to", default="", help="高级覆盖：咨询时间终点 YYYY-MM-DD")
    return parser.parse_args()


def default_query_output_path() -> Path:
    return Path("query") / f"{datetime.now().strftime('%Y%m%d%H%M')}.xlsx"


def validate_output_path(output_path: Path | None, overwrite: bool) -> None:
    if output_path is None:
        return
    if output_path.exists() and output_path.is_dir():
        raise ValueError(f"输出路径是目录，不是文件: {output_path}")
    if output_path.exists() and not overwrite:
        raise ValueError(f"输出已存在，请加 --overwrite 或更换输出路径: {output_path}")


def load_embedding_model(model_name: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("缺少依赖 sentence-transformers，请先安装 requirements.txt") from exc

    try:
        return SentenceTransformer(model_name, device="cpu")
    except Exception as exc:
        raise RuntimeError(f"embedding 模型加载失败: {model_name}: {exc}") from exc


def release_embedding_model(model: Any) -> None:
    del model
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


def encode_query(model: Any, text: str) -> np.ndarray:
    embedding = model.encode(
        [text or ""],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return normalize_embeddings(embedding)[0]


def normalize_project_weights(project_name_weight: float, project_detail_weight: float) -> tuple[float, float]:
    if project_name_weight < 0 or project_detail_weight < 0:
        raise ValueError("--project-name-weight 和 --project-detail-weight 必须大于等于 0")
    total = project_name_weight + project_detail_weight
    if total <= 0:
        raise ValueError("--project-name-weight 和 --project-detail-weight 不能同时为 0")
    return project_name_weight / total, project_detail_weight / total


def load_index(index_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    samples_path = index_dir / "samples.parquet"
    groups_path = index_dir / "project_groups.parquet"
    name_path = index_dir / "project_name_embeddings.npy"
    detail_path = index_dir / "project_detail_embeddings.npy"
    meta_path = index_dir / "index_meta.json"

    missing = [path.name for path in [samples_path, groups_path, name_path, detail_path, meta_path] if not path.exists()]
    if missing:
        raise ValueError(f"索引目录缺少文件: {', '.join(missing)}")

    samples = pd.read_parquet(samples_path)
    project_groups = pd.read_parquet(groups_path)
    project_name_embeddings = np.load(name_path)
    project_detail_embeddings = np.load(detail_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if len(project_groups) != project_name_embeddings.shape[0]:
        raise ValueError("工程分组数量与工程名称 embedding 数量不一致")
    if len(project_groups) != project_detail_embeddings.shape[0]:
        raise ValueError("工程分组数量与工程明细 embedding 数量不一致")
    if project_name_embeddings.ndim != 2 or project_detail_embeddings.ndim != 2:
        raise ValueError("工程分组 embedding 维度不正确")
    if project_name_embeddings.shape[1] != project_detail_embeddings.shape[1]:
        raise ValueError("工程名称 embedding 与工程明细 embedding 维度不一致")
    return samples, project_groups, project_name_embeddings, project_detail_embeddings, meta


def cell_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def safe_positive_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number <= 0:
        return None
    return number


def normalize_query_unit(value: Any) -> str:
    text = cell_text(value).lower().replace(" ", "")
    mapping = {
        "平": "m²",
        "平方": "m²",
        "平米": "m²",
        "平方米": "m²",
        "㎡": "m²",
        "m2": "m²",
        "m^2": "m²",
        "m²": "m²",
        "米": "m",
        "m": "m",
        "项": "项",
        "台": "台",
        "套": "套",
        "次": "次",
        "个": "个",
    }
    return mapping.get(text, "")


def parse_date_text(value: Any) -> date | None:
    text = cell_text(value)
    if not text:
        return None
    parsed = pd.to_datetime(pd.Series([text]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return None
    return parsed.date()


def time_range_to_dates(time_range_type: Any, base_date: date | None = None) -> tuple[date | None, date | None]:
    range_type = cell_text(time_range_type)
    if range_type == "none" or not range_type:
        return None, None
    days = TIME_RANGE_DAYS.get(range_type)
    if days is None:
        return None, None
    today = base_date or date.today()
    return today - timedelta(days=days), today


def map_location_hint(location_hint: Any, project_groups: pd.DataFrame) -> tuple[str, list[str]]:
    hint = cell_text(location_hint)
    if not hint or "location" not in project_groups.columns:
        return "", []

    locations = sorted(
        {cell_text(value) for value in project_groups["location"].dropna().tolist() if cell_text(value)},
        key=lambda item: (len(item), item),
    )
    if hint in locations:
        return hint, []

    matches = [location for location in locations if hint in location]
    if not matches:
        return "", [f"location_hint 未匹配样本库 location: {hint}"]
    if len(matches) > 1:
        return matches[0], [f"location_hint 匹配多个 location，已选择最短值: {matches[0]}"]
    return matches[0], []


def build_parse_query_prompt(query: str) -> str:
    return f"""
你是一个维修工程造价查询意图解析器。请把用户口语化查询解析成严格 JSON。

只允许输出 JSON，不要解释，不要 Markdown。

字段固定为：
- semantic_query_text: 用户 query 中抽取出的“用于相似项目检索的工程语义文本”。必须和历史项目侧 project_name_text 使用同一套抽取口径。
- quantity: 用户提到的工程量数字，没有则 null。
- unit: 用户提到的单位，统一成 m²、m、项、台、套、次、个等，没有则 null。
- location_hint: 用户提到的地点词，没有则 null。
- time_range_type: 只能是 last_year、last_half_year、last_3_months、none。

不要推断价格。
不要推荐清单。
不要输出目录分类。
不要输出 catalog_id。
不要输出 answer。

semantic_query_text 抽取规则：
{SEMANTIC_PROJECT_TEXT_RULES}

用户查询：{query}
""".strip()


def fallback_parsed_query(query: str, note: str) -> ParsedQuery:
    return ParsedQuery(
        raw_query=query,
        semantic_query_text=query,
        quantity=None,
        unit="",
        location="",
        consultation_time_from=None,
        consultation_time_to=None,
        parse_notes=[note],
    )


def parse_query_requirements(query: str, project_groups: pd.DataFrame) -> ParsedQuery:
    try:
        result = request_llm_json(
            build_parse_query_prompt(query),
            max_tokens=512,
            system_prompt="你只输出一个 JSON object，不输出解释、Markdown 或思考过程。",
        )
    except (LLMServiceError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        return fallback_parsed_query(query, f"LLM 解析失败，已回退为原始 query 检索: {exc}")

    if not isinstance(result, dict):
        return fallback_parsed_query(query, "LLM 输出不是 JSON object，已回退为原始 query 检索")

    notes: list[str] = []
    semantic_query_text = cell_text(result.get("semantic_query_text")) or query
    quantity = safe_positive_float(result.get("quantity"))
    unit = normalize_query_unit(result.get("unit"))
    if result.get("unit") is not None and not unit:
        notes.append(f"单位无法识别，已置空: {cell_text(result.get('unit'))}")

    location, location_notes = map_location_hint(result.get("location_hint"), project_groups)
    notes.extend(location_notes)

    time_from, time_to = time_range_to_dates(result.get("time_range_type"))
    time_range_type = cell_text(result.get("time_range_type"))
    if time_range_type and time_range_type not in {*TIME_RANGE_DAYS.keys(), "none"}:
        notes.append(f"time_range_type 无效，未启用时间过滤: {time_range_type}")

    return ParsedQuery(
        raw_query=query,
        semantic_query_text=semantic_query_text,
        quantity=quantity,
        unit=unit,
        location=location,
        consultation_time_from=time_from,
        consultation_time_to=time_to,
        parse_notes=notes,
    )


def apply_parsed_overrides(
    parsed: ParsedQuery,
    project_groups: pd.DataFrame,
    parsed_quantity: float | None = None,
    parsed_unit: str = "",
    parsed_location: str = "",
    consultation_time_from: str = "",
    consultation_time_to: str = "",
) -> ParsedQuery:
    notes = list(parsed.parse_notes)
    quantity = safe_positive_float(parsed_quantity) if parsed_quantity is not None else parsed.quantity
    unit = normalize_query_unit(parsed_unit) if parsed_unit else parsed.unit
    location = parsed.location
    if parsed_location:
        mapped_location, location_notes = map_location_hint(parsed_location, project_groups)
        location = mapped_location
        notes.extend(location_notes)
        if not mapped_location:
            notes.append(f"覆盖 location 未匹配样本库，已置空: {parsed_location}")

    time_from = parse_date_text(consultation_time_from) if consultation_time_from else parsed.consultation_time_from
    time_to = parse_date_text(consultation_time_to) if consultation_time_to else parsed.consultation_time_to
    if consultation_time_from and time_from is None:
        notes.append(f"consultation-time-from 无法解析，已忽略: {consultation_time_from}")
    if consultation_time_to and time_to is None:
        notes.append(f"consultation-time-to 无法解析，已忽略: {consultation_time_to}")

    return ParsedQuery(
        raw_query=parsed.raw_query,
        semantic_query_text=parsed.semantic_query_text,
        quantity=quantity,
        unit=unit,
        location=location,
        consultation_time_from=time_from,
        consultation_time_to=time_to,
        parse_notes=notes,
    )


def filter_project_groups(
    project_groups: pd.DataFrame,
    project_name_embeddings: np.ndarray,
    project_detail_embeddings: np.ndarray,
    parsed: ParsedQuery,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    mask = np.ones(len(project_groups), dtype=bool)

    if parsed.location:
        if "location" not in project_groups.columns:
            mask &= False
        else:
            mask &= project_groups["location"].fillna("").astype(str).str.strip().eq(parsed.location).to_numpy()

    if parsed.consultation_time_from or parsed.consultation_time_to:
        if "consultation_time" not in project_groups.columns:
            mask &= False
        else:
            dates = pd.to_datetime(project_groups["consultation_time"], errors="coerce")
            date_mask = dates.notna().to_numpy()
            if parsed.consultation_time_from:
                date_mask &= (dates >= pd.Timestamp(parsed.consultation_time_from)).to_numpy()
            if parsed.consultation_time_to:
                date_mask &= (dates <= pd.Timestamp(parsed.consultation_time_to)).to_numpy()
            mask &= date_mask

    return project_groups.loc[mask].copy(), project_name_embeddings[mask], project_detail_embeddings[mask]


def score_project_groups(
    project_groups: pd.DataFrame,
    project_name_embeddings: np.ndarray,
    project_detail_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    top_k: int,
    project_name_weight: float,
    project_detail_weight: float,
) -> pd.DataFrame:
    if project_groups.empty or top_k <= 0:
        rows = project_groups.head(0).copy()
        rows.insert(0, "project_detail_score", [])
        rows.insert(0, "project_name_score", [])
        rows.insert(0, "project_score", [])
        rows.insert(0, "project_rank", [])
        return rows

    normalized_name_weight, normalized_detail_weight = normalize_project_weights(project_name_weight, project_detail_weight)
    project_name_scores = project_name_embeddings @ query_embedding
    project_detail_scores = project_detail_embeddings @ query_embedding
    scores = normalized_name_weight * project_name_scores + normalized_detail_weight * project_detail_scores
    rows = project_groups.copy()
    rows["project_score"] = scores
    rows["project_name_score"] = project_name_scores
    rows["project_detail_score"] = project_detail_scores
    rows = rows.sort_values("project_score", ascending=False).head(top_k).copy()
    rows.insert(0, "project_rank", range(1, len(rows) + 1))
    return rows


def expand_matched_samples(samples: pd.DataFrame, matched_projects: pd.DataFrame) -> pd.DataFrame:
    if matched_projects.empty:
        return pd.DataFrame(columns=[*MATCH_COLUMNS, *DEBUG_MATCH_COLUMNS])

    if "project_key" not in samples.columns or "project_key" not in matched_projects.columns:
        raise ValueError("samples 和 matched_projects 必须包含 project_key")

    project_columns = [
        "project_key",
        "project_rank",
        "project_score",
        "project_name_score",
        "project_detail_score",
        "project_detail_text",
    ]
    available_project_columns = [column for column in project_columns if column in matched_projects.columns]
    project_meta = matched_projects[available_project_columns].copy()
    project_keys = project_meta["project_key"].tolist()

    rows = samples[samples["project_key"].isin(project_keys)].copy()
    if rows.empty:
        return pd.DataFrame(columns=[*MATCH_COLUMNS, *DEBUG_MATCH_COLUMNS])
    rows = rows.merge(project_meta, on="project_key", how="left")
    rows["_source_order"] = rows["project_key"].map({project_key: index for index, project_key in enumerate(project_keys)})
    sort_columns = ["project_rank", "_source_order"]
    if "seq" in rows.columns:
        sort_columns.append("seq")
    rows = rows.sort_values(sort_columns).drop(columns=["_source_order"])
    return rows


def numeric_values(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").dropna()


def range_stats(frame: pd.DataFrame, column: str) -> dict[str, float | None]:
    values = numeric_values(frame, column)
    if values.empty:
        return {f"{column}_min": None, f"{column}_median": None, f"{column}_max": None}
    return {
        f"{column}_min": float(values.min()),
        f"{column}_median": float(values.median()),
        f"{column}_max": float(values.max()),
    }


def ordered_join(values: pd.Series) -> str:
    output: list[str] = []
    for value in values.tolist():
        text = cell_text(value)
        if text and text not in output:
            output.append(text)
    return ", ".join(output)


def first_non_empty(values: pd.Series) -> str:
    for value in values.tolist():
        text = cell_text(value)
        if text:
            return text
    return ""


def source_item_refs(group: pd.DataFrame) -> pd.Series:
    if "project_key" not in group.columns or "item_row_id" not in group.columns:
        return pd.Series(dtype=object)
    return group.apply(
        lambda row: "::".join(
            part for part in [cell_text(row.get("project_key")), cell_text(row.get("item_row_id"))] if part
        ),
        axis=1,
    )


def aggregate_recommend_items(matches: pd.DataFrame, parsed: ParsedQuery) -> pd.DataFrame:
    if matches.empty:
        return pd.DataFrame(columns=RECOMMENDED_ITEM_COLUMNS)

    rows: list[dict[str, Any]] = []
    group_columns = ["cost_item_name", "unit_normalized", "catalog_id", "维修状态"]
    for column in group_columns:
        if column not in matches.columns:
            matches[column] = ""

    for _signature, group in matches.groupby(group_columns, sort=False, dropna=False):
        first = group.iloc[0]
        unit_normalized = cell_text(first.get("unit_normalized"))
        row: dict[str, Any] = {
            "清单项名称": cell_text(first.get("cost_item_name")),
            "项目特征/施工工艺": first_non_empty(group["project_description"]) if "project_description" in group.columns else "",
            "单位": unit_normalized,
            "历史样本数": int(len(group)),
            "catalog_id": cell_text(first.get("catalog_id")),
            "一级分类": cell_text(first.get("一级分类")),
            "二级分类": cell_text(first.get("二级分类")),
            "维修状态": cell_text(first.get("维修状态")),
            "unit_normalized": unit_normalized,
            "来源清单行": ordered_join(source_item_refs(group)),
            "max_project_score": float(pd.to_numeric(group["project_score"], errors="coerce").max()),
        }
        row.update(rename_stats(range_stats(group, "quantity"), "quantity", "历史工程量"))
        row.update(rename_stats(range_stats(group, "unit_price"), "unit_price", "历史综合单价"))
        row.update(rename_stats(range_stats(group, "total_price"), "total_price", "历史总价"))
        row.update(rename_stats(range_stats(group, "labor_unit_price"), "labor_unit_price", "其中包含人工单价"))
        row.update(rename_stats(range_stats(group, "machinery_unit_price"), "machinery_unit_price", "其中包含机械单价"))

        if parsed.quantity is not None and parsed.unit and row["unit_normalized"] == parsed.unit:
            row["本次估算金额最小值"] = multiply_or_none(row.get("历史综合单价最小值"), parsed.quantity)
            row["本次估算金额中位数"] = multiply_or_none(row.get("历史综合单价中位数"), parsed.quantity)
            row["本次估算金额最大值"] = multiply_or_none(row.get("历史综合单价最大值"), parsed.quantity)
        else:
            row["本次估算金额最小值"] = row.get("历史总价最小值")
            row["本次估算金额中位数"] = row.get("历史总价中位数")
            row["本次估算金额最大值"] = row.get("历史总价最大值")
        rows.append(row)

    recommended = pd.DataFrame(rows)
    recommended["_has_unit_price"] = recommended["历史综合单价中位数"].notna()
    recommended["_has_total_price"] = recommended["历史总价中位数"].notna()
    recommended = recommended.sort_values(
        ["max_project_score", "历史样本数", "_has_unit_price", "_has_total_price"],
        ascending=[False, False, False, False],
    ).drop(columns=["max_project_score", "_has_unit_price", "_has_total_price", "catalog_id"])
    recommended.insert(0, "序号", range(1, len(recommended) + 1))

    for column in RECOMMENDED_ITEM_COLUMNS:
        if column not in recommended.columns:
            recommended[column] = None

    return recommended[RECOMMENDED_ITEM_COLUMNS]


def rename_stats(stats: dict[str, float | None], source_prefix: str, target_prefix: str) -> dict[str, float | None]:
    return {
        f"{target_prefix}最小值": stats.get(f"{source_prefix}_min"),
        f"{target_prefix}中位数": stats.get(f"{source_prefix}_median"),
        f"{target_prefix}最大值": stats.get(f"{source_prefix}_max"),
    }


def multiply_or_none(value: Any, quantity: float) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return None
    return float(number) * quantity


def matches_for_output(matches: pd.DataFrame, include_debug_text: bool) -> pd.DataFrame:
    columns = [*MATCH_COLUMNS, *(DEBUG_MATCH_COLUMNS if include_debug_text else [])]
    if matches.empty:
        return pd.DataFrame(columns=columns)
    output = matches.copy()
    for column in columns:
        if column not in output.columns:
            output[column] = ""
    return output[columns]


QUANTITY_DISPLAY_COLUMNS = (
    "历史工程量最小值",
    "历史工程量中位数",
    "历史工程量最大值",
)

UNIT_PRICE_DISPLAY_COLUMNS = (
    "历史综合单价最小值",
    "历史综合单价中位数",
    "历史综合单价最大值",
    "其中包含人工单价最小值",
    "其中包含人工单价中位数",
    "其中包含人工单价最大值",
    "其中包含机械单价最小值",
    "其中包含机械单价中位数",
    "其中包含机械单价最大值",
)

AMOUNT_DISPLAY_COLUMNS = (
    "本次估算金额最小值",
    "本次估算金额中位数",
    "本次估算金额最大值",
    "历史总价最小值",
    "历史总价中位数",
    "历史总价最大值",
)


def display_number(value: Any) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number) or not np.isfinite(float(number)):
        return ""
    return f"{float(number):.2f}"


def format_recommend_value(value: Any, column: str, unit: str) -> str:
    number = display_number(value)
    if not number:
        return ""
    if column in QUANTITY_DISPLAY_COLUMNS:
        return f"{number} {unit}".strip()
    if column in UNIT_PRICE_DISPLAY_COLUMNS:
        return f"{number} 元/{unit}" if unit else f"{number} 元"
    if column in AMOUNT_DISPLAY_COLUMNS:
        return f"{number} 元"
    return number


def recommend_items_for_output(recommend_items: pd.DataFrame, display: bool = False) -> pd.DataFrame:
    working = recommend_items.copy()
    display_columns = [*QUANTITY_DISPLAY_COLUMNS, *UNIT_PRICE_DISPLAY_COLUMNS, *AMOUNT_DISPLAY_COLUMNS]

    for column in RECOMMENDED_ITEM_COLUMNS:
        if column not in working.columns:
            working[column] = ""

    if display:
        for column in display_columns:
            if column in working.columns:
                working[column] = working[column].astype(object)
        for row_index, row in working.iterrows():
            unit = cell_text(row.get("单位"))
            for column in display_columns:
                working.at[row_index, column] = format_recommend_value(row.get(column), column, unit)

    output = working[RECOMMENDED_ITEM_COLUMNS].copy()
    return output.fillna("")


def time_range_text(parsed: ParsedQuery) -> str:
    if not parsed.consultation_time_from or not parsed.consultation_time_to:
        return "未限制"
    days = (parsed.consultation_time_to - parsed.consultation_time_from).days
    mapping = {
        TIME_RANGE_DAYS["last_year"]: "一年内",
        TIME_RANGE_DAYS["last_half_year"]: "半年内",
        TIME_RANGE_DAYS["last_3_months"]: "近三个月",
    }
    return mapping.get(days, "未限制")


def parsed_result_text(parsed: ParsedQuery) -> str:
    quantity = "未识别"
    if parsed.quantity is not None and parsed.unit:
        quantity = f"{parsed.quantity:g} {parsed.unit}"
    location = parsed.location or "未识别"
    return f"识别工程量：{quantity}；地点：{location}；时间：{time_range_text(parsed)}"


def write_query_result_workbook(
    output_path: Path,
    parsed: ParsedQuery,
    recommend_items: pd.DataFrame,
    matches: pd.DataFrame,
    include_debug_text: bool,
    display: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        recommend_items_for_output(recommend_items, display=display).to_excel(
            writer,
            sheet_name="recommend_items",
            index=False,
            startrow=2,
        )
        worksheet = writer.sheets["recommend_items"]
        worksheet.cell(row=1, column=1, value="查询内容")
        worksheet.cell(row=1, column=2, value=parsed.raw_query)
        worksheet.cell(row=2, column=1, value="解析结果")
        worksheet.cell(row=2, column=2, value=parsed_result_text(parsed))
        matches_for_output(matches, include_debug_text).to_excel(writer, sheet_name="matches", index=False)
    apply_workbook_style(output_path)


def apply_workbook_style(path: Path) -> None:
    try:
        import openpyxl
        from openpyxl.styles import Alignment, Font
    except ImportError:
        return

    workbook = openpyxl.load_workbook(path)
    for worksheet in workbook.worksheets:
        header_row = 3 if worksheet.title == "recommend_items" else 1

        # 不冻结窗格，避免用户打开 Excel 时出现固定区域。
        worksheet.freeze_panes = None

        for cell in worksheet[header_row]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(wrap_text=False, vertical="top")

        if worksheet.title == "recommend_items":
            for row_index in (1, 2):
                for cell in worksheet[row_index]:
                    cell.alignment = Alignment(wrap_text=False, vertical="top")

        for row in worksheet.iter_rows():
            for cell in row:
                cell.alignment = Alignment(wrap_text=False, vertical="top")

    workbook.save(path)
    workbook.close()


def run_query(
    index_dir: Path,
    raw_text: str,
    top_k: int,
    output: Path | None,
    include_debug_text: bool = False,
    parsed_quantity: float | None = None,
    parsed_unit: str = "",
    parsed_location: str = "",
    consultation_time_from: str = "",
    consultation_time_to: str = "",
    project_name_weight: float = 0.85,
    project_detail_weight: float = 0.15,
    display: bool = False,
) -> tuple[ParsedQuery, pd.DataFrame, pd.DataFrame]:
    normalized_name_weight, normalized_detail_weight = normalize_project_weights(project_name_weight, project_detail_weight)
    samples, project_groups, project_name_embeddings, project_detail_embeddings, meta = load_index(index_dir)
    parsed = parse_query_requirements(raw_text, project_groups)
    parsed = apply_parsed_overrides(
        parsed,
        project_groups,
        parsed_quantity=parsed_quantity,
        parsed_unit=parsed_unit,
        parsed_location=parsed_location,
        consultation_time_from=consultation_time_from,
        consultation_time_to=consultation_time_to,
    )
    filtered_groups, filtered_name_embeddings, filtered_detail_embeddings = filter_project_groups(
        project_groups,
        project_name_embeddings,
        project_detail_embeddings,
        parsed,
    )

    if filtered_groups.empty:
        matches = pd.DataFrame(columns=[*MATCH_COLUMNS, *DEBUG_MATCH_COLUMNS])
        recommend_items = pd.DataFrame(columns=RECOMMENDED_ITEM_COLUMNS)
        if output:
            write_query_result_workbook(output, parsed, recommend_items, matches, include_debug_text, display=display)
        return parsed, recommend_items, matches

    model = load_embedding_model(str(meta.get("model") or "BAAI/bge-m3"))
    try:
        query_embedding = encode_query(model, parsed.semantic_query_text)
        if query_embedding.shape[0] != filtered_name_embeddings.shape[1]:
            raise ValueError("query embedding 维度与工程分组 embedding 维度不一致")
        matched_projects = score_project_groups(
            filtered_groups,
            filtered_name_embeddings,
            filtered_detail_embeddings,
            query_embedding,
            top_k,
            normalized_name_weight,
            normalized_detail_weight,
        )
        matches = expand_matched_samples(samples, matched_projects)
        recommend_items = aggregate_recommend_items(matches, parsed)
    finally:
        release_embedding_model(model)
        del model
        gc.collect()

    if output:
        write_query_result_workbook(output, parsed, recommend_items, matches, include_debug_text, display=display)

    return parsed, recommend_items, matches


def print_terminal_summary(
    parsed: ParsedQuery,
    recommend_items: pd.DataFrame,
    matches: pd.DataFrame,
    output_path: Path | None,
) -> None:
    if matches.empty:
        print("[WARN] 硬过滤或工程组召回后没有匹配结果。")
    print(f"[DONE] semantic query: {parsed.semantic_query_text}")
    if parsed.location:
        print(f"location filter: {parsed.location}")
    if parsed.consultation_time_from or parsed.consultation_time_to:
        print(f"time filter: {parsed.consultation_time_from or ''} -> {parsed.consultation_time_to or ''}")
    if parsed.quantity is not None and parsed.unit:
        print(f"parsed quantity: {parsed.quantity:g} {parsed.unit}")
    print(f"[DONE] matched item rows: {len(matches)}")
    print(f"[DONE] recommend items: {len(recommend_items)}")
    if parsed.parse_notes:
        print(f"parse notes: {'；'.join(parsed.parse_notes)}")
    if output_path:
        print(f"输出文件: {output_path}")


def main() -> int:
    args = parse_args()
    index_dir = Path(args.index_dir).expanduser().resolve()
    output_arg = Path(args.output) if args.output is not None else default_query_output_path()
    output_path = output_arg.expanduser().resolve()

    try:
        validate_output_path(output_path, args.overwrite)
        parsed, recommend_items, matches = run_query(
            index_dir=index_dir,
            raw_text=args.text,
            top_k=args.top_k,
            output=output_path,
            include_debug_text=args.include_debug_text,
            parsed_quantity=args.parsed_quantity,
            parsed_unit=args.parsed_unit,
            parsed_location=args.parsed_location,
            consultation_time_from=args.consultation_time_from,
            consultation_time_to=args.consultation_time_to,
            project_name_weight=args.project_name_weight,
            project_detail_weight=args.project_detail_weight,
            display=args.display,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    print_terminal_summary(parsed, recommend_items, matches, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
