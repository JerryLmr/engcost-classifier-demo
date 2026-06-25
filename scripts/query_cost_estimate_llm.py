#!/usr/bin/env python3
import argparse
import gc
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from classifier.llm_client import request_llm_json  # noqa: E402
from services.standard_classifier import classify_project_standard  # noqa: E402


CATALOG_EXACT_BONUS = 0.06
HAS_UNIT_PRICE_BONUS = 0.01
LOW_PROJECT_SCORE_WARNING_THRESHOLD = 0.55
NO_EXACT_CATALOG_WARNING = "样本库缺少该标准目录下的历史工程/清单项，不能形成可靠价格参考。"

SUMMARY_COLUMNS = [
    "原始输入",
    "标准目录ID",
    "一级分类",
    "二级分类",
    "维修状态",
    "标准对象",
    "最高工程相似度",
    "相似工程数",
    "推荐清单项数",
    "总结来源",
    "提示",
]

RECOMMENDED_ITEM_COLUMNS = [
    "rank",
    "cost_item_name",
    "project_description",
    "unit_normalized",
    "source_project_count",
    "occurrence_count",
    "support_ratio",
    "unit_price_count",
    "unit_price_p25",
    "unit_price_median",
    "unit_price_p75",
    "labor_unit_price_count",
    "labor_unit_price_p25",
    "labor_unit_price_median",
    "labor_unit_price_p75",
    "machinery_unit_price_count",
    "machinery_unit_price_p25",
    "machinery_unit_price_median",
    "machinery_unit_price_p75",
    "unit_price_coverage",
    "labor_unit_price_coverage",
    "machinery_unit_price_coverage",
    "price_breakdown_status",
    "example_source_row_ids",
    "example_item_row_ids",
]

DEBUG_ITEM_MATCH_COLUMNS = [
    "rank",
    "final_score",
    "item_score",
    "project_score",
    "full_score",
    "catalog_match",
    "source_row_id",
    "item_row_id",
    "工程名称",
    "catalog_id",
    "一级分类",
    "二级分类",
    "维修状态",
    "标准对象",
    "cost_item_name",
    "project_description",
    "unit_normalized",
    "quantity",
    "unit_price",
    "labor_unit_price",
    "machinery_unit_price",
    "score_below_min_score",
    "catalog_mismatch",
]

PROJECT_MATCH_COLUMNS = [
    "rank",
    "project_score",
    "source_row_id",
    "工程名称",
    "catalog_id",
    "一级分类",
    "二级分类",
    "维修状态",
    "标准对象",
    "item_count",
]

PRICE_COLUMNS = {
    "unit_price_p25",
    "unit_price_median",
    "unit_price_p75",
    "labor_unit_price_p25",
    "labor_unit_price_median",
    "labor_unit_price_p75",
    "machinery_unit_price_p25",
    "machinery_unit_price_median",
    "machinery_unit_price_p75",
    "unit_price",
    "labor_unit_price",
    "machinery_unit_price",
    "quantity",
}

SCORE_COLUMNS = {
    "final_score",
    "item_score",
    "project_score",
    "full_score",
    "support_ratio",
    "unit_price_coverage",
    "labor_unit_price_coverage",
    "machinery_unit_price_coverage",
    "最高工程相似度",
}

ANSWER_AUXILIARY_KEYWORDS = (
    "垂直运输",
    "大型机械",
    "进出场",
    "安拆",
    "吊篮",
    "脚手架",
    "措施",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM 自然语言造价估计实验入口")
    parser.add_argument("--index-dir", required=True, help="索引目录")
    parser.add_argument("--text", required=True, help="口语化维修需求")
    parser.add_argument("--output", default="", help="可选 xlsx 输出路径")
    parser.add_argument("--overwrite", action="store_true", help="若输出文件已存在则覆盖")
    parser.add_argument("--top-k", type=int, default=20, help="输出工程、推荐清单项和调试匹配数量")
    parser.add_argument("--min-score", type=float, default=0.6, help="debug_item_matches 低分标记阈值")
    return parser.parse_args()


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


def load_index(
    index_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    samples_path = index_dir / "samples.parquet"
    groups_path = index_dir / "project_groups.parquet"
    item_path = index_dir / "item_embeddings.npy"
    project_path = index_dir / "project_embeddings.npy"
    full_path = index_dir / "full_embeddings.npy"
    group_path = index_dir / "project_group_embeddings.npy"
    meta_path = index_dir / "index_meta.json"
    paths = [samples_path, groups_path, item_path, project_path, full_path, group_path, meta_path]

    missing = [path.name for path in paths if not path.exists()]
    if missing:
        raise ValueError(f"索引目录缺少文件: {', '.join(missing)}")

    samples = pd.read_parquet(samples_path)
    project_groups = pd.read_parquet(groups_path)
    item_embeddings = np.load(item_path)
    project_embeddings = np.load(project_path)
    full_embeddings = np.load(full_path)
    project_group_embeddings = np.load(group_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if (
        len(samples) != item_embeddings.shape[0]
        or len(samples) != project_embeddings.shape[0]
        or len(samples) != full_embeddings.shape[0]
    ):
        raise ValueError("样本数量与清单 embedding 数量不一致")
    if item_embeddings.shape != project_embeddings.shape or item_embeddings.shape != full_embeddings.shape:
        raise ValueError("三路清单 embedding 维度或样本数不一致")
    if len(project_groups) != project_group_embeddings.shape[0]:
        raise ValueError("工程分组数量与 embedding 数量不一致")
    if project_group_embeddings.shape[1] != item_embeddings.shape[1]:
        raise ValueError("工程分组 embedding 维度与清单 embedding 维度不一致")
    return samples, project_groups, item_embeddings, project_embeddings, full_embeddings, project_group_embeddings, meta


def cell_text(value: Any) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def classify_raw_text(raw_text: str) -> tuple[dict[str, Any], list[str]]:
    result = classify_project_standard(raw_text)
    warnings: list[str] = []
    pipeline_status = cell_text(result.get("pipeline_status"))
    if pipeline_status and pipeline_status != "ok":
        reason = cell_text(result.get("reason"))
        warnings.append(f"工程分类未能稳定匹配标准目录：{reason or pipeline_status}")
    return result, warnings


def top_indexes(scores: np.ndarray, limit: int) -> np.ndarray:
    if limit <= 0 or scores.size == 0:
        return np.array([], dtype=int)
    return np.argsort(scores)[::-1][: min(limit, scores.size)]


def select_project_groups(
    project_groups: pd.DataFrame,
    project_group_embeddings: np.ndarray,
    raw_query_embedding: np.ndarray,
    predicted_catalog_id: str,
    top_k: int,
) -> tuple[pd.DataFrame, bool, int]:
    scores = project_group_embeddings @ raw_query_embedding
    groups = project_groups.copy()
    groups["project_score"] = scores

    if not predicted_catalog_id:
        return groups.head(0).copy(), False, 0

    exact = groups[groups["catalog_id"].fillna("").astype(str).str.strip() == predicted_catalog_id].copy()
    exact_catalog_total_count = int(len(exact))
    if exact.empty:
        return exact, False, exact_catalog_total_count

    selected = exact.sort_values("project_score", ascending=False).head(max(top_k, 0)).copy()

    selected.insert(0, "rank", range(1, len(selected) + 1))
    return selected, True, exact_catalog_total_count


def normalize_signature_text(value: Any) -> str:
    text = cell_text(value).lower()
    text = re.sub(r"\s+", "", text)
    return re.sub(r"[，,。；;：:/\\|()（）【】\[\]{}<>《》\"'“”‘’]", "", text)


def item_signature(row: pd.Series) -> str:
    return "|".join(
        [
            normalize_signature_text(row.get("cost_item_name")),
            normalize_signature_text(row.get("project_description")),
            cell_text(row.get("unit_normalized")),
        ]
    )


def numeric_values(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").dropna()


def price_stats(frame: pd.DataFrame, column: str) -> dict[str, Any]:
    values = numeric_values(frame, column)
    if values.empty:
        return {
            f"{column}_count": 0,
            f"{column}_p25": None,
            f"{column}_median": None,
            f"{column}_p75": None,
        }
    return {
        f"{column}_count": int(values.count()),
        f"{column}_p25": float(values.quantile(0.25)),
        f"{column}_median": float(values.quantile(0.5)),
        f"{column}_p75": float(values.quantile(0.75)),
    }


def ordered_examples(values: pd.Series, limit: int = 5) -> str:
    examples: list[str] = []
    for value in values:
        text = cell_text(value)
        if text and text not in examples:
            examples.append(text)
        if len(examples) >= limit:
            break
    return ", ".join(examples)


def coverage_ratio(count: int, occurrence_count: int) -> float:
    if occurrence_count <= 0:
        return 0.0
    return count / occurrence_count


def price_breakdown_status(row: dict[str, Any]) -> str:
    unit_count = int(row.get("unit_price_count") or 0)
    labor_count = int(row.get("labor_unit_price_count") or 0)
    machinery_count = int(row.get("machinery_unit_price_count") or 0)
    if unit_count <= 0:
        return "无综合单价参考"
    if labor_count > 0 and machinery_count > 0:
        return "有人工和机械费用拆分"
    if labor_count > 0:
        return "有人工费用拆分"
    if machinery_count > 0:
        return "有机械费用拆分"
    return "仅有综合单价，未见人工/机械费用拆分"


def aggregate_recommended_items(samples: pd.DataFrame, matched_projects: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if matched_projects.empty:
        return pd.DataFrame(columns=RECOMMENDED_ITEM_COLUMNS)

    source_ids = set(matched_projects["source_row_id"].tolist())
    item_rows = samples[samples["source_row_id"].isin(source_ids)].copy()
    if item_rows.empty:
        return pd.DataFrame(columns=RECOMMENDED_ITEM_COLUMNS)
    item_rows["item_signature"] = item_rows.apply(item_signature, axis=1)

    matched_project_count = len(source_ids)
    rows: list[dict[str, Any]] = []
    for _signature, group in item_rows.groupby("item_signature", sort=False, dropna=False):
        first = group.iloc[0]
        source_project_count = int(group["source_row_id"].nunique())
        row = {
            "cost_item_name": cell_text(first.get("cost_item_name")),
            "project_description": cell_text(first.get("project_description")),
            "unit_normalized": cell_text(first.get("unit_normalized")),
            "source_project_count": source_project_count,
            "occurrence_count": int(len(group)),
            "support_ratio": source_project_count / matched_project_count if matched_project_count else 0.0,
            "example_source_row_ids": ordered_examples(group["source_row_id"]),
            "example_item_row_ids": ordered_examples(group["item_row_id"]),
        }
        for column in ["unit_price", "labor_unit_price", "machinery_unit_price"]:
            row.update(price_stats(group, column))
        occurrence_count = int(row["occurrence_count"])
        row["unit_price_coverage"] = coverage_ratio(int(row["unit_price_count"]), occurrence_count)
        row["labor_unit_price_coverage"] = coverage_ratio(int(row["labor_unit_price_count"]), occurrence_count)
        row["machinery_unit_price_coverage"] = coverage_ratio(int(row["machinery_unit_price_count"]), occurrence_count)
        row["price_breakdown_status"] = price_breakdown_status(row)
        rows.append(row)

    recommended = pd.DataFrame(rows)
    if recommended.empty:
        return pd.DataFrame(columns=RECOMMENDED_ITEM_COLUMNS)
    recommended = recommended.sort_values(
        ["source_project_count", "occurrence_count", "unit_price_count"],
        ascending=[False, False, False],
    ).head(max(top_k, 0)).copy()
    recommended.insert(0, "rank", range(1, len(recommended) + 1))
    for column in RECOMMENDED_ITEM_COLUMNS:
        if column not in recommended.columns:
            recommended[column] = None
    return recommended[RECOMMENDED_ITEM_COLUMNS]


def numeric_or_none(value: Any) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return None
    return float(number)


def build_debug_item_matches(
    samples: pd.DataFrame,
    item_embeddings: np.ndarray,
    project_embeddings: np.ndarray,
    full_embeddings: np.ndarray,
    item_query_embedding: np.ndarray,
    project_query_embedding: np.ndarray,
    full_query_embedding: np.ndarray,
    predicted_catalog_id: str,
    top_k: int,
    min_score: float,
) -> pd.DataFrame:
    item_scores = item_embeddings @ item_query_embedding
    project_scores = project_embeddings @ project_query_embedding
    full_scores = full_embeddings @ full_query_embedding
    recall_limit = max(top_k, 0) * 3

    candidate_indexes = set(top_indexes(item_scores, recall_limit).tolist())
    candidate_indexes.update(top_indexes(project_scores, recall_limit).tolist())
    candidate_indexes.update(top_indexes(full_scores, recall_limit).tolist())

    rows: list[dict[str, Any]] = []
    for index in candidate_indexes:
        sample = samples.iloc[index]
        sample_catalog_id = cell_text(sample.get("catalog_id"))
        catalog_match = bool(predicted_catalog_id and sample_catalog_id == predicted_catalog_id)
        has_unit_price = numeric_or_none(sample.get("unit_price")) is not None
        semantic_score = max(float(item_scores[index]), float(project_scores[index]), float(full_scores[index]))
        final_score = semantic_score
        if catalog_match:
            final_score += CATALOG_EXACT_BONUS
        if has_unit_price:
            final_score += HAS_UNIT_PRICE_BONUS

        row = {column: sample.get(column, "") for column in DEBUG_ITEM_MATCH_COLUMNS if column not in {
            "rank",
            "final_score",
            "item_score",
            "project_score",
            "full_score",
            "catalog_match",
            "score_below_min_score",
            "catalog_mismatch",
        }}
        row.update(
            {
                "final_score": final_score,
                "item_score": float(item_scores[index]),
                "project_score": float(project_scores[index]),
                "full_score": float(full_scores[index]),
                "catalog_match": catalog_match,
                "score_below_min_score": final_score < min_score,
                "catalog_mismatch": bool(predicted_catalog_id and sample_catalog_id != predicted_catalog_id),
            }
        )
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=DEBUG_ITEM_MATCH_COLUMNS)
    matches = pd.DataFrame(rows).sort_values("final_score", ascending=False).head(max(top_k, 0)).copy()
    matches.insert(0, "rank", range(1, len(matches) + 1))
    for column in DEBUG_ITEM_MATCH_COLUMNS:
        if column not in matches.columns:
            matches[column] = ""
    return matches[DEBUG_ITEM_MATCH_COLUMNS]


def project_matches_for_output(matched_projects: pd.DataFrame) -> pd.DataFrame:
    if matched_projects.empty:
        return pd.DataFrame(columns=PROJECT_MATCH_COLUMNS)
    output = matched_projects.copy()
    for column in PROJECT_MATCH_COLUMNS:
        if column not in output.columns:
            output[column] = ""
    return output[PROJECT_MATCH_COLUMNS]


def build_warnings(
    classify_warnings: list[str],
    exact_catalog_available: bool,
    exact_catalog_total_count: int,
    matched_projects: pd.DataFrame,
    recommended_count: int,
) -> list[str]:
    warnings = list(classify_warnings)
    if not exact_catalog_available:
        warnings.append(NO_EXACT_CATALOG_WARNING)
    elif exact_catalog_total_count < 3:
        warnings.append("同标准目录历史样本较少，推荐清单项可靠性有限。")
    if not matched_projects.empty:
        top_project_score = numeric_or_none(matched_projects["project_score"].iloc[0])
        if top_project_score is not None and top_project_score < LOW_PROJECT_SCORE_WARNING_THRESHOLD:
            warnings.append("同目录样本存在，但与输入语义相似度偏低，请人工复核。")
    if recommended_count == 0:
        warnings.append("未形成推荐清单项，请补充材料、做法、面积或设备规格。")
    return warnings


def build_summary(
    raw_text: str,
    classification: dict[str, Any],
    matched_projects: pd.DataFrame,
    recommended_items: pd.DataFrame,
    warnings: list[str],
) -> dict[str, Any]:
    top_project_score = float(matched_projects["project_score"].iloc[0]) if not matched_projects.empty else None
    return {
        "原始输入": raw_text,
        "标准目录ID": cell_text(classification.get("catalog_id")),
        "一级分类": cell_text(classification.get("category")),
        "二级分类": cell_text(classification.get("item")),
        "维修状态": cell_text(classification.get("repair_status")),
        "标准对象": cell_text(classification.get("standard_group")),
        "最高工程相似度": top_project_score,
        "相似工程数": int(len(matched_projects)),
        "推荐清单项数": int(len(recommended_items)),
        "总结来源": "",
        "提示": "；".join(warnings),
    }


def occurrence_level(support_ratio: Any) -> str:
    ratio = numeric_or_none(support_ratio) or 0.0
    if ratio >= 0.6:
        return "高"
    if ratio >= 0.3:
        return "中"
    return "较低/条件项"


def rounded_number(value: Any, digits: int = 2) -> float | None:
    number = numeric_or_none(value)
    if number is None:
        return None
    return round(number, digits)


def format_money(value: Any) -> str:
    number = rounded_number(value, digits=2)
    if number is None:
        return ""
    text = f"{number:.2f}".rstrip("0").rstrip(".")
    return text or "0"


def format_price_with_unit(value: Any, unit: Any) -> str:
    money = format_money(value)
    if not money:
        return ""
    unit_text = cell_text(unit) or "未注明"
    return f"{money} 元/{unit_text}"


def row_count(row: pd.Series, column: str) -> int:
    return int(numeric_or_none(row.get(column)) or 0)


def format_unit_price_text(row: pd.Series) -> str:
    if row_count(row, "unit_price_count") <= 0:
        return "参考综合单价：暂无可靠综合单价参考"

    unit = cell_text(row.get("unit_normalized")) or "未注明"
    median = format_price_with_unit(row.get("unit_price_median"), unit)
    if not median:
        return "参考综合单价：暂无可靠综合单价参考"

    p25 = format_money(row.get("unit_price_p25"))
    median_money = format_money(row.get("unit_price_median"))
    p75 = format_money(row.get("unit_price_p75"))
    if p25 and median_money and p75 and len({p25, median_money, p75}) > 1:
        return f"参考综合单价：约 {median}，历史样本区间约 {p25}-{p75} 元/{unit}"
    return f"参考综合单价：{median}"


def format_price_breakdown_text(row: pd.Series) -> str:
    if row_count(row, "unit_price_count") <= 0:
        return "价格拆分：缺少可用综合单价，暂不展开费用拆分"

    unit = cell_text(row.get("unit_normalized")) or "未注明"
    labor_count = row_count(row, "labor_unit_price_count")
    machinery_count = row_count(row, "machinery_unit_price_count")
    labor_price = format_price_with_unit(row.get("labor_unit_price_median"), unit)
    machinery_price = format_price_with_unit(row.get("machinery_unit_price_median"), unit)

    if labor_count > 0 and machinery_count > 0 and labor_price and machinery_price:
        return f"价格拆分：其中包含人工约 {labor_price}、机械约 {machinery_price}"
    if labor_count > 0 and labor_price:
        return f"价格拆分：其中包含人工约 {labor_price}，历史样本未见机械费用拆分"
    if machinery_count > 0 and machinery_price:
        return f"价格拆分：其中包含机械约 {machinery_price}，历史样本未见人工费用拆分"
    return "价格拆分：历史样本未见人工、机械费用拆分"


def format_occurrence_text(row: pd.Series) -> str:
    return occurrence_level(row.get("support_ratio"))


def item_feature_text(row: pd.Series) -> str:
    name = cell_text(row.get("cost_item_name"))
    description = cell_text(row.get("project_description"))
    if description and description != name:
        return description
    return name or description or "该清单项"


def is_auxiliary_answer_item(row: pd.Series) -> bool:
    text = f"{cell_text(row.get('cost_item_name'))} {cell_text(row.get('project_description'))}"
    return any(keyword in text for keyword in ANSWER_AUXILIARY_KEYWORDS)


def answer_item_display_name(row: pd.Series) -> str:
    name = cell_text(row.get("cost_item_name"))
    if is_auxiliary_answer_item(row) and name:
        return f"{name}（措施/辅助项）"
    return name


def answer_ordered_items(recommended_items: pd.DataFrame, limit: int = 8) -> pd.DataFrame:
    if recommended_items.empty:
        return recommended_items.head(0).copy()
    answer_items = recommended_items.head(limit).copy()
    auxiliary_mask = answer_items.apply(is_auxiliary_answer_item, axis=1)
    return pd.concat([answer_items[~auxiliary_mask], answer_items[auxiliary_mask]])


def answer_items_payload(recommended_items: pd.DataFrame, limit: int = 8) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if recommended_items.empty:
        return rows

    for _index, row in answer_ordered_items(recommended_items, limit=limit).iterrows():
        is_auxiliary = is_auxiliary_answer_item(row)
        item = {
            "rank": int(numeric_or_none(row.get("rank")) or 0),
            "清单项": cell_text(row.get("cost_item_name")),
            "展示名称": answer_item_display_name(row),
            "是否措施/辅助项": is_auxiliary,
            "单位": cell_text(row.get("unit_normalized")) or "未注明",
            "历史样本类似工程常见程度": format_occurrence_text(row),
            "参考综合单价文本": format_unit_price_text(row),
            "价格拆分文本": format_price_breakdown_text(row),
            "施工工艺或项目特征": item_feature_text(row),
            "支持工程数": int(numeric_or_none(row.get("source_project_count")) or 0),
            "出现次数": int(numeric_or_none(row.get("occurrence_count")) or 0),
            "支持率": rounded_number(row.get("support_ratio"), digits=4),
        }
        rows.append(item)
    return rows


def build_answer_prompt(raw_text: str, summary: dict[str, Any], recommended_items: pd.DataFrame) -> str:
    payload = {
        "原始输入": raw_text,
        "标准目录分类结果": {
            "标准目录ID": summary.get("标准目录ID"),
            "一级分类": summary.get("一级分类"),
            "二级分类": summary.get("二级分类"),
            "维修状态": summary.get("维修状态"),
            "标准对象": summary.get("标准对象"),
        },
        "warnings": summary.get("提示"),
        "recommended_items_top8": answer_items_payload(recommended_items, limit=8),
    }
    return f"""
/no_think
你是维修工程造价问答助手。只基于输入 JSON 生成 answer，不要推理过程。

规则：
- recommended_items_top8 中的“参考综合单价文本”“价格拆分文本”“历史样本类似工程常见程度”“施工工艺或项目特征”已经由程序生成。
- 必须直接使用上述字段，不得重新解释统计 count，不得根据人工/机械为空推断不需要人工或机械。
- 只能使用 recommended_items_top8 中的清单项，不得新增清单项/工艺。
- 不得编造价格；不得输出 recommended_items_top8 中没有的价格。
- 面向普通业主/物业人员，语言简洁，不使用内部技术口吻。
- 不要输出 markdown 表格。
- recommended_items_top8 已按主体施工项在前、措施/辅助项在后排序；直接按输入顺序输出。
- 每项必须包含：历史样本类似工程常见程度、单位、参考综合单价、价格拆分、施工工艺/项目特征。
- recommended_items_top8 为空时，说明同目录样本不足，不能形成可靠价格参考。
- 施工工艺/项目特征只能使用输入字段“施工工艺或项目特征”，不得发散或新增工程事实。
- answer 总字数控制在 900 字以内。

answer 固定格式：
根据历史已审定样本，类似【原始输入】通常可能包含以下清单项/工艺：
1. 【展示名称】：
   历史样本类似工程常见程度：【历史样本类似工程常见程度】
   单位：【单位】
   【参考综合单价文本】
   【价格拆分文本】
   施工工艺/项目特征：【施工工艺或项目特征】

需要补充的信息/风险提示：
...

只输出 JSON object：{{"answer":"..."}}

输入 JSON：
{json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}
""".strip()


def build_answer_fallback(raw_text: str, summary: dict[str, Any], recommended_items: pd.DataFrame) -> str:
    warnings = cell_text(summary.get("提示"))
    if recommended_items.empty:
        risk = warnings or "同目录推荐清单项为空，样本不足，不能形成可靠价格参考。"
        return (
            f"根据历史已审定样本，类似{raw_text}暂未形成可用的推荐清单项/工艺。\n\n"
            f"需要补充的信息/风险提示：{risk}"
        )

    lines = [f"根据历史已审定样本，类似{raw_text}通常可能包含以下清单项/工艺："]
    for index, (_row_index, row) in enumerate(answer_ordered_items(recommended_items, limit=8).iterrows(), start=1):
        name = answer_item_display_name(row) or f"清单项{index}"
        unit = cell_text(row.get("unit_normalized")) or "未注明"
        lines.append(
            f"{index}. {name}：\n"
            f"   历史样本类似工程常见程度：{format_occurrence_text(row)}\n"
            f"   单位：{unit}\n"
            f"   {format_unit_price_text(row)}\n"
            f"   {format_price_breakdown_text(row)}\n"
            f"   施工工艺/项目特征：{item_feature_text(row)}"
        )

    risk = warnings or "仍需人工确认工程量、施工做法、材料规格和现场条件。"
    lines.append("")
    lines.append(f"需要补充的信息/风险提示：{risk}")
    return "\n".join(lines)


def build_answer(raw_text: str, summary: dict[str, Any], recommended_items: pd.DataFrame) -> dict[str, str]:
    try:
        data = request_llm_json(
            build_answer_prompt(raw_text, summary, recommended_items),
            max_tokens=512,
            timeout_seconds=45,
            system_prompt=(
                "/no_think\n"
                "你是维修工程造价问答助手。不要输出思考过程，不要输出 <think>，"
                "不要输出 markdown。最终答案只能是一个 JSON object。"
            ),
        )
        answer = cell_text(data.get("answer"))
        if answer:
            return {"answer": answer, "answer_source": "llm", "answer_error": ""}
        return {
            "answer": build_answer_fallback(raw_text, summary, recommended_items),
            "answer_source": "fallback",
            "answer_error": "LLM 返回空 answer",
        }
    except Exception as exc:
        return {
            "answer": build_answer_fallback(raw_text, summary, recommended_items),
            "answer_source": "fallback",
            "answer_error": str(exc),
        }


def apply_workbook_style(output_path: Path) -> None:
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment, Font

    workbook = load_workbook(output_path)
    for worksheet in workbook.worksheets:
        worksheet.freeze_panes = "A2"
        for cell in worksheet[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
        for column_cells in worksheet.columns:
            header = cell_text(column_cells[0].value)
            max_length = len(header)
            for cell in column_cells[1: min(len(column_cells), 80)]:
                max_length = max(max_length, len(cell_text(cell.value)))
                cell.alignment = Alignment(vertical="top", wrap_text=True)
                if header in SCORE_COLUMNS:
                    cell.number_format = "0.0000"
                elif header in PRICE_COLUMNS:
                    cell.number_format = "0.00"
            width = min(max(max_length + 2, 10), 42)
            if worksheet.title == "answer" and header == "内容":
                width = 100
            elif header == "提示":
                width = 60
            elif header == "project_description":
                width = 60
            elif header in {"cost_item_name", "工程名称"}:
                width = 32 if header == "cost_item_name" else 30
            worksheet.column_dimensions[column_cells[0].column_letter].width = width
    workbook.save(output_path)
    workbook.close()


def write_query_result_workbook(
    output_path: Path,
    answer_result: dict[str, str],
    summary: dict[str, Any],
    recommended_items: pd.DataFrame,
    debug_item_matches: pd.DataFrame,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame(
            [
                {"字段": "总结来源", "内容": answer_result.get("answer_source", "")},
                {"字段": "错误信息", "内容": answer_result.get("answer_error", "")},
                {"字段": "总结", "内容": answer_result.get("answer", "")},
            ]
        ).to_excel(
            writer,
            sheet_name="answer",
            index=False,
        )
        pd.DataFrame([{column: summary.get(column) for column in SUMMARY_COLUMNS}], columns=SUMMARY_COLUMNS).to_excel(
            writer,
            sheet_name="summary",
            index=False,
        )
        recommended_items.to_excel(writer, sheet_name="recommended_items", index=False)
        debug_item_matches.to_excel(writer, sheet_name="debug_item_matches", index=False)
    apply_workbook_style(output_path)


def run_query(
    index_dir: Path,
    raw_text: str,
    top_k: int,
    min_score: float,
    output: Path | None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, str]]:
    (
        samples,
        project_groups,
        item_embeddings,
        project_embeddings,
        full_embeddings,
        project_group_embeddings,
        meta,
    ) = load_index(index_dir)
    classification, classify_warnings = classify_raw_text(raw_text)

    model = load_embedding_model(str(meta.get("model") or "BAAI/bge-m3"))
    try:
        raw_query_embedding = encode_query(model, raw_text)
        predicted_catalog_id = cell_text(classification.get("catalog_id"))
        matched_projects, exact_catalog_available, exact_catalog_total_count = select_project_groups(
            project_groups=project_groups,
            project_group_embeddings=project_group_embeddings,
            raw_query_embedding=raw_query_embedding,
            predicted_catalog_id=predicted_catalog_id,
            top_k=top_k,
        )
        recommended_items = aggregate_recommended_items(samples, matched_projects, top_k)

        debug_item_matches = build_debug_item_matches(
            samples=samples,
            item_embeddings=item_embeddings,
            project_embeddings=project_embeddings,
            full_embeddings=full_embeddings,
            item_query_embedding=raw_query_embedding,
            project_query_embedding=raw_query_embedding,
            full_query_embedding=raw_query_embedding,
            predicted_catalog_id=predicted_catalog_id,
            top_k=top_k,
            min_score=min_score,
        )
    finally:
        release_embedding_model(model)
        del model
        gc.collect()

    warnings = build_warnings(
        classify_warnings=classify_warnings,
        exact_catalog_available=exact_catalog_available,
        exact_catalog_total_count=exact_catalog_total_count,
        matched_projects=matched_projects,
        recommended_count=len(recommended_items),
    )
    summary = build_summary(raw_text, classification, matched_projects, recommended_items, warnings)
    answer_result = build_answer(raw_text, summary, recommended_items)
    summary["总结来源"] = answer_result.get("answer_source", "")

    if output:
        write_query_result_workbook(output, answer_result, summary, recommended_items, debug_item_matches)

    return summary, recommended_items, debug_item_matches, answer_result


def print_terminal_summary(summary: dict[str, Any], answer: str, output_path: Path | None) -> None:
    print(f"[DONE] matched projects: {summary.get('相似工程数')}")
    print(f"[DONE] recommended items: {summary.get('推荐清单项数')}")
    print(
        "预测目录: "
        f"{summary.get('标准目录ID')} "
        f"{summary.get('一级分类')} / {summary.get('二级分类')}"
    )
    if answer:
        print(f"总结预览: {answer[:500]}")
    if summary.get("提示"):
        print(f"注意事项: {summary['提示']}")
    if output_path:
        print(f"输出文件: {output_path}")


def main() -> int:
    args = parse_args()
    index_dir = Path(args.index_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else None

    try:
        validate_output_path(output_path, args.overwrite)
        summary, _recommended_items, _debug_item_matches, answer_result = run_query(
            index_dir=index_dir,
            raw_text=args.text,
            top_k=args.top_k,
            min_score=args.min_score,
            output=output_path,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    print_terminal_summary(summary, answer_result.get("answer", ""), output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
