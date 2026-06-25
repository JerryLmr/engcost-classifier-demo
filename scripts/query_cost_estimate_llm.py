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

from classifier.unit_normalizer import normalize_unit  # noqa: E402
from services.standard_classifier import classify_project_standard  # noqa: E402


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
    "识别工程量",
    "工程量单位",
    "可计算清单项数",
    "简单合计参考金额",
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
    "input_quantity",
    "input_quantity_unit",
    "estimated_amount_p25",
    "estimated_amount_median",
    "estimated_amount_p75",
    "estimated_amount_note",
    "example_source_row_ids",
    "example_item_row_ids",
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
    "input_quantity",
    "estimated_amount_p25",
    "estimated_amount_median",
    "estimated_amount_p75",
}

SCORE_COLUMNS = {
    "project_score",
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
    parser.add_argument("--top-k", type=int, default=20, help="输出工程和推荐清单项数量")
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
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict[str, Any]]:
    samples_path = index_dir / "samples.parquet"
    groups_path = index_dir / "project_groups.parquet"
    group_path = index_dir / "project_group_embeddings.npy"
    meta_path = index_dir / "index_meta.json"
    paths = [samples_path, groups_path, group_path, meta_path]

    missing = [path.name for path in paths if not path.exists()]
    if missing:
        raise ValueError(f"索引目录缺少文件: {', '.join(missing)}")

    samples = pd.read_parquet(samples_path)
    project_groups = pd.read_parquet(groups_path)
    project_group_embeddings = np.load(group_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if len(project_groups) != project_group_embeddings.shape[0]:
        raise ValueError("工程分组数量与 embedding 数量不一致")
    if project_group_embeddings.ndim != 2:
        raise ValueError("工程分组 embedding 维度不正确")
    return samples, project_groups, project_group_embeddings, meta


def cell_text(value: Any) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def extract_simple_quantity(raw_text: str) -> dict[str, Any] | None:
    unit_pattern = r"(平方米|平米|平方|平|m\s*\^\s*2|m\s*2|m²|㎡)"
    match = re.search(rf"(?P<quantity>\d+(?:\.\d+)?)\s*(?P<unit>{unit_pattern})", raw_text or "", re.IGNORECASE)
    if not match:
        return None

    raw_unit = match.group("unit")
    normalized = normalize_unit(raw_unit)
    return {
        "quantity": float(match.group("quantity")),
        "raw_unit": raw_unit,
        "unit": normalized,
        "source": "regex" if normalized else "regex_unrecognized_unit",
    }


def classify_raw_text(raw_text: str) -> tuple[dict[str, Any], list[str]]:
    result = classify_project_standard(raw_text)
    warnings: list[str] = []
    pipeline_status = cell_text(result.get("pipeline_status"))
    if pipeline_status and pipeline_status != "ok":
        reason = cell_text(result.get("reason"))
        warnings.append(f"工程分类未能稳定匹配标准目录：{reason or pipeline_status}")
    return result, warnings


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


def apply_simple_amount_estimates(
    recommended_items: pd.DataFrame,
    quantity_info: dict[str, Any] | None,
) -> pd.DataFrame:
    recommended = recommended_items.copy()
    for column in [
        "input_quantity",
        "input_quantity_unit",
        "estimated_amount_p25",
        "estimated_amount_median",
        "estimated_amount_p75",
        "estimated_amount_note",
    ]:
        if column not in recommended.columns:
            recommended[column] = None

    if recommended.empty:
        return recommended[RECOMMENDED_ITEM_COLUMNS]

    if not quantity_info:
        recommended["estimated_amount_note"] = "未识别工程量，暂不计算参考金额"
        return recommended[RECOMMENDED_ITEM_COLUMNS]

    quantity = numeric_or_none(quantity_info.get("quantity"))
    input_unit = cell_text(quantity_info.get("unit"))
    recommended["input_quantity"] = quantity
    recommended["input_quantity_unit"] = input_unit

    if quantity is None or not input_unit:
        recommended["estimated_amount_note"] = "输入工程量单位未识别，暂不计算参考金额"
        return recommended[RECOMMENDED_ITEM_COLUMNS]

    for index, row in recommended.iterrows():
        item_unit = cell_text(row.get("unit_normalized"))
        if item_unit != input_unit:
            recommended.at[index, "estimated_amount_note"] = "单位不一致，未按输入工程量计算"
            continue

        p25 = numeric_or_none(row.get("unit_price_p25"))
        median = numeric_or_none(row.get("unit_price_median"))
        p75 = numeric_or_none(row.get("unit_price_p75"))
        if median is None:
            recommended.at[index, "estimated_amount_note"] = "缺少综合单价样本，无法计算参考金额"
            continue

        recommended.at[index, "estimated_amount_median"] = quantity * median
        if p25 is not None and p75 is not None:
            recommended.at[index, "estimated_amount_p25"] = quantity * p25
            recommended.at[index, "estimated_amount_p75"] = quantity * p75
            recommended.at[index, "estimated_amount_note"] = "按输入工程量和综合单价历史区间简单估算"
        else:
            recommended.at[index, "estimated_amount_note"] = (
                "缺少完整 P25-P75 区间，仅保留中位数参考；answer 中不要展示为确定报价"
            )

    return recommended[RECOMMENDED_ITEM_COLUMNS]


def count_calculable_amount_items(recommended_items: pd.DataFrame) -> int:
    if recommended_items.empty:
        return 0
    p25 = numeric_frame_column(recommended_items, "estimated_amount_p25")
    p75 = numeric_frame_column(recommended_items, "estimated_amount_p75")
    return int((p25.notna() & p75.notna()).sum())


def numeric_frame_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([None] * len(frame), index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def simple_total_amounts(recommended_items: pd.DataFrame) -> dict[str, float | None]:
    if recommended_items.empty:
        return {"p25": None, "median": None, "p75": None}
    p25 = numeric_frame_column(recommended_items, "estimated_amount_p25")
    p75 = numeric_frame_column(recommended_items, "estimated_amount_p75")
    mask = p25.notna() & p75.notna()
    if not bool(mask.any()):
        return {"p25": None, "median": None, "p75": None}
    median = numeric_frame_column(recommended_items, "estimated_amount_median")
    return {
        "p25": float(p25[mask].sum()),
        "median": float(median[mask].sum()) if bool(median[mask].notna().any()) else None,
        "p75": float(p75[mask].sum()),
    }


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


def format_amount_value(value: Any) -> str:
    number = numeric_or_none(value)
    if number is None:
        return ""
    rounded = round(number, 2)
    if rounded.is_integer():
        return f"{rounded:,.0f}"
    return f"{rounded:,.2f}".rstrip("0").rstrip(".")


def format_amount_range(p25: Any, p75: Any) -> str:
    low = format_amount_value(p25)
    high = format_amount_value(p75)
    if not low or not high:
        return ""
    return f"{low}-{high} 元"


def build_summary(
    raw_text: str,
    classification: dict[str, Any],
    matched_projects: pd.DataFrame,
    recommended_items: pd.DataFrame,
    quantity_info: dict[str, Any] | None,
    warnings: list[str],
) -> dict[str, Any]:
    top_project_score = float(matched_projects["project_score"].iloc[0]) if not matched_projects.empty else None
    totals = simple_total_amounts(recommended_items)
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
        "识别工程量": quantity_info.get("quantity") if quantity_info else "",
        "工程量单位": quantity_info.get("unit") if quantity_info else "",
        "可计算清单项数": count_calculable_amount_items(recommended_items),
        "简单合计参考金额": format_amount_range(totals.get("p25"), totals.get("p75")),
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


def format_simple_estimate_text(row: pd.Series) -> str:
    quantity = numeric_or_none(row.get("input_quantity"))
    unit = cell_text(row.get("input_quantity_unit"))
    if quantity is None or not unit:
        return ""

    p25 = numeric_or_none(row.get("estimated_amount_p25"))
    median = numeric_or_none(row.get("estimated_amount_median"))
    p75 = numeric_or_none(row.get("estimated_amount_p75"))
    quantity_text = format_amount_value(quantity)

    if p25 is not None and p75 is not None:
        amount_range = format_amount_range(p25, p75)
        if amount_range:
            return f"简单估算：按 {quantity_text} {unit} 计算，参考金额约 {amount_range}"
    if median is not None:
        return (
            f"简单估算：按 {quantity_text} {unit} 计算，"
            f"历史样本区间不足，仅可参考中位金额约 {format_amount_value(median)} 元"
        )
    return ""


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
        estimate_text = format_simple_estimate_text(row)
        if estimate_text:
            lines.append(f"   {estimate_text}")

    risk = warnings or "仍需人工确认工程量、施工做法、材料规格和现场条件。"
    totals = simple_total_amounts(recommended_items)
    total_range = format_amount_range(totals.get("p25"), totals.get("p75"))
    if total_range:
        lines.append("")
        lines.append(f"按已能匹配单位的清单项简单合计，参考金额约 {total_range}。")

    has_quantity = bool(cell_text(summary.get("工程量单位")))
    notes = recommended_items.get("estimated_amount_note")
    has_unit_mismatch = bool(notes.astype(str).str.contains("单位不一致", na=False).any()) if notes is not None else False
    if has_quantity and has_unit_mismatch:
        risk = f"{risk}；单位与输入工程量不一致的项目未纳入简单合计，例如台次、项、套等措施或设备类费用。"
    if total_range:
        risk = (
            f"{risk}；该合计仅按历史样本综合单价区间和输入工程量粗略计算，"
            "未包含单位不匹配、现场条件不明确或需单独确认的措施项。"
        )
    lines.append("")
    lines.append(f"需要补充的信息/风险提示：{risk}")
    return "\n".join(lines)


def build_answer(raw_text: str, summary: dict[str, Any], recommended_items: pd.DataFrame) -> dict[str, str]:
    return {
        "answer": build_answer_fallback(raw_text, summary, recommended_items),
        "answer_source": "template",
        "answer_error": "",
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
    matched_projects: pd.DataFrame,
    recommended_items: pd.DataFrame,
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
        project_matches_for_output(matched_projects).to_excel(writer, sheet_name="matched_projects", index=False)
        recommended_items.to_excel(writer, sheet_name="recommended_items", index=False)
    apply_workbook_style(output_path)


def run_query(
    index_dir: Path,
    raw_text: str,
    top_k: int,
    output: Path | None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, str]]:
    samples, project_groups, project_group_embeddings, meta = load_index(index_dir)
    classification, classify_warnings = classify_raw_text(raw_text)
    quantity_info = extract_simple_quantity(raw_text)

    model = load_embedding_model(str(meta.get("model") or "BAAI/bge-m3"))
    try:
        raw_query_embedding = encode_query(model, raw_text)
        if raw_query_embedding.shape[0] != project_group_embeddings.shape[1]:
            raise ValueError("query embedding 维度与工程分组 embedding 维度不一致")
        predicted_catalog_id = cell_text(classification.get("catalog_id"))
        matched_projects, exact_catalog_available, exact_catalog_total_count = select_project_groups(
            project_groups=project_groups,
            project_group_embeddings=project_group_embeddings,
            raw_query_embedding=raw_query_embedding,
            predicted_catalog_id=predicted_catalog_id,
            top_k=top_k,
        )
        recommended_items = aggregate_recommended_items(samples, matched_projects, top_k)
        recommended_items = apply_simple_amount_estimates(recommended_items, quantity_info)
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
    summary = build_summary(raw_text, classification, matched_projects, recommended_items, quantity_info, warnings)
    answer_result = build_answer(raw_text, summary, recommended_items)
    summary["总结来源"] = answer_result.get("answer_source", "")

    if output:
        write_query_result_workbook(output, answer_result, summary, matched_projects, recommended_items)

    return summary, recommended_items, matched_projects, answer_result


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
        summary, _recommended_items, _matched_projects, answer_result = run_query(
            index_dir=index_dir,
            raw_text=args.text,
            top_k=args.top_k,
            output=output_path,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    print_terminal_summary(summary, answer_result.get("answer", ""), output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
