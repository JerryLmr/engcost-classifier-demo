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
from classifier.llm_client import LLMServiceError, request_llm_json  # noqa: E402
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
    "item_id",
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

ANSWER_PLAN_COLUMNS = [
    "row_type",
    "section_title",
    "display_order",
    "item_id",
    "cost_item_name",
    "project_description",
    "unit_normalized",
    "plan_action",
    "representative_item_id",
    "similar_group_title",
    "reason",
    "note",
]

ANSWER_PLAN_ITEM_LIMIT = 12

ANSWER_PLAN_COMPACT_ITEM_FIELDS = [
    "item_id",
    "rank",
    "cost_item_name",
    "project_description",
    "unit_normalized",
    "source_project_count",
    "occurrence_count",
    "support_ratio",
    "price_breakdown_status",
    "input_quantity",
    "input_quantity_unit",
    "estimated_amount_note",
    "has_unit_price",
    "has_amount_estimate",
]

NO_AUTO_TOTAL_NOTE = (
    "以上为各清单项按输入工程量计算的单项参考金额；"
    "由于部分清单项可能属于替代做法、重复候选或条件措施项，当前不自动汇总为总价。"
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
    recommended.insert(1, "item_id", [f"rec_{index:03d}" for index in range(1, len(recommended) + 1)])
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
        "简单合计参考金额": "未自动合计",
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


def answer_item_display_name(row: pd.Series, is_conditional: bool = False) -> str:
    name = cell_text(row.get("cost_item_name"))
    if is_conditional and name:
        return f"{name}（需现场确认）"
    return name


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
        rounded_p25 = round(p25, 2)
        rounded_p75 = round(p75, 2)
        if rounded_p25 < rounded_p75:
            amount_range = format_amount_range(p25, p75)
            if amount_range:
                return f"简单估算：按 {quantity_text} {unit} 计算，参考金额约 {amount_range}"
        amount = format_amount_value(median if median is not None else p25)
        if amount:
            return f"简单估算：按 {quantity_text} {unit} 计算，历史样本价格集中，参考金额约 {amount} 元"
    if median is not None:
        return (
            f"简单估算：按 {quantity_text} {unit} 计算，"
            f"历史样本区间不足，仅可参考中位金额约 {format_amount_value(median)} 元"
        )
    return ""


def json_safe_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value


def ensure_recommended_item_ids(recommended_items: pd.DataFrame) -> pd.DataFrame:
    recommended = recommended_items.copy()
    for column in RECOMMENDED_ITEM_COLUMNS:
        if column not in recommended.columns:
            recommended[column] = None
    if recommended.empty:
        return recommended[RECOMMENDED_ITEM_COLUMNS]

    item_ids: list[str] = []
    for index, value in enumerate(recommended["item_id"].tolist(), start=1):
        item_id = cell_text(value)
        item_ids.append(item_id or f"rec_{index:03d}")
    recommended["item_id"] = item_ids
    return recommended[RECOMMENDED_ITEM_COLUMNS]


def build_answer_plan_payload(
    raw_text: str,
    summary: dict[str, Any],
    recommended_items: pd.DataFrame,
) -> dict[str, Any]:
    recommended = ensure_recommended_item_ids(recommended_items)
    compact_items: list[dict[str, Any]] = []
    for _index, row in recommended.iterrows():
        item = {
            "item_id": cell_text(row.get("item_id")),
            "rank": json_safe_value(row.get("rank")),
            "cost_item_name": cell_text(row.get("cost_item_name")),
            "project_description": cell_text(row.get("project_description")),
            "unit_normalized": cell_text(row.get("unit_normalized")),
            "source_project_count": json_safe_value(row.get("source_project_count")),
            "occurrence_count": json_safe_value(row.get("occurrence_count")),
            "support_ratio": json_safe_value(row.get("support_ratio")),
            "price_breakdown_status": cell_text(row.get("price_breakdown_status")),
            "input_quantity": json_safe_value(row.get("input_quantity")),
            "input_quantity_unit": cell_text(row.get("input_quantity_unit")),
            "estimated_amount_note": cell_text(row.get("estimated_amount_note")),
            "has_unit_price": row_count(row, "unit_price_count") > 0,
            "has_amount_estimate": numeric_or_none(row.get("estimated_amount_median")) is not None,
        }
        compact_items.append({field: item.get(field) for field in ANSWER_PLAN_COMPACT_ITEM_FIELDS})
    return {
        "raw_text": raw_text,
        "summary": {key: json_safe_value(value) for key, value in summary.items()},
        "recommended_items": compact_items,
    }


def build_answer_plan_prompt(
    raw_text: str,
    summary: dict[str, Any],
    recommended_items: pd.DataFrame,
) -> str:
    payload = build_answer_plan_payload(raw_text, summary, recommended_items)
    payload_text = json.dumps(payload, ensure_ascii=False, indent=2)
    return f"""
你是维修工程造价问答的 answer planner。你的任务不是重新检索、不是报价、不是生成最终 answer，而是在输入的 recommended_items 候选池内做语义归并、业务分组和展示选择。

输入数据：
{payload_text}

输出要求：
1. 只能使用输入中存在的 item_id。
2. 不得新增清单项。
3. 不得输出价格。
4. 不得输出最终 answer 正文。
5. 如果多个 item 表达相近、属于相近做法、同类设备或同类工序，可以放入 similar_groups。
6. similar_groups.display_item_id 表示 answer 中优先详细展示的代表项。
7. 如果某些项目是条件项、措施项、需现场确认项，可以放入 conditional_item_ids。
8. 如果某些项目明显是替代方案、重复候选或与用户输入不匹配，可以放入 excluded_item_ids。
9. sections 的 title 由你根据候选项语义自行生成，不要依赖固定标题。
10. 每个 item_id 最多出现在一个 section 中。
11. notes 最多 3 条。
12. 输出必须是 JSON object，不要 markdown，不要解释过程，不要 <think>。

JSON 字段固定为：
{{
  "mode": "flat 或 sectioned",
  "sections": [
    {{
      "title": "分组标题",
      "item_ids": ["rec_001", "rec_002"]
    }}
  ],
  "similar_groups": [
    {{
      "title": "相近做法或同类项标题",
      "item_ids": ["rec_002", "rec_004"],
      "display_item_id": "rec_002",
      "reason": "简短原因"
    }}
  ],
  "conditional_item_ids": ["rec_006"],
  "excluded_item_ids": ["rec_005"],
  "notes": ["简短提示"]
}}
""".strip()


def valid_item_id_set(recommended_items: pd.DataFrame) -> set[str]:
    if recommended_items.empty or "item_id" not in recommended_items.columns:
        return set()
    return {cell_text(value) for value in recommended_items["item_id"].tolist() if cell_text(value)}


def unique_valid_item_ids(values: Any, valid_ids: set[str], used_ids: set[str] | None = None) -> list[str]:
    if not isinstance(values, list):
        return []
    result: list[str] = []
    for value in values:
        item_id = cell_text(value)
        if not item_id or item_id not in valid_ids or item_id in result:
            continue
        if used_ids is not None and item_id in used_ids:
            continue
        result.append(item_id)
        if used_ids is not None:
            used_ids.add(item_id)
    return result


def validate_answer_plan(
    plan: dict[str, Any],
    recommended_items: pd.DataFrame,
) -> dict[str, Any]:
    if not isinstance(plan, dict):
        raise ValueError("planner 返回结构不是 JSON object")

    recommended = ensure_recommended_item_ids(recommended_items)
    valid_ids = valid_item_id_set(recommended)
    sections_input = plan.get("sections")
    if not isinstance(sections_input, list):
        raise ValueError("planner sections 格式错误")

    used_section_ids: set[str] = set()
    sections: list[dict[str, Any]] = []
    for section in sections_input:
        if not isinstance(section, dict):
            continue
        item_ids = unique_valid_item_ids(section.get("item_ids"), valid_ids, used_section_ids)
        if not item_ids:
            continue
        sections.append(
            {
                "title": cell_text(section.get("title")) or "历史样本候选清单项",
                "item_ids": item_ids,
            }
        )
    if not sections:
        raise ValueError("planner sections 为空或无有效 item_id")

    similar_groups: list[dict[str, Any]] = []
    similar_input = plan.get("similar_groups", [])
    if isinstance(similar_input, list):
        for group in similar_input:
            if not isinstance(group, dict):
                continue
            item_ids = unique_valid_item_ids(group.get("item_ids"), valid_ids)
            if not item_ids:
                continue
            display_item_id = cell_text(group.get("display_item_id"))
            if display_item_id not in item_ids:
                display_item_id = item_ids[0]
            similar_groups.append(
                {
                    "title": cell_text(group.get("title")) or "相近做法或同类项",
                    "item_ids": item_ids,
                    "display_item_id": display_item_id,
                    "reason": cell_text(group.get("reason")),
                }
            )

    conditional_item_ids = unique_valid_item_ids(plan.get("conditional_item_ids", []), valid_ids)
    excluded_item_ids = unique_valid_item_ids(plan.get("excluded_item_ids", []), valid_ids)
    notes_input = plan.get("notes", [])
    notes = [cell_text(note) for note in notes_input if cell_text(note)] if isinstance(notes_input, list) else []

    mode = cell_text(plan.get("mode"))
    if mode not in {"flat", "sectioned"}:
        mode = "flat" if len(sections) == 1 else "sectioned"

    return {
        "mode": mode,
        "sections": sections,
        "similar_groups": similar_groups,
        "conditional_item_ids": conditional_item_ids,
        "excluded_item_ids": excluded_item_ids,
        "notes": notes[:3],
    }


def build_fallback_answer_plan(
    recommended_items: pd.DataFrame,
    planner_error: str = "",
) -> dict[str, Any]:
    recommended = ensure_recommended_item_ids(recommended_items)
    item_ids = [cell_text(value) for value in recommended["item_id"].head(ANSWER_PLAN_ITEM_LIMIT).tolist()]
    item_ids = [item_id for item_id in item_ids if item_id]
    notes = ["部分清单项可能属于替代做法、措施项或条件项，需人工复核是否计入。"]
    if len(recommended) > ANSWER_PLAN_ITEM_LIMIT:
        notes.insert(0, "候选项较多，更多候选见 recommended_items sheet。")
    return {
        "mode": "flat",
        "sections": [{"title": "历史样本候选清单项", "item_ids": item_ids}] if item_ids else [],
        "similar_groups": [],
        "conditional_item_ids": [],
        "excluded_item_ids": [],
        "notes": notes[:3],
        "planner_source": "fallback_template",
        "planner_error": planner_error,
    }


def build_answer_plan(
    raw_text: str,
    summary: dict[str, Any],
    recommended_items: pd.DataFrame,
) -> dict[str, Any]:
    recommended = ensure_recommended_item_ids(recommended_items)
    if recommended.empty:
        return {
            "mode": "flat",
            "sections": [],
            "similar_groups": [],
            "conditional_item_ids": [],
            "excluded_item_ids": [],
            "notes": [],
            "planner_source": "fallback_template",
            "planner_error": "",
        }

    try:
        prompt = build_answer_plan_prompt(raw_text, summary, recommended)
        plan = request_llm_json(
            prompt,
            system_prompt=(
                "你是维修工程造价问答的 answer planner。"
                "不要输出思考过程，不要输出 <think>，不要输出 markdown。"
                "最终答案只能是一个 JSON object。"
            ),
        )
        validated = validate_answer_plan(plan, recommended)
        validated["planner_source"] = "planner_template"
        validated["planner_error"] = ""
        return validated
    except (LLMServiceError, RuntimeError, ValueError, KeyError, TypeError) as exc:
        return build_fallback_answer_plan(recommended, planner_error=str(exc))


def item_lookup_by_id(recommended_items: pd.DataFrame) -> dict[str, pd.Series]:
    recommended = ensure_recommended_item_ids(recommended_items)
    return {cell_text(row.get("item_id")): row for _index, row in recommended.iterrows() if cell_text(row.get("item_id"))}


def render_answer_from_plan(
    raw_text: str,
    summary: dict[str, Any],
    recommended_items: pd.DataFrame,
    answer_plan: dict[str, Any],
) -> str:
    warnings = cell_text(summary.get("提示"))
    recommended = ensure_recommended_item_ids(recommended_items)
    if recommended.empty:
        risk_parts = [warnings or "同目录推荐清单项为空，样本不足，不能形成可靠价格参考。", NO_AUTO_TOTAL_NOTE]
        return (
            f"根据历史已审定样本，类似{raw_text}暂未形成可用的推荐清单项/工艺。\n\n"
            f"需要补充的信息/风险提示：{'；'.join(part for part in risk_parts if part)}"
        )

    item_lookup = item_lookup_by_id(recommended)
    conditional_ids = set(answer_plan.get("conditional_item_ids") or [])
    excluded_ids = set(answer_plan.get("excluded_item_ids") or [])

    similar_by_display: dict[str, dict[str, Any]] = {}
    similar_hidden_ids: set[str] = set()
    for group in answer_plan.get("similar_groups") or []:
        if not isinstance(group, dict):
            continue
        display_item_id = cell_text(group.get("display_item_id"))
        item_ids = [cell_text(item_id) for item_id in group.get("item_ids", []) if cell_text(item_id)]
        if display_item_id and item_ids:
            similar_by_display[display_item_id] = group
            similar_hidden_ids.update(item_id for item_id in item_ids if item_id != display_item_id)

    lines = [f"根据历史已审定样本，类似{raw_text}通常可能包含以下清单项/工艺："]
    display_order = 1
    rendered_any = False
    for section in answer_plan.get("sections") or []:
        section_title = cell_text(section.get("title")) if isinstance(section, dict) else ""
        item_ids = section.get("item_ids", []) if isinstance(section, dict) else []
        if section_title:
            lines.append("")
            lines.append(f"【{section_title}】")
        for item_id in item_ids:
            item_id = cell_text(item_id)
            if not item_id or item_id in excluded_ids or item_id in similar_hidden_ids:
                continue
            row = item_lookup.get(item_id)
            if row is None:
                continue
            rendered_any = True
            name = answer_item_display_name(row, is_conditional=item_id in conditional_ids) or f"清单项{display_order}"
            unit = cell_text(row.get("unit_normalized")) or "未注明"
            lines.append(
                f"{display_order}. {name}：\n"
                f"   历史样本类似工程常见程度：{format_occurrence_text(row)}\n"
                f"   单位：{unit}\n"
                f"   {format_unit_price_text(row)}\n"
                f"   {format_price_breakdown_text(row)}\n"
                f"   施工工艺/项目特征：{item_feature_text(row)}"
            )
            estimate_text = format_simple_estimate_text(row)
            if estimate_text:
                lines.append(f"   {estimate_text}")
            similar_group = similar_by_display.get(item_id)
            if similar_group:
                similar_names: list[str] = []
                for similar_id in similar_group.get("item_ids", []):
                    similar_id = cell_text(similar_id)
                    if not similar_id or similar_id == item_id:
                        continue
                    similar_row = item_lookup.get(similar_id)
                    similar_name = cell_text(similar_row.get("cost_item_name")) if similar_row is not None else ""
                    if similar_name and similar_name not in similar_names:
                        similar_names.append(similar_name)
                if similar_names:
                    lines.append(f"   历史样本中另有相近做法/同类项：{'、'.join(similar_names)}，详见 recommended_items。")
            display_order += 1

    if not rendered_any:
        return render_answer_from_plan(raw_text, summary, recommended, build_fallback_answer_plan(recommended))

    risk_parts = []
    if warnings:
        risk_parts.append(warnings)
    risk_parts.extend(cell_text(note) for note in answer_plan.get("notes", []) if cell_text(note))
    risk_parts.append(NO_AUTO_TOTAL_NOTE)
    risk = "；".join(dict.fromkeys(part for part in risk_parts if part))
    lines.append("")
    lines.append(f"需要补充的信息/风险提示：{risk or '仍需人工确认工程量、施工做法、材料规格和现场条件。'}")
    return "\n".join(lines)


def build_answer_fallback(raw_text: str, summary: dict[str, Any], recommended_items: pd.DataFrame) -> str:
    return render_answer_from_plan(raw_text, summary, recommended_items, build_fallback_answer_plan(recommended_items))


def build_answer(raw_text: str, summary: dict[str, Any], recommended_items: pd.DataFrame) -> dict[str, Any]:
    answer_plan = build_answer_plan(raw_text, summary, recommended_items)
    answer_source = "planner_template" if answer_plan.get("planner_source") == "planner_template" else "fallback_template"
    return {
        "answer": render_answer_from_plan(raw_text, summary, recommended_items, answer_plan),
        "answer_source": answer_source,
        "answer_error": cell_text(answer_plan.get("planner_error")),
        "answer_plan": answer_plan,
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
            elif header in {"planner_error", "reason", "note"}:
                width = 60
            elif header == "project_description":
                width = 60
            elif header in {"cost_item_name", "工程名称"}:
                width = 32 if header == "cost_item_name" else 30
            worksheet.column_dimensions[column_cells[0].column_letter].width = width
    workbook.save(output_path)
    workbook.close()


def answer_plan_item_fields(item_id: str, item_lookup: dict[str, pd.Series]) -> dict[str, str]:
    row = item_lookup.get(item_id)
    if row is None:
        return {
            "cost_item_name": "",
            "project_description": "",
            "unit_normalized": "",
        }
    return {
        "cost_item_name": cell_text(row.get("cost_item_name")),
        "project_description": cell_text(row.get("project_description")),
        "unit_normalized": cell_text(row.get("unit_normalized")),
    }


def shown_item_plan_action(item_id: str, conditional_ids: set[str], representative_ids: set[str]) -> str:
    is_representative = item_id in representative_ids
    is_conditional = item_id in conditional_ids
    if is_representative and is_conditional:
        return "展示，代表相近做法；需现场确认"
    if is_representative:
        return "展示，代表相近做法"
    if is_conditional:
        return "展示，需现场确认"
    return "展示"


def answer_plan_for_output(answer_plan: dict[str, Any], recommended_items: pd.DataFrame) -> pd.DataFrame:
    recommended = ensure_recommended_item_ids(recommended_items)
    item_lookup = item_lookup_by_id(recommended)
    conditional_ids = set(answer_plan.get("conditional_item_ids") or [])
    excluded_ids = set(answer_plan.get("excluded_item_ids") or [])
    rows: list[dict[str, Any]] = []

    similar_by_item: dict[str, dict[str, Any]] = {}
    representative_ids: set[str] = set()
    hidden_similar_ids: set[str] = set()
    for group in answer_plan.get("similar_groups") or []:
        if not isinstance(group, dict):
            continue
        display_item_id = cell_text(group.get("display_item_id"))
        if display_item_id:
            representative_ids.add(display_item_id)
        for item_id in group.get("item_ids", []) or []:
            item_id = cell_text(item_id)
            if item_id:
                similar_by_item[item_id] = group
                if display_item_id and item_id != display_item_id:
                    hidden_similar_ids.add(item_id)

    section_by_item: dict[str, str] = {}
    for section in answer_plan.get("sections") or []:
        if not isinstance(section, dict):
            continue
        section_title = cell_text(section.get("title"))
        for item_id in section.get("item_ids", []) or []:
            item_id = cell_text(item_id)
            if item_id and item_id not in section_by_item:
                section_by_item[item_id] = section_title

    emitted_ids: set[str] = set()
    display_order = 1
    for section in answer_plan.get("sections") or []:
        if not isinstance(section, dict):
            continue
        section_title = cell_text(section.get("title"))
        for item_id in section.get("item_ids", []) or []:
            item_id = cell_text(item_id)
            if not item_id or item_id in excluded_ids or item_id in hidden_similar_ids:
                continue
            group = similar_by_item.get(item_id, {})
            rows.append(
                {
                    "row_type": "shown_item",
                    "section_title": section_title,
                    "display_order": display_order,
                    "item_id": item_id,
                    **answer_plan_item_fields(item_id, item_lookup),
                    "plan_action": shown_item_plan_action(item_id, conditional_ids, representative_ids),
                    "representative_item_id": item_id if item_id in representative_ids else "",
                    "similar_group_title": cell_text(group.get("title")) if group else "",
                    "reason": cell_text(group.get("reason")) if group else "",
                    "note": "",
                }
            )
            emitted_ids.add(item_id)
            display_order += 1

    for group in answer_plan.get("similar_groups") or []:
        if not isinstance(group, dict):
            continue
        display_item_id = cell_text(group.get("display_item_id"))
        section_title = section_by_item.get(display_item_id, "") if display_item_id else ""
        for item_id in group.get("item_ids", []) or []:
            item_id = cell_text(item_id)
            if not item_id or item_id == display_item_id or item_id in emitted_ids:
                continue
            rows.append(
                {
                    "row_type": "hidden_similar_item",
                    "section_title": section_title,
                    "display_order": "",
                    "item_id": item_id,
                    **answer_plan_item_fields(item_id, item_lookup),
                    "plan_action": "作为相近做法隐藏",
                    "representative_item_id": display_item_id,
                    "similar_group_title": cell_text(group.get("title")),
                    "reason": cell_text(group.get("reason")),
                    "note": "",
                }
            )
            emitted_ids.add(item_id)

    for item_id in sorted(excluded_ids - emitted_ids):
        group = similar_by_item.get(item_id, {})
        rows.append(
            {
                "row_type": "excluded_item",
                "section_title": "",
                "display_order": "",
                "item_id": item_id,
                **answer_plan_item_fields(item_id, item_lookup),
                "plan_action": "排除",
                "representative_item_id": "",
                "similar_group_title": cell_text(group.get("title")) if group else "",
                "reason": cell_text(group.get("reason")) if group else "",
                "note": "",
            }
        )
        emitted_ids.add(item_id)

    for note in answer_plan.get("notes") or []:
        note = cell_text(note)
        if not note:
            continue
        rows.append(
            {
                "row_type": "note",
                "section_title": "",
                "display_order": "",
                "item_id": "",
                "cost_item_name": "",
                "project_description": "",
                "unit_normalized": "",
                "plan_action": "全局提示",
                "representative_item_id": "",
                "similar_group_title": "",
                "reason": "",
                "note": note,
            }
        )

    if not rows:
        rows.append(
            {
                "row_type": "",
                "section_title": "",
                "display_order": "",
                "item_id": "",
                "cost_item_name": "",
                "project_description": "",
                "unit_normalized": "",
                "plan_action": "",
                "representative_item_id": "",
                "similar_group_title": "",
                "reason": "",
                "note": "",
            }
        )

    return pd.DataFrame(rows, columns=ANSWER_PLAN_COLUMNS)


def write_query_result_workbook(
    output_path: Path,
    answer_result: dict[str, Any],
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
        answer_plan_for_output(answer_result.get("answer_plan", {}), recommended_items).to_excel(
            writer,
            sheet_name="answer_plan",
            index=False,
        )
        project_matches_for_output(matched_projects).to_excel(writer, sheet_name="matched_projects", index=False)
        ensure_recommended_item_ids(recommended_items).to_excel(writer, sheet_name="recommended_items", index=False)
    apply_workbook_style(output_path)


def run_query(
    index_dir: Path,
    raw_text: str,
    top_k: int,
    output: Path | None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
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
