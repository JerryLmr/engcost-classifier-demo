#!/usr/bin/env python3
import argparse
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
from classifier.unit_normalizer import normalize_unit  # noqa: E402
from services.standard_classifier import classify_project_standard  # noqa: E402


CATALOG_EXACT_BONUS = 0.06
UNIT_MATCH_BONUS = 0.03
HAS_UNIT_PRICE_BONUS = 0.01
NO_EXACT_CATALOG_WARNING = "样本库缺少该标准目录下的历史工程/清单项，不能形成可靠价格参考。"

SUMMARY_COLUMNS = [
    "raw_text",
    "classified_project_text",
    "llm_item_query",
    "llm_feature_query",
    "inferred_quantity",
    "inferred_unit",
    "predicted_catalog_id",
    "predicted_一级分类",
    "predicted_二级分类",
    "predicted_维修状态",
    "predicted_标准对象",
    "top_project_score",
    "matched_project_count",
    "recommended_item_count",
    "warnings",
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
    "top_project_score",
}


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
        return SentenceTransformer(model_name)
    except Exception as exc:
        raise RuntimeError(f"embedding 模型加载失败: {model_name}: {exc}") from exc


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


def join_text(parts: list[str], separator: str = " ") -> str:
    return separator.join(part for part in parts if part).strip()


def parse_quantity(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)
    return None


def build_query_profile_prompt(raw_text: str) -> str:
    return f"""
请把用户的口语化维修需求抽取成检索辅助文本。你只负责改写检索词，不要做标准目录分类，不要估算价格，不要生成 catalog_id。

用户输入：
{raw_text}

字段要求：
- llm_item_query 表示清单项、材料、设备或维修对象，可为空。
- llm_feature_query 表示做法、规格、项目特征，可为空。
- quantity/unit 只在用户明确给出时填写，不要猜。
- missing_info 填写影响估价可靠性的缺失信息。
- confidence 只能是 high、medium、low。
- 不要输出 project_text、一级分类、二级分类或 catalog_id。

只输出 JSON object，不要 markdown，不要解释。

JSON 字段固定为：
{{
  "llm_item_query": "",
  "llm_feature_query": "",
  "quantity": null,
  "unit": "",
  "missing_info": [],
  "confidence": "medium"
}}
""".strip()


def normalize_query_profile(raw_profile: dict[str, Any]) -> dict[str, Any]:
    missing_info = raw_profile.get("missing_info")
    if not isinstance(missing_info, list):
        missing_info = []
    confidence = cell_text(raw_profile.get("confidence")).lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    return {
        "llm_item_query": cell_text(raw_profile.get("llm_item_query")),
        "llm_feature_query": cell_text(raw_profile.get("llm_feature_query")),
        "quantity": parse_quantity(raw_profile.get("quantity")),
        "unit": normalize_unit(raw_profile.get("unit")),
        "missing_info": [cell_text(item) for item in missing_info if cell_text(item)],
        "confidence": confidence,
    }


def extract_query_profile(raw_text: str) -> dict[str, Any]:
    return normalize_query_profile(request_llm_json(build_query_profile_prompt(raw_text)))


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
) -> tuple[pd.DataFrame, bool]:
    scores = project_group_embeddings @ raw_query_embedding
    groups = project_groups.copy()
    groups["project_score"] = scores

    if predicted_catalog_id:
        exact = groups[groups["catalog_id"].fillna("").astype(str).str.strip() == predicted_catalog_id].copy()
        if exact.empty:
            return exact, False
        selected = exact.sort_values("project_score", ascending=False).head(max(top_k, 0)).copy()
    else:
        selected = groups.sort_values("project_score", ascending=False).head(max(top_k, 0)).copy()

    selected.insert(0, "rank", range(1, len(selected) + 1))
    return selected, True


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
    unit: str,
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
        unit_match = bool(unit and cell_text(sample.get("unit_normalized")) == unit)
        has_unit_price = numeric_or_none(sample.get("unit_price")) is not None
        semantic_score = max(float(item_scores[index]), float(project_scores[index]), float(full_scores[index]))
        final_score = semantic_score
        if catalog_match:
            final_score += CATALOG_EXACT_BONUS
        if unit_match:
            final_score += UNIT_MATCH_BONUS
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
    profile: dict[str, Any],
    classify_warnings: list[str],
    exact_catalog_available: bool,
    recommended_count: int,
) -> list[str]:
    warnings = list(classify_warnings)
    if not exact_catalog_available:
        warnings.append(NO_EXACT_CATALOG_WARNING)
    if profile.get("quantity") is None:
        warnings.append("未识别工程量，推荐清单项仅提供单价参考。")
    if not profile.get("unit"):
        warnings.append("未识别计量单位，价格参考按推荐清单项自身单位展示。")
    if recommended_count == 0:
        warnings.append("未形成推荐清单项，请补充材料、做法、面积或设备规格。")
    if profile.get("confidence") == "low":
        warnings.append("输入信息较模糊，系统解析置信度低。")
    return warnings


def build_summary(
    raw_text: str,
    profile: dict[str, Any],
    classification: dict[str, Any],
    matched_projects: pd.DataFrame,
    recommended_items: pd.DataFrame,
    warnings: list[str],
) -> dict[str, Any]:
    top_project_score = float(matched_projects["project_score"].iloc[0]) if not matched_projects.empty else None
    return {
        "raw_text": raw_text,
        "classified_project_text": raw_text,
        "llm_item_query": profile.get("llm_item_query"),
        "llm_feature_query": profile.get("llm_feature_query"),
        "inferred_quantity": profile.get("quantity"),
        "inferred_unit": profile.get("unit"),
        "predicted_catalog_id": cell_text(classification.get("catalog_id")),
        "predicted_一级分类": cell_text(classification.get("category")),
        "predicted_二级分类": cell_text(classification.get("item")),
        "predicted_维修状态": cell_text(classification.get("repair_status")),
        "predicted_标准对象": cell_text(classification.get("standard_group")),
        "top_project_score": top_project_score,
        "matched_project_count": int(len(matched_projects)),
        "recommended_item_count": int(len(recommended_items)),
        "warnings": "；".join(warnings),
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
            if header in {"project_description", "warnings", "llm_feature_query"}:
                width = 48
            elif header in {"cost_item_name", "工程名称"}:
                width = 30
            worksheet.column_dimensions[column_cells[0].column_letter].width = width
    workbook.save(output_path)
    workbook.close()


def write_query_result_workbook(
    output_path: Path,
    summary: dict[str, Any],
    recommended_items: pd.DataFrame,
    debug_item_matches: pd.DataFrame,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
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
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    (
        samples,
        project_groups,
        item_embeddings,
        project_embeddings,
        full_embeddings,
        project_group_embeddings,
        meta,
    ) = load_index(index_dir)
    profile = extract_query_profile(raw_text)
    classification, classify_warnings = classify_raw_text(raw_text)

    model = load_embedding_model(str(meta.get("model") or "BAAI/bge-m3"))
    raw_query_embedding = encode_query(model, raw_text)
    predicted_catalog_id = cell_text(classification.get("catalog_id"))
    matched_projects, exact_catalog_available = select_project_groups(
        project_groups=project_groups,
        project_group_embeddings=project_group_embeddings,
        raw_query_embedding=raw_query_embedding,
        predicted_catalog_id=predicted_catalog_id,
        top_k=top_k,
    )
    recommended_items = aggregate_recommended_items(samples, matched_projects, top_k)

    item_query = join_text([profile.get("llm_item_query", ""), profile.get("llm_feature_query", "")]) or raw_text
    full_query = join_text([raw_text, profile.get("llm_item_query", ""), profile.get("llm_feature_query", "")])
    debug_item_matches = build_debug_item_matches(
        samples=samples,
        item_embeddings=item_embeddings,
        project_embeddings=project_embeddings,
        full_embeddings=full_embeddings,
        item_query_embedding=encode_query(model, item_query),
        project_query_embedding=raw_query_embedding,
        full_query_embedding=encode_query(model, full_query),
        predicted_catalog_id=predicted_catalog_id,
        unit=profile.get("unit", ""),
        top_k=top_k,
        min_score=min_score,
    )

    warnings = build_warnings(profile, classify_warnings, exact_catalog_available, len(recommended_items))
    summary = build_summary(raw_text, profile, classification, matched_projects, recommended_items, warnings)

    if output:
        write_query_result_workbook(output, summary, recommended_items, debug_item_matches)

    return summary, recommended_items, debug_item_matches


def print_terminal_summary(summary: dict[str, Any], recommended_items: pd.DataFrame, output_path: Path | None) -> None:
    print(f"[DONE] matched projects: {summary.get('matched_project_count')}")
    print(f"[DONE] recommended items: {summary.get('recommended_item_count')}")
    print(
        "预测目录: "
        f"{summary.get('predicted_catalog_id')} "
        f"{summary.get('predicted_一级分类')} / {summary.get('predicted_二级分类')}"
    )
    if not recommended_items.empty:
        first = recommended_items.iloc[0].to_dict()
        print(
            "首位推荐清单项: "
            f"{first.get('cost_item_name')} / {first.get('unit_normalized')} / "
            f"单价中位数 {first.get('unit_price_median')}"
        )
    if summary.get("warnings"):
        print(f"注意事项: {summary['warnings']}")
    if output_path:
        print(f"输出文件: {output_path}")


def main() -> int:
    args = parse_args()
    index_dir = Path(args.index_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else None

    try:
        validate_output_path(output_path, args.overwrite)
        summary, recommended_items, _debug_item_matches = run_query(
            index_dir=index_dir,
            raw_text=args.text,
            top_k=args.top_k,
            min_score=args.min_score,
            output=output_path,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    print_terminal_summary(summary, recommended_items, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
