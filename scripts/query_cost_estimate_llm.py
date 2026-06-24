#!/usr/bin/env python3
import argparse
import json
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
from classifier.llm_client import request_llm_json  # noqa: E402
from services.standard_classifier import classify_project_standard  # noqa: E402


CATALOG_EXACT_BONUS = 0.06
CATALOG_PREFIX_BONUS = 0.02
UNIT_MATCH_BONUS = 0.03
HAS_UNIT_PRICE_BONUS = 0.01

SUMMARY_COLUMNS = [
    "raw_text",
    "project_text",
    "item_text",
    "feature_text",
    "inferred_quantity",
    "inferred_unit",
    "predicted_catalog_id",
    "predicted_category",
    "predicted_item",
    "min_score",
    "top_score",
    "priced_count",
    "warnings",
]

UNIT_PRICE_COLUMNS = [
    "unit",
    "sample_count",
    "priced_count",
    "min",
    "p25",
    "median",
    "p75",
    "max",
    "estimated_total_p25",
    "estimated_total_median",
    "estimated_total_p75",
]

MATCH_COLUMNS = [
    "rank",
    "final_score",
    "item_score",
    "project_score",
    "full_score",
    "catalog_match",
    "unit_match",
    "工程名称",
    "catalog_id",
    "一级分类",
    "二级分类",
    "维修状态",
    "cost_item_name",
    "project_description",
    "unit_normalized",
    "quantity",
    "unit_price",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LLM 自然语言造价估计实验入口")
    parser.add_argument("--index-dir", required=True, help="索引目录")
    parser.add_argument("--text", required=True, help="口语化维修需求")
    parser.add_argument("--output", default="", help="可选 xlsx 输出路径")
    parser.add_argument("--overwrite", action="store_true", help="若输出文件已存在则覆盖")
    parser.add_argument("--top-k", type=int, default=20, help="输出相似样本数量")
    parser.add_argument("--min-score", type=float, default=0.6, help="价格统计最小 final_score")
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


def load_samples_and_embeddings(index_dir: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    samples_path = index_dir / "samples.parquet"
    item_path = index_dir / "item_embeddings.npy"
    project_path = index_dir / "project_embeddings.npy"
    full_path = index_dir / "full_embeddings.npy"
    meta_path = index_dir / "index_meta.json"

    missing = [path.name for path in [samples_path, item_path, project_path, full_path, meta_path] if not path.exists()]
    if missing:
        raise ValueError(f"索引目录缺少文件: {', '.join(missing)}")

    samples = pd.read_parquet(samples_path)
    item_embeddings = np.load(item_path)
    project_embeddings = np.load(project_path)
    full_embeddings = np.load(full_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if (
        len(samples) != item_embeddings.shape[0]
        or len(samples) != project_embeddings.shape[0]
        or len(samples) != full_embeddings.shape[0]
    ):
        raise ValueError("样本数量与 embedding 数量不一致")
    if item_embeddings.shape != project_embeddings.shape or item_embeddings.shape != full_embeddings.shape:
        raise ValueError("三路 embedding 维度或样本数不一致")
    return samples, item_embeddings, project_embeddings, full_embeddings, meta


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
请把用户的口语化维修需求抽取成检索 profile。你只负责解析输入，不要估算价格，不要生成 catalog_id。

用户输入：
{raw_text}

字段要求：
- project_text 表示工程/维修场景。
- item_text 表示清单项、材料或设备对象。
- feature_text 表示做法、规格、项目特征。
- quantity/unit 只在用户明确给出时填写，不要猜。
- catalog_hint_text 可填写有助于工程分类的自然语言线索，但不要填写 catalog_id。
- missing_info 填写影响估价可靠性的缺失信息。
- confidence 只能是 high、medium、low。

只输出 JSON object，不要 markdown，不要解释。

JSON 字段固定为：
{{
  "project_text": "...",
  "item_text": "...",
  "feature_text": "...",
  "quantity": null,
  "unit": "",
  "catalog_hint_text": "",
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

    unit = normalize_unit(raw_profile.get("unit"))
    return {
        "project_text": cell_text(raw_profile.get("project_text")),
        "item_text": cell_text(raw_profile.get("item_text")),
        "feature_text": cell_text(raw_profile.get("feature_text")),
        "quantity": parse_quantity(raw_profile.get("quantity")),
        "unit": unit,
        "catalog_hint_text": cell_text(raw_profile.get("catalog_hint_text")),
        "missing_info": [cell_text(item) for item in missing_info if cell_text(item)],
        "confidence": confidence,
    }


def extract_query_profile(raw_text: str) -> dict[str, Any]:
    raw_profile = request_llm_json(build_query_profile_prompt(raw_text))
    return normalize_query_profile(raw_profile)


def classify_query_profile(profile: dict[str, Any], raw_text: str) -> tuple[dict[str, Any], list[str]]:
    classify_text = profile.get("project_text") or raw_text
    result = classify_project_standard(str(classify_text))
    warnings: list[str] = []
    pipeline_status = cell_text(result.get("pipeline_status"))
    if pipeline_status and pipeline_status != "ok":
        reason = cell_text(result.get("reason"))
        warnings.append(f"工程分类未能稳定匹配标准目录：{reason or pipeline_status}")
    return result, warnings


def catalog_prefix(catalog_id: str) -> str:
    text = cell_text(catalog_id)
    prefix = text.split("-", 1)[0]
    return prefix if prefix in {"CP", "CF"} else ""


def top_indexes(scores: np.ndarray, limit: int) -> np.ndarray:
    if limit <= 0 or scores.size == 0:
        return np.array([], dtype=int)
    return np.argsort(scores)[::-1][: min(limit, scores.size)]


def numeric_or_none(value: Any) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return None
    return float(number)


def score_and_rerank_candidates(
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
) -> pd.DataFrame:
    item_scores = item_embeddings @ item_query_embedding
    project_scores = project_embeddings @ project_query_embedding
    full_scores = full_embeddings @ full_query_embedding

    recall_limit = max(top_k, 0) * 3
    candidate_indexes = set(top_indexes(item_scores, recall_limit).tolist())
    candidate_indexes.update(top_indexes(project_scores, recall_limit).tolist())
    candidate_indexes.update(top_indexes(full_scores, recall_limit).tolist())

    predicted_prefix = catalog_prefix(predicted_catalog_id)
    rows: list[dict[str, Any]] = []
    for index in candidate_indexes:
        sample = samples.iloc[index]
        sample_catalog_id = cell_text(sample.get("catalog_id"))
        exact_match = bool(predicted_catalog_id and sample_catalog_id == predicted_catalog_id)
        prefix_match = bool(
            not exact_match
            and predicted_prefix
            and catalog_prefix(sample_catalog_id) == predicted_prefix
        )
        sample_unit = cell_text(sample.get("unit_normalized"))
        unit_match = bool(unit and sample_unit == unit)
        unit_price = numeric_or_none(sample.get("unit_price"))
        has_unit_price = unit_price is not None
        semantic_score = max(float(item_scores[index]), float(project_scores[index]), float(full_scores[index]))
        final_score = semantic_score
        if exact_match:
            final_score += CATALOG_EXACT_BONUS
        if prefix_match:
            final_score += CATALOG_PREFIX_BONUS
        if unit_match:
            final_score += UNIT_MATCH_BONUS
        if has_unit_price:
            final_score += HAS_UNIT_PRICE_BONUS

        row = {column: sample.get(column, "") for column in MATCH_COLUMNS if column not in {
            "rank",
            "final_score",
            "item_score",
            "project_score",
            "full_score",
            "catalog_match",
            "unit_match",
        }}
        row.update(
            {
                "final_score": final_score,
                "item_score": float(item_scores[index]),
                "project_score": float(project_scores[index]),
                "full_score": float(full_scores[index]),
                "catalog_match": "exact" if exact_match else "prefix" if prefix_match else "",
                "unit_match": unit_match,
            }
        )
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=MATCH_COLUMNS)

    matches = pd.DataFrame(rows).sort_values(["final_score"], ascending=False).head(max(top_k, 0)).copy()
    matches.insert(0, "rank", range(1, len(matches) + 1))
    for column in MATCH_COLUMNS:
        if column not in matches.columns:
            matches[column] = ""
    return matches[MATCH_COLUMNS]


def price_sample_pool(matches: pd.DataFrame, min_score: float) -> pd.DataFrame:
    if matches.empty:
        return matches.copy()
    pool = matches.copy()
    pool["unit_price_numeric"] = pd.to_numeric(pool["unit_price"], errors="coerce")
    pool["final_score_numeric"] = pd.to_numeric(pool["final_score"], errors="coerce")
    return pool[(pool["final_score_numeric"] >= min_score) & pool["unit_price_numeric"].notna()].copy()


def quantile_or_none(values: pd.Series, q: float) -> float | None:
    if values.empty:
        return None
    return float(values.quantile(q))


def multiply_or_none(value: float | None, quantity: float | None) -> float | None:
    if value is None or quantity is None:
        return None
    return float(value) * quantity


def summarize_unit_prices(price_pool: pd.DataFrame, quantity: float | None) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if price_pool.empty:
        return pd.DataFrame(columns=UNIT_PRICE_COLUMNS)

    for unit, group in price_pool.groupby("unit_normalized", dropna=False):
        values = pd.to_numeric(group["unit_price"], errors="coerce").dropna()
        p25 = quantile_or_none(values, 0.25)
        median = quantile_or_none(values, 0.5)
        p75 = quantile_or_none(values, 0.75)
        rows.append(
            {
                "unit": cell_text(unit),
                "sample_count": int(len(group)),
                "priced_count": int(values.count()),
                "min": float(values.min()) if not values.empty else None,
                "p25": p25,
                "median": median,
                "p75": p75,
                "max": float(values.max()) if not values.empty else None,
                "estimated_total_p25": multiply_or_none(p25, quantity),
                "estimated_total_median": multiply_or_none(median, quantity),
                "estimated_total_p75": multiply_or_none(p75, quantity),
            }
        )

    return pd.DataFrame(rows, columns=UNIT_PRICE_COLUMNS).sort_values(
        ["priced_count", "unit"],
        ascending=[False, True],
    )


def main_priced_count(price_pool: pd.DataFrame, unit: str) -> int:
    if price_pool.empty:
        return 0
    if unit:
        return int((price_pool["unit_normalized"].fillna("").astype(str).str.strip() == unit).sum())
    return int(len(price_pool))


def build_warnings(
    profile: dict[str, Any],
    classify_warnings: list[str],
    priced_count: int,
    top_score: float | None,
    min_score: float,
) -> list[str]:
    warnings = list(classify_warnings)
    if profile.get("quantity") is None:
        warnings.append("未识别工程量，只能给单价参考，不能估算总价。")
    if not profile.get("unit"):
        warnings.append("未识别计量单位，已按历史样本单位分组展示，不能混合不同单位直接估价。")
    if priced_count < 3:
        warnings.append("高相似可计价样本不足，建议补充材料、做法、面积或设备规格。")
    if top_score is not None and top_score < min_score:
        warnings.append("相似度偏低，结果仅供线索参考。")
    if profile.get("confidence") == "low":
        warnings.append("输入信息较模糊，系统解析置信度低。")
    return warnings


def build_summary(
    raw_text: str,
    profile: dict[str, Any],
    classification: dict[str, Any],
    min_score: float,
    top_score: float | None,
    priced_count: int,
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "raw_text": raw_text,
        "project_text": profile.get("project_text"),
        "item_text": profile.get("item_text"),
        "feature_text": profile.get("feature_text"),
        "inferred_quantity": profile.get("quantity"),
        "inferred_unit": profile.get("unit"),
        "predicted_catalog_id": cell_text(classification.get("catalog_id")),
        "predicted_category": cell_text(classification.get("category")),
        "predicted_item": cell_text(classification.get("item")),
        "min_score": min_score,
        "top_score": top_score,
        "priced_count": priced_count,
        "warnings": "；".join(warnings),
    }


def write_query_result_workbook(
    output_path: Path,
    summary: dict[str, Any],
    unit_price_by_unit: pd.DataFrame,
    matches: pd.DataFrame,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame([{column: summary.get(column) for column in SUMMARY_COLUMNS}], columns=SUMMARY_COLUMNS).to_excel(
            writer,
            sheet_name="summary",
            index=False,
        )
        unit_price_by_unit.to_excel(writer, sheet_name="unit_price_by_unit", index=False)
        matches.to_excel(writer, sheet_name="matches", index=False)


def run_query(
    index_dir: Path,
    raw_text: str,
    top_k: int,
    min_score: float,
    output: Path | None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    samples, item_embeddings, project_embeddings, full_embeddings, meta = load_samples_and_embeddings(index_dir)
    profile = extract_query_profile(raw_text)
    classification, classify_warnings = classify_query_profile(profile, raw_text)

    model = load_embedding_model(str(meta.get("model") or "BAAI/bge-m3"))
    item_query = join_text([profile.get("item_text", ""), profile.get("feature_text", "")]) or raw_text
    project_query = profile.get("project_text") or raw_text
    full_query = join_text(
        [
            raw_text,
            profile.get("project_text", ""),
            profile.get("item_text", ""),
            profile.get("feature_text", ""),
        ]
    )

    matches = score_and_rerank_candidates(
        samples=samples,
        item_embeddings=item_embeddings,
        project_embeddings=project_embeddings,
        full_embeddings=full_embeddings,
        item_query_embedding=encode_query(model, item_query),
        project_query_embedding=encode_query(model, project_query),
        full_query_embedding=encode_query(model, full_query),
        predicted_catalog_id=cell_text(classification.get("catalog_id")),
        unit=profile.get("unit", ""),
        top_k=top_k,
    )

    price_pool = price_sample_pool(matches, min_score)
    unit_price_by_unit = summarize_unit_prices(price_pool, profile.get("quantity"))
    priced_count = main_priced_count(price_pool, profile.get("unit", ""))
    top_score = float(matches["final_score"].iloc[0]) if not matches.empty else None
    warnings = build_warnings(profile, classify_warnings, priced_count, top_score, min_score)
    summary = build_summary(
        raw_text=raw_text,
        profile=profile,
        classification=classification,
        min_score=min_score,
        top_score=top_score,
        priced_count=priced_count,
        warnings=warnings,
    )

    if output:
        write_query_result_workbook(output, summary, unit_price_by_unit, matches)

    return summary, unit_price_by_unit, matches


def print_terminal_summary(summary: dict[str, Any], unit_price_by_unit: pd.DataFrame, output_path: Path | None) -> None:
    print(f"[DONE] top_score: {summary.get('top_score')}")
    print(f"预测目录: {summary.get('predicted_catalog_id')} {summary.get('predicted_category')} / {summary.get('predicted_item')}")
    if unit_price_by_unit.empty:
        print("[WARN] 没有满足 min-score 的可计价样本。")
    else:
        first = unit_price_by_unit.iloc[0].to_dict()
        print(
            "综合单价参考: "
            f"{first.get('p25')} - {first.get('p75')}，中位数 {first.get('median')} / {first.get('unit')}"
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
        summary, unit_price_by_unit, _matches = run_query(
            index_dir=index_dir,
            raw_text=args.text,
            top_k=args.top_k,
            min_score=args.min_score,
            output=output_path,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    print_terminal_summary(summary, unit_price_by_unit, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
