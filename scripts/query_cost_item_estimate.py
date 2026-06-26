#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

NOTE = "参考区间来自已审定清单样本的相似项检索结果，仅用于维修项目初案估算参考，不替代正式造价审核。"

MATCH_COLUMNS = [
    "rank",
    "final_score",
    "item_score",
    "context_score",
    "source_row_id",
    "item_row_id",
    "工程名称",
    "consultation_time",
    "location",
    "sub_project_id",
    "catalog_id",
    "一级分类",
    "二级分类",
    "维修状态",
    "标准对象",
    "是否复合工程",
    "复合目录",
    "seq",
    "cost_item_name",
    "project_description",
    "unit_normalized",
    "quantity",
    "unit_price",
    "labor_unit_price",
    "machinery_unit_price",
    "item_similarity_text",
    "item_context_text",
]

SUMMARY_COLUMNS = [
    "query",
    "context",
    "unit",
    "catalog_id",
    "consultation_time",
    "location",
    "quantity",
    "top_k",
    "matched_count",
    "filter_notes",
    "unit_price_count",
    "unit_price_min",
    "unit_price_median",
    "unit_price_max",
    "estimated_total_min",
    "estimated_total_median",
    "estimated_total_max",
    "labor_unit_price_count",
    "labor_unit_price_min",
    "labor_unit_price_median",
    "labor_unit_price_max",
    "estimated_labor_min",
    "estimated_labor_median",
    "estimated_labor_max",
    "machinery_unit_price_count",
    "machinery_unit_price_min",
    "machinery_unit_price_median",
    "machinery_unit_price_max",
    "estimated_machinery_min",
    "estimated_machinery_median",
    "estimated_machinery_max",
    "note",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于相似清单样本估算初案价格参考区间")
    parser.add_argument("--index-dir", required=True, help="索引目录")
    parser.add_argument("--query", required=True, help="清单项自身语义文本")
    parser.add_argument("--context", default="", help="工程上下文文本")
    parser.add_argument("--unit", default="", help="单位，优先按 unit_normalized 过滤")
    parser.add_argument("--catalog-id", default="", help="标准目录 id，优先按 catalog_id 过滤")
    parser.add_argument("--consultation-time", default="", help="咨询时间，按 consultation_time 精确过滤")
    parser.add_argument("--location", default="", help="地点，按 location 精确过滤")
    parser.add_argument("--quantity", type=float, default=None, help="估算数量")
    parser.add_argument("--top-k", type=int, default=10, help="输出相似样本数量")
    parser.add_argument("--output", default="", help="可选 xlsx 输出路径")
    parser.add_argument("--overwrite", action="store_true", help="若输出文件已存在则覆盖")
    parser.add_argument("--item-weight", type=float, default=0.75, help="清单项语义权重")
    parser.add_argument("--context-weight", type=float, default=0.25, help="上下文语义权重")
    parser.add_argument("--min-candidates", type=int, default=20, help="catalog_id 过滤最小候选数")
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


def load_samples_and_embeddings(index_dir: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    samples_path = index_dir / "samples.parquet"
    item_path = index_dir / "item_similarity_embeddings.npy"
    context_path = index_dir / "item_context_embeddings.npy"
    meta_path = index_dir / "index_meta.json"

    missing = [path.name for path in [samples_path, item_path, context_path, meta_path] if not path.exists()]
    if missing:
        raise ValueError(f"索引目录缺少文件: {', '.join(missing)}")

    samples = pd.read_parquet(samples_path)
    item_embeddings = np.load(item_path)
    context_embeddings = np.load(context_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if len(samples) != item_embeddings.shape[0] or len(samples) != context_embeddings.shape[0]:
        raise ValueError("样本数量与 embedding 数量不一致")
    if item_embeddings.shape != context_embeddings.shape:
        raise ValueError("两路 embedding 维度或样本数不一致")
    return samples, item_embeddings, context_embeddings, meta


def normalize_weights(item_weight: float, context_weight: float, has_context: bool) -> tuple[float, float, list[str]]:
    if not has_context:
        return 1.0, 0.0, []

    total = item_weight + context_weight
    if total <= 0:
        raise ValueError("item-weight 和 context-weight 之和必须大于 0")
    if abs(total - 1.0) < 1e-9:
        return item_weight, context_weight, []
    return item_weight / total, context_weight / total, ["权重和不为1，已自动归一化"]


def safe_text_series(samples: pd.DataFrame, column: str) -> pd.Series:
    if column not in samples.columns:
        return pd.Series([""] * len(samples), index=samples.index, dtype=str)
    return samples[column].fillna("").astype(str)


def build_candidate_mask(
    samples: pd.DataFrame,
    unit: str,
    catalog_id: str,
    consultation_time: str,
    location: str,
    min_candidates: int,
) -> tuple[np.ndarray, list[str]]:
    notes: list[str] = []
    mask = np.ones(len(samples), dtype=bool)

    if unit:
        unit_values = safe_text_series(samples, "unit_normalized")
        unit_mask = (unit_values == unit).to_numpy()
        if unit_mask.any():
            mask &= unit_mask
        else:
            notes.append("单位过滤无匹配，已放宽")

    if catalog_id:
        catalog_values = safe_text_series(samples, "catalog_id")
        composite_values = safe_text_series(samples, "复合目录")
        current_pool = mask.copy()
        primary_mask = current_pool & (catalog_values == catalog_id).to_numpy()
        candidate_mask = primary_mask.copy()

        if candidate_mask.sum() < min_candidates:
            composite_mask = current_pool & composite_values.str.contains(catalog_id, regex=False, na=False).to_numpy()
            if (composite_mask & ~candidate_mask).any():
                notes.append("复合目录已补充相似样本候选")
            candidate_mask |= composite_mask

        if candidate_mask.sum() < min_candidates:
            notes.append("分类候选数量不足，已放宽")
            candidate_mask = current_pool

        mask = candidate_mask

    for column, value, label in [
        ("consultation_time", consultation_time, "consultation_time"),
        ("location", location, "location"),
    ]:
        if not value:
            continue
        if column not in samples.columns:
            notes.append(f"{label} 过滤字段缺失，未匹配到样本")
            mask &= np.zeros(len(samples), dtype=bool)
            continue
        exact_mask = (safe_text_series(samples, column) == value).to_numpy()
        mask &= exact_mask
        if not mask.any():
            notes.append(f"{label} 过滤无匹配")

    return mask, notes


def score_candidates(
    samples: pd.DataFrame,
    item_embeddings: np.ndarray,
    context_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    context_embedding: np.ndarray | None,
    candidate_mask: np.ndarray,
    top_k: int,
    item_weight: float,
    context_weight: float,
) -> pd.DataFrame:
    if not candidate_mask.any():
        return pd.DataFrame(columns=MATCH_COLUMNS)

    item_scores = item_embeddings @ query_embedding
    if context_embedding is None:
        context_scores = np.zeros(len(samples), dtype=np.float32)
        final_scores = item_scores
    else:
        context_scores = context_embeddings @ context_embedding
        final_scores = item_weight * item_scores + context_weight * context_scores

    candidate_indexes = np.flatnonzero(candidate_mask)
    ordered = candidate_indexes[np.argsort(final_scores[candidate_indexes])[::-1]]
    selected = ordered[: max(top_k, 0)]

    rows = samples.iloc[selected].copy()
    rows.insert(0, "context_score", context_scores[selected])
    rows.insert(0, "item_score", item_scores[selected])
    rows.insert(0, "final_score", final_scores[selected])
    rows.insert(0, "rank", range(1, len(rows) + 1))

    for column in MATCH_COLUMNS:
        if column not in rows.columns:
            rows[column] = ""
    return rows[MATCH_COLUMNS]


def numeric_values(matches: pd.DataFrame, column: str) -> pd.Series:
    if column not in matches.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(matches[column], errors="coerce").dropna()


def summarize_one_price_field(matches: pd.DataFrame, column: str) -> dict[str, float | int | None]:
    values = numeric_values(matches, column)
    if values.empty:
        return {"count": 0, "min": None, "median": None, "max": None}
    return {
        "count": int(values.count()),
        "min": float(values.min()),
        "median": float(values.median()),
        "max": float(values.max()),
    }


def multiply_or_none(value: float | int | None, quantity: float | None) -> float | None:
    if value is None or quantity is None:
        return None
    return float(value) * quantity


def summarize_price_ranges(
    matches: pd.DataFrame,
    query: str,
    context: str,
    unit: str,
    catalog_id: str,
    consultation_time: str,
    location: str,
    quantity: float | None,
    top_k: int,
    filter_notes: list[str],
) -> dict[str, Any]:
    unit_price = summarize_one_price_field(matches, "unit_price")
    labor_price = summarize_one_price_field(matches, "labor_unit_price")
    machinery_price = summarize_one_price_field(matches, "machinery_unit_price")

    notes = list(filter_notes)
    if any(summary["count"] and summary["count"] < 3 for summary in [unit_price, labor_price, machinery_price]):
        notes.append("样本数不足，区间仅供参考")

    return {
        "query": query,
        "context": context,
        "unit": unit,
        "catalog_id": catalog_id,
        "consultation_time": consultation_time,
        "location": location,
        "quantity": quantity,
        "top_k": top_k,
        "matched_count": len(matches),
        "filter_notes": "；".join(notes),
        "unit_price_count": unit_price["count"],
        "unit_price_min": unit_price["min"],
        "unit_price_median": unit_price["median"],
        "unit_price_max": unit_price["max"],
        "estimated_total_min": multiply_or_none(unit_price["min"], quantity),
        "estimated_total_median": multiply_or_none(unit_price["median"], quantity),
        "estimated_total_max": multiply_or_none(unit_price["max"], quantity),
        "labor_unit_price_count": labor_price["count"],
        "labor_unit_price_min": labor_price["min"],
        "labor_unit_price_median": labor_price["median"],
        "labor_unit_price_max": labor_price["max"],
        "estimated_labor_min": multiply_or_none(labor_price["min"], quantity),
        "estimated_labor_median": multiply_or_none(labor_price["median"], quantity),
        "estimated_labor_max": multiply_or_none(labor_price["max"], quantity),
        "machinery_unit_price_count": machinery_price["count"],
        "machinery_unit_price_min": machinery_price["min"],
        "machinery_unit_price_median": machinery_price["median"],
        "machinery_unit_price_max": machinery_price["max"],
        "estimated_machinery_min": multiply_or_none(machinery_price["min"], quantity),
        "estimated_machinery_median": multiply_or_none(machinery_price["median"], quantity),
        "estimated_machinery_max": multiply_or_none(machinery_price["max"], quantity),
        "note": NOTE,
    }


def write_query_result_workbook(output_path: Path, summary: dict[str, Any], matches: pd.DataFrame) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_row = {column: summary.get(column) for column in SUMMARY_COLUMNS}
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        pd.DataFrame([summary_row], columns=SUMMARY_COLUMNS).to_excel(writer, sheet_name="summary", index=False)
        matches.to_excel(writer, sheet_name="matches", index=False)


def format_range(summary: dict[str, Any], prefix: str) -> str:
    minimum = summary.get(f"{prefix}_min")
    median = summary.get(f"{prefix}_median")
    maximum = summary.get(f"{prefix}_max")
    if minimum is None or median is None or maximum is None:
        return "暂无参考区间"
    return f"{minimum:g} - {maximum:g}，中位数 {median:g}"


def print_terminal_summary(summary: dict[str, Any], output_path: Path | None) -> None:
    matched_count = summary["matched_count"]
    if matched_count == 0:
        print("[WARN] 未匹配到相似样本，请检查 unit/catalog_id 是否过严，或放宽过滤条件。")
        return

    print(f"[DONE] matched samples: {matched_count}")
    print(f"综合单价参考：{format_range(summary, 'unit_price')} 元/{summary.get('unit') or '单位'}")
    if summary.get("quantity") is not None:
        print(f"估算综合费用：{format_range(summary, 'estimated_total')} 元")
    if summary.get("filter_notes"):
        print(f"注意事项：{summary['filter_notes']}")
    if output_path:
        print(f"输出文件：{output_path}")


def run_query(
    index_dir: Path,
    query: str,
    context: str,
    unit: str,
    catalog_id: str,
    consultation_time: str,
    location: str,
    quantity: float | None,
    top_k: int,
    output: Path | None,
    item_weight: float,
    context_weight: float,
    min_candidates: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    samples, item_embeddings, context_embeddings, meta = load_samples_and_embeddings(index_dir)
    model = load_embedding_model(str(meta.get("model") or "BAAI/bge-m3"))

    has_context = bool(context.strip())
    normalized_item_weight, normalized_context_weight, weight_notes = normalize_weights(
        item_weight,
        context_weight,
        has_context,
    )
    candidate_mask, filter_notes = build_candidate_mask(
        samples,
        unit,
        catalog_id,
        consultation_time,
        location,
        min_candidates,
    )
    filter_notes = weight_notes + filter_notes

    query_embedding = encode_query(model, query)
    context_embedding = encode_query(model, context) if has_context else None
    matches = score_candidates(
        samples=samples,
        item_embeddings=item_embeddings,
        context_embeddings=context_embeddings,
        query_embedding=query_embedding,
        context_embedding=context_embedding,
        candidate_mask=candidate_mask,
        top_k=top_k,
        item_weight=normalized_item_weight,
        context_weight=normalized_context_weight,
    )
    summary = summarize_price_ranges(
        matches=matches,
        query=query,
        context=context,
        unit=unit,
        catalog_id=catalog_id,
        consultation_time=consultation_time,
        location=location,
        quantity=quantity,
        top_k=top_k,
        filter_notes=filter_notes,
    )

    if output:
        write_query_result_workbook(output, summary, matches)

    return summary, matches


def main() -> int:
    args = parse_args()
    index_dir = Path(args.index_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else None

    try:
        validate_output_path(output_path, args.overwrite)
        summary, _matches = run_query(
            index_dir=index_dir,
            query=args.query,
            context=args.context,
            unit=args.unit,
            catalog_id=args.catalog_id,
            consultation_time=args.consultation_time,
            location=args.location,
            quantity=args.quantity,
            top_k=args.top_k,
            output=output_path,
            item_weight=args.item_weight,
            context_weight=args.context_weight,
            min_candidates=args.min_candidates,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    print_terminal_summary(summary, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
