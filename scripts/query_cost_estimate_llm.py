#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from classifier.llm_client import LLMServiceError, request_llm_json  # noqa: E402


MATCHED_PROJECT_PACKAGE_COLUMNS = [
    "rank",
    "package_score",
    "project_package_id",
    "工程名称",
    "project_name_text",
    "分类摘要",
    "包含清单摘要",
    "consultation_time",
    "location",
    "item_count",
]

CANDIDATE_ITEM_STATS_COLUMNS = [
    "fine_signature",
    "family_signature",
    "cost_item_name",
    "project_description",
    "unit",
    "历史样本数",
    "来源工程包数",
    "历史工程量最小值",
    "历史工程量中位数",
    "历史工程量最大值",
    "历史综合单价最小值",
    "历史综合单价中位数",
    "历史综合单价最大值",
    "历史合价最小值",
    "历史合价中位数",
    "历史合价最大值",
    "package_score最大值",
    "item_score最大值",
    "cooccur_score",
    "catalog_score",
    "final_score",
    "是否被LLM采用",
    "evidence_refs",
]

EVIDENCE_ITEM_COLUMNS = [
    "evidence_ref",
    "来源工程名称",
    "project_package_id",
    "consultation_time",
    "location",
    "catalog_id",
    "一级分类",
    "二级分类",
    "维修状态",
    "标准对象",
    "cost_item_name",
    "project_description",
    "unit",
    "quantity",
    "unit_price",
    "total_price",
    "labor_unit_price",
    "machinery_unit_price",
    "package_rank",
    "package_score",
    "item_score",
    "cooccur_score",
    "catalog_score",
    "final_score",
]

SUGGESTED_BILL_COLUMNS = [
    "序号",
    "推荐类型",
    "清单项名称",
    "项目特征/施工工艺",
    "单位",
    "建议工程量",
    "工程量依据",
    "综合单价低值",
    "综合单价中值",
    "综合单价高值",
    "估算金额低值",
    "估算金额中值",
    "估算金额高值",
    "采用理由",
    "不确定性说明",
    "来源证据",
]


@dataclass(frozen=True)
class QueryUnderstanding:
    raw_query: str
    semantic_query_text: str
    need_summary: str
    known_constraints: dict[str, Any]
    likely_catalog: dict[str, Any]
    calculation_notes: str
    parse_notes: list[str]
    llm_success: bool


@dataclass(frozen=True)
class QueryResult:
    understanding: QueryUnderstanding
    suggested_bill: pd.DataFrame
    matched_project_packages: pd.DataFrame
    candidate_item_stats: pd.DataFrame
    evidence_items: pd.DataFrame
    parse_info: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="自然语言维修工程造价离线查询入口")
    parser.add_argument("--index-dir", default="embeddings", help="索引目录，默认 embeddings")
    parser.add_argument("--text", required=True, help="口语化维修需求")
    parser.add_argument("--top-packages", type=int, default=20, help="召回相似历史工程包数量")
    parser.add_argument("--top-items", type=int, default=300, help="召回直接相关历史清单行数量")
    parser.add_argument("--output", default=None, help="xlsx 输出路径，默认 query/YYYYMMDDHHMM.xlsx")
    parser.add_argument("--overwrite", action="store_true", help="若输出文件已存在则覆盖")
    parser.add_argument("--include-debug-text", action="store_true", help="在 parse_info 中保留 LLM 调试文本摘要")
    parser.add_argument("--display", action="store_true", help="输出时将部分数值格式化为易读文本")
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


def cell_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number):
        return None
    return number


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def parse_jsonish_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (array / norms).astype(np.float32, copy=False)


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


def encode_query(model: Any, text: str) -> np.ndarray:
    embedding = model.encode(
        [text or ""],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return normalize_embeddings(embedding)[0]


def load_index(index_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    samples_path = index_dir / "samples.parquet"
    packages_path = index_dir / "project_packages.parquet"
    package_embeddings_path = index_dir / "project_package_embeddings.npy"
    item_embeddings_path = index_dir / "item_embeddings.npy"
    meta_path = index_dir / "index_meta.json"

    missing = [
        path.name
        for path in [samples_path, packages_path, package_embeddings_path, item_embeddings_path, meta_path]
        if not path.exists()
    ]
    if missing:
        raise ValueError(f"索引目录缺少文件: {', '.join(missing)}")

    samples = pd.read_parquet(samples_path)
    project_packages = pd.read_parquet(packages_path)
    project_package_embeddings = np.load(package_embeddings_path)
    item_embeddings = np.load(item_embeddings_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if item_embeddings.ndim != 2 or project_package_embeddings.ndim != 2:
        raise ValueError("embedding 必须是二维矩阵")
    if len(samples) != item_embeddings.shape[0]:
        raise ValueError("样本数量与 item_embeddings 数量不一致")
    if len(project_packages) != project_package_embeddings.shape[0]:
        raise ValueError("工程包数量与 project_package_embeddings 数量不一致")
    if item_embeddings.shape[1] != project_package_embeddings.shape[1]:
        raise ValueError("item embedding 与 project_package embedding 维度不一致")
    if "sample_index" not in samples.columns:
        raise ValueError("samples.parquet 缺少 sample_index")
    if "project_package_id" not in samples.columns or "project_package_id" not in project_packages.columns:
        raise ValueError("索引缺少 project_package_id")
    return samples, project_packages, project_package_embeddings, item_embeddings, meta


def build_query_understanding_prompt(query: str) -> str:
    return f"""
你是维修工程造价查询理解器。请把用户自然语言需求解析成严格 JSON object。

只允许输出 JSON，不要 Markdown，不要解释。

字段固定为：
{{
  "semantic_query_text": "用于 embedding 检索的语义文本",
  "need_summary": "一句话概括用户需求",
  "known_constraints": {{
    "部位": "",
    "材料": "",
    "用户明确数量": "",
    "地点": "",
    "时间范围": ""
  }},
  "likely_catalog": {{
    "一级分类": "",
    "二级分类": "",
    "维修状态": "",
    "标准对象": ""
  }},
  "calculation_notes": "对工程量或估价类比有帮助的备注"
}}

要求：
- semantic_query_text 用于同时召回历史工程包和历史清单行。
- 保留用户明确表达的数量、材料、部位、地点、时间，不要编造。
- likely_catalog 可以为空，不确定就留空。
- 不要输出建议清单，不要计算价格。

用户需求：{query}
""".strip()


def fallback_query_understanding(query: str, note: str) -> QueryUnderstanding:
    return QueryUnderstanding(
        raw_query=query,
        semantic_query_text=query,
        need_summary="",
        known_constraints={},
        likely_catalog={},
        calculation_notes="",
        parse_notes=[note],
        llm_success=False,
    )


def understand_query(query: str) -> QueryUnderstanding:
    try:
        result = request_llm_json(
            build_query_understanding_prompt(query),
            max_tokens=768,
            system_prompt="你只输出一个 JSON object，不输出解释、Markdown 或思考过程。",
        )
    except (LLMServiceError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        return fallback_query_understanding(query, f"LLM 查询理解失败，已回退为原始 query 检索: {exc}")

    if not isinstance(result, dict):
        return fallback_query_understanding(query, "LLM 查询理解输出不是 JSON object，已回退为原始 query 检索")

    semantic_query_text = cell_text(result.get("semantic_query_text")) or query
    return QueryUnderstanding(
        raw_query=query,
        semantic_query_text=semantic_query_text,
        need_summary=cell_text(result.get("need_summary")),
        known_constraints=parse_jsonish_dict(result.get("known_constraints")),
        likely_catalog=parse_jsonish_dict(result.get("likely_catalog")),
        calculation_notes=cell_text(result.get("calculation_notes")),
        parse_notes=[],
        llm_success=True,
    )


def top_score_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 0 or scores.size == 0:
        return np.array([], dtype=int)
    count = min(int(top_k), int(scores.size))
    if count == scores.size:
        return np.argsort(-scores)
    candidate = np.argpartition(-scores, count - 1)[:count]
    return candidate[np.argsort(-scores[candidate])]


def score_project_packages(
    project_packages: pd.DataFrame,
    project_package_embeddings: np.ndarray,
    query_embedding: np.ndarray,
    top_packages: int,
) -> pd.DataFrame:
    scores = project_package_embeddings @ query_embedding
    indices = top_score_indices(scores, top_packages)
    rows = project_packages.iloc[indices].copy()
    rows.insert(0, "package_score", scores[indices].astype(float))
    rows.insert(0, "rank", range(1, len(rows) + 1))
    return rows


def matched_project_packages_for_output(matched: pd.DataFrame) -> pd.DataFrame:
    output = matched.copy()
    rename_map = {
        "catalog_summary": "分类摘要",
        "item_summary": "包含清单摘要",
    }
    output = output.rename(columns=rename_map)
    for column in MATCHED_PROJECT_PACKAGE_COLUMNS:
        if column not in output.columns:
            output[column] = ""
    return output[MATCHED_PROJECT_PACKAGE_COLUMNS].fillna("")


def score_direct_items(
    samples: pd.DataFrame,
    item_scores: np.ndarray,
    top_items: int,
) -> pd.DataFrame:
    indices = top_score_indices(item_scores, top_items)
    rows = samples.iloc[indices].copy()
    rows["item_score"] = item_scores[indices].astype(float)
    return rows


def value_set(values: dict[str, Any]) -> set[str]:
    return {cell_text(value) for value in values.values() if cell_text(value)}


def catalog_score(row: pd.Series, likely_catalog: dict[str, Any]) -> float:
    expected = {key: cell_text(likely_catalog.get(key)) for key in ["一级分类", "二级分类", "维修状态", "标准对象"]}
    expected = {key: value for key, value in expected.items() if value}
    if not expected:
        return 0.5

    matched = [key for key, value in expected.items() if cell_text(row.get(key)) == value]
    if len(matched) == len(expected):
        return 1.0
    if "二级分类" in matched:
        return 0.8
    if "一级分类" in matched:
        return 0.6
    return 0.2


def unit_score(row: pd.Series, known_constraints: dict[str, Any]) -> float:
    constraint_text = " ".join(value_set(known_constraints))
    if not constraint_text:
        return 0.5
    unit = cell_text(row.get("unit_normalized")) or cell_text(row.get("unit"))
    raw_unit = cell_text(row.get("unit"))
    if unit and unit in constraint_text:
        return 1.0
    if raw_unit and raw_unit in constraint_text:
        return 1.0
    return 0.5


def matched_package_maps(matched_project_packages: pd.DataFrame) -> tuple[dict[str, float], dict[str, int]]:
    score_map: dict[str, float] = {}
    rank_map: dict[str, int] = {}
    for _index, row in matched_project_packages.iterrows():
        package_id = cell_text(row.get("project_package_id"))
        if not package_id:
            continue
        score_map[package_id] = float(row.get("package_score") or 0.0)
        rank_map[package_id] = int(row.get("rank") or 0)
    return score_map, rank_map


def cooccur_scores(samples: pd.DataFrame, matched_package_ids: list[str]) -> dict[str, float]:
    if not matched_package_ids:
        return {}
    matched = samples[samples["project_package_id"].astype(str).isin(matched_package_ids)]
    if matched.empty or "family_signature" not in matched.columns:
        return {}
    counts = matched.groupby("family_signature", dropna=False)["project_package_id"].nunique()
    denominator = max(len(matched_package_ids), 1)
    return {cell_text(signature): float(count) / denominator for signature, count in counts.items()}


def candidate_pool(
    samples: pd.DataFrame,
    matched_project_packages: pd.DataFrame,
    direct_item_hits: pd.DataFrame,
    item_scores: np.ndarray,
    understanding: QueryUnderstanding,
) -> pd.DataFrame:
    package_score_map, package_rank_map = matched_package_maps(matched_project_packages)
    matched_package_ids = list(package_score_map.keys())
    direct_indices = {
        int(index)
        for index in pd.to_numeric(direct_item_hits.get("sample_index", pd.Series(dtype=int)), errors="coerce").dropna()
    }

    package_rows = samples[samples["project_package_id"].astype(str).isin(matched_package_ids)].copy()
    candidate_indices = set(pd.to_numeric(package_rows["sample_index"], errors="coerce").dropna().astype(int).tolist())
    candidate_indices.update(direct_indices)
    if not candidate_indices:
        return samples.head(0).copy()

    rows = samples[pd.to_numeric(samples["sample_index"], errors="coerce").astype("Int64").isin(candidate_indices)].copy()
    rows["sample_index"] = pd.to_numeric(rows["sample_index"], errors="raise").astype(int)
    if rows["sample_index"].min() < 0 or rows["sample_index"].max() >= len(item_scores):
        raise ValueError("sample_index 超出 item_embeddings 范围")

    rows["package_score"] = rows["project_package_id"].map(package_score_map).fillna(0.0).astype(float)
    rows["package_rank"] = rows["project_package_id"].map(package_rank_map)
    rows["item_score"] = rows["sample_index"].map(lambda sample_index: float(item_scores[int(sample_index)]))
    family_cooccur = cooccur_scores(samples, matched_package_ids)
    rows["cooccur_score"] = rows["family_signature"].map(lambda signature: family_cooccur.get(cell_text(signature), 0.0))
    rows["catalog_score"] = rows.apply(lambda row: catalog_score(row, understanding.likely_catalog), axis=1)
    rows["unit_score"] = rows.apply(lambda row: unit_score(row, understanding.known_constraints), axis=1)
    rows["direct_hit"] = rows["sample_index"].isin(direct_indices)
    rows["final_score"] = (
        0.30 * rows["package_score"]
        + 0.25 * rows["item_score"]
        + 0.25 * rows["cooccur_score"]
        + 0.15 * rows["catalog_score"]
        + 0.05 * rows["unit_score"]
    )
    rows["evidence_ref"] = rows["sample_index"].map(lambda value: f"E{int(value)}")
    sort_columns = ["final_score", "item_score", "package_score", "cooccur_score"]
    return rows.sort_values(sort_columns, ascending=[False, False, False, False]).reset_index(drop=True)


def numeric_values(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").dropna()


def min_median_max(frame: pd.DataFrame, column: str) -> tuple[float | None, float | None, float | None]:
    values = numeric_values(frame, column)
    if values.empty:
        return None, None, None
    return float(values.min()), float(values.median()), float(values.max())


def first_value(group: pd.DataFrame, column: str) -> str:
    if column not in group.columns:
        return ""
    for value in group[column].tolist():
        text = cell_text(value)
        if text:
            return text
    return ""


def ordered_refs(values: pd.Series) -> str:
    seen: set[str] = set()
    refs: list[str] = []
    for value in values.tolist():
        text = cell_text(value)
        if text and text not in seen:
            refs.append(text)
            seen.add(text)
    return ", ".join(refs)


def build_candidate_item_stats(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=CANDIDATE_ITEM_STATS_COLUMNS)

    rows: list[dict[str, Any]] = []
    for fine_signature, group in candidates.groupby("fine_signature", sort=False, dropna=False):
        quantity_min, quantity_median, quantity_max = min_median_max(group, "quantity")
        unit_price_min, unit_price_median, unit_price_max = min_median_max(group, "unit_price")
        total_price_min, total_price_median, total_price_max = min_median_max(group, "total_price")
        rows.append(
            {
                "fine_signature": cell_text(fine_signature),
                "family_signature": first_value(group, "family_signature"),
                "cost_item_name": first_value(group, "cost_item_name"),
                "project_description": first_value(group, "project_description"),
                "unit": first_value(group, "unit_normalized") or first_value(group, "unit"),
                "历史样本数": int(len(group)),
                "来源工程包数": int(group["project_package_id"].nunique()) if "project_package_id" in group.columns else 0,
                "历史工程量最小值": quantity_min,
                "历史工程量中位数": quantity_median,
                "历史工程量最大值": quantity_max,
                "历史综合单价最小值": unit_price_min,
                "历史综合单价中位数": unit_price_median,
                "历史综合单价最大值": unit_price_max,
                "历史合价最小值": total_price_min,
                "历史合价中位数": total_price_median,
                "历史合价最大值": total_price_max,
                "package_score最大值": float(numeric_values(group, "package_score").max()) if not numeric_values(group, "package_score").empty else 0.0,
                "item_score最大值": float(numeric_values(group, "item_score").max()) if not numeric_values(group, "item_score").empty else 0.0,
                "cooccur_score": float(numeric_values(group, "cooccur_score").max()) if not numeric_values(group, "cooccur_score").empty else 0.0,
                "catalog_score": float(numeric_values(group, "catalog_score").max()) if not numeric_values(group, "catalog_score").empty else 0.0,
                "final_score": float(numeric_values(group, "final_score").max()) if not numeric_values(group, "final_score").empty else 0.0,
                "是否被LLM采用": "否",
                "evidence_refs": ordered_refs(group["evidence_ref"]),
            }
        )

    output = pd.DataFrame(rows)
    output = output.sort_values(["final_score", "item_score最大值", "历史样本数"], ascending=[False, False, False])
    for column in CANDIDATE_ITEM_STATS_COLUMNS:
        if column not in output.columns:
            output[column] = None
    return output[CANDIDATE_ITEM_STATS_COLUMNS].reset_index(drop=True)


def build_evidence_items(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=EVIDENCE_ITEM_COLUMNS)
    output = pd.DataFrame(
        {
            "evidence_ref": candidates.get("evidence_ref", ""),
            "来源工程名称": candidates.get("工程名称", ""),
            "project_package_id": candidates.get("project_package_id", ""),
            "consultation_time": candidates.get("consultation_time", ""),
            "location": candidates.get("location", ""),
            "catalog_id": candidates.get("catalog_id", ""),
            "一级分类": candidates.get("一级分类", ""),
            "二级分类": candidates.get("二级分类", ""),
            "维修状态": candidates.get("维修状态", ""),
            "标准对象": candidates.get("标准对象", ""),
            "cost_item_name": candidates.get("cost_item_name", ""),
            "project_description": candidates.get("project_description", ""),
            "unit": candidates.get("unit_normalized", candidates.get("unit", "")),
            "quantity": candidates.get("quantity", ""),
            "unit_price": candidates.get("unit_price", ""),
            "total_price": candidates.get("total_price", ""),
            "labor_unit_price": candidates.get("labor_unit_price", ""),
            "machinery_unit_price": candidates.get("machinery_unit_price", ""),
            "package_rank": candidates.get("package_rank", ""),
            "package_score": candidates.get("package_score", ""),
            "item_score": candidates.get("item_score", ""),
            "cooccur_score": candidates.get("cooccur_score", ""),
            "catalog_score": candidates.get("catalog_score", ""),
            "final_score": candidates.get("final_score", ""),
        }
    )
    for column in EVIDENCE_ITEM_COLUMNS:
        if column not in output.columns:
            output[column] = ""
    return output[EVIDENCE_ITEM_COLUMNS].fillna("")


def records_for_llm(frame: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    limited = frame.head(limit).replace({np.nan: None})
    return json.loads(limited.to_json(orient="records", force_ascii=False))


def build_suggested_bill_prompt(
    understanding: QueryUnderstanding,
    matched_project_packages: pd.DataFrame,
    candidate_item_stats: pd.DataFrame,
    evidence_items: pd.DataFrame,
) -> str:
    payload = {
        "用户原始需求": understanding.raw_query,
        "query_understanding": {
            "semantic_query_text": understanding.semantic_query_text,
            "need_summary": understanding.need_summary,
            "known_constraints": understanding.known_constraints,
            "likely_catalog": understanding.likely_catalog,
            "calculation_notes": understanding.calculation_notes,
        },
        "matched_project_packages": records_for_llm(matched_project_packages, 20),
        "candidate_item_stats": records_for_llm(candidate_item_stats, 80),
        "evidence_items": records_for_llm(evidence_items, 200),
    }
    return f"""
你是维修工程造价建议清单生成器。请基于历史工程包、候选项统计和历史明细证据，生成本次 suggested_bill。

只允许输出 JSON object，不要 Markdown，不要解释。JSON 格式：
{{
  "suggested_bill": [
    {{
      "seq": 1,
      "recommend_type": "直接匹配项",
      "cost_item_name": "",
      "project_description": "",
      "unit": "",
      "suggested_quantity": null,
      "quantity_basis": "",
      "unit_price_low": null,
      "unit_price_mid": null,
      "unit_price_high": null,
      "estimated_amount_low": null,
      "estimated_amount_mid": null,
      "estimated_amount_high": null,
      "adopt_reason": "",
      "uncertainty_note": "",
      "evidence_refs": []
    }}
  ]
}}

要求：
- 从 candidate_item_stats 中选择本次建议清单，不要输出明显无关项。
- 工程量、单位和估价由你结合用户需求与历史工程包/明细证据类比判断。
- 不确定时也尽量给参考估计，并在 uncertainty_note 标注“需人工确认”及原因。
- recommend_type 只能是：直接匹配项、工程包共现项、补充候选项。
- evidence_refs 必须引用 evidence_items 中的 evidence_ref。

输入数据：
{json_text(payload)}
""".strip()


def list_from_refs(value: Any) -> list[str]:
    if isinstance(value, list):
        return [cell_text(item) for item in value if cell_text(item)]
    text = cell_text(value)
    if not text:
        return []
    return [part.strip() for part in text.replace("；", ",").split(",") if part.strip()]


def bill_value(item: dict[str, Any], english_key: str, chinese_key: str = "") -> Any:
    if english_key in item:
        return item.get(english_key)
    if chinese_key and chinese_key in item:
        return item.get(chinese_key)
    return ""


def suggested_bill_from_llm_result(result: dict[str, Any]) -> pd.DataFrame:
    bill = result.get("suggested_bill")
    if not isinstance(bill, list):
        raise ValueError("LLM 输出缺少 suggested_bill list")

    rows: list[dict[str, Any]] = []
    for index, item in enumerate(bill, start=1):
        if not isinstance(item, dict):
            continue
        refs = list_from_refs(bill_value(item, "evidence_refs", "来源证据"))
        rows.append(
            {
                "序号": bill_value(item, "seq", "序号") or index,
                "推荐类型": bill_value(item, "recommend_type", "推荐类型"),
                "清单项名称": bill_value(item, "cost_item_name", "清单项名称"),
                "项目特征/施工工艺": bill_value(item, "project_description", "项目特征/施工工艺"),
                "单位": bill_value(item, "unit", "单位"),
                "建议工程量": bill_value(item, "suggested_quantity", "建议工程量"),
                "工程量依据": bill_value(item, "quantity_basis", "工程量依据"),
                "综合单价低值": bill_value(item, "unit_price_low", "综合单价低值"),
                "综合单价中值": bill_value(item, "unit_price_mid", "综合单价中值"),
                "综合单价高值": bill_value(item, "unit_price_high", "综合单价高值"),
                "估算金额低值": bill_value(item, "estimated_amount_low", "估算金额低值"),
                "估算金额中值": bill_value(item, "estimated_amount_mid", "估算金额中值"),
                "估算金额高值": bill_value(item, "estimated_amount_high", "估算金额高值"),
                "采用理由": bill_value(item, "adopt_reason", "采用理由"),
                "不确定性说明": bill_value(item, "uncertainty_note", "不确定性说明"),
                "来源证据": ", ".join(refs),
            }
        )
    if not rows:
        raise ValueError("LLM suggested_bill 为空")
    return pd.DataFrame(rows, columns=SUGGESTED_BILL_COLUMNS)


def fallback_suggested_bill(candidate_item_stats: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, row in candidate_item_stats.head(limit).iterrows():
        rows.append(
            {
                "序号": len(rows) + 1,
                "推荐类型": "fallback_candidate",
                "清单项名称": row.get("cost_item_name", ""),
                "项目特征/施工工艺": row.get("project_description", ""),
                "单位": row.get("unit", ""),
                "建议工程量": "",
                "工程量依据": "LLM 生成失败，未生成建议工程量",
                "综合单价低值": row.get("历史综合单价最小值"),
                "综合单价中值": row.get("历史综合单价中位数"),
                "综合单价高值": row.get("历史综合单价最大值"),
                "估算金额低值": row.get("历史合价最小值"),
                "估算金额中值": row.get("历史合价中位数"),
                "估算金额高值": row.get("历史合价最大值"),
                "采用理由": "按 final_score 排序输出候选项，不代表完整建议清单",
                "不确定性说明": "需人工确认；LLM suggested_bill 生成失败",
                "来源证据": row.get("evidence_refs", ""),
            }
        )
    return pd.DataFrame(rows, columns=SUGGESTED_BILL_COLUMNS)


def mark_adopted_candidates(candidate_item_stats: pd.DataFrame, suggested_bill: pd.DataFrame) -> pd.DataFrame:
    output = candidate_item_stats.copy()
    if output.empty or suggested_bill.empty:
        return output
    adopted_refs: set[str] = set()
    for value in suggested_bill["来源证据"].tolist():
        adopted_refs.update(list_from_refs(value))

    if not adopted_refs:
        return output

    def adopted(row: pd.Series) -> str:
        refs = set(list_from_refs(row.get("evidence_refs")))
        return "是" if refs & adopted_refs else "否"

    output["是否被LLM采用"] = output.apply(adopted, axis=1)
    return output


def generate_suggested_bill(
    understanding: QueryUnderstanding,
    matched_project_packages: pd.DataFrame,
    candidate_item_stats: pd.DataFrame,
    evidence_items: pd.DataFrame,
) -> tuple[pd.DataFrame, bool, bool, str, str]:
    prompt = build_suggested_bill_prompt(understanding, matched_project_packages, candidate_item_stats, evidence_items)
    try:
        result = request_llm_json(
            prompt,
            max_tokens=4096,
            system_prompt="你只输出一个 JSON object，不输出解释、Markdown 或思考过程。",
        )
        suggested_bill = suggested_bill_from_llm_result(result)
        return suggested_bill, True, False, "", prompt
    except (LLMServiceError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        return fallback_suggested_bill(candidate_item_stats), False, True, str(exc), prompt


def display_frame(frame: pd.DataFrame, display: bool) -> pd.DataFrame:
    if not display:
        return frame
    output = frame.copy()
    for column in output.columns:
        if pd.api.types.is_numeric_dtype(output[column]):
            output[column] = output[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.2f}")
    return output


def build_parse_info(
    understanding: QueryUnderstanding,
    top_packages: int,
    top_items: int,
    meta: dict[str, Any],
    sample_count: int,
    package_count: int,
    suggested_success: bool,
    fallback: bool,
    output_path: Path | None,
    started_at: datetime,
    index_dir: Path,
    llm_error: str,
    include_debug_text: bool,
    suggested_prompt: str,
) -> pd.DataFrame:
    rows = [
        ("原始用户需求", understanding.raw_query),
        ("semantic_query_text", understanding.semantic_query_text),
        ("need_summary", understanding.need_summary),
        ("known_constraints", json_text(understanding.known_constraints)),
        ("likely_catalog", json_text(understanding.likely_catalog)),
        ("calculation_notes", understanding.calculation_notes),
        ("top_packages", top_packages),
        ("top_items", top_items),
        ("embedding_model", meta.get("model", "")),
        ("sample_count", sample_count),
        ("package_count", package_count),
        ("LLM query understanding 是否成功", "是" if understanding.llm_success else "否"),
        ("LLM suggested_bill 是否成功", "是" if suggested_success else "否"),
        ("是否 fallback", "是" if fallback else "否"),
        ("LLM error", llm_error),
        ("output_path", str(output_path or "")),
        ("运行时间", f"{(datetime.now() - started_at).total_seconds():.2f}s"),
        ("index_dir", str(index_dir)),
        ("主要文件路径", json_text((meta.get("files") or {}))),
        ("parse_notes", "；".join(understanding.parse_notes)),
    ]
    if include_debug_text:
        rows.append(("suggested_bill_prompt_preview", suggested_prompt[:3000]))
    return pd.DataFrame(rows, columns=["字段", "值"])


def write_query_result_workbook(
    output_path: Path,
    result: QueryResult,
    display: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        display_frame(result.suggested_bill, display).to_excel(writer, sheet_name="suggested_bill", index=False)
        display_frame(result.matched_project_packages, display).to_excel(
            writer,
            sheet_name="matched_project_packages",
            index=False,
        )
        display_frame(result.candidate_item_stats, display).to_excel(
            writer,
            sheet_name="candidate_item_stats",
            index=False,
        )
        display_frame(result.evidence_items, display).to_excel(writer, sheet_name="evidence_items", index=False)
        result.parse_info.to_excel(writer, sheet_name="parse_info", index=False)
    apply_workbook_style(output_path)


def apply_workbook_style(path: Path) -> None:
    try:
        import openpyxl
        from openpyxl.styles import Alignment, Font
    except ImportError:
        return

    workbook = openpyxl.load_workbook(path)
    for worksheet in workbook.worksheets:
        worksheet.freeze_panes = None
        for cell in worksheet[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(wrap_text=False, vertical="top")
        for row in worksheet.iter_rows():
            for cell in row:
                cell.alignment = Alignment(wrap_text=False, vertical="top")
    workbook.save(path)
    workbook.close()


def run_query(
    index_dir: Path,
    raw_text: str,
    top_packages: int,
    top_items: int,
    output: Path | None,
    include_debug_text: bool = False,
    display: bool = False,
) -> QueryResult:
    started_at = datetime.now()
    samples, project_packages, project_package_embeddings, item_embeddings, meta = load_index(index_dir)
    understanding = understand_query(raw_text)

    model = load_embedding_model(str(meta.get("model") or "BAAI/bge-m3"))
    try:
        query_embedding = encode_query(model, understanding.semantic_query_text)
    finally:
        release_embedding_model(model)
        gc.collect()

    if query_embedding.shape[0] != project_package_embeddings.shape[1]:
        raise ValueError("query embedding 维度与索引 embedding 维度不一致")

    matched_raw = score_project_packages(project_packages, project_package_embeddings, query_embedding, top_packages)
    item_scores = item_embeddings @ query_embedding
    direct_item_hits = score_direct_items(samples, item_scores, top_items)
    candidates = candidate_pool(samples, matched_raw, direct_item_hits, item_scores, understanding)
    candidate_item_stats = build_candidate_item_stats(candidates)
    evidence_items = build_evidence_items(candidates)
    matched_project_packages = matched_project_packages_for_output(matched_raw)

    suggested_bill, suggested_success, fallback, llm_error, suggested_prompt = generate_suggested_bill(
        understanding,
        matched_project_packages,
        candidate_item_stats,
        evidence_items,
    )
    if suggested_success:
        candidate_item_stats = mark_adopted_candidates(candidate_item_stats, suggested_bill)
    parse_info = build_parse_info(
        understanding=understanding,
        top_packages=top_packages,
        top_items=top_items,
        meta=meta,
        sample_count=len(samples),
        package_count=len(project_packages),
        suggested_success=suggested_success,
        fallback=fallback,
        output_path=output,
        started_at=started_at,
        index_dir=index_dir,
        llm_error=llm_error,
        include_debug_text=include_debug_text,
        suggested_prompt=suggested_prompt,
    )

    result = QueryResult(
        understanding=understanding,
        suggested_bill=suggested_bill,
        matched_project_packages=matched_project_packages,
        candidate_item_stats=candidate_item_stats,
        evidence_items=evidence_items,
        parse_info=parse_info,
    )
    if output:
        write_query_result_workbook(output, result, display=display)
    return result


def print_terminal_summary(result: QueryResult, output_path: Path | None) -> None:
    print(f"[DONE] semantic query: {result.understanding.semantic_query_text}")
    print(f"[DONE] matched project packages: {len(result.matched_project_packages)}")
    print(f"[DONE] candidate item stats: {len(result.candidate_item_stats)}")
    print(f"[DONE] evidence items: {len(result.evidence_items)}")
    print(f"[DONE] suggested bill rows: {len(result.suggested_bill)}")
    if result.understanding.parse_notes:
        print(f"parse notes: {'；'.join(result.understanding.parse_notes)}")
    if output_path:
        print(f"输出文件: {output_path}")


def main() -> int:
    args = parse_args()
    index_dir = Path(args.index_dir).expanduser().resolve()
    output_arg = Path(args.output) if args.output is not None else default_query_output_path()
    output_path = output_arg.expanduser().resolve()

    try:
        validate_output_path(output_path, args.overwrite)
        result = run_query(
            index_dir=index_dir,
            raw_text=args.text,
            top_packages=args.top_packages,
            top_items=args.top_items,
            output=output_path,
            include_debug_text=args.include_debug_text,
            display=args.display,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    print_terminal_summary(result, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
