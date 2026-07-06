#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import re
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

from classifier.llm_client import LLMServiceError, check_lmstudio_service, request_llm_json  # noqa: E402
from services.standard_classifier import classify_project_standard  # noqa: E402


MATCHED_PROJECT_PACKAGE_COLUMNS = [
    "rank",
    "package_score",
    "project_package_id",
    "工程名称",
    "project_name_text",
    "cost_item_names_summary",
    "consultation_time",
    "location",
    "cache_subject",
    "item_count",
]

CANDIDATE_ITEM_STATS_COLUMNS = [
    "candidate_id",
    "fine_signature",
    "family_signature",
    "cost_item_name",
    "project_description",
    "unit",
    "unit_normalized",
    "历史样本数",
    "来源工程包数",
    "历史工程量最低值",
    "历史工程量中位数",
    "历史工程量最高值",
    "历史综合单价最低值",
    "历史综合单价中位数",
    "历史综合单价最高值",
    "历史合价最低值",
    "历史合价中位数",
    "历史合价最高值",
    "历史人工单价最低值",
    "历史人工单价中位数",
    "历史人工单价最高值",
    "历史机械单价最低值",
    "历史机械单价中位数",
    "历史机械单价最高值",
    "package_score最大值",
    "item_score最大值",
    "cooccur_score",
    "catalog_score",
    "final_score",
    "是否被LLM采用",
    "source_refs",
]

CANDIDATE_FAMILY_COLUMNS = [
    "family_id",
    "family_signature",
    "representative_cost_item_name",
    "representative_project_description",
    "unit",
    "unit_normalized",
    "覆盖candidate_ids",
    "历史样本数",
    "来源工程包数",
    "历史工程量最低值",
    "历史工程量中位数",
    "历史工程量最高值",
    "历史综合单价最低值",
    "历史综合单价中位数",
    "历史综合单价最高值",
    "历史合价最低值",
    "历史合价中位数",
    "历史合价最高值",
    "历史人工单价最低值",
    "历史人工单价中位数",
    "历史人工单价最高值",
    "历史机械单价最低值",
    "历史机械单价中位数",
    "历史机械单价最高值",
    "package_score最大值",
    "item_score最大值",
    "cooccur_score",
    "catalog_score",
    "final_score",
    "source_refs",
]

EVIDENCE_ITEM_COLUMNS = [
    "source_ref",
    "project_key",
    "item_row_id",
    "stable_sample_id",
    "batch_id",
    "source_row_id",
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
    "unit_normalized",
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
    "family_id",
    "candidate_id",
    "推荐类型",
    "项目角色",
    "计量/估算口径",
    "清单项名称",
    "项目特征/施工工艺",
    "单位",
    "建议工程量",
    "工程量依据",
    "是否计入金额汇总",
    "综合单价最低值",
    "综合单价中位数",
    "综合单价最高值",
    "其中包含人工费单价最低值",
    "其中包含人工费单价中位数",
    "其中包含人工费单价最高值",
    "其中包含机械费单价最低值",
    "其中包含机械费单价中位数",
    "其中包含机械费单价最高值",
    "估算金额最低值",
    "估算金额中位数",
    "估算金额最高值",
    "估算人工费最低值",
    "估算人工费中位数",
    "估算人工费最高值",
    "估算机械费最低值",
    "估算机械费中位数",
    "估算机械费最高值",
    "金额计算说明",
    "采用理由",
    "不确定性说明",
    "来源样本",
]

ESTIMATE_SUMMARY_COLUMNS = [
    "字段",
    "值",
]

LLM_TRACE_COLUMNS = [
    "step",
    "purpose",
    "success",
    "error",
    "prompt_chars",
    "estimated_tokens",
    "max_tokens",
    "input_summary",
]


@dataclass(frozen=True)
class QueryRewrite:
    raw_query: str
    project_package_query_text: str
    item_query_text: str
    parsed_quantities: list[dict[str, Any]]
    materials_or_specs: list[str]
    repair_object: str
    uncertainties: list[str]
    notes: list[str]
    success: bool


@dataclass(frozen=True)
class QueryCatalog:
    catalog_id: str
    一级分类: str
    二级分类: str
    维修状态: str
    标准对象: str
    confidence: float | None
    raw_result: dict[str, Any]
    success: bool
    notes: list[str]


@dataclass(frozen=True)
class QueryResult:
    rewrite: QueryRewrite
    query_catalog: QueryCatalog
    estimate_summary: pd.DataFrame
    suggested_bill: pd.DataFrame
    matched_project_packages: pd.DataFrame
    candidate_families: pd.DataFrame
    candidate_item_stats: pd.DataFrame
    evidence_items: pd.DataFrame
    parse_info: pd.DataFrame
    llm_trace: pd.DataFrame


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
    parser.add_argument(
        "--llm-check-timeout",
        type=float,
        default=3.0,
        help="启动前检查 LLM 服务可用性的超时时间，默认 3 秒",
    )
    parser.add_argument(
        "--max-packages-per-cache-subject",
        type=int,
        default=1,
        help="同一 cache_subject 最多保留的相似历史工程包数量，默认 1；设为 0 表示不限制",
    )
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


def truncate_text(value: Any, limit: int) -> str:
    text = cell_text(value)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip()


def normalize_dedupe_text(value: Any) -> str:
    return re.sub(r"\s+", " ", cell_text(value).lower()).strip()


def normalize_source_row_id(value: Any) -> str:
    text = cell_text(value)
    if not text:
        return ""
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return str(int(number))
    return text


def append_warning(warnings: list[str] | None, code: str) -> None:
    if warnings is not None and code not in warnings:
        warnings.append(code)


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value in (None, ""):
        return []
    return [value]


def estimated_tokens(text: str) -> int:
    return max(1, int(len(text) / 2))


def trace_row(
    step: str,
    purpose: str,
    success: bool,
    error: str = "",
    prompt: str = "",
    max_tokens: int | str = "",
    input_summary: str = "",
) -> dict[str, Any]:
    return {
        "step": step,
        "purpose": purpose,
        "success": "是" if success else "否",
        "error": error,
        "prompt_chars": len(prompt),
        "estimated_tokens": estimated_tokens(prompt) if prompt else "",
        "max_tokens": max_tokens,
        "input_summary": input_summary,
    }


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


def build_query_rewrite_prompt(query: str) -> str:
    return f"""
你是维修工程需求解析和 embedding query rewrite 助手。请把用户原始需求解析为严格 JSON object。

只能输出 JSON object，不要 Markdown，不要解释，不要建议清单，不要计算价格。

输出格式：
{{
  "project_package_query_text": "",
  "item_query_text": "",
  "parsed_quantities": [
    {{
      "raw_text": "500平",
      "value": 500,
      "unit": "m²",
      "meaning": "维修面积",
      "confidence": 0.95
    }}
  ],
  "materials_or_specs": ["材料或规格"],
  "repair_object": "维修对象",
  "uncertainties": ["现场条件未知"]
}}

当前 embedding 结构：
1. project_package_text 由“工程名称、project_name_text、cost_item_name 去重列表”组成。
   project_package_query_text 用于匹配相似历史工程包，应描述用户明确表达或直接相关的维修工程场景，保持短检索 query，不要预设建议清单、前置项、措施项或替代工艺。
2. item_retrieval_text 由“cost_item_name、project_description、unit_normalized”组成。
   item_query_text 用于匹配相似清单行，应贴近用户明确表达的维修对象、材料规格和做法，不要扩展未明确发生的清单项。
3. item_query_text 必须非空。如果用户问得很粗，也输出宽泛 item query，不要留空。
4. parsed_quantities 只解析用户原文明确或强烈暗示的工程量、面积、长度、数量等，不要为了估价编造工程量。
5. repair_object、materials_or_specs、uncertainties 用于后续估价 LLM 判断口径，不要输出价格，不要预设 suggested_bill、前置项、措施项或替代工艺。

示例：
用户：屋面漏水，想做3mm SBS防水，面积大概500平
输出：{{"project_package_query_text":"屋面漏水维修工程 屋面防水维修 3mm SBS防水","item_query_text":"屋面防水 3.0mm SBS防水卷材","parsed_quantities":[{{"raw_text":"500平","value":500,"unit":"m²","meaning":"屋面防水面积","confidence":0.95}}],"materials_or_specs":["3mm SBS"],"repair_object":"屋面防水层","uncertainties":["现场做法未知","基层状况未知"]}}

用户：屋面漏水帮我估价
输出：{{"project_package_query_text":"屋面漏水维修工程 屋面防水维修","item_query_text":"屋面防水 防水层维修","parsed_quantities":[],"materials_or_specs":[],"repair_object":"屋面防水层","uncertainties":["维修面积未知","现场做法未知","基层状况未知"]}}

用户：地下室渗水维修
输出：{{"project_package_query_text":"地下室渗水维修工程 地下室防水维修 防水层维修 墙面修复 地面修复","item_query_text":"地下室防水 渗水维修 防水层维修","parsed_quantities":[],"materials_or_specs":[],"repair_object":"地下室防水层","uncertainties":["渗水范围未知","基层状况未知"]}}

用户需求：{query}
""".strip()


def fallback_query_rewrite(query: str, note: str) -> QueryRewrite:
    return QueryRewrite(
        raw_query=query,
        project_package_query_text=query,
        item_query_text=query,
        parsed_quantities=[],
        materials_or_specs=[],
        repair_object="",
        uncertainties=[],
        notes=[note],
        success=False,
    )


def query_rewrite_for_embedding(query: str) -> tuple[QueryRewrite, dict[str, Any]]:
    prompt = build_query_rewrite_prompt(query)
    max_tokens = 512
    try:
        result = request_llm_json(
            prompt,
            max_tokens=max_tokens,
            system_prompt="你只输出一个 JSON object，不输出解释、Markdown 或思考过程。",
        )
    except (LLMServiceError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        rewrite = fallback_query_rewrite(query, f"LLM query rewrite 失败，已回退为原始 query: {exc}")
        return rewrite, trace_row(
            "query_rewrite_for_embedding",
            "生成 project_package_query_text 和 item_query_text",
            False,
            error=str(exc),
            prompt=prompt,
            max_tokens=max_tokens,
            input_summary=query,
        )

    notes: list[str] = []
    package_text = cell_text(result.get("project_package_query_text")) if isinstance(result, dict) else ""
    item_text = cell_text(result.get("item_query_text")) if isinstance(result, dict) else ""
    if not package_text:
        package_text = query
        notes.append("project_package_query_text 为空，已回退为原始 query")
    if not item_text:
        item_text = package_text or query
        notes.append("item_query_text 为空，已回退为 project_package_query_text 或原始 query")
    rewrite = QueryRewrite(
        raw_query=query,
        project_package_query_text=package_text,
        item_query_text=item_text,
        parsed_quantities=[item for item in as_list(result.get("parsed_quantities")) if isinstance(item, dict)],
        materials_or_specs=[cell_text(item) for item in as_list(result.get("materials_or_specs")) if cell_text(item)],
        repair_object=cell_text(result.get("repair_object")),
        uncertainties=[cell_text(item) for item in as_list(result.get("uncertainties")) if cell_text(item)],
        notes=notes,
        success=True,
    )
    return rewrite, trace_row(
        "query_rewrite_for_embedding",
        "生成 project_package_query_text 和 item_query_text",
        True,
        prompt=prompt,
        max_tokens=max_tokens,
        input_summary=query,
    )


def empty_query_catalog(raw_query: str, note: str, raw_result: dict[str, Any] | None = None) -> QueryCatalog:
    return QueryCatalog(
        catalog_id="",
        一级分类="",
        二级分类="",
        维修状态="",
        标准对象="",
        confidence=None,
        raw_result=raw_result or {},
        success=False,
        notes=[note],
    )


def classify_query_catalog(
    raw_query: str,
    project_package_query_text: str,
    item_query_text: str,
) -> tuple[QueryCatalog, dict[str, Any]]:
    classify_subject = project_package_query_text or raw_query
    item_summary = [item_query_text] if item_query_text else None
    input_summary = json_text(
        {
            "consultation_project_name": raw_query,
            "classify_subject": classify_subject,
            "item_summary": item_summary or [],
        }
    )
    try:
        result = classify_project_standard(
            classify_subject,
            consultation_project_name=raw_query,
            item_summary=item_summary,
        )
    except Exception as exc:  # noqa: BLE001
        catalog = empty_query_catalog(raw_query, f"标准目录分类异常，catalog_score 使用 0.5: {exc}")
        return catalog, trace_row(
            "query_catalog_classification",
            "复用标准目录分类器选择主目录",
            False,
            error=str(exc),
            prompt=input_summary,
            max_tokens="standard_classifier",
            input_summary=input_summary,
        )

    success = cell_text(result.get("catalog_id")) and cell_text(result.get("catalog_id")) != "OUT_OF_SCOPE"
    notes: list[str] = []
    if not success:
        notes.append("标准目录分类未命中有效主目录，catalog_score 使用 0.5")
    catalog = QueryCatalog(
        catalog_id=cell_text(result.get("catalog_id")),
        一级分类=cell_text(result.get("category") or result.get("一级分类")),
        二级分类=cell_text(result.get("item") or result.get("二级分类")),
        维修状态=cell_text(result.get("repair_status") or result.get("维修状态")),
        标准对象=cell_text(result.get("standard_group") or result.get("标准对象")),
        confidence=None,
        raw_result=result,
        success=bool(success),
        notes=notes,
    )
    return catalog, trace_row(
        "query_catalog_classification",
        "复用标准目录分类器选择主目录",
        bool(success),
        error="；".join(notes),
        prompt=input_summary,
        max_tokens="standard_classifier",
        input_summary=input_summary,
    )


def top_score_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    if top_k <= 0 or scores.size == 0:
        return np.array([], dtype=int)
    count = min(int(top_k), int(scores.size))
    if count == scores.size:
        return np.argsort(-scores)
    candidate = np.argpartition(-scores, count - 1)[:count]
    return candidate[np.argsort(-scores[candidate])]


def package_dedupe_key(row: pd.Series) -> str:
    return " | ".join(
        [
            normalize_dedupe_text(row.get("工程名称")),
            normalize_dedupe_text(row.get("project_name_text")),
            normalize_dedupe_text(row.get("cost_item_names_summary")),
        ]
    )


def score_project_packages(
    project_packages: pd.DataFrame,
    project_package_embeddings: np.ndarray,
    package_query_embedding: np.ndarray,
    top_packages: int,
    max_packages_per_cache_subject: int = 1,
) -> pd.DataFrame:
    scores = project_package_embeddings @ package_query_embedding
    indices = top_score_indices(scores, max(top_packages * 20, top_packages))
    rows = project_packages.iloc[indices].copy()
    rows["package_score"] = scores[indices].astype(float)
    rows["package_dedupe_key"] = rows.apply(package_dedupe_key, axis=1)
    rows = rows.sort_values("package_score", ascending=False)
    rows = rows.drop_duplicates("package_dedupe_key", keep="first").copy()
    if "cache_subject" in rows.columns and max_packages_per_cache_subject > 0:
        rows["_cache_subject_key"] = rows["cache_subject"].map(normalize_dedupe_text)
        empty_mask = rows["_cache_subject_key"].eq("")
        rows.loc[empty_mask, "_cache_subject_key"] = rows.loc[empty_mask, "package_dedupe_key"]

        rows["_cache_subject_rank"] = rows.groupby("_cache_subject_key").cumcount()
        rows = rows[rows["_cache_subject_rank"] < max_packages_per_cache_subject]
        rows = rows.drop(columns=["_cache_subject_key", "_cache_subject_rank"])
    rows = rows.head(top_packages).copy()
    rows = rows.drop(columns=["package_dedupe_key"])
    rows.insert(0, "rank", range(1, len(rows) + 1))
    return rows


def matched_project_packages_for_output(matched: pd.DataFrame) -> pd.DataFrame:
    output = matched.copy()
    for column in MATCHED_PROJECT_PACKAGE_COLUMNS:
        if column not in output.columns:
            output[column] = ""
    return output[MATCHED_PROJECT_PACKAGE_COLUMNS].fillna("")


def score_direct_items(samples: pd.DataFrame, item_scores: np.ndarray, top_items: int) -> pd.DataFrame:
    indices = top_score_indices(item_scores, top_items)
    rows = samples.iloc[indices].copy()
    rows["item_score"] = item_scores[indices].astype(float)
    return rows


def catalog_score(row: pd.Series, query_catalog: QueryCatalog) -> float:
    if not query_catalog.success:
        return 0.5
    item_catalog_id = cell_text(row.get("catalog_id"))
    if item_catalog_id and item_catalog_id == query_catalog.catalog_id:
        return 1.0

    scores: list[float] = []
    comparisons = [
        ("二级分类", query_catalog.二级分类, 0.8),
        ("一级分类", query_catalog.一级分类, 0.6),
        ("维修状态", query_catalog.维修状态, 0.55),
        ("标准对象", query_catalog.标准对象, 0.55),
    ]
    for column, expected, score in comparisons:
        if not expected:
            continue
        actual = cell_text(row.get(column))
        if not actual:
            scores.append(0.5)
        elif actual == expected:
            scores.append(score)
    if not scores:
        return 0.5
    return max(scores)


def unit_score(_row: pd.Series) -> float:
    return 0.5


def source_identity_for_row(row: pd.Series, warnings: list[str] | None = None) -> tuple[str, str, str]:
    project_key = cell_text(row.get("project_key"))
    if not project_key:
        batch_id = cell_text(row.get("batch_id"))
        source_row_id = normalize_source_row_id(row.get("source_row_id"))
        if batch_id and source_row_id:
            project_key = f"{batch_id}::{source_row_id}"
            append_warning(warnings, "source_ref_recovered_from_batch_source_row")

    item_row_id = cell_text(row.get("item_row_id"))
    if not item_row_id:
        source_row_id = normalize_source_row_id(row.get("source_row_id"))
        seq = cell_text(row.get("seq"))
        if source_row_id and seq:
            item_row_id = f"{source_row_id}-{seq}"
            append_warning(warnings, "source_ref_recovered_from_source_row_seq")

    source_ref = f"{project_key}::{item_row_id}" if project_key and item_row_id else ""
    if not source_ref:
        append_warning(warnings, "source_ref_missing")
    return project_key, item_row_id, source_ref


def attach_source_refs(rows: pd.DataFrame, warnings: list[str] | None = None) -> pd.DataFrame:
    if rows.empty:
        return rows
    output = rows.copy()
    identities = output.apply(lambda row: source_identity_for_row(row, warnings), axis=1)
    output["project_key"] = [item[0] for item in identities]
    output["item_row_id"] = [item[1] for item in identities]
    output["source_ref"] = [item[2] for item in identities]
    return output


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
    query_catalog: QueryCatalog,
    warnings: list[str] | None = None,
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

    sample_index_series = pd.to_numeric(samples["sample_index"], errors="coerce").astype("Int64")
    rows = samples[sample_index_series.isin(candidate_indices)].copy()
    rows["sample_index"] = pd.to_numeric(rows["sample_index"], errors="raise").astype(int)
    if rows["sample_index"].min() < 0 or rows["sample_index"].max() >= len(item_scores):
        raise ValueError("sample_index 超出 item_embeddings 范围")

    rows["package_score"] = rows["project_package_id"].map(package_score_map).fillna(0.0).astype(float)
    rows["package_rank"] = rows["project_package_id"].map(package_rank_map)
    rows["item_score"] = rows["sample_index"].map(lambda sample_index: float(item_scores[int(sample_index)]))
    family_cooccur = cooccur_scores(samples, matched_package_ids)
    rows["cooccur_score"] = rows["family_signature"].map(lambda signature: family_cooccur.get(cell_text(signature), 0.0))
    rows["catalog_score"] = rows.apply(lambda row: catalog_score(row, query_catalog), axis=1)
    rows["unit_score"] = rows.apply(unit_score, axis=1)
    rows["direct_hit"] = rows["sample_index"].isin(direct_indices)
    rows["final_score"] = (
        0.30 * rows["package_score"]
        + 0.25 * rows["item_score"]
        + 0.25 * rows["cooccur_score"]
        + 0.15 * rows["catalog_score"]
        + 0.05 * rows["unit_score"]
    )
    rows = attach_source_refs(rows, warnings)
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


def max_numeric_or_zero(frame: pd.DataFrame, column: str) -> float:
    values = numeric_values(frame, column)
    if values.empty:
        return 0.0
    return float(values.max())


def first_value(group: pd.DataFrame, column: str) -> str:
    if column not in group.columns:
        return ""
    for value in group[column].tolist():
        text = cell_text(value)
        if text:
            return text
    return ""


def ordered_refs(values: pd.Series, limit: int = 10) -> str:
    seen: set[str] = set()
    refs: list[str] = []
    for value in values.tolist():
        text = cell_text(value)
        if text and text not in seen:
            refs.append(text)
            seen.add(text)
        if len(refs) >= limit:
            break
    return ", ".join(refs)


def source_package_count(group: pd.DataFrame) -> int:
    for column in ["project_key", "project_package_id"]:
        if column not in group.columns:
            continue
        values = group[column].map(cell_text)
        non_empty = values[values.ne("")]
        if not non_empty.empty:
            return int(non_empty.nunique())
    if "source_ref" not in group.columns:
        return 0
    package_keys = []
    for value in group["source_ref"].tolist():
        text = cell_text(value)
        if "::" in text:
            package_keys.append(text.rsplit("::", 1)[0])
    return len(set(package_keys))


def build_candidate_item_stats(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=CANDIDATE_ITEM_STATS_COLUMNS)

    rows: list[dict[str, Any]] = []
    for fine_signature, group in candidates.groupby("fine_signature", sort=False, dropna=False):
        group = group.sort_values(["final_score", "item_score", "package_score"], ascending=[False, False, False])
        quantity_min, quantity_median, quantity_max = min_median_max(group, "quantity")
        unit_price_min, unit_price_median, unit_price_max = min_median_max(group, "unit_price")
        total_price_min, total_price_median, total_price_max = min_median_max(group, "total_price")
        labor_min, labor_median, labor_max = min_median_max(group, "labor_unit_price")
        machinery_min, machinery_median, machinery_max = min_median_max(group, "machinery_unit_price")
        rows.append(
            {
                "fine_signature": cell_text(fine_signature),
                "family_signature": first_value(group, "family_signature"),
                "cost_item_name": first_value(group, "cost_item_name"),
                "project_description": first_value(group, "project_description"),
                "unit": first_value(group, "unit"),
                "unit_normalized": first_value(group, "unit_normalized") or first_value(group, "unit"),
                "历史样本数": int(len(group)),
                "来源工程包数": int(group["project_package_id"].nunique()) if "project_package_id" in group.columns else 0,
                "历史工程量最低值": quantity_min,
                "历史工程量中位数": quantity_median,
                "历史工程量最高值": quantity_max,
                "历史综合单价最低值": unit_price_min,
                "历史综合单价中位数": unit_price_median,
                "历史综合单价最高值": unit_price_max,
                "历史合价最低值": total_price_min,
                "历史合价中位数": total_price_median,
                "历史合价最高值": total_price_max,
                "历史人工单价最低值": labor_min,
                "历史人工单价中位数": labor_median,
                "历史人工单价最高值": labor_max,
                "历史机械单价最低值": machinery_min,
                "历史机械单价中位数": machinery_median,
                "历史机械单价最高值": machinery_max,
                "package_score最大值": max_numeric_or_zero(group, "package_score"),
                "item_score最大值": max_numeric_or_zero(group, "item_score"),
                "cooccur_score": max_numeric_or_zero(group, "cooccur_score"),
                "catalog_score": max_numeric_or_zero(group, "catalog_score"),
                "final_score": max_numeric_or_zero(group, "final_score"),
                "是否被LLM采用": "",
                "source_refs": ordered_refs(group.get("source_ref", pd.Series(dtype=object)), limit=10),
            }
        )

    output = pd.DataFrame(rows)
    output = output.sort_values(["final_score", "item_score最大值", "历史样本数"], ascending=[False, False, False])
    output.insert(0, "candidate_id", [f"C{index:03d}" for index in range(1, len(output) + 1)])
    for column in CANDIDATE_ITEM_STATS_COLUMNS:
        if column not in output.columns:
            output[column] = None
    return output[CANDIDATE_ITEM_STATS_COLUMNS].reset_index(drop=True)


def build_candidate_families(candidates: pd.DataFrame, candidate_item_stats: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=CANDIDATE_FAMILY_COLUMNS)

    candidate_ids_by_family: dict[str, str] = {}
    if not candidate_item_stats.empty and "family_signature" in candidate_item_stats.columns:
        for family_signature, stats_group in candidate_item_stats.groupby("family_signature", sort=False, dropna=False):
            ids = [cell_text(value) for value in stats_group.get("candidate_id", pd.Series(dtype=object)).tolist()]
            candidate_ids_by_family[cell_text(family_signature)] = ", ".join([value for value in ids if value])

    rows: list[dict[str, Any]] = []
    for family_signature, group in candidates.groupby("family_signature", sort=False, dropna=False):
        group = group.sort_values(["final_score", "item_score", "package_score"], ascending=[False, False, False])
        representative = group.iloc[0]
        quantity_min, quantity_median, quantity_max = min_median_max(group, "quantity")
        unit_price_min, unit_price_median, unit_price_max = min_median_max(group, "unit_price")
        total_price_min, total_price_median, total_price_max = min_median_max(group, "total_price")
        labor_min, labor_median, labor_max = min_median_max(group, "labor_unit_price")
        machinery_min, machinery_median, machinery_max = min_median_max(group, "machinery_unit_price")
        rows.append(
            {
                "family_signature": cell_text(family_signature),
                "representative_cost_item_name": cell_text(representative.get("cost_item_name")),
                "representative_project_description": cell_text(representative.get("project_description")),
                "unit": cell_text(representative.get("unit")),
                "unit_normalized": cell_text(representative.get("unit_normalized")) or cell_text(representative.get("unit")),
                "覆盖candidate_ids": candidate_ids_by_family.get(cell_text(family_signature), ""),
                "历史样本数": int(len(group)),
                "来源工程包数": source_package_count(group),
                "历史工程量最低值": quantity_min,
                "历史工程量中位数": quantity_median,
                "历史工程量最高值": quantity_max,
                "历史综合单价最低值": unit_price_min,
                "历史综合单价中位数": unit_price_median,
                "历史综合单价最高值": unit_price_max,
                "历史合价最低值": total_price_min,
                "历史合价中位数": total_price_median,
                "历史合价最高值": total_price_max,
                "历史人工单价最低值": labor_min,
                "历史人工单价中位数": labor_median,
                "历史人工单价最高值": labor_max,
                "历史机械单价最低值": machinery_min,
                "历史机械单价中位数": machinery_median,
                "历史机械单价最高值": machinery_max,
                "package_score最大值": max_numeric_or_zero(group, "package_score"),
                "item_score最大值": max_numeric_or_zero(group, "item_score"),
                "cooccur_score": max_numeric_or_zero(group, "cooccur_score"),
                "catalog_score": max_numeric_or_zero(group, "catalog_score"),
                "final_score": max_numeric_or_zero(group, "final_score"),
                "source_refs": ordered_refs(group.get("source_ref", pd.Series(dtype=object)), limit=10),
            }
        )

    output = pd.DataFrame(rows)
    output = output.sort_values(["final_score", "item_score最大值", "历史样本数"], ascending=[False, False, False])
    output.insert(0, "family_id", [f"F{index:03d}" for index in range(1, len(output) + 1)])
    for column in CANDIDATE_FAMILY_COLUMNS:
        if column not in output.columns:
            output[column] = None
    return output[CANDIDATE_FAMILY_COLUMNS].reset_index(drop=True)


def build_evidence_items(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=EVIDENCE_ITEM_COLUMNS)
    output = pd.DataFrame(
        {
            "source_ref": candidates.get("source_ref", ""),
            "project_key": candidates.get("project_key", ""),
            "item_row_id": candidates.get("item_row_id", ""),
            "stable_sample_id": candidates.get("stable_sample_id", ""),
            "batch_id": candidates.get("batch_id", ""),
            "source_row_id": candidates.get("source_row_id", ""),
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
            "unit": candidates.get("unit", ""),
            "unit_normalized": candidates.get("unit_normalized", candidates.get("unit", "")),
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


def replace_nan_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return json.loads(frame.replace({np.nan: None}).to_json(orient="records", force_ascii=False))


def split_refs(value: Any, limit: int | None = None) -> list[str]:
    if isinstance(value, list):
        refs = [cell_text(item) for item in value if cell_text(item)]
    else:
        refs = [part.strip() for part in cell_text(value).replace("；", ",").split(",") if part.strip()]
    return refs[:limit] if limit is not None else refs


def compressed_packages_for_llm(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    columns = [
        "rank",
        "package_score",
        "project_package_id",
        "工程名称",
        "project_name_text",
        "cost_item_names_summary",
        "cache_subject",
        "item_count",
    ]
    rows = frame.head(limit).copy()
    for column in columns:
        if column not in rows.columns:
            rows[column] = ""
    rows["工程名称"] = rows["工程名称"].map(lambda value: truncate_text(value, 80))
    rows["project_name_text"] = rows["project_name_text"].map(lambda value: truncate_text(value, 80))
    rows["cost_item_names_summary"] = rows["cost_item_names_summary"].map(lambda value: truncate_text(value, 500))
    return rows[columns]


def compressed_candidates_for_llm(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    columns = [
        "candidate_id",
        "fine_signature",
        "family_signature",
        "cost_item_name",
        "project_description",
        "unit",
        "unit_normalized",
        "历史样本数",
        "来源工程包数",
        "历史工程量最低值",
        "历史工程量中位数",
        "历史工程量最高值",
        "历史综合单价最低值",
        "历史综合单价中位数",
        "历史综合单价最高值",
        "历史合价最低值",
        "历史合价中位数",
        "历史合价最高值",
        "历史人工单价最低值",
        "历史人工单价中位数",
        "历史人工单价最高值",
        "历史机械单价最低值",
        "历史机械单价中位数",
        "历史机械单价最高值",
        "package_score最大值",
        "item_score最大值",
        "cooccur_score",
        "catalog_score",
        "final_score",
    ]
    rows = frame.head(limit).copy()
    for column in columns:
        if column not in rows.columns:
            rows[column] = None
    rows["source_refs_sample"] = rows.get("source_refs", pd.Series(dtype=object)).map(lambda value: split_refs(value, 5))
    return rows[[*columns, "source_refs_sample"]]


def compressed_families_for_llm(frame: pd.DataFrame, limit: int) -> pd.DataFrame:
    columns = [
        "family_id",
        "family_signature",
        "representative_cost_item_name",
        "representative_project_description",
        "unit",
        "unit_normalized",
        "覆盖candidate_ids",
        "历史样本数",
        "来源工程包数",
        "历史工程量最低值",
        "历史工程量中位数",
        "历史工程量最高值",
        "历史综合单价最低值",
        "历史综合单价中位数",
        "历史综合单价最高值",
        "历史合价最低值",
        "历史合价中位数",
        "历史合价最高值",
        "历史人工单价最低值",
        "历史人工单价中位数",
        "历史人工单价最高值",
        "历史机械单价最低值",
        "历史机械单价中位数",
        "历史机械单价最高值",
        "package_score最大值",
        "item_score最大值",
        "cooccur_score",
        "catalog_score",
        "final_score",
    ]
    rows = frame.head(limit).copy()
    for column in columns:
        if column not in rows.columns:
            rows[column] = None
    rows["source_refs_sample"] = rows.get("source_refs", pd.Series(dtype=object)).map(lambda value: split_refs(value, 5))
    return rows[[*columns, "source_refs_sample"]]


def selected_evidence_for_llm(
    evidence_items: pd.DataFrame,
    compressed_candidates: pd.DataFrame,
    total_limit: int,
    per_candidate_limit: int = 3,
) -> pd.DataFrame:
    if evidence_items.empty or compressed_candidates.empty:
        return pd.DataFrame(columns=[
            "source_ref",
            "来源工程名称",
            "project_package_id",
            "cost_item_name",
            "project_description",
            "unit",
            "unit_normalized",
            "quantity",
            "unit_price",
            "total_price",
            "labor_unit_price",
            "machinery_unit_price",
            "package_rank",
            "package_score",
            "item_score",
        ])
    refs: list[str] = []
    for value in compressed_candidates["source_refs_sample"].tolist():
        for ref in split_refs(value, per_candidate_limit):
            if ref not in refs:
                refs.append(ref)
            if len(refs) >= total_limit:
                break
        if len(refs) >= total_limit:
            break
    columns = [
        "source_ref",
        "来源工程名称",
        "project_package_id",
        "cost_item_name",
        "project_description",
        "unit",
        "unit_normalized",
        "quantity",
        "unit_price",
        "total_price",
        "labor_unit_price",
        "machinery_unit_price",
        "package_rank",
        "package_score",
        "item_score",
    ]
    rows = evidence_items[evidence_items["source_ref"].isin(refs)].copy()
    order = {ref: index for index, ref in enumerate(refs)}
    rows["_order"] = rows["source_ref"].map(order)
    rows = rows.sort_values("_order").drop(columns=["_order"])
    for column in columns:
        if column not in rows.columns:
            rows[column] = ""
    return rows[columns].head(total_limit)


def build_suggested_bill_prompt(
    rewrite: QueryRewrite,
    query_catalog: QueryCatalog,
    matched_project_packages: pd.DataFrame,
    candidate_families: pd.DataFrame,
    candidate_item_stats: pd.DataFrame,
    evidence_items: pd.DataFrame,
    package_limit: int = 8,
    family_limit: int = 20,
    candidate_limit: int = 30,
    evidence_limit: int = 60,
) -> str:
    compressed_packages = compressed_packages_for_llm(matched_project_packages, package_limit)
    compressed_families = compressed_families_for_llm(candidate_families, family_limit)
    compressed_candidates = compressed_candidates_for_llm(candidate_item_stats, candidate_limit)
    evidence_ref_source = compressed_families if not compressed_families.empty else compressed_candidates
    compressed_evidence = selected_evidence_for_llm(evidence_items, evidence_ref_source, evidence_limit)
    payload = {
        "用户原始需求": rewrite.raw_query,
        "query_rewrite": {
            "project_package_query_text": rewrite.project_package_query_text,
            "item_query_text": rewrite.item_query_text,
        },
        "ParsedQuery": {
            "project_package_query_text": rewrite.project_package_query_text,
            "item_query_text": rewrite.item_query_text,
            "parsed_quantities": rewrite.parsed_quantities,
            "materials_or_specs": rewrite.materials_or_specs,
            "repair_object": rewrite.repair_object,
            "uncertainties": rewrite.uncertainties,
        },
        "query_catalog": {
            "catalog_id": query_catalog.catalog_id,
            "一级分类": query_catalog.一级分类,
            "二级分类": query_catalog.二级分类,
            "维修状态": query_catalog.维修状态,
            "标准对象": query_catalog.标准对象,
            "success": query_catalog.success,
        },
        "matched_project_packages": replace_nan_records(compressed_packages),
        "candidate_families": replace_nan_records(compressed_families),
        "candidate_item_stats": replace_nan_records(compressed_candidates),
        "evidence_items": replace_nan_records(compressed_evidence),
    }
    return f"""
你是维修工程造价建议清单生成器。请基于用户需求、ParsedQuery、相似历史工程包、候选项统计和历史明细证据，生成本次 suggested_bill。

只允许输出 JSON object，不要 Markdown，不要解释。用户需求即使较粗，也必须基于相似历史工程包和候选项给出参考 suggested_bill；不确定内容写入 uncertainty_note，不要直接放弃估价。

JSON 格式：
{{
  "suggested_bill": [
    {{
      "seq": 1,
      "family_id": "F001",
      "candidate_id": "C001",
      "recommend_type": "核心施工项",
      "item_role": "核心施工项",
      "estimate_method": "基于用户给出的约500㎡面积，参考历史样本综合单价区间估算",
      "cost_item_name": "",
      "project_description": "",
      "unit": "",
      "suggested_quantity": null,
      "quantity_basis": "",
      "include_in_amount": true,
      "unit_price_low": null,
      "unit_price_mid": null,
      "unit_price_high": null,
      "labor_unit_price_low": null,
      "labor_unit_price_mid": null,
      "labor_unit_price_high": null,
      "machinery_unit_price_low": null,
      "machinery_unit_price_mid": null,
      "machinery_unit_price_high": null,
      "estimated_amount_low": null,
      "estimated_amount_mid": null,
      "estimated_amount_high": null,
      "estimated_labor_amount_low": null,
      "estimated_labor_amount_mid": null,
      "estimated_labor_amount_high": null,
      "estimated_machinery_amount_low": null,
      "estimated_machinery_amount_mid": null,
      "estimated_machinery_amount_high": null,
      "amount_calc_note": "",
      "adopt_reason": "",
      "uncertainty_note": ""
    }}
  ]
}}

要求：
- suggested_bill 必须优先从 candidate_families 中选择本次建议清单，必须填写对应 family_id，不要输出明显无关项。
- candidate_id 是可选兼容字段，只作为代表细项或细粒度参考；不能因为同一 family 下存在多个 candidate_id 就重复输出多行。
- 同一个 family_id 只能输出一次；不要从多个同一 family 的 candidate_id 中重复选择。
- candidate_item_stats 只作为细粒度参考，用于理解 family 覆盖的明细候选，不作为优先选择入口。
- suggested_bill 默认输出 3-6 行。除非 candidate_families 和 candidate_item_stats 中确实没有可用候选，否则不要只输出 1-2 行。
- 不要只选择同一种清单项的多个近似重复项。对于语义高度重复的候选族，只选择最能代表本次需求的一项；优先选择 final_score 高、历史样本数多、项目特征更匹配用户需求的 family。
- 必须输出 include_in_amount。include_in_amount=true 表示该行金额参与 estimate_summary 的总金额区间汇总；include_in_amount=false 表示该行只是建议参考、待确认项或替代方案，不参与总金额汇总。
- include_in_amount 规则：
  1. 核心施工项：如果工程量明确，通常 include_in_amount=true。
  2. 常见前置项：如果工程量可由用户明确工程量合理推导，可以 include_in_amount=true；否则 false。
  3. 恢复/收尾项、措施/条件项：工程量或计价口径不明确时 include_in_amount=false。
  4. 可选/替代工艺：默认 include_in_amount=false，除非用户明确选择该工艺作为核心方案。
- 生成 suggested_bill 时按跨目录通用候选角色判断：
  1. 核心施工项：直接解决用户需求的主要维修、更新、改造内容。
  2. 常见前置项：核心施工前通常需要发生的拆除、清理、检测、基层处理、开挖、保护等。
  3. 恢复/收尾项：核心施工后恢复原状、饰面恢复、回填、调试、试运行、验收等。
  4. 措施/条件项：为完成施工所需的脚手架、吊篮、垂直运输、机械设备进出场及安拆、吊装、安全文明、临时设施等。
  5. 可选/替代工艺：与核心施工项解决同一目标，但属于不同材料、规格、设备型号或技术路线，默认不应重复计价。
- 同一目标的替代工艺默认只选择最匹配用户需求的一项；其它可作为待确认或替代参考，不进入金额汇总。
- 措施/条件项没有明确工程量时可以列为待确认，但金额留空。
- 如果用户明确给出面积，例如 500㎡：
  - 单位为 m² / ㎡ 且与该面积对应的候选项，可以使用该面积作为建议工程量；
  - 项、台班、次、部、套等非面积单位，不要机械按面积线性放大；如果候选单位确实是 m²，才可参考面积，并在 uncertainty_note 中说明需确认适用范围和是否已包含在综合单价中。
- LLM 可以判断工程量、单位、估价口径和金额计算方式，但历史综合单价、人工单价、机械单价仍必须来自对应 family_id 的 candidate_families；缺少 family_id 时才回退到 candidate_id 的 candidate_item_stats，不允许自行估价。
- 不要输出 source_ref、source_refs、evidence_ref、evidence_refs、stable_sample_id、project_key、item_key 或任何来源编号。
- 人工/机械单价或金额没有证据时保留 null，不要填 0。
- recommend_type 建议使用：核心施工项、常见前置项、恢复/收尾项、措施/条件项、可选/替代工艺、补充候选项。
- item_role 建议使用：核心施工项、常见前置项、恢复/收尾项、措施/条件项、可选/替代工艺、补充候选项。
- adopt_reason 要写清楚为什么采用该 candidate，特别是它在通用候选角色中的作用，而不是只写“与需求匹配”。
- uncertainty_note 要说明现场需要确认什么，例如施工范围、材料规格、设备型号、基层或原状条件、施工高度、运输距离、临时措施、恢复范围、调试验收要求等是否明确。
- 如果楼栋数、设备数量、运输高度、基层状况、节点复杂度未知，也要在 uncertainty_note 中指出。
- 如果无法可靠估算工程量，可以给历史价格参考但金额为空，或给保守范围并说明原因。

输入数据：
{json_text(payload)}
""".strip()


def guarded_suggested_bill_prompt(
    rewrite: QueryRewrite,
    query_catalog: QueryCatalog,
    matched_project_packages: pd.DataFrame,
    candidate_families: pd.DataFrame,
    candidate_item_stats: pd.DataFrame,
    evidence_items: pd.DataFrame,
) -> str:
    prompt = build_suggested_bill_prompt(
        rewrite,
        query_catalog,
        matched_project_packages,
        candidate_families,
        candidate_item_stats,
        evidence_items,
        package_limit=8,
        family_limit=20,
        candidate_limit=30,
        evidence_limit=60,
    )
    if len(prompt) <= 30000:
        return prompt
    prompt = build_suggested_bill_prompt(
        rewrite,
        query_catalog,
        matched_project_packages,
        candidate_families,
        candidate_item_stats,
        evidence_items,
        package_limit=5,
        family_limit=14,
        candidate_limit=20,
        evidence_limit=30,
    )
    if len(prompt) <= 22000:
        return prompt
    prompt = build_suggested_bill_prompt(
        rewrite,
        query_catalog,
        matched_project_packages,
        candidate_families,
        candidate_item_stats,
        evidence_items,
        package_limit=4,
        family_limit=8,
        candidate_limit=12,
        evidence_limit=18,
    )
    if len(prompt) <= 12000:
        return prompt
    return build_suggested_bill_prompt(
        rewrite,
        query_catalog,
        matched_project_packages,
        candidate_families,
        candidate_item_stats,
        evidence_items,
        package_limit=2,
        family_limit=4,
        candidate_limit=6,
        evidence_limit=6,
    )


def bill_value(item: dict[str, Any], english_key: str, chinese_key: str = "") -> Any:
    if english_key in item:
        return item.get(english_key)
    if chinese_key and chinese_key in item:
        return item.get(chinese_key)
    return ""


def normalize_include_in_amount(value: Any, item_role: Any, suggested_quantity: Any) -> str:
    if isinstance(value, bool):
        return "是" if value else "否"

    text = cell_text(value).lower()
    if text in {"true", "1", "yes", "y", "是", "计入"}:
        return "是"
    if text in {"false", "0", "no", "n", "否", "不计入"}:
        return "否"

    role = cell_text(item_role)
    if role == "核心施工项":
        return "是"
    if role == "常见前置项":
        return "是" if numeric_or_none(suggested_quantity) is not None else "否"
    return "否"


def suggested_bill_from_llm_result(result: dict[str, Any], warnings: list[str] | None = None) -> pd.DataFrame:
    bill = result.get("suggested_bill")
    if not isinstance(bill, list):
        raise ValueError("LLM 输出缺少 suggested_bill list")

    forbidden_source_keys = {
        "source_ref",
        "source_refs",
        "evidence_ref",
        "evidence_refs",
        "stable_sample_id",
        "project_key",
        "item_key",
        "来源样本",
        "来源证据",
    }
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(bill, start=1):
        if not isinstance(item, dict):
            continue
        if forbidden_source_keys & set(item.keys()):
            append_warning(warnings, "llm_source_fields_ignored")
        item_role = bill_value(item, "item_role", "项目角色")
        suggested_quantity = bill_value(item, "suggested_quantity", "建议工程量")
        rows.append(
            {
                "序号": bill_value(item, "seq", "序号") or index,
                "family_id": bill_value(item, "family_id", "family_id"),
                "candidate_id": bill_value(item, "candidate_id", "candidate_id"),
                "推荐类型": bill_value(item, "recommend_type", "推荐类型"),
                "项目角色": item_role,
                "计量/估算口径": bill_value(item, "estimate_method", "计量/估算口径"),
                "清单项名称": bill_value(item, "cost_item_name", "清单项名称"),
                "项目特征/施工工艺": bill_value(item, "project_description", "项目特征/施工工艺"),
                "单位": bill_value(item, "unit", "单位"),
                "建议工程量": suggested_quantity,
                "工程量依据": bill_value(item, "quantity_basis", "工程量依据"),
                "是否计入金额汇总": normalize_include_in_amount(
                    bill_value(item, "include_in_amount", "是否计入金额汇总"),
                    item_role,
                    suggested_quantity,
                ),
                "综合单价最低值": bill_value(item, "unit_price_low", "综合单价最低值"),
                "综合单价中位数": bill_value(item, "unit_price_mid", "综合单价中位数"),
                "综合单价最高值": bill_value(item, "unit_price_high", "综合单价最高值"),
                "其中包含人工费单价最低值": bill_value(item, "labor_unit_price_low", "其中包含人工费单价最低值"),
                "其中包含人工费单价中位数": bill_value(item, "labor_unit_price_mid", "其中包含人工费单价中位数"),
                "其中包含人工费单价最高值": bill_value(item, "labor_unit_price_high", "其中包含人工费单价最高值"),
                "其中包含机械费单价最低值": bill_value(item, "machinery_unit_price_low", "其中包含机械费单价最低值"),
                "其中包含机械费单价中位数": bill_value(item, "machinery_unit_price_mid", "其中包含机械费单价中位数"),
                "其中包含机械费单价最高值": bill_value(item, "machinery_unit_price_high", "其中包含机械费单价最高值"),
                "估算金额最低值": bill_value(item, "estimated_amount_low", "估算金额最低值"),
                "估算金额中位数": bill_value(item, "estimated_amount_mid", "估算金额中位数"),
                "估算金额最高值": bill_value(item, "estimated_amount_high", "估算金额最高值"),
                "估算人工费最低值": bill_value(item, "estimated_labor_amount_low", "估算人工费最低值"),
                "估算人工费中位数": bill_value(item, "estimated_labor_amount_mid", "估算人工费中位数"),
                "估算人工费最高值": bill_value(item, "estimated_labor_amount_high", "估算人工费最高值"),
                "估算机械费最低值": bill_value(item, "estimated_machinery_amount_low", "估算机械费最低值"),
                "估算机械费中位数": bill_value(item, "estimated_machinery_amount_mid", "估算机械费中位数"),
                "估算机械费最高值": bill_value(item, "estimated_machinery_amount_high", "估算机械费最高值"),
                "金额计算说明": bill_value(item, "amount_calc_note", "金额计算说明"),
                "采用理由": bill_value(item, "adopt_reason", "采用理由"),
                "不确定性说明": bill_value(item, "uncertainty_note", "不确定性说明"),
                "来源样本": "",
            }
        )
    if not rows:
        raise ValueError("LLM suggested_bill 为空")
    return pd.DataFrame(rows, columns=SUGGESTED_BILL_COLUMNS)


def fallback_suggested_bill(
    candidate_item_stats: pd.DataFrame,
    candidate_families: pd.DataFrame | None = None,
    limit: int = 20,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if candidate_families is not None and not candidate_families.empty:
        for _index, row in candidate_families.head(limit).iterrows():
            candidate_id = split_refs(row.get("覆盖candidate_ids"), 1)
            rows.append(
                {
                    "序号": len(rows) + 1,
                    "family_id": row.get("family_id", ""),
                    "candidate_id": candidate_id[0] if candidate_id else "",
                    "推荐类型": "fallback_family",
                    "项目角色": "补充候选项",
                    "计量/估算口径": "LLM suggested_bill 生成失败，未判断估算口径",
                    "清单项名称": row.get("representative_cost_item_name", ""),
                    "项目特征/施工工艺": row.get("representative_project_description", ""),
                    "单位": row.get("unit_normalized", "") or row.get("unit", ""),
                    "建议工程量": "",
                    "工程量依据": "LLM suggested_bill 生成失败，未生成建议工程量",
                    "是否计入金额汇总": "否",
                    "综合单价最低值": row.get("历史综合单价最低值"),
                    "综合单价中位数": row.get("历史综合单价中位数"),
                    "综合单价最高值": row.get("历史综合单价最高值"),
                    "其中包含人工费单价最低值": row.get("历史人工单价最低值"),
                    "其中包含人工费单价中位数": row.get("历史人工单价中位数"),
                    "其中包含人工费单价最高值": row.get("历史人工单价最高值"),
                    "其中包含机械费单价最低值": row.get("历史机械单价最低值"),
                    "其中包含机械费单价中位数": row.get("历史机械单价中位数"),
                    "其中包含机械费单价最高值": row.get("历史机械单价最高值"),
                    "估算金额最低值": "",
                    "估算金额中位数": "",
                    "估算金额最高值": "",
                    "估算人工费最低值": "",
                    "估算人工费中位数": "",
                    "估算人工费最高值": "",
                    "估算机械费最低值": "",
                    "估算机械费中位数": "",
                    "估算机械费最高值": "",
                    "金额计算说明": "缺少 LLM 判断工程量，未计算估算金额",
                    "采用理由": "LLM suggested_bill 生成失败，本行仅为候选族统计结果，不代表最终建议清单",
                    "不确定性说明": "需修复 LLM 上下文或降低候选规模后重新生成",
                    "来源样本": row.get("source_refs", ""),
                }
            )
        return pd.DataFrame(rows, columns=SUGGESTED_BILL_COLUMNS)

    family_id_by_signature: dict[str, str] = {}
    if candidate_families is not None and not candidate_families.empty:
        family_id_by_signature = {
            cell_text(row.get("family_signature")): cell_text(row.get("family_id"))
            for _index, row in candidate_families.iterrows()
        }
    for _index, row in candidate_item_stats.head(limit).iterrows():
        rows.append(
            {
                "序号": len(rows) + 1,
                "family_id": family_id_by_signature.get(cell_text(row.get("family_signature")), ""),
                "candidate_id": row.get("candidate_id", ""),
                "推荐类型": "fallback_candidate",
                "项目角色": "补充候选项",
                "计量/估算口径": "LLM suggested_bill 生成失败，未判断估算口径",
                "清单项名称": row.get("cost_item_name", ""),
                "项目特征/施工工艺": row.get("project_description", ""),
                "单位": row.get("unit_normalized", "") or row.get("unit", ""),
                "建议工程量": "",
                "工程量依据": "LLM suggested_bill 生成失败，未生成建议工程量",
                "是否计入金额汇总": "否",
                "综合单价最低值": row.get("历史综合单价最低值"),
                "综合单价中位数": row.get("历史综合单价中位数"),
                "综合单价最高值": row.get("历史综合单价最高值"),
                "其中包含人工费单价最低值": row.get("历史人工单价最低值"),
                "其中包含人工费单价中位数": row.get("历史人工单价中位数"),
                "其中包含人工费单价最高值": row.get("历史人工单价最高值"),
                "其中包含机械费单价最低值": row.get("历史机械单价最低值"),
                "其中包含机械费单价中位数": row.get("历史机械单价中位数"),
                "其中包含机械费单价最高值": row.get("历史机械单价最高值"),
                "估算金额最低值": "",
                "估算金额中位数": "",
                "估算金额最高值": "",
                "估算人工费最低值": "",
                "估算人工费中位数": "",
                "估算人工费最高值": "",
                "估算机械费最低值": "",
                "估算机械费中位数": "",
                "估算机械费最高值": "",
                "金额计算说明": "缺少 LLM 判断工程量，未计算估算金额",
                "采用理由": "LLM suggested_bill 生成失败，本行仅为候选项统计结果，不代表最终建议清单",
                "不确定性说明": "需修复 LLM 上下文或降低候选规模后重新生成",
                "来源样本": row.get("source_refs", ""),
            }
        )
    return pd.DataFrame(rows, columns=SUGGESTED_BILL_COLUMNS)


def numeric_or_none(value: Any) -> float | None:
    if value is None or cell_text(value) == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def numbers_close(left: Any, right: Any, tolerance: float = 1.0) -> bool:
    left_number = numeric_or_none(left)
    right_number = numeric_or_none(right)
    if left_number is None or right_number is None:
        return left_number is None and right_number is None
    return abs(left_number - right_number) <= max(tolerance, abs(right_number) * 0.01)


def strict_candidate_match(candidate_item_stats: pd.DataFrame, row: pd.Series) -> pd.Series | None:
    if candidate_item_stats.empty:
        return None
    mask = (
        candidate_item_stats["cost_item_name"].map(cell_text).eq(cell_text(row.get("清单项名称")))
        & candidate_item_stats["project_description"].map(cell_text).eq(cell_text(row.get("项目特征/施工工艺")))
        & (
            candidate_item_stats["unit"].map(cell_text).eq(cell_text(row.get("单位")))
            | candidate_item_stats["unit_normalized"].map(cell_text).eq(cell_text(row.get("单位")))
        )
    )
    matches = candidate_item_stats[mask]
    if len(matches) == 1:
        return matches.iloc[0]
    return None


def calc_amount(quantity: Any, unit_price: Any) -> float | None:
    quantity_number = numeric_or_none(quantity)
    price_number = numeric_or_none(unit_price)
    if quantity_number is None or price_number is None:
        return None
    return round(quantity_number * price_number, 2)


def postprocess_suggested_bill(
    suggested_bill: pd.DataFrame,
    candidate_item_stats: pd.DataFrame,
    candidate_families: pd.DataFrame | None = None,
    warnings: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if isinstance(candidate_families, list) and warnings is None:
        warnings = candidate_families
        candidate_families = None
    output = suggested_bill.copy().astype("object")
    stats = candidate_item_stats.copy().astype("object")
    families = candidate_families.copy().astype("object") if candidate_families is not None else pd.DataFrame(
        columns=CANDIDATE_FAMILY_COLUMNS
    )
    for column in SUGGESTED_BILL_COLUMNS:
        if column not in output.columns:
            output[column] = ""
    for index, row in output.iterrows():
        output.at[index, "是否计入金额汇总"] = normalize_include_in_amount(
            row.get("是否计入金额汇总"),
            row.get("项目角色"),
            row.get("建议工程量"),
        )
    stats["是否被LLM采用"] = ""
    candidate_map = {cell_text(row.get("candidate_id")): row for _index, row in stats.iterrows()}
    family_map = {cell_text(row.get("family_id")): row for _index, row in families.iterrows()}
    adoption: dict[str, str] = {}

    if not output.empty and "family_id" in output.columns:
        keep_indices: list[int] = []
        seen_family_ids: set[str] = set()
        for index, row in output.iterrows():
            family_id = cell_text(row.get("family_id"))
            if family_id and family_id in seen_family_ids:
                append_warning(warnings, "duplicate_family_id_dropped")
                continue
            if family_id:
                seen_family_ids.add(family_id)
            keep_indices.append(index)
        output = output.loc[keep_indices].reset_index(drop=True)

    price_pairs = [
        ("综合单价最低值", "历史综合单价最低值"),
        ("综合单价中位数", "历史综合单价中位数"),
        ("综合单价最高值", "历史综合单价最高值"),
        ("其中包含人工费单价最低值", "历史人工单价最低值"),
        ("其中包含人工费单价中位数", "历史人工单价中位数"),
        ("其中包含人工费单价最高值", "历史人工单价最高值"),
        ("其中包含机械费单价最低值", "历史机械单价最低值"),
        ("其中包含机械费单价中位数", "历史机械单价中位数"),
        ("其中包含机械费单价最高值", "历史机械单价最高值"),
    ]
    amount_pairs = [
        ("估算金额最低值", "综合单价最低值"),
        ("估算金额中位数", "综合单价中位数"),
        ("估算金额最高值", "综合单价最高值"),
        ("估算人工费最低值", "其中包含人工费单价最低值"),
        ("估算人工费中位数", "其中包含人工费单价中位数"),
        ("估算人工费最高值", "其中包含人工费单价最高值"),
        ("估算机械费最低值", "其中包含机械费单价最低值"),
        ("估算机械费中位数", "其中包含机械费单价中位数"),
        ("估算机械费最高值", "其中包含机械费单价最高值"),
    ]

    for index, row in output.iterrows():
        family_id = cell_text(row.get("family_id"))
        candidate_id = cell_text(row.get("candidate_id"))
        matched_stats = family_map.get(family_id)
        candidate = None
        match_status = "是"
        adopted_candidate_ids: list[str] = []
        if matched_stats is not None:
            adopted_candidate_ids = split_refs(matched_stats.get("覆盖candidate_ids"))
            if not candidate_id and adopted_candidate_ids:
                candidate_id = adopted_candidate_ids[0]
                output.at[index, "candidate_id"] = candidate_id
            if not cell_text(output.at[index, "清单项名称"]):
                output.at[index, "清单项名称"] = matched_stats.get("representative_cost_item_name", "")
            if not cell_text(output.at[index, "项目特征/施工工艺"]):
                output.at[index, "项目特征/施工工艺"] = matched_stats.get("representative_project_description", "")
            if not cell_text(output.at[index, "单位"]):
                output.at[index, "单位"] = matched_stats.get("unit_normalized", "") or matched_stats.get("unit", "")
        else:
            if family_id:
                append_warning(warnings, "family_id_missing_or_ambiguous")
            candidate = candidate_map.get(candidate_id)

        if matched_stats is None and candidate is None:
            candidate = strict_candidate_match(stats, row)
            if candidate is not None:
                append_warning(warnings, "candidate_id_missing_strict_matched")
                candidate_id = cell_text(candidate.get("candidate_id"))
                output.at[index, "candidate_id"] = candidate_id
                match_status = "是（退化匹配）"
            else:
                append_warning(warnings, "candidate_id_missing_or_ambiguous")
                for final_column, _stats_column in price_pairs:
                    output.at[index, final_column] = ""
                for amount_column, _price_column in amount_pairs:
                    output.at[index, amount_column] = ""
                output.at[index, "来源样本"] = ""
                continue

        if matched_stats is None and candidate is not None:
            matched_stats = candidate
            adopted_candidate_ids = [candidate_id]

        for adopted_candidate_id in adopted_candidate_ids:
            if adopted_candidate_id in adoption and adoption[adopted_candidate_id] != match_status:
                adoption[adopted_candidate_id] = "匹配冲突"
                append_warning(warnings, "candidate_id_missing_or_ambiguous")
            else:
                adoption[adopted_candidate_id] = match_status

        if matched_stats is None:
            append_warning(warnings, "candidate_id_missing_or_ambiguous")
            for final_column, _stats_column in price_pairs:
                output.at[index, final_column] = ""
            for amount_column, _price_column in amount_pairs:
                output.at[index, amount_column] = ""
            output.at[index, "来源样本"] = ""
            continue

        if candidate_id and candidate_id not in adopted_candidate_ids and candidate_id in candidate_map:
            adopted_candidate_ids.append(candidate_id)
            adoption[candidate_id] = match_status

        if family_id and matched_stats is not None:
            output.at[index, "family_id"] = family_id
        else:
            output.at[index, "family_id"] = ""

        for final_column, stats_column in price_pairs:
            candidate_value = matched_stats.get(stats_column)
            if numeric_or_none(candidate_value) is None:
                if cell_text(output.at[index, final_column]):
                    append_warning(warnings, "empty_price_field_kept_empty")
                output.at[index, final_column] = ""
                continue
            if not numbers_close(output.at[index, final_column], candidate_value):
                append_warning(warnings, "llm_unit_price_overridden_by_candidate_stats")
            output.at[index, final_column] = candidate_value

        output.at[index, "来源样本"] = ", ".join(split_refs(matched_stats.get("source_refs"), 10))

        for amount_column, price_column in amount_pairs:
            calculated = calc_amount(output.at[index, "建议工程量"], output.at[index, price_column])
            if calculated is None:
                if cell_text(output.at[index, amount_column]):
                    append_warning(warnings, "empty_price_field_kept_empty")
                output.at[index, amount_column] = ""
                continue
            if not numbers_close(output.at[index, amount_column], calculated):
                append_warning(warnings, "llm_amount_overridden_by_program_calculation")
                output.at[index, amount_column] = calculated

    if not output.empty:
        output["序号"] = range(1, len(output) + 1)
    for index, row in stats.iterrows():
        candidate_id = cell_text(row.get("candidate_id"))
        stats.at[index, "是否被LLM采用"] = adoption.get(candidate_id, "")
    return output[SUGGESTED_BILL_COLUMNS], stats[CANDIDATE_ITEM_STATS_COLUMNS]


def generate_suggested_bill(
    rewrite: QueryRewrite,
    query_catalog: QueryCatalog,
    matched_project_packages: pd.DataFrame,
    candidate_families: pd.DataFrame,
    candidate_item_stats: pd.DataFrame,
    evidence_items: pd.DataFrame,
    warnings: list[str] | None = None,
) -> tuple[pd.DataFrame, bool, bool, str, str, dict[str, Any]]:
    prompt = guarded_suggested_bill_prompt(
        rewrite,
        query_catalog,
        matched_project_packages,
        candidate_families,
        candidate_item_stats,
        evidence_items,
    )
    max_tokens = 4096
    try:
        result = request_llm_json(
            prompt,
            max_tokens=max_tokens,
            system_prompt="你只输出一个 JSON object，不输出解释、Markdown 或思考过程。",
        )
        suggested_bill = suggested_bill_from_llm_result(result, warnings)
        trace = trace_row(
            "suggested_bill_generation",
            "基于压缩证据生成 suggested_bill",
            True,
            prompt=prompt,
            max_tokens=max_tokens,
            input_summary=f"families={len(candidate_families)}, candidates={len(candidate_item_stats)}, evidence={len(evidence_items)}",
        )
        return suggested_bill, True, False, "", prompt, trace
    except (LLMServiceError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        trace = trace_row(
            "suggested_bill_generation",
            "基于压缩证据生成 suggested_bill",
            False,
            error=str(exc),
            prompt=prompt,
            max_tokens=max_tokens,
            input_summary=f"families={len(candidate_families)}, candidates={len(candidate_item_stats)}, evidence={len(evidence_items)}",
        )
        return fallback_suggested_bill(candidate_item_stats, candidate_families), False, True, str(exc), prompt, trace


def display_frame(frame: pd.DataFrame, display: bool) -> pd.DataFrame:
    if not display:
        return frame
    output = frame.copy()
    for column in output.columns:
        if pd.api.types.is_numeric_dtype(output[column]):
            output[column] = output[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.2f}")
    return output


def query_catalog_dict(query_catalog: QueryCatalog) -> dict[str, Any]:
    return {
        "catalog_id": query_catalog.catalog_id,
        "一级分类": query_catalog.一级分类,
        "二级分类": query_catalog.二级分类,
        "维修状态": query_catalog.维修状态,
        "标准对象": query_catalog.标准对象,
        "confidence": query_catalog.confidence,
        "success": query_catalog.success,
        "notes": query_catalog.notes,
        "raw_result": query_catalog.raw_result,
    }


def parsed_query_dict(rewrite: QueryRewrite) -> dict[str, Any]:
    return {
        "project_package_query_text": rewrite.project_package_query_text,
        "item_query_text": rewrite.item_query_text,
        "parsed_quantities": rewrite.parsed_quantities,
        "materials_or_specs": rewrite.materials_or_specs,
        "repair_object": rewrite.repair_object,
        "uncertainties": rewrite.uncertainties,
    }


def join_non_empty(values: list[Any], limit: int | None = None) -> str:
    texts: list[str] = []
    for value in values:
        text = cell_text(value)
        if text and text not in texts:
            texts.append(text)
        if limit is not None and len(texts) >= limit:
            break
    return "；".join(texts)


def amount_sum(frame: pd.DataFrame, column: str) -> float | None:
    values = numeric_values(frame, column)
    if values.empty:
        return None
    return round(float(values.sum()), 2)


def build_estimate_summary(
    rewrite: QueryRewrite,
    query_catalog: QueryCatalog,
    suggested_bill: pd.DataFrame,
) -> pd.DataFrame:
    quantity_texts: list[str] = []
    for item in rewrite.parsed_quantities:
        if not isinstance(item, dict):
            continue
        raw_text = cell_text(item.get("raw_text"))
        meaning = cell_text(item.get("meaning"))
        value = numeric_or_none(item.get("value"))
        unit = cell_text(item.get("unit"))
        if value is not None:
            value_text = f"{value:g}{unit}"
        else:
            value_text = raw_text
        if not value_text:
            continue
        if meaning:
            quantity_texts.append(f"{meaning}约 {value_text}")
        else:
            quantity_texts.append(f"工程量约 {value_text}")

    demand_parts = []
    if rewrite.repair_object:
        demand_parts.append(f"用户拟对{rewrite.repair_object}进行维修")
    else:
        demand_parts.append(f"用户需求为“{rewrite.raw_query}”")
    if rewrite.materials_or_specs:
        demand_parts.append(f"涉及{join_non_empty(rewrite.materials_or_specs)}")
    if quantity_texts:
        demand_parts.append(f"明确{join_non_empty(quantity_texts)}")
    demand_understanding = "，".join(demand_parts) + "。"

    suggested_reference_items = join_non_empty(suggested_bill["清单项名称"].tolist(), limit=20)
    if suggested_reference_items:
        suggested_reference_items += "。"
    else:
        suggested_reference_items = "当前未形成可展示的建议清单项。"

    amount_bill = suggested_bill.copy()
    if "是否计入金额汇总" not in amount_bill.columns:
        amount_bill["是否计入金额汇总"] = ""
    for index, row in amount_bill.iterrows():
        amount_bill.at[index, "是否计入金额汇总"] = normalize_include_in_amount(
            row.get("是否计入金额汇总"),
            row.get("项目角色"),
            row.get("建议工程量"),
        )
    amount_bill = amount_bill[amount_bill["是否计入金额汇总"].map(cell_text).eq("是")]

    low_amount = amount_sum(amount_bill, "估算金额最低值")
    mid_amount = amount_sum(amount_bill, "估算金额中位数")
    high_amount = amount_sum(amount_bill, "估算金额最高值")
    if low_amount is not None and mid_amount is not None and high_amount is not None:
        amount_range = (
            f"当前计入金额汇总项目的参考金额约为 {low_amount:,.2f} - {high_amount:,.2f} 元，"
            f"中位参考值约 {mid_amount:,.2f} 元。"
        )
    elif low_amount is not None or mid_amount is not None or high_amount is not None:
        amount_parts = []
        if low_amount is not None:
            amount_parts.append(f"最低参考值约 {low_amount:,.2f} 元")
        if mid_amount is not None:
            amount_parts.append(f"中位参考值约 {mid_amount:,.2f} 元")
        if high_amount is not None:
            amount_parts.append(f"最高参考值约 {high_amount:,.2f} 元")
        amount_range = (
            f"当前计入金额汇总项目已有部分金额参考：{join_non_empty(amount_parts)}；"
            "因部分项目缺少可计算工程量，区间可能不完整。"
        )
    else:
        amount_range = "当前没有可计入金额汇总的可计算项目，暂不汇总总价，仅提供历史单价参考。"

    confirmation_values = []
    for value in [*rewrite.uncertainties, *suggested_bill["不确定性说明"].tolist()]:
        text = cell_text(value).strip("。；; ")
        if text.endswith("未知"):
            text = text[:-2]
        if text.endswith("不确定"):
            text = text[:-3]
        if text.endswith("需要确认"):
            text = text[:-4]
        if text.endswith("需确认"):
            text = text[:-3]
        if text.endswith("待确认"):
            text = text[:-3]
        if text:
            confirmation_values.append(text)
    confirmation_text = join_non_empty(confirmation_values, limit=6)
    if confirmation_text:
        site_confirmation = f"需确认{confirmation_text}。"
    else:
        site_confirmation = "需结合现场踏勘确认实际工程量、施工条件和细部做法。"

    rows = [
        ("需求理解", demand_understanding),
        ("建议参考清单", suggested_reference_items),
        ("参考金额区间", amount_range),
        ("需现场确认", site_confirmation),
    ]
    return pd.DataFrame(rows, columns=ESTIMATE_SUMMARY_COLUMNS)


def build_parse_info(
    rewrite: QueryRewrite,
    query_catalog: QueryCatalog,
    top_packages: int,
    top_items: int,
    max_packages_per_cache_subject: int,
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
    warnings: list[str] | None = None,
) -> pd.DataFrame:
    rows = [
        ("原始用户需求", rewrite.raw_query),
        ("project_package_query_text", rewrite.project_package_query_text),
        ("item_query_text", rewrite.item_query_text),
        ("ParsedQuery", json_text(parsed_query_dict(rewrite))),
        ("query_catalog", json_text(query_catalog_dict(query_catalog))),
        ("item_retrieval_text_fields", "cost_item_name + project_description + unit_normalized"),
        ("package_retrieval_text_fields", "工程名称 + project_name_text + cost_item_names_summary"),
        ("top_packages", top_packages),
        ("top_items", top_items),
        ("max_packages_per_cache_subject", max_packages_per_cache_subject),
        ("embedding_model", meta.get("model", "")),
        ("sample_count", sample_count),
        ("package_count", package_count),
        ("LLM query rewrite 是否成功", "是" if rewrite.success else "否"),
        ("query_catalog_classification 是否成功", "是" if query_catalog.success else "否"),
        ("LLM suggested_bill 是否成功", "是" if suggested_success else "否"),
        ("是否 fallback", "是" if fallback else "否"),
        ("LLM error", llm_error),
        ("prompt_chars", len(suggested_prompt)),
        ("estimated_tokens", estimated_tokens(suggested_prompt)),
        ("output_path", str(output_path or "")),
        ("运行时间", f"{(datetime.now() - started_at).total_seconds():.2f}s"),
        ("index_dir", str(index_dir)),
        ("主要文件路径", json_text((meta.get("files") or {}))),
        ("rewrite_notes", "；".join(rewrite.notes)),
        ("catalog_notes", "；".join(query_catalog.notes)),
        ("warnings", "；".join(warnings or [])),
    ]
    if include_debug_text:
        rows.append(("suggested_bill_prompt_preview", suggested_prompt[:3000]))
    return pd.DataFrame(rows, columns=["字段", "值"])


def write_query_result_workbook(output_path: Path, result: QueryResult, display: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        display_frame(result.estimate_summary, display).to_excel(writer, sheet_name="estimate_summary", index=False)
        display_frame(result.suggested_bill, display).to_excel(writer, sheet_name="suggested_bill", index=False)
        display_frame(result.matched_project_packages, display).to_excel(
            writer,
            sheet_name="matched_project_packages",
            index=False,
        )
        display_frame(result.candidate_families, display).to_excel(
            writer,
            sheet_name="candidate_families",
            index=False,
        )
        display_frame(result.candidate_item_stats, display).to_excel(
            writer,
            sheet_name="candidate_item_stats",
            index=False,
        )
        display_frame(result.evidence_items, display).to_excel(writer, sheet_name="evidence_items", index=False)
        result.parse_info.to_excel(writer, sheet_name="parse_info", index=False)
        result.llm_trace.to_excel(writer, sheet_name="llm_trace", index=False)
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
    max_packages_per_cache_subject: int = 1,
    include_debug_text: bool = False,
    display: bool = False,
) -> QueryResult:
    started_at = datetime.now()
    warnings: list[str] = []
    samples, project_packages, project_package_embeddings, item_embeddings, meta = load_index(index_dir)
    rewrite, rewrite_trace = query_rewrite_for_embedding(raw_text)
    query_catalog, catalog_trace = classify_query_catalog(
        raw_text,
        rewrite.project_package_query_text,
        rewrite.item_query_text,
    )

    model = load_embedding_model(str(meta.get("model") or "BAAI/bge-m3"))
    try:
        package_query_embedding = encode_query(model, rewrite.project_package_query_text)
        item_query_embedding = encode_query(model, rewrite.item_query_text)
    finally:
        release_embedding_model(model)
        gc.collect()

    if package_query_embedding.shape[0] != project_package_embeddings.shape[1]:
        raise ValueError("package query embedding 维度与索引 embedding 维度不一致")
    if item_query_embedding.shape[0] != item_embeddings.shape[1]:
        raise ValueError("item query embedding 维度与索引 embedding 维度不一致")

    matched_raw = score_project_packages(
        project_packages,
        project_package_embeddings,
        package_query_embedding,
        top_packages,
        max_packages_per_cache_subject=max_packages_per_cache_subject,
    )
    item_scores = item_embeddings @ item_query_embedding
    direct_item_hits = score_direct_items(samples, item_scores, top_items)
    candidates = candidate_pool(samples, matched_raw, direct_item_hits, item_scores, query_catalog, warnings=warnings)
    candidate_item_stats = build_candidate_item_stats(candidates)
    candidate_families = build_candidate_families(candidates, candidate_item_stats)
    evidence_items = build_evidence_items(candidates)
    matched_project_packages = matched_project_packages_for_output(matched_raw)

    suggested_bill, suggested_success, fallback, llm_error, suggested_prompt, suggested_trace = generate_suggested_bill(
        rewrite,
        query_catalog,
        matched_project_packages,
        candidate_families,
        candidate_item_stats,
        evidence_items,
        warnings=warnings,
    )
    suggested_bill, candidate_item_stats = postprocess_suggested_bill(
        suggested_bill,
        candidate_item_stats,
        candidate_families,
        warnings,
    )
    estimate_summary = build_estimate_summary(rewrite, query_catalog, suggested_bill)
    if warnings:
        suggested_trace["input_summary"] = f"{suggested_trace.get('input_summary', '')}; warnings={';'.join(warnings)}"
    parse_info = build_parse_info(
        rewrite=rewrite,
        query_catalog=query_catalog,
        top_packages=top_packages,
        top_items=top_items,
        max_packages_per_cache_subject=max_packages_per_cache_subject,
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
        warnings=warnings,
    )
    llm_trace = pd.DataFrame([rewrite_trace, catalog_trace, suggested_trace], columns=LLM_TRACE_COLUMNS)

    result = QueryResult(
        rewrite=rewrite,
        query_catalog=query_catalog,
        estimate_summary=estimate_summary,
        suggested_bill=suggested_bill,
        matched_project_packages=matched_project_packages,
        candidate_families=candidate_families,
        candidate_item_stats=candidate_item_stats,
        evidence_items=evidence_items,
        parse_info=parse_info,
        llm_trace=llm_trace,
    )
    if output:
        write_query_result_workbook(output, result, display=display)
    return result


def print_terminal_summary(result: QueryResult, output_path: Path | None) -> None:
    print(f"[DONE] package query: {result.rewrite.project_package_query_text}")
    print(f"[DONE] item query: {result.rewrite.item_query_text}")
    if result.query_catalog.success:
        print(
            "[DONE] query catalog: "
            f"{result.query_catalog.catalog_id} "
            f"{result.query_catalog.一级分类}/{result.query_catalog.二级分类}/{result.query_catalog.维修状态}"
        )
    else:
        print("[WARN] query catalog classification failed; catalog_score used neutral 0.5")
    print(f"[DONE] matched project packages: {len(result.matched_project_packages)}")
    print(f"[DONE] candidate families: {len(result.candidate_families)}")
    print(f"[DONE] candidate item stats: {len(result.candidate_item_stats)}")
    print(f"[DONE] evidence items: {len(result.evidence_items)}")
    print(f"[DONE] suggested bill rows: {len(result.suggested_bill)}")
    if result.rewrite.notes:
        print(f"rewrite notes: {'；'.join(result.rewrite.notes)}")
    if result.query_catalog.notes:
        print(f"catalog notes: {'；'.join(result.query_catalog.notes)}")
    if output_path:
        print(f"输出文件: {output_path}")


def main() -> int:
    args = parse_args()
    index_dir = Path(args.index_dir).expanduser().resolve()
    output_arg = Path(args.output) if args.output is not None else default_query_output_path()
    output_path = output_arg.expanduser().resolve()

    try:
        validate_output_path(output_path, args.overwrite)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 1

    try:
        check_lmstudio_service(timeout_seconds=args.llm_check_timeout)
    except RuntimeError as exc:
        detail = str(exc)
        marker = "LMSTUDIO_BASE_URL="
        suffix = detail[detail.find(marker) :] if marker in detail else detail
        print(f"[ERROR] LLM 服务不可用，请先启动 LM Studio Server，并检查 {suffix}")
        return 1

    try:
        result = run_query(
            index_dir=index_dir,
            raw_text=args.text,
            top_packages=args.top_packages,
            top_items=args.top_items,
            output=output_path,
            max_packages_per_cache_subject=args.max_packages_per_cache_subject,
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
