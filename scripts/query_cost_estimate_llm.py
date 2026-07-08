#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import unicodedata
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

from classifier.llm_client import LLMServiceError, check_lmstudio_service, request_llm_json, request_llm_json_with_usage  # noqa: E402
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

CANDIDATE_FAMILY_COLUMNS = [
    "family_id",
    "fine_signature",
    "representative_cost_item_name",
    "representative_project_description",
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
    "package_path_score最大值",
    "item_path_score最大值",
    "item_score最大值",
    "cooccur_score",
    "catalog_score",
    "final_score",
    "source_refs",
]

EVIDENCE_ITEM_COLUMNS = [
    "source_ref",
    "family_id",
    "fine_signature",
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
    "package_path_score",
    "item_path_score",
    "item_score",
    "cooccur_score",
    "catalog_score",
    "final_score",
]

SUGGESTED_BILL_COLUMNS = [
    "序号",
    "display_id",
    "清单名称",
    "单位",
    "默认参考做法",
    "selected_family_id",
    "其他历史做法",
    "历史family数量",
    "默认family历史样本数",
    "默认family来源工程包数",
    "工程量来源",
    "建议工程量最低值",
    "建议工程量中位数",
    "建议工程量最高值",
    "工程量依据",
    "是否计入参考金额区间",
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
    "估算金额中包含人工费最低值",
    "估算金额中包含人工费中位数",
    "估算金额中包含人工费最高值",
    "估算金额中包含机械费最低值",
    "估算金额中包含机械费中位数",
    "估算金额中包含机械费最高值",
    "金额计算口径",
    "推荐依据",
    "需确认事项",
    "来源样本",
]

CANDIDATE_DISPLAY_GROUP_COLUMNS = [
    "display_id",
    "display_key",
    "display_name",
    "unit",
    "family_count",
    "family_ids",
    "历史样本数合计",
    "来源工程包数去重",
    "final_score最大值",
    "item_score最大值",
    "cooccur_score最大值",
    "catalog_score最大值",
    "top_family_examples",
]

DISPLAY_GROUP_FAMILY_COLUMNS = [
    "display_id",
    "display_key",
    "display_name",
    "family_id",
    "fine_signature",
    "representative_cost_item_name",
    "representative_project_description",
    "unit",
    "历史样本数",
    "来源工程包数",
    "final_score",
    "item_score最大值",
    "cooccur_score",
    "catalog_score",
]

DISPLAY_SELECTION_TRACE_COLUMNS = [
    "selection_rank",
    "display_id",
    "display_key",
    "display_name",
    "unit",
    "family_count",
    "family_ids",
    "历史样本数合计",
    "来源工程包数去重",
    "final_score最大值",
    "item_score最大值",
    "cooccur_score最大值",
    "candidate_source",
    "selected_by_llm",
    "selection_reason",
    "selection_source",
]

FAMILY_SELECTION_TRACE_COLUMNS = [
    "selection_rank",
    "family_id",
    "fine_signature",
    "representative_cost_item_name",
    "representative_project_description",
    "unit",
    "历史样本数",
    "来源工程包数",
    "final_score",
    "item_score最大值",
    "cooccur_score",
    "candidate_source",
    "selected_by_llm",
    "selection_reason",
    "selection_source",
]

DISPLAY_FAMILY_SELECTION_TRACE_COLUMNS = [
    "display_id",
    "display_name",
    "family_id",
    "fine_signature",
    "representative_cost_item_name",
    "representative_project_description",
    "display_project_description",
    "历史样本数",
    "来源工程包数",
    "final_score",
    "item_score最大值",
    "unit_price_min",
    "unit_price_median",
    "unit_price_max",
    "is_selected_family",
    "is_other_practice",
    "other_practice_difference",
    "selection_reason",
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
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
]

ALLOWED_QUANTITY_SOURCES = {
    "用户明确给定",
    "历史样本估算",
    "需现场确认",
}


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
    candidate_display_groups: pd.DataFrame
    display_group_families: pd.DataFrame
    display_selection_trace: pd.DataFrame
    display_family_selection_trace: pd.DataFrame
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
    parser.add_argument("--family-selection-limit", type=int, default=50, help="发送给 display_selection LLM 的 display 上限，默认 50")
    parser.add_argument("--family-exploration-limit", type=int, default=5, help="display_selection 等距探索候选上限，默认 5")
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


def normalize_display_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", cell_text(value)).lower()
    replacements = {
        "，": ",",
        "。": ".",
        "；": ";",
        "：": ":",
        "（": "(",
        "）": ")",
        "【": "[",
        "】": "]",
        "、": ",",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" \t\r\n,.;:!?，。；：、()[]【】")
    return text


def normalize_display_description(value: Any) -> str:
    text = cell_text(value)
    if not text:
        return ""
    return re.sub(r"^1\.(\d+\.\d+\s*mm)", r"\1", text, count=1, flags=re.IGNORECASE)


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
    usage: dict[str, Any] | None = None,
) -> dict[str, Any]:
    usage = usage or {}
    return {
        "step": step,
        "purpose": purpose,
        "success": "是" if success else "否",
        "error": error,
        "prompt_chars": len(prompt),
        "estimated_tokens": estimated_tokens(prompt) if prompt else "",
        "max_tokens": max_tokens,
        "input_summary": input_summary,
        "prompt_tokens": usage.get("prompt_tokens", ""),
        "completion_tokens": usage.get("completion_tokens", ""),
        "total_tokens": usage.get("total_tokens", ""),
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

    weighted_comparisons = [
        ("一级分类", query_catalog.一级分类, 0.20),
        ("二级分类", query_catalog.二级分类, 0.30),
        ("维修状态", query_catalog.维修状态, 0.15),
        ("标准对象", query_catalog.标准对象, 0.15),
    ]
    score = 0.5
    compared = False
    for column, expected, weight in weighted_comparisons:
        if not expected:
            continue
        compared = True
        actual = cell_text(row.get(column))
        if not actual:
            continue
        elif actual == expected:
            score += weight
        else:
            score -= weight * 0.75
    if not compared:
        return 0.5
    return round(max(0.0, min(0.95, score)), 4)


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
    if matched.empty or "fine_signature" not in matched.columns:
        return {}
    counts = matched.groupby("fine_signature", dropna=False)["project_package_id"].nunique()
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
    rows["cooccur_score"] = rows["fine_signature"].map(lambda signature: family_cooccur.get(cell_text(signature), 0.0))
    rows["catalog_score"] = rows.apply(lambda row: catalog_score(row, query_catalog), axis=1)
    rows["unit_score"] = rows.apply(unit_score, axis=1)
    rows["direct_hit"] = rows["sample_index"].isin(direct_indices)
    rows["package_path_score"] = (
        0.40 * rows["package_score"]
        + 0.25 * rows["cooccur_score"]
        + 0.20 * rows["item_score"]
        + 0.10 * rows["catalog_score"]
        + 0.05 * rows["unit_score"]
    )
    rows["item_path_score"] = 0.70 * rows["item_score"] + 0.20 * rows["catalog_score"] + 0.10 * rows["unit_score"]
    rows["final_score"] = rows[["package_path_score", "item_path_score"]].max(axis=1)
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


def build_candidate_families(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=CANDIDATE_FAMILY_COLUMNS)

    rows: list[dict[str, Any]] = []
    for fine_signature, group in candidates.groupby("fine_signature", sort=False, dropna=False):
        group = group.sort_values(["final_score", "item_score", "package_score"], ascending=[False, False, False])
        representative = group.iloc[0]
        quantity_min, quantity_median, quantity_max = min_median_max(group, "quantity")
        unit_price_min, unit_price_median, unit_price_max = min_median_max(group, "unit_price")
        total_price_min, total_price_median, total_price_max = min_median_max(group, "total_price")
        labor_min, labor_median, labor_max = min_median_max(group, "labor_unit_price")
        machinery_min, machinery_median, machinery_max = min_median_max(group, "machinery_unit_price")
        rows.append(
            {
                "fine_signature": cell_text(fine_signature),
                "representative_cost_item_name": cell_text(representative.get("cost_item_name")),
                "representative_project_description": cell_text(representative.get("project_description")),
                "unit": cell_text(representative.get("unit")),
                "unit_normalized": cell_text(representative.get("unit_normalized")) or cell_text(representative.get("unit")),
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
                "package_path_score最大值": max_numeric_or_zero(group, "package_path_score"),
                "item_path_score最大值": max_numeric_or_zero(group, "item_path_score"),
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


def build_evidence_items(candidates: pd.DataFrame, candidate_families: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=EVIDENCE_ITEM_COLUMNS)
    signature_to_family_id = {
        cell_text(row.get("fine_signature")): cell_text(row.get("family_id"))
        for _index, row in candidate_families.iterrows()
        if cell_text(row.get("fine_signature"))
    }
    fine_signatures = candidates.get("fine_signature", pd.Series([""] * len(candidates), index=candidates.index))
    output = pd.DataFrame(
        {
            "source_ref": candidates.get("source_ref", ""),
            "family_id": fine_signatures.map(lambda value: signature_to_family_id.get(cell_text(value), "")),
            "fine_signature": fine_signatures,
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
            "package_path_score": candidates.get("package_path_score", ""),
            "item_path_score": candidates.get("item_path_score", ""),
            "item_score": candidates.get("item_score", ""),
            "cooccur_score": candidates.get("cooccur_score", ""),
            "catalog_score": candidates.get("catalog_score", ""),
            "final_score": candidates.get("final_score", ""),
        }
    )
    for column in EVIDENCE_ITEM_COLUMNS:
        if column not in output.columns:
            output[column] = ""
    output = output[EVIDENCE_ITEM_COLUMNS].where(pd.notna(output), "")
    if len(candidates) != len(output):
        raise ValueError("evidence_items 行数与 candidate pool 不一致")
    missing_family_ids = output["family_id"].map(cell_text).eq("")
    if missing_family_ids.any():
        raise ValueError(f"存在 {int(missing_family_ids.sum())} 条 evidence item 未映射到 family_id")
    return output


def display_unit_for_family(row: pd.Series) -> str:
    return cell_text(row.get("unit_normalized")) or normalized_unit(row.get("unit"))


def top_family_examples(group: pd.DataFrame, limit: int = 3) -> list[dict[str, Any]]:
    if group.empty:
        return []
    ordered = group.sort_values(
        ["item_score最大值", "历史样本数", "来源工程包数", "final_score"],
        ascending=[False, False, False, False],
    )
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for _index, row in ordered.iterrows():
        key = (
            normalize_display_description(row.get("representative_project_description")),
            cell_text(row.get("family_id")),
        )
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            {
                "family_id": cell_text(row.get("family_id")),
                "项目特征简述": truncate_text(normalize_display_description(row.get("representative_project_description")), 60),
                "samples": int(row.get("历史样本数") or 0),
                "packages": int(row.get("来源工程包数") or 0),
            }
        )
        if len(rows) >= limit:
            break
    return rows


def build_candidate_display_groups(
    candidate_families: pd.DataFrame,
    evidence_items: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidate_families.empty:
        return (
            pd.DataFrame(columns=CANDIDATE_DISPLAY_GROUP_COLUMNS),
            pd.DataFrame(columns=DISPLAY_GROUP_FAMILY_COLUMNS),
        )

    families = candidate_families.copy()
    families["_display_name_normalized"] = families["representative_cost_item_name"].map(normalize_display_name)
    families["_display_unit_normalized"] = families.apply(display_unit_for_family, axis=1).map(cell_text)
    families["_display_key"] = families["_display_name_normalized"] + "|" + families["_display_unit_normalized"]

    group_rows: list[dict[str, Any]] = []
    grouped_family_rows: list[dict[str, Any]] = []
    display_id_by_key: dict[str, str] = {}
    raw_groups: list[tuple[dict[str, Any], pd.DataFrame]] = []
    for display_key, group in families.groupby("_display_key", sort=False, dropna=False):
        group = group.sort_values(["final_score", "item_score最大值", "历史样本数"], ascending=[False, False, False])
        representative = group.iloc[0]
        family_ids = [cell_text(value) for value in group["family_id"].tolist() if cell_text(value)]
        evidence = evidence_items[evidence_items.get("family_id", pd.Series(dtype=object)).map(cell_text).isin(family_ids)].copy()
        package_values = evidence.get("project_package_id", pd.Series(dtype=object)).map(cell_text)
        package_values = package_values[package_values.ne("")]
        if package_values.empty:
            package_values = evidence.get("project_key", pd.Series(dtype=object)).map(cell_text)
            package_values = package_values[package_values.ne("")]
        raw_groups.append(
            (
                {
                    "display_key": cell_text(display_key),
                    "display_name": cell_text(representative.get("representative_cost_item_name")),
                    "unit": display_unit_for_family(representative),
                    "family_count": int(len(group)),
                    "family_ids": ",".join(family_ids),
                    "历史样本数合计": int(pd.to_numeric(group.get("历史样本数", pd.Series(dtype=object)), errors="coerce").fillna(0).sum()),
                    "来源工程包数去重": int(package_values.nunique()),
                    "final_score最大值": max_numeric_or_zero(group, "final_score"),
                    "item_score最大值": max_numeric_or_zero(group, "item_score最大值"),
                    "cooccur_score最大值": max_numeric_or_zero(group, "cooccur_score"),
                    "catalog_score最大值": max_numeric_or_zero(group, "catalog_score"),
                    "top_family_examples": json_text(top_family_examples(group)),
                },
                group,
            )
        )

    raw_groups.sort(
        key=lambda item: (
            -float(item[0].get("final_score最大值") or 0.0),
            -float(item[0].get("item_score最大值") or 0.0),
            -int(item[0].get("历史样本数合计") or 0),
            cell_text(item[0].get("display_name")),
        )
    )
    for index, (row, _group) in enumerate(raw_groups, start=1):
        display_id = f"D{index:03d}"
        row["display_id"] = display_id
        display_id_by_key[cell_text(row.get("display_key"))] = display_id
        group_rows.append(row)

    group_by_key = {cell_text(row.get("display_key")): group for row, group in raw_groups}
    for display_key, display_id in display_id_by_key.items():
        group = group_by_key.get(display_key, pd.DataFrame())
        display_row = next(row for row in group_rows if cell_text(row.get("display_key")) == display_key)
        for _index, family in group.iterrows():
            grouped_family_rows.append(
                {
                    "display_id": display_id,
                    "display_key": display_key,
                    "display_name": cell_text(display_row.get("display_name")),
                    "family_id": cell_text(family.get("family_id")),
                    "fine_signature": cell_text(family.get("fine_signature")),
                    "representative_cost_item_name": cell_text(family.get("representative_cost_item_name")),
                    "representative_project_description": cell_text(family.get("representative_project_description")),
                    "unit": display_unit_for_family(family),
                    "历史样本数": family.get("历史样本数", ""),
                    "来源工程包数": family.get("来源工程包数", ""),
                    "final_score": family.get("final_score", ""),
                    "item_score最大值": family.get("item_score最大值", ""),
                    "cooccur_score": family.get("cooccur_score", ""),
                    "catalog_score": family.get("catalog_score", ""),
                }
            )

    display_groups = pd.DataFrame(group_rows)
    display_families = pd.DataFrame(grouped_family_rows)
    for column in CANDIDATE_DISPLAY_GROUP_COLUMNS:
        if column not in display_groups.columns:
            display_groups[column] = ""
    for column in DISPLAY_GROUP_FAMILY_COLUMNS:
        if column not in display_families.columns:
            display_families[column] = ""
    missing_display = set(candidate_families["family_id"].map(cell_text)) - set(display_families["family_id"].map(cell_text))
    if missing_display:
        raise ValueError(f"存在 candidate family 未映射到 display_id: {join_non_empty(sorted(missing_display))}")
    return (
        display_groups[CANDIDATE_DISPLAY_GROUP_COLUMNS].reset_index(drop=True),
        display_families[DISPLAY_GROUP_FAMILY_COLUMNS].reset_index(drop=True),
    )


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


def select_displays_for_llm(
    candidate_display_groups: pd.DataFrame,
    display_selection_limit: int = 50,
    exploration_limit: int = 5,
) -> pd.DataFrame:
    if candidate_display_groups.empty or display_selection_limit <= 0:
        return candidate_display_groups.head(0).copy()

    display_selection_limit = max(int(display_selection_limit), 0)
    exploration_limit = max(int(exploration_limit), 0)
    if len(candidate_display_groups) <= display_selection_limit:
        output = candidate_display_groups.copy().reset_index(drop=True)
        output.insert(0, "selection_rank", range(1, len(output) + 1))
        output["candidate_source"] = "all_under_limit"
        return output

    ranking_limit = max(display_selection_limit - min(exploration_limit, display_selection_limit), 0)
    rankings = [
        (
            "final_score",
            candidate_display_groups.sort_values(
                ["final_score最大值", "item_score最大值", "历史样本数合计"],
                ascending=[False, False, False],
            ),
        ),
        (
            "item_score",
            candidate_display_groups.sort_values(
                ["item_score最大值", "final_score最大值", "历史样本数合计"],
                ascending=[False, False, False],
            ),
        ),
        (
            "cooccur_score",
            candidate_display_groups.sort_values(
                ["cooccur_score最大值", "final_score最大值", "历史样本数合计"],
                ascending=[False, False, False],
            ),
        ),
        (
            "source_package_count",
            candidate_display_groups.sort_values(
                ["来源工程包数去重", "历史样本数合计", "final_score最大值"],
                ascending=[False, False, False],
            ),
        ),
    ]
    display_rows = {cell_text(row.get("display_id")): row for _index, row in candidate_display_groups.iterrows()}
    selected_ids: list[str] = []
    selected_set: set[str] = set()
    selected_sources: dict[str, list[str]] = {}
    positions = {name: 0 for name, _frame in rankings}

    while len(selected_ids) < ranking_limit:
        advanced = False
        for source_name, ranked in rankings:
            if len(selected_ids) >= ranking_limit:
                break
            while positions[source_name] < len(ranked):
                advanced = True
                row = ranked.iloc[positions[source_name]]
                positions[source_name] += 1
                display_id = cell_text(row.get("display_id"))
                if not display_id:
                    continue
                if display_id in selected_set:
                    sources = selected_sources.setdefault(display_id, [])
                    if source_name not in sources:
                        sources.append(source_name)
                    continue
                selected_set.add(display_id)
                selected_ids.append(display_id)
                selected_sources[display_id] = [source_name]
                break
        if not advanced:
            break

    remaining = candidate_display_groups[~candidate_display_groups["display_id"].map(cell_text).isin(selected_set)].reset_index(drop=True)
    exploration_count = min(exploration_limit, display_selection_limit - len(selected_ids), len(remaining))
    if exploration_count > 0:
        indices = np.linspace(0, len(remaining) - 1, num=exploration_count, dtype=int).tolist()
        indices = list(dict.fromkeys(indices))
        used_indices = set(indices)
        for index in range(len(remaining)):
            if len(indices) >= exploration_count:
                break
            if index not in used_indices:
                indices.append(index)
                used_indices.add(index)
        for index in indices[:exploration_count]:
            display_id = cell_text(remaining.iloc[index].get("display_id"))
            if not display_id or display_id in selected_set:
                continue
            selected_set.add(display_id)
            selected_ids.append(display_id)
            selected_sources[display_id] = ["exploration"]

    output = pd.DataFrame([display_rows[display_id] for display_id in selected_ids], columns=candidate_display_groups.columns).reset_index(drop=True)
    output.insert(0, "selection_rank", range(1, len(output) + 1))
    output["candidate_source"] = [
        ",".join(selected_sources.get(cell_text(row.get("display_id")), []))
        for _index, row in output.iterrows()
    ]
    return output


def display_selection_records(candidate_displays: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    frame = candidate_displays if limit is None else candidate_displays.head(limit)
    for _index, row in frame.iterrows():
        examples = []
        try:
            raw_examples = json.loads(cell_text(row.get("top_family_examples")) or "[]")
        except ValueError:
            raw_examples = []
        if isinstance(raw_examples, list):
            for item in raw_examples[:3]:
                if not isinstance(item, dict):
                    continue
                examples.append(
                    {
                        "family_id": cell_text(item.get("family_id")),
                        "spec": truncate_text(item.get("项目特征简述") or item.get("spec"), 60),
                        "samples": int(item.get("samples") or 0),
                        "packages": int(item.get("packages") or 0),
                    }
                )
        rows.append(
            {
                "id": cell_text(row.get("display_id")),
                "name": truncate_text(row.get("display_name"), 40),
                "unit": cell_text(row.get("unit")),
                "families": int(row.get("family_count") or 0),
                "samples": int(row.get("历史样本数合计") or 0),
                "packages": int(row.get("来源工程包数去重") or 0),
                "examples": examples,
                "item_score": round(float(row.get("item_score最大值") or 0.0), 4),
                "cooccur_score": round(float(row.get("cooccur_score最大值") or 0.0), 4),
            }
        )
    return rows


def build_display_selection_prompt(
    rewrite: QueryRewrite,
    query_catalog: QueryCatalog,
    candidate_displays: pd.DataFrame,
) -> tuple[str, list[dict[str, Any]]]:
    records = display_selection_records(candidate_displays)
    payload = {
        "raw_query": rewrite.raw_query,
        "parsed_query": parsed_query_dict(rewrite),
        "query_catalog": query_catalog_dict(query_catalog),
        "candidate_displays": records,
    }
    prompt = f"""
你是物业维修工程建议清单选择器。

【任务】

根据用户需求和历史候选证据，从 candidate_displays 中选择应进入本次建议清单的工作项。

每个 display 代表一个清单工作项，内部可能包含多个不同历史做法。
本阶段只判断哪些 display 应进入建议清单：

- 不选择具体 family；
- 不判断价格和工程量；
- 不撰写最终方案总结。

请综合判断：

1. 用户明确描述的维修对象、问题、材料、规格和工程量；
2. display 与用户维修目标的直接相关性；
3. display 在相似历史工程中的出现情况；
4. display 与其他拟选工作项之间是否存在合理的施工或配套关系；
5. 该工作项是否可能因现场条件、原有构造、施工组织或实施方案而需要。

用户通常只描述维修目标，不会完整列出实际工程中的全部清单工作项。
因此，不要只选择与用户原文措辞最相似的项目。

但历史工程中出现过，也不代表当前工程一定需要。
只有当候选与本次需求存在清楚、可解释的关系时，才应选择。

【选择规则】

1. 只能选择 candidate_displays 中存在的 display_id。
2. 同一 display_id 最多选择一次。
3. 不设置固定选择数量。
4. 不得为了凑数量选择无关工作项。
5. 不得创造候选中不存在的工作项。
6. 不得仅凭一般施工常识补充缺乏候选证据的工作项。
7. 不得仅因为现场条件尚未明确，就自动排除一个与当前工程有较强关系的候选。
8. 如果某个工作项是否实施依赖现场条件，可以选择，但 selection_reason 必须说明需要确认的条件。
9. 对实质重复、相互包含或通常互为替代的 display，不要同时选择，除非它们确实代表可以独立计价且同时实施的不同工作内容。
10. 用户已明确材料、设备或工艺时，不应选择与其明显冲突的替代工作项。
11. 不输出价格、工程量，也不自行改写清单名称。

【selection_reason】

每个已选 display 的 selection_reason 必须说明：

1. 它为什么与用户需求相关；
2. 它与用户明确需求或其他拟选工作项之间有什么关系；
3. 如果是否实施依赖现场条件，需要确认什么条件。

不要只写“历史中常见”“与需求相关”等空泛理由。

【输出 JSON】

{{
  "selected_displays": [
    {{
      "display_id": "D001",
      "selection_reason": "该工作项直接对应用户提出的维修对象和施工目标"
    }}
  ]
}}

只能输出一个 JSON object，不得输出解释、Markdown 或思考过程。

【输入数据】

{json_text(payload)}
""".strip()
    return prompt, records


def build_display_selection_trace_frame(displays_for_llm: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fallback_rank, (_index, row) in enumerate(displays_for_llm.iterrows(), start=1):
        rows.append(
            {
                "selection_rank": int(row.get("selection_rank") or fallback_rank),
                "display_id": cell_text(row.get("display_id")),
                "display_key": cell_text(row.get("display_key")),
                "display_name": cell_text(row.get("display_name")),
                "unit": cell_text(row.get("unit")),
                "family_count": row.get("family_count", ""),
                "family_ids": cell_text(row.get("family_ids")),
                "历史样本数合计": row.get("历史样本数合计", ""),
                "来源工程包数去重": row.get("来源工程包数去重", ""),
                "final_score最大值": row.get("final_score最大值", ""),
                "item_score最大值": row.get("item_score最大值", ""),
                "cooccur_score最大值": row.get("cooccur_score最大值", ""),
                "candidate_source": cell_text(row.get("candidate_source")),
                "selected_by_llm": "否",
                "selection_reason": "",
                "selection_source": "not_selected",
            }
        )
    return pd.DataFrame(rows, columns=DISPLAY_SELECTION_TRACE_COLUMNS)


def apply_display_selection_trace_result(
    trace_frame: pd.DataFrame,
    selected: pd.DataFrame,
    selection_source: str,
) -> pd.DataFrame:
    output = trace_frame.copy()
    if output.empty or selected.empty:
        return output
    selected_map = {cell_text(row.get("display_id")): row for _index, row in selected.iterrows()}
    for index, row in output.iterrows():
        display_id = cell_text(row.get("display_id"))
        selected_row = selected_map.get(display_id)
        if selected_row is None:
            continue
        output.at[index, "selected_by_llm"] = "是" if selection_source == "llm" else "否"
        output.at[index, "selection_reason"] = cell_text(selected_row.get("selection_reason"))
        output.at[index, "selection_source"] = selection_source
    return output


def display_selection_candidate_source_counts(trace_frame: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    if trace_frame.empty or "candidate_source" not in trace_frame.columns:
        return counts
    for value in trace_frame["candidate_source"].tolist():
        for source in split_refs(value):
            counts[source] = counts.get(source, 0) + 1
    return counts


def parse_display_selection_result(
    result: dict[str, Any],
    allowed_display_ids: set[str],
    warnings: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_rows = result.get("selected_displays")
    if not isinstance(raw_rows, list):
        raise ValueError("LLM 输出缺少 selected_displays list")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    meta = {"invalid_display_ids": [], "duplicate_display_ids": []}
    for item in raw_rows:
        if not isinstance(item, dict):
            continue
        display_id = cell_text(item.get("display_id"))
        if display_id not in allowed_display_ids:
            if display_id:
                meta["invalid_display_ids"].append(display_id)
                append_warning(warnings, "invalid_display_ids")
            continue
        if display_id in seen:
            meta["duplicate_display_ids"].append(display_id)
            append_warning(warnings, "duplicate_display_ids")
            continue
        seen.add(display_id)
        rows.append(
            {
                "display_id": display_id,
                "selection_reason": cell_text(item.get("selection_reason")),
            }
        )
    return pd.DataFrame(rows, columns=["display_id", "selection_reason"]), meta


def generate_display_selection(
    rewrite: QueryRewrite,
    query_catalog: QueryCatalog,
    candidate_display_groups: pd.DataFrame,
    display_selection_limit: int = 50,
    exploration_limit: int = 5,
    warnings: list[str] | None = None,
) -> tuple[pd.DataFrame, bool, bool, str, str, dict[str, Any], dict[str, Any], pd.DataFrame]:
    displays_for_llm = select_displays_for_llm(
        candidate_display_groups,
        display_selection_limit=display_selection_limit,
        exploration_limit=exploration_limit,
    )
    trace_frame = build_display_selection_trace_frame(displays_for_llm)
    prompt, records = build_display_selection_prompt(rewrite, query_catalog, displays_for_llm)
    allowed_display_ids = {cell_text(row.get("id")) for row in records}
    candidate_ids = [cell_text(row.get("id")) for row in records if cell_text(row.get("id"))]
    source_counts = display_selection_candidate_source_counts(trace_frame)
    exploration_count = source_counts.get("exploration", 0)
    max_tokens = 2048
    try:
        response = request_llm_json_with_usage(
            prompt,
            max_tokens=max_tokens,
            system_prompt="你只输出一个 JSON object，不输出解释、Markdown 或思考过程。",
        )
        selected, meta = parse_display_selection_result(response.content, allowed_display_ids, warnings)
        selected_ids = selected["display_id"].map(cell_text).tolist() if "display_id" in selected.columns else []
        selected_detail = replace_nan_records(selected)
        meta.update(
            {
                "candidate_ids": candidate_ids,
                "selected_ids": selected_ids,
                "selected_detail": selected_detail,
                "candidate_source_counts": source_counts,
                "exploration_count": exploration_count,
            }
        )
        trace_frame = apply_display_selection_trace_result(trace_frame, selected, "llm")
        trace = trace_row(
            "display_selection",
            "选择进入建议清单的 display_group",
            True,
            prompt=prompt,
            max_tokens=max_tokens,
            input_summary=json_text(
                {
                    "displays_sent": len(records),
                    "selected": len(selected),
                    "candidate_ids": trace_id_summary(candidate_ids),
                    "selected_ids": trace_id_summary(selected_ids),
                    "exploration_count": exploration_count,
                }
            ),
            usage=response.usage,
        )
        return selected, True, False, "", prompt, trace, meta, trace_frame
    except (LLMServiceError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        append_warning(warnings, "display_selection_failed")
        if not trace_frame.empty:
            trace_frame = trace_frame.copy()
            trace_frame["selection_source"] = "fallback"
        meta = {
            "invalid_display_ids": [],
            "duplicate_display_ids": [],
            "candidate_ids": candidate_ids,
            "selected_ids": [],
            "selected_detail": [],
            "candidate_source_counts": source_counts,
            "exploration_count": exploration_count,
        }
        trace = trace_row(
            "display_selection",
            "选择进入建议清单的 display_group",
            False,
            error=str(exc),
            prompt=prompt,
            max_tokens=max_tokens,
            input_summary=json_text(
                {
                    "displays_sent": len(records),
                    "selected": 0,
                    "candidate_ids": trace_id_summary(candidate_ids),
                    "selected_ids": [],
                    "exploration_count": exploration_count,
                }
            ),
        )
        return pd.DataFrame(columns=["display_id", "selection_reason"]), False, True, str(exc), prompt, trace, meta, trace_frame


def select_families_for_llm(
    candidate_families: pd.DataFrame,
    family_selection_limit: int = 50,
    exploration_limit: int = 5,
) -> pd.DataFrame:
    if candidate_families.empty or family_selection_limit <= 0:
        return candidate_families.head(0).copy()

    family_selection_limit = max(int(family_selection_limit), 0)
    exploration_limit = max(int(exploration_limit), 0)
    ranking_limit = max(family_selection_limit - min(exploration_limit, family_selection_limit), 0)

    final_ranked = candidate_families.sort_values(
        ["final_score", "item_score最大值", "历史样本数"],
        ascending=[False, False, False],
    )
    item_ranked = candidate_families.sort_values(
        ["item_score最大值", "final_score", "历史样本数"],
        ascending=[False, False, False],
    )
    cooccur_ranked = candidate_families.sort_values(
        ["cooccur_score", "final_score", "历史样本数"],
        ascending=[False, False, False],
    )
    source_ranked = candidate_families.sort_values(
        ["来源工程包数", "历史样本数", "final_score"],
        ascending=[False, False, False],
    )
    rankings = [
        ("final_score", final_ranked),
        ("item_score", item_ranked),
        ("cooccur_score", cooccur_ranked),
        ("source_package_count", source_ranked),
    ]
    family_rows = {cell_text(row.get("family_id")): row for _index, row in candidate_families.iterrows()}

    selected_ids: list[str] = []
    selected_set: set[str] = set()
    selected_sources: dict[str, list[str]] = {}
    positions = {name: 0 for name, _frame in rankings}

    while len(selected_ids) < ranking_limit:
        advanced = False
        for source_name, ranked in rankings:
            if len(selected_ids) >= ranking_limit:
                break
            while positions[source_name] < len(ranked):
                advanced = True
                row = ranked.iloc[positions[source_name]]
                positions[source_name] += 1
                family_id = cell_text(row.get("family_id"))
                if not family_id:
                    continue
                if family_id in selected_set:
                    sources = selected_sources.setdefault(family_id, [])
                    if source_name not in sources:
                        sources.append(source_name)
                    continue
                selected_set.add(family_id)
                selected_ids.append(family_id)
                selected_sources[family_id] = [source_name]
                break
        if not advanced:
            break

    remaining = candidate_families[~candidate_families["family_id"].map(cell_text).isin(selected_set)].reset_index(drop=True)
    exploration_count = min(exploration_limit, family_selection_limit - len(selected_ids), len(remaining))
    if exploration_count > 0:
        indices = np.linspace(0, len(remaining) - 1, num=exploration_count, dtype=int).tolist()
        indices = list(dict.fromkeys(indices))
        used_indices = set(indices)
        for index in range(len(remaining)):
            if len(indices) >= exploration_count:
                break
            if index not in used_indices:
                indices.append(index)
                used_indices.add(index)
        for index in indices[:exploration_count]:
            family_id = cell_text(remaining.iloc[index].get("family_id"))
            if not family_id or family_id in selected_set:
                continue
            selected_set.add(family_id)
            selected_ids.append(family_id)
            selected_sources[family_id] = ["exploration"]

    output = pd.DataFrame([family_rows[family_id] for family_id in selected_ids], columns=candidate_families.columns).reset_index(drop=True)
    output.insert(0, "selection_rank", range(1, len(output) + 1))
    output["candidate_source"] = [
        ",".join(selected_sources.get(cell_text(row.get("family_id")), []))
        for _index, row in output.iterrows()
    ]
    return output


def family_selection_records(candidate_families: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    frame = candidate_families if limit is None else candidate_families.head(limit)
    for _index, row in frame.iterrows():
        rows.append(
            {
                "id": cell_text(row.get("family_id")),
                "name": truncate_text(row.get("representative_cost_item_name"), 40),
                "spec": truncate_text(row.get("representative_project_description"), 80),
                "unit": cell_text(row.get("unit_normalized")) or cell_text(row.get("unit")),
                "samples": int(row.get("历史样本数") or 0),
                "packages": int(row.get("来源工程包数") or 0),
                "item_score": round(float(row.get("item_score最大值") or 0.0), 4),
                "cooccur_score": round(float(row.get("cooccur_score") or 0.0), 4),
            }
        )
    return rows


def build_family_selection_prompt(
    rewrite: QueryRewrite,
    query_catalog: QueryCatalog,
    candidate_families: pd.DataFrame,
) -> tuple[str, list[dict[str, Any]]]:
    records = family_selection_records(candidate_families)
    payload = {
        "raw_query": rewrite.raw_query,
        "parsed_query": parsed_query_dict(rewrite),
        "query_catalog": query_catalog_dict(query_catalog),
        "candidate_families": records,
    }
    prompt = f"""
你是物业维修工程建议清单选择器。

【任务】

根据用户需求，从 candidate_families 中选择应进入建议清单的历史施工做法。

每个 family 代表一种历史做法。本阶段只判断哪些 family 与本次需求存在清楚、可解释的关系。

【选择规则】

1. 只能选择 candidate_families 中存在的 family_id，同一 family_id 最多选择一次。
2. 不设置固定选择数量，不得为了凑数量选择无关项。
3. 不得创造候选中不存在的项目。
4. 用户已明确材料、设备或工艺时，不应选择与其明显冲突的替代做法。
5. 实质相同、仅措辞不同的 family 不要重复选择；
   存在明确施工边界差异的项目可以分别选择。
6. 不输出价格、工程量或自行改写清单内容。
7. 只能输出一个 JSON object，不得输出解释、Markdown 或思考过程。

【输出 JSON】

{{
  "selected_families": [
    {{
      "family_id": "F001",
      "selection_reason": "该施工做法与用户提出的维修对象和施工目标直接相关"
    }}
  ]
}}

【输入数据】

{json_text(payload)}
""".strip()
    return prompt, records


def build_family_selection_trace_frame(families_for_llm: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for fallback_rank, (_index, row) in enumerate(families_for_llm.iterrows(), start=1):
        rows.append(
            {
                "selection_rank": int(row.get("selection_rank") or fallback_rank),
                "family_id": cell_text(row.get("family_id")),
                "fine_signature": cell_text(row.get("fine_signature")),
                "representative_cost_item_name": cell_text(row.get("representative_cost_item_name")),
                "representative_project_description": cell_text(row.get("representative_project_description")),
                "unit": cell_text(row.get("unit_normalized")) or cell_text(row.get("unit")),
                "历史样本数": row.get("历史样本数", ""),
                "来源工程包数": row.get("来源工程包数", ""),
                "final_score": row.get("final_score", ""),
                "item_score最大值": row.get("item_score最大值", ""),
                "cooccur_score": row.get("cooccur_score", ""),
                "candidate_source": cell_text(row.get("candidate_source")),
                "selected_by_llm": "否",
                "selection_reason": "",
                "selection_source": "not_selected",
            }
        )
    return pd.DataFrame(rows, columns=FAMILY_SELECTION_TRACE_COLUMNS)


def apply_family_selection_trace_result(
    trace_frame: pd.DataFrame,
    selected: pd.DataFrame,
    selection_source: str,
) -> pd.DataFrame:
    output = trace_frame.copy()
    if output.empty or selected.empty:
        return output
    selected_map = {cell_text(row.get("family_id")): row for _index, row in selected.iterrows()}
    for index, row in output.iterrows():
        family_id = cell_text(row.get("family_id"))
        selected_row = selected_map.get(family_id)
        if selected_row is None:
            continue
        output.at[index, "selected_by_llm"] = "是" if selection_source == "llm" else "否"
        output.at[index, "selection_reason"] = cell_text(selected_row.get("selection_reason"))
        output.at[index, "selection_source"] = selection_source
    return output


def family_selection_candidate_source_counts(trace_frame: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    if trace_frame.empty or "candidate_source" not in trace_frame.columns:
        return counts
    for value in trace_frame["candidate_source"].tolist():
        for source in split_refs(value):
            counts[source] = counts.get(source, 0) + 1
    return counts


def trace_id_summary(ids: list[str], limit: int = 20) -> dict[str, Any]:
    if len(ids) <= limit:
        return {"ids": ids}
    return {"ids": ids[:limit], "total": len(ids)}


def append_trace_warnings(trace: dict[str, Any], warnings: list[str]) -> None:
    summary = cell_text(trace.get("input_summary"))
    if not summary:
        trace["input_summary"] = json_text({"warnings": warnings})
        return
    try:
        payload = json.loads(summary)
    except (TypeError, ValueError):
        trace["input_summary"] = f"{summary}; warnings={';'.join(warnings)}"
        return
    if isinstance(payload, dict):
        payload["warnings"] = warnings
        trace["input_summary"] = json_text(payload)
    else:
        trace["input_summary"] = f"{summary}; warnings={';'.join(warnings)}"


def parse_family_selection_result(
    result: dict[str, Any],
    allowed_family_ids: set[str],
    warnings: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_rows = result.get("selected_families")
    if not isinstance(raw_rows, list):
        raise ValueError("LLM 输出缺少 selected_families list")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    meta = {"invalid_family_ids": [], "duplicate_family_ids": []}
    for item in raw_rows:
        if not isinstance(item, dict):
            continue
        family_id = cell_text(item.get("family_id"))
        if family_id not in allowed_family_ids:
            if family_id:
                meta["invalid_family_ids"].append(family_id)
                append_warning(warnings, "invalid_family_ids")
            continue
        if family_id in seen:
            meta["duplicate_family_ids"].append(family_id)
            append_warning(warnings, "duplicate_family_ids")
            continue
        seen.add(family_id)
        rows.append(
            {
                "family_id": family_id,
                "selection_reason": cell_text(item.get("selection_reason")),
            }
        )
    return pd.DataFrame(rows, columns=["family_id", "selection_reason"]), meta


def generate_family_selection(
    rewrite: QueryRewrite,
    query_catalog: QueryCatalog,
    candidate_families: pd.DataFrame,
    family_selection_limit: int = 50,
    exploration_limit: int = 5,
    warnings: list[str] | None = None,
) -> tuple[pd.DataFrame, bool, bool, str, str, dict[str, Any], dict[str, Any], pd.DataFrame]:
    families_for_llm = select_families_for_llm(
        candidate_families,
        family_selection_limit=family_selection_limit,
        exploration_limit=exploration_limit,
    )
    trace_frame = build_family_selection_trace_frame(families_for_llm)
    prompt, records = build_family_selection_prompt(rewrite, query_catalog, families_for_llm)
    allowed_family_ids = {cell_text(row.get("id")) for row in records}
    candidate_ids = [cell_text(row.get("id")) for row in records if cell_text(row.get("id"))]
    source_counts = family_selection_candidate_source_counts(trace_frame)
    exploration_count = source_counts.get("exploration", 0)
    max_tokens = 2048
    try:
        response = request_llm_json_with_usage(
            prompt,
            max_tokens=max_tokens,
            system_prompt="你只输出一个 JSON object，不输出解释、Markdown 或思考过程。",
        )
        selected, meta = parse_family_selection_result(response.content, allowed_family_ids, warnings)
        selected_ids = selected["family_id"].map(cell_text).tolist() if "family_id" in selected.columns else []
        selected_detail = replace_nan_records(selected)
        meta.update(
            {
                "candidate_ids": candidate_ids,
                "selected_ids": selected_ids,
                "selected_detail": selected_detail,
                "candidate_source_counts": source_counts,
                "exploration_count": exploration_count,
            }
        )
        trace_frame = apply_family_selection_trace_result(trace_frame, selected, "llm")
        trace = trace_row(
            "family_selection",
            "选择相关 fine_signature family 并判断项目角色",
            True,
            prompt=prompt,
            max_tokens=max_tokens,
            input_summary=json_text(
                {
                    "families_sent": len(records),
                    "selected": len(selected),
                    "candidate_ids": trace_id_summary(candidate_ids),
                    "selected_ids": trace_id_summary(selected_ids),
                    "exploration_count": exploration_count,
                }
            ),
            usage=response.usage,
        )
        return selected, True, False, "", prompt, trace, meta, trace_frame
    except (LLMServiceError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        append_warning(warnings, "family_selection_failed")
        if not trace_frame.empty:
            trace_frame = trace_frame.copy()
            trace_frame["selection_source"] = "fallback"
        meta = {
            "invalid_family_ids": [],
            "duplicate_family_ids": [],
            "candidate_ids": candidate_ids,
            "selected_ids": [],
            "selected_detail": [],
            "candidate_source_counts": source_counts,
            "exploration_count": exploration_count,
        }
        trace = trace_row(
            "family_selection",
            "选择相关 fine_signature family 并判断项目角色",
            False,
            error=str(exc),
            prompt=prompt,
            max_tokens=max_tokens,
            input_summary=json_text(
                {
                    "families_sent": len(records),
                    "selected": 0,
                    "candidate_ids": trace_id_summary(candidate_ids),
                    "selected_ids": [],
                    "exploration_count": exploration_count,
                }
            ),
        )
        return pd.DataFrame(columns=["family_id", "selection_reason"]), False, True, str(exc), prompt, trace, meta, trace_frame


def family_selection_payload_for_display(
    display_id: str,
    display_group_families: pd.DataFrame,
    candidate_families: pd.DataFrame,
) -> list[dict[str, Any]]:
    family_map = {cell_text(row.get("family_id")): row for _index, row in candidate_families.iterrows()}
    rows = display_group_families[display_group_families["display_id"].map(cell_text).eq(display_id)].copy()
    rows = rows.sort_values(["item_score最大值", "历史样本数", "来源工程包数", "final_score"], ascending=[False, False, False, False])
    payload: list[dict[str, Any]] = []
    for _index, row in rows.iterrows():
        family_id = cell_text(row.get("family_id"))
        family = family_map.get(family_id)
        if family is None:
            continue
        payload.append(
            {
                "family_id": family_id,
                "name": truncate_text(row.get("representative_cost_item_name"), 40),
                "spec": truncate_text(normalize_display_description(row.get("representative_project_description")), 80),
                "samples": int(row.get("历史样本数") or 0),
                "packages": int(row.get("来源工程包数") or 0),
                "item_score": round(float(row.get("item_score最大值") or 0.0), 4),
                "final_score": round(float(row.get("final_score") or 0.0), 4),
                "unit_price_min": family.get("历史综合单价最低值"),
                "unit_price_median": family.get("历史综合单价中位数"),
                "unit_price_max": family.get("历史综合单价最高值"),
            }
        )
    return payload


def build_display_family_selection_prompt(
    rewrite: QueryRewrite,
    selected_displays: pd.DataFrame,
    candidate_display_groups: pd.DataFrame,
    display_group_families: pd.DataFrame,
    candidate_families: pd.DataFrame,
) -> tuple[str, list[dict[str, Any]]]:
    display_map = {cell_text(row.get("display_id")): row for _index, row in candidate_display_groups.iterrows()}
    records: list[dict[str, Any]] = []
    for _index, selected in selected_displays.iterrows():
        display_id = cell_text(selected.get("display_id"))
        display = display_map.get(display_id)
        if display is None:
            continue
        records.append(
            {
                "display_id": display_id,
                "display_name": truncate_text(display.get("display_name"), 40),
                "selection_reason": cell_text(selected.get("selection_reason")),
                "candidate_families": family_selection_payload_for_display(display_id, display_group_families, candidate_families),
            }
        )
    payload = {
        "raw_query": rewrite.raw_query,
        "parsed_query": parsed_query_dict(rewrite),
        "selected_displays": records,
    }
    prompt = f"""
你是物业维修工程历史做法选择器。

【任务】

对于每个 selected display，从其内部 candidate_families 中：

1. 选择一个 selected_family_id，作为默认参考做法和价格来源；
2. 选择 0～3 个 other_family_ids，作为其他历史做法；
3. 概括默认做法；
4. 说明其他历史做法与默认做法的主要差异。

【默认参考 family 选择原则】

1. 优先匹配用户明确提出的材料、规格、部位和工艺。
2. 优先选择描述清晰、没有明显 OCR 歧义的 family。
3. 优先选择施工边界适合作为默认参考口径的 family。
4. 历史样本数和来源工程包数越充分越优先。
5. 样本数最多是重要依据，但不能覆盖材料、规格或施工边界不匹配。

【其他历史做法】

1. 每个 display 最多选择 3 个。
2. 必须与默认 family 存在有意义的项目特征差异。
3. 仅空格、标点、3mm/3.0mm、序号或措辞差异，不属于其他历史做法。
4. 优先展示：
   - 施工边界差异；
   - 材料或工艺差异；
   - 厚度或规格差异；
   - 特殊性能差异。
5. difference 只概括主要差异，不复制整段项目特征。
6. 只能选择该 display 内存在的 family_id。

【输出 JSON】

{{
  "display_results": [
    {{
      "display_id": "D001",
      "selected_family_id": "F004",
      "default_practice": "与用户需求匹配的默认历史做法",
      "selection_reason": "与用户明确要求一致，描述清晰，历史样本可追溯",
      "other_practices": [
        {{
          "family_id": "F007",
          "difference": "施工边界与默认做法不同"
        }}
      ]
    }}
  ]
}}

【输入数据】

{json_text(payload)}
""".strip()
    return prompt, records


def fallback_family_for_display(display_id: str, display_group_families: pd.DataFrame) -> str:
    rows = display_group_families[display_group_families["display_id"].map(cell_text).eq(display_id)].copy()
    if rows.empty:
        return ""
    rows = rows.sort_values(["item_score最大值", "历史样本数", "来源工程包数", "final_score"], ascending=[False, False, False, False])
    return cell_text(rows.iloc[0].get("family_id"))


def parse_display_family_selection_result(
    result: dict[str, Any],
    selected_displays: pd.DataFrame,
    candidate_display_groups: pd.DataFrame,
    display_group_families: pd.DataFrame,
    warnings: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw_rows = result.get("display_results")
    if not isinstance(raw_rows, list):
        raise ValueError("display_family_selection 输出缺少 display_results list")

    display_ids = [cell_text(value) for value in selected_displays.get("display_id", pd.Series(dtype=object)).tolist()]
    allowed_by_display = {
        display_id: set(
            display_group_families[display_group_families["display_id"].map(cell_text).eq(display_id)]["family_id"].map(cell_text)
        )
        for display_id in display_ids
    }
    selected_by_display = {cell_text(row.get("display_id")): row for _index, row in selected_displays.iterrows()}
    display_map = {cell_text(row.get("display_id")): row for _index, row in candidate_display_groups.iterrows()}
    result_by_display: dict[str, dict[str, Any]] = {}
    meta = {
        "invalid_display_ids": [],
        "invalid_family_ids": [],
        "duplicate_display_ids": [],
        "selected_family_ids": [],
        "other_family_ids": [],
        "other_family_count": 0,
    }

    for item in raw_rows:
        if not isinstance(item, dict):
            continue
        display_id = cell_text(item.get("display_id"))
        if display_id not in allowed_by_display:
            meta["invalid_display_ids"].append(display_id)
            append_warning(warnings, "invalid_display_family_display_ids")
            continue
        if display_id in result_by_display:
            meta["duplicate_display_ids"].append(display_id)
            append_warning(warnings, "duplicate_display_family_display_ids")
            continue
        allowed_family_ids = allowed_by_display[display_id]
        selected_family_id = cell_text(item.get("selected_family_id"))
        if selected_family_id not in allowed_family_ids:
            raise ValueError(f"selected_family_id 不属于 display: {display_id}/{selected_family_id}")
        raw_other = item.get("other_practices", [])
        if not isinstance(raw_other, list):
            raise ValueError(f"other_practices 必须为 list: {display_id}")
        other_practices: list[dict[str, str]] = []
        seen_other: set[str] = set()
        for other in raw_other:
            if not isinstance(other, dict):
                continue
            family_id = cell_text(other.get("family_id"))
            if family_id not in allowed_family_ids:
                raise ValueError(f"other_family_id 不属于 display: {display_id}/{family_id}")
            if family_id == selected_family_id:
                raise ValueError(f"selected_family_id 不能同时出现在 other_family_ids: {display_id}/{family_id}")
            if family_id in seen_other:
                continue
            seen_other.add(family_id)
            other_practices.append({"family_id": family_id, "difference": cell_text(other.get("difference"))})
        if len(other_practices) > 3:
            raise ValueError(f"other_family_ids 最多 3 个: {display_id}")
        selected_row = selected_by_display[display_id]
        display_row = display_map.get(display_id, pd.Series(dtype=object))
        result_by_display[display_id] = {
            "display_id": display_id,
            "display_name": cell_text(display_row.get("display_name")),
            "unit": cell_text(display_row.get("unit")),
            "family_count": display_row.get("family_count", ""),
            "selection_reason": cell_text(selected_row.get("selection_reason")),
            "selected_family_id": selected_family_id,
            "default_practice": truncate_text(normalize_display_description(item.get("default_practice")), 120),
            "family_selection_reason": cell_text(item.get("selection_reason")),
            "other_practices": other_practices,
        }
        meta["selected_family_ids"].append(selected_family_id)
        meta["other_family_ids"].extend([other["family_id"] for other in other_practices])

    missing = [display_id for display_id in display_ids if display_id not in result_by_display]
    if missing:
        raise ValueError(f"每个 selected_display 必须有 selected_family_id: {join_non_empty(missing)}")
    meta["other_family_count"] = len(meta["other_family_ids"])
    return pd.DataFrame([result_by_display[display_id] for display_id in display_ids]), meta


def fallback_display_family_selection(
    selected_displays: pd.DataFrame,
    candidate_display_groups: pd.DataFrame,
    display_group_families: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    display_map = {cell_text(row.get("display_id")): row for _index, row in candidate_display_groups.iterrows()}
    family_map = {cell_text(row.get("family_id")): row for _index, row in display_group_families.iterrows()}
    rows: list[dict[str, Any]] = []
    selected_family_ids: list[str] = []
    for _index, selected in selected_displays.iterrows():
        display_id = cell_text(selected.get("display_id"))
        selected_family_id = fallback_family_for_display(display_id, display_group_families)
        if not selected_family_id:
            continue
        family = family_map.get(selected_family_id, pd.Series(dtype=object))
        display = display_map.get(display_id, pd.Series(dtype=object))
        selected_family_ids.append(selected_family_id)
        rows.append(
            {
                "display_id": display_id,
                "display_name": cell_text(display.get("display_name")),
                "unit": cell_text(display.get("unit")),
                "family_count": display.get("family_count", ""),
                "selection_reason": cell_text(selected.get("selection_reason")),
                "selected_family_id": selected_family_id,
                "default_practice": truncate_text(normalize_display_description(family.get("representative_project_description")), 120),
                "family_selection_reason": "按组内 item_score、样本数、来源工程包数和 final_score 确定默认参考做法",
                "other_practices": [],
            }
        )
    meta = {
        "invalid_display_ids": [],
        "invalid_family_ids": [],
        "duplicate_display_ids": [],
        "selected_family_ids": selected_family_ids,
        "other_family_ids": [],
        "other_family_count": 0,
    }
    return pd.DataFrame(rows), meta


def build_display_family_selection_trace_frame(
    selected_display_practices: pd.DataFrame,
    display_group_families: pd.DataFrame,
    candidate_families: pd.DataFrame,
) -> pd.DataFrame:
    family_map = {cell_text(row.get("family_id")): row for _index, row in candidate_families.iterrows()}
    selected_map = {cell_text(row.get("display_id")): row for _index, row in selected_display_practices.iterrows()}
    rows: list[dict[str, Any]] = []
    selected_display_ids = set(selected_map)
    for _index, row in display_group_families[display_group_families["display_id"].map(cell_text).isin(selected_display_ids)].iterrows():
        display_id = cell_text(row.get("display_id"))
        family_id = cell_text(row.get("family_id"))
        selected = selected_map.get(display_id, pd.Series(dtype=object))
        candidate = family_map.get(family_id, pd.Series(dtype=object))
        other_practices = selected.get("other_practices") if isinstance(selected.get("other_practices"), list) else []
        other_map = {cell_text(item.get("family_id")): cell_text(item.get("difference")) for item in other_practices if isinstance(item, dict)}
        rows.append(
            {
                "display_id": display_id,
                "display_name": cell_text(row.get("display_name")),
                "family_id": family_id,
                "fine_signature": cell_text(row.get("fine_signature")),
                "representative_cost_item_name": cell_text(row.get("representative_cost_item_name")),
                "representative_project_description": cell_text(row.get("representative_project_description")),
                "display_project_description": normalize_display_description(row.get("representative_project_description")),
                "历史样本数": row.get("历史样本数", ""),
                "来源工程包数": row.get("来源工程包数", ""),
                "final_score": row.get("final_score", ""),
                "item_score最大值": row.get("item_score最大值", ""),
                "unit_price_min": candidate.get("历史综合单价最低值", ""),
                "unit_price_median": candidate.get("历史综合单价中位数", ""),
                "unit_price_max": candidate.get("历史综合单价最高值", ""),
                "is_selected_family": "是" if family_id == cell_text(selected.get("selected_family_id")) else "否",
                "is_other_practice": "是" if family_id in other_map else "否",
                "other_practice_difference": other_map.get(family_id, ""),
                "selection_reason": cell_text(selected.get("family_selection_reason")) if family_id == cell_text(selected.get("selected_family_id")) else "",
            }
        )
    return pd.DataFrame(rows, columns=DISPLAY_FAMILY_SELECTION_TRACE_COLUMNS)


def generate_display_family_selection(
    rewrite: QueryRewrite,
    selected_displays: pd.DataFrame,
    candidate_display_groups: pd.DataFrame,
    display_group_families: pd.DataFrame,
    candidate_families: pd.DataFrame,
    warnings: list[str] | None = None,
) -> tuple[pd.DataFrame, bool, bool, str, str, dict[str, Any], dict[str, Any], pd.DataFrame]:
    prompt, records = build_display_family_selection_prompt(
        rewrite,
        selected_displays,
        candidate_display_groups,
        display_group_families,
        candidate_families,
    )
    max_tokens = 3072
    if selected_displays.empty:
        trace = trace_row(
            "display_family_selection",
            "为已选 display 选择默认 family 和其他历史做法",
            True,
            prompt=prompt,
            max_tokens=max_tokens,
            input_summary=json_text({"display_count": 0, "family_count": 0, "selected_family_ids": [], "other_family_ids": []}),
        )
        return pd.DataFrame(), True, False, "", prompt, trace, {
            "selected_family_ids": [],
            "other_family_ids": [],
            "other_family_count": 0,
        }, pd.DataFrame(columns=DISPLAY_FAMILY_SELECTION_TRACE_COLUMNS)
    try:
        response = request_llm_json_with_usage(
            prompt,
            max_tokens=max_tokens,
            system_prompt="你只输出一个 JSON object，不输出解释、Markdown 或思考过程。",
        )
        selected, meta = parse_display_family_selection_result(
            response.content,
            selected_displays,
            candidate_display_groups,
            display_group_families,
            warnings,
        )
        trace_frame = build_display_family_selection_trace_frame(selected, display_group_families, candidate_families)
        family_count = sum(len(item.get("candidate_families") or []) for item in records)
        trace = trace_row(
            "display_family_selection",
            "为已选 display 选择默认 family 和其他历史做法",
            True,
            prompt=prompt,
            max_tokens=max_tokens,
            input_summary=json_text(
                {
                    "display_count": len(records),
                    "family_count": family_count,
                    "selected_family_ids": trace_id_summary(meta.get("selected_family_ids") or []),
                    "other_family_ids": trace_id_summary(meta.get("other_family_ids") or []),
                }
            ),
            usage=response.usage,
        )
        return selected, True, False, "", prompt, trace, meta, trace_frame
    except (LLMServiceError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        append_warning(warnings, "display_family_selection_failed")
        fallback, meta = fallback_display_family_selection(selected_displays, candidate_display_groups, display_group_families)
        trace_frame = build_display_family_selection_trace_frame(fallback, display_group_families, candidate_families)
        family_count = sum(len(item.get("candidate_families") or []) for item in records)
        trace = trace_row(
            "display_family_selection",
            "为已选 display 选择默认 family 和其他历史做法",
            False,
            error=str(exc),
            prompt=prompt,
            max_tokens=max_tokens,
            input_summary=json_text(
                {
                    "display_count": len(records),
                    "family_count": family_count,
                    "selected_family_ids": trace_id_summary(meta.get("selected_family_ids") or []),
                    "other_family_ids": trace_id_summary(meta.get("other_family_ids") or []),
                }
            ),
        )
        return fallback, False, True, str(exc), prompt, trace, meta, trace_frame


def normalize_dedup_text(value: Any) -> str:
    text = cell_text(value).lower()
    text = text.replace("㎡", "m²").replace("平方米", "m²").replace("平方", "m²")
    text = re.sub(r"m\s*2|m\^2|m\^\{2\}", "m²", text)
    text = re.sub(r"\s+", "", text)
    text = text.replace("（", "(").replace("）", ")")
    text = text.replace("，", ",").replace("；", ";").replace("：", ":")
    return text.strip()


def display_strength(row: pd.Series) -> tuple[int, int, float]:
    return (
        int(row.get("family_count") or 0),
        int(row.get("默认family历史样本数") or 0),
        float(row.get("final_score最大值") or 0.0),
    )


def apply_display_dedup_suppression(selected_displays: pd.DataFrame, suppressed_by: dict[str, str]) -> pd.DataFrame:
    if selected_displays.empty or not suppressed_by:
        return selected_displays.reset_index(drop=True)
    suppressed_ids = set(suppressed_by)
    output = selected_displays[~selected_displays["display_id"].map(cell_text).isin(suppressed_ids)].copy()
    return output.reset_index(drop=True)


def deterministic_display_dedup_selection(selected_displays: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    if selected_displays.empty:
        return selected_displays.copy(), {}
    groups: dict[tuple[str, ...], list[str]] = {}
    row_map = {cell_text(row.get("display_id")): row for _index, row in selected_displays.iterrows()}
    for _index, row in selected_displays.iterrows():
        display_id = cell_text(row.get("display_id"))
        exact_key = (
            "fields",
            normalize_dedup_text(row.get("display_name")),
            normalize_dedup_text(row.get("default_practice")),
            normalize_dedup_text(row.get("unit")),
        )
        groups.setdefault(exact_key, []).append(display_id)

    suppressed_by: dict[str, str] = {}
    for display_ids in groups.values():
        active_ids = [display_id for display_id in display_ids if display_id not in suppressed_by]
        if len(active_ids) <= 1:
            continue
        representative_id = max(active_ids, key=lambda display_id: display_strength(row_map.get(display_id, pd.Series(dtype=object))))
        for display_id in active_ids:
            if display_id != representative_id:
                suppressed_by[display_id] = representative_id
    return apply_display_dedup_suppression(selected_displays, suppressed_by), suppressed_by


def selected_display_payload(selected_displays: pd.DataFrame, display_ids: list[str]) -> list[dict[str, str]]:
    display_map = {cell_text(row.get("display_id")): row for _index, row in selected_displays.iterrows()}
    records: list[dict[str, str]] = []
    for display_id in display_ids:
        row = display_map.get(display_id)
        if row is None:
            continue
        records.append(
            {
                "id": display_id,
                "display_name": truncate_text(row.get("display_name"), 80),
                "selected_family_id": cell_text(row.get("selected_family_id")),
                "default_practice": truncate_text(row.get("default_practice"), 140),
                "unit": cell_text(row.get("unit")),
            }
        )
    return records


def suspicious_dedup_display_ids(selected_displays: pd.DataFrame) -> list[str]:
    selected_rows = [(cell_text(row.get("display_id")), row) for _index, row in selected_displays.iterrows()]
    if len(selected_rows) <= 6:
        return [display_id for display_id, _row in selected_rows]
    suspicious: set[str] = set()
    for index, (left_id, left) in enumerate(selected_rows):
        left_unit = normalize_dedup_text(left.get("unit"))
        for right_id, right in selected_rows[index + 1 :]:
            right_unit = normalize_dedup_text(right.get("unit"))
            if left_unit != right_unit:
                continue
            name_overlap = dedup_text_overlap(left.get("display_name"), right.get("display_name"))
            practice_overlap = dedup_text_overlap(left.get("default_practice"), right.get("default_practice"))
            if name_overlap or practice_overlap:
                suspicious.update([left_id, right_id])
    return [display_id for display_id, _row in selected_rows if display_id in suspicious]


def build_display_dedup_selection_prompt(raw_query: str, selected_records: list[dict[str, str]]) -> str:
    payload = {"query": raw_query, "selected": selected_records}
    return f"""
你是物业维修工程建议清单的重复展示抑制器。

只判断输入 selected 中是否存在会让用户误以为需要重复施工的实质重复项。
你不能新增、改名或合并 display，只能在输入 display_id 中选择保留或抑制展示。

判断时同时比较 display 工作内容和默认参考做法。
如果一个 display 的 selected family 已包含另一个 display 的全部施工内容，可以抑制被包含项。

【不得抑制】
- 不同施工目标。
- 拆除与新做工作内容。
- 拆除与垃圾运输在独立计价时。
- 一个 display 额外包含会显著影响价格或施工边界的内容。

只输出一个 JSON object，不得输出解释、Markdown 或思考过程。

【输出 JSON】
{{
  "keep": ["D001", "D003"],
  "suppress": [
    {{
      "display_id": "D002",
      "representative_id": "D001",
      "reason": "D001的默认做法已覆盖D002的施工内容"
    }}
  ],
  "keep_reasons": [
    {{
      "display_id": "D001",
      "reason": "与其他 display 不是重复施工"
    }}
  ]
}}

【输入数据】
{json_text(payload)}
""".strip()


def parse_display_dedup_selection_result(
    result: dict[str, Any],
    allowed_display_ids: set[str],
) -> tuple[set[str], dict[str, str], list[dict[str, str]], list[dict[str, str]]]:
    keep_raw = result.get("keep")
    suppress_raw = result.get("suppress")
    keep_reasons_raw = result.get("keep_reasons", [])
    if not isinstance(keep_raw, list) or not isinstance(suppress_raw, list):
        raise ValueError("dedup_selection 输出缺少 keep/suppress list")
    if not isinstance(keep_reasons_raw, list):
        raise ValueError("dedup_selection keep_reasons 必须为 list")

    keep_ids = {cell_text(item) for item in keep_raw if cell_text(item)}
    if not keep_ids.issubset(allowed_display_ids):
        raise ValueError("dedup_selection keep 包含非法 display_id")

    suppress_map: dict[str, str] = {}
    suppress_detail: list[dict[str, str]] = []
    for item in suppress_raw:
        if not isinstance(item, dict):
            raise ValueError("dedup_selection suppress 项必须为 object")
        suppressed_id = cell_text(item.get("display_id"))
        representative_id = cell_text(item.get("representative_id"))
        reason = cell_text(item.get("reason"))
        if suppressed_id not in allowed_display_ids or representative_id not in allowed_display_ids:
            raise ValueError("dedup_selection suppress 包含非法 display_id")
        if not reason:
            raise ValueError("dedup_selection suppress 缺少 reason")
        if suppressed_id == representative_id:
            raise ValueError("dedup_selection suppress 不能自引用")
        if suppressed_id in suppress_map and suppress_map[suppressed_id] != representative_id:
            raise ValueError("dedup_selection suppress 存在冲突")
        suppress_map[suppressed_id] = representative_id
        suppress_detail.append({"display_id": suppressed_id, "representative_id": representative_id, "reason": reason})

    keep_reasons: list[dict[str, str]] = []
    for item in keep_reasons_raw:
        if not isinstance(item, dict):
            raise ValueError("dedup_selection keep_reasons 项必须为 object")
        display_id = cell_text(item.get("display_id"))
        if display_id not in allowed_display_ids:
            raise ValueError("dedup_selection keep_reasons 包含非法 display_id")
        keep_reasons.append({"display_id": display_id, "reason": cell_text(item.get("reason"))})

    if keep_ids & set(suppress_map):
        raise ValueError("dedup_selection display 不能同时 keep 和 suppress")
    if has_suppression_cycle(suppress_map):
        raise ValueError("dedup_selection suppress 形成环")
    return keep_ids, suppress_map, suppress_detail, keep_reasons


def generate_display_dedup_selection(
    rewrite: QueryRewrite,
    selected_displays: pd.DataFrame,
    warnings: list[str] | None = None,
) -> tuple[pd.DataFrame, bool, bool, str, str, dict[str, Any], dict[str, Any]]:
    auto_selected, auto_suppressed = deterministic_display_dedup_selection(selected_displays)
    display_ids = suspicious_dedup_display_ids(auto_selected)
    records = selected_display_payload(auto_selected, display_ids)
    input_ids = [record["id"] for record in records]
    auto_keep_ids = [cell_text(row.get("display_id")) for _index, row in auto_selected.iterrows() if cell_text(row.get("display_id"))]
    meta: dict[str, Any] = {
        "auto_suppressed": [f"{left}->{right}" for left, right in auto_suppressed.items()],
        "llm_suppressed": [],
        "input_ids": input_ids,
        "keep_ids": auto_keep_ids,
        "suppress_detail": [],
        "keep_reasons": [],
        "invalid": [],
    }
    if len(records) <= 1:
        trace = trace_row(
            "dedup_selection",
            "抑制已选 display 中的重复或包含展示项",
            True,
            prompt="",
            max_tokens=0,
            input_summary=json_text(
                {
                    "selected_before_dedup": len(selected_displays),
                    "input_ids": trace_id_summary(input_ids),
                    "kept_ids": trace_id_summary(auto_keep_ids),
                    "suppressed_ids": trace_id_summary(list(auto_suppressed)),
                }
            ),
        )
        return auto_selected, True, False, "", "", trace, meta

    prompt = build_display_dedup_selection_prompt(rewrite.raw_query, records)
    allowed_display_ids = {record["id"] for record in records}
    max_tokens = 512
    try:
        response = request_llm_json_with_usage(
            prompt,
            max_tokens=max_tokens,
            system_prompt="你只输出一个 JSON object，不输出解释、Markdown 或思考过程。",
        )
        keep_ids, llm_suppressed, suppress_detail, keep_reasons = parse_display_dedup_selection_result(response.content, allowed_display_ids)
        deduped = apply_display_dedup_suppression(auto_selected, llm_suppressed)
        kept_ids = [cell_text(row.get("display_id")) for _index, row in deduped.iterrows() if cell_text(row.get("display_id"))]
        meta["llm_suppressed"] = [f"{left}->{right}" for left, right in llm_suppressed.items()]
        meta["keep_ids"] = kept_ids
        meta["suppress_detail"] = suppress_detail
        meta["keep_reasons"] = keep_reasons
        meta["llm_keep_ids"] = sorted(keep_ids)
        trace = trace_row(
            "dedup_selection",
            "抑制已选 display 中的重复或包含展示项",
            True,
            prompt=prompt,
            max_tokens=max_tokens,
            input_summary=json_text(
                {
                    "selected_before_dedup": len(selected_displays),
                    "input_ids": trace_id_summary(input_ids),
                    "kept_ids": trace_id_summary(kept_ids),
                    "suppressed_ids": trace_id_summary(list(llm_suppressed)),
                }
            ),
            usage=response.usage,
        )
        return deduped, True, False, "", prompt, trace, meta
    except (LLMServiceError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        append_warning(warnings, "dedup_selection_failed")
        meta["invalid"] = [str(exc)]
        trace = trace_row(
            "dedup_selection",
            "抑制已选 display 中的重复或包含展示项",
            False,
            error=str(exc),
            prompt=prompt,
            max_tokens=max_tokens,
            input_summary=json_text(
                {
                    "selected_before_dedup": len(selected_displays),
                    "input_ids": trace_id_summary(input_ids),
                    "kept_ids": trace_id_summary(auto_keep_ids),
                    "suppressed_ids": trace_id_summary(list(auto_suppressed)),
                }
            ),
        )
        return auto_selected, False, True, str(exc), prompt, trace, meta


def family_strength(row: pd.Series) -> tuple[int, int, float]:
    return (
        int(row.get("历史样本数") or 0),
        int(row.get("来源工程包数") or 0),
        float(row.get("final_score") or 0.0),
    )


def apply_dedup_suppression(selected_families: pd.DataFrame, suppressed_by: dict[str, str]) -> pd.DataFrame:
    if selected_families.empty or not suppressed_by:
        return selected_families.reset_index(drop=True)
    suppressed_ids = set(suppressed_by)
    output = selected_families[~selected_families["family_id"].map(cell_text).isin(suppressed_ids)].copy()
    return output.reset_index(drop=True)


def deterministic_dedup_selection(
    selected_families: pd.DataFrame,
    candidate_families: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    if selected_families.empty:
        return selected_families.copy(), {}
    family_map = {cell_text(row.get("family_id")): row for _index, row in candidate_families.iterrows()}
    groups: dict[tuple[str, ...], list[str]] = {}
    for _index, selected in selected_families.iterrows():
        family_id = cell_text(selected.get("family_id"))
        family = family_map.get(family_id)
        if family is None:
            continue
        signature_key = ("signature", cell_text(family.get("fine_signature")))
        exact_key = (
            "fields",
            normalize_dedup_text(family.get("representative_cost_item_name")),
            normalize_dedup_text(family.get("representative_project_description")),
            normalize_dedup_text(family.get("unit_normalized")) or normalize_dedup_text(family.get("unit")),
        )
        if signature_key[1]:
            groups.setdefault(signature_key, []).append(family_id)
        groups.setdefault(exact_key, []).append(family_id)

    suppressed_by: dict[str, str] = {}
    for family_ids in groups.values():
        active_ids = [family_id for family_id in family_ids if family_id not in suppressed_by]
        if len(active_ids) <= 1:
            continue
        representative_id = max(active_ids, key=lambda family_id: family_strength(family_map.get(family_id, pd.Series(dtype=object))))
        for family_id in active_ids:
            if family_id != representative_id:
                suppressed_by[family_id] = representative_id

    return apply_dedup_suppression(selected_families, suppressed_by), suppressed_by


def selected_family_payload(selected_families: pd.DataFrame, candidate_families: pd.DataFrame, family_ids: list[str]) -> list[dict[str, str]]:
    family_map = {cell_text(row.get("family_id")): row for _index, row in candidate_families.iterrows()}
    records: list[dict[str, str]] = []
    for family_id in family_ids:
        family = family_map.get(family_id)
        if family is None:
            continue
        records.append(
            {
                "id": family_id,
                "name": truncate_text(family.get("representative_cost_item_name"), 80),
                "spec": truncate_text(family.get("representative_project_description"), 140),
                "unit": cell_text(family.get("unit_normalized")) or cell_text(family.get("unit")),
            }
        )
    return records


def dedup_text_overlap(left: str, right: str) -> bool:
    left = normalize_dedup_text(left)
    right = normalize_dedup_text(right)
    if not left or not right:
        return False
    if left in right or right in left:
        return True
    left_pairs = {left[index : index + 2] for index in range(max(len(left) - 1, 0))}
    right_pairs = {right[index : index + 2] for index in range(max(len(right) - 1, 0))}
    return bool(left_pairs and right_pairs and len(left_pairs & right_pairs) >= 4)


def suspicious_dedup_family_ids(selected_families: pd.DataFrame, candidate_families: pd.DataFrame) -> list[str]:
    if len(selected_families) <= 6:
        return [cell_text(row.get("family_id")) for _index, row in selected_families.iterrows()]
    family_map = {cell_text(row.get("family_id")): row for _index, row in candidate_families.iterrows()}
    selected_rows = [(cell_text(row.get("family_id")), row) for _index, row in selected_families.iterrows()]
    suspicious: set[str] = set()
    for index, (left_id, left_selected) in enumerate(selected_rows):
        left_family = family_map.get(left_id)
        if left_family is None:
            continue
        left_unit = normalize_dedup_text(left_family.get("unit_normalized")) or normalize_dedup_text(left_family.get("unit"))
        for right_id, right_selected in selected_rows[index + 1 :]:
            right_family = family_map.get(right_id)
            if right_family is None:
                continue
            right_unit = normalize_dedup_text(right_family.get("unit_normalized")) or normalize_dedup_text(right_family.get("unit"))
            if left_unit != right_unit:
                continue
            name_overlap = dedup_text_overlap(left_family.get("representative_cost_item_name"), right_family.get("representative_cost_item_name"))
            spec_overlap = dedup_text_overlap(left_family.get("representative_project_description"), right_family.get("representative_project_description"))
            if name_overlap or spec_overlap:
                suspicious.update([left_id, right_id])
    return [family_id for family_id, _row in selected_rows if family_id in suspicious]


def build_dedup_selection_prompt(raw_query: str, selected_records: list[dict[str, str]]) -> str:
    payload = {"query": raw_query, "selected": selected_records}
    return f"""
你是物业维修工程建议清单的重复展示抑制器。

只判断输入 selected 中是否存在会让用户误以为需要重复施工的实质重复项。
你不能新增、改名或合并 family，只能在输入 id 中选择保留或抑制展示。

【只抑制】
- 施工目标相同。
- 材料、厚度、工艺、部位和主要施工边界基本相同。
- 一个 family 的内容已基本覆盖另一个。

【不得抑制】
- 不同材料、厚度、工艺、层数、部位或单位。
- 拆除与后续处理。
- 拆除与垃圾运输在独立计价时。
- 同一系统中不同功能部件。
- 一个 family 额外包含会显著影响价格的施工内容。

只输出一个 JSON object，不得输出解释、Markdown 或思考过程。

【输出 JSON】
{{
  "keep": ["F065", "F013"],
  "suppress": [
    {{
      "family_id": "F001",
      "representative_id": "F065",
      "reason": "F065已覆盖同一施工边界，保留样本更充分的项目"
    }}
  ],
  "keep_reasons": [
    {{
      "family_id": "F065",
      "reason": "主要施工项，与其他独立工作内容不是重复施工"
    }}
  ]
}}

【输入数据】
{json_text(payload)}
""".strip()


def has_suppression_cycle(suppress_map: dict[str, str]) -> bool:
    for family_id in suppress_map:
        seen: set[str] = set()
        current = family_id
        while current in suppress_map:
            if current in seen:
                return True
            seen.add(current)
            current = suppress_map[current]
    return False


def parse_dedup_selection_result(
    result: dict[str, Any],
    allowed_family_ids: set[str],
) -> tuple[set[str], dict[str, str], list[dict[str, str]], list[dict[str, str]]]:
    keep_raw = result.get("keep")
    suppress_raw = result.get("suppress")
    keep_reasons_raw = result.get("keep_reasons", [])
    if not isinstance(keep_raw, list) or not isinstance(suppress_raw, list):
        raise ValueError("dedup_selection 输出缺少 keep/suppress list")
    if not isinstance(keep_reasons_raw, list):
        raise ValueError("dedup_selection keep_reasons 必须为 list")

    keep_ids = {cell_text(item) for item in keep_raw if cell_text(item)}
    if not keep_ids.issubset(allowed_family_ids):
        raise ValueError("dedup_selection keep 包含非法 family_id")

    suppress_map: dict[str, str] = {}
    suppress_detail: list[dict[str, str]] = []
    for item in suppress_raw:
        if not isinstance(item, dict):
            raise ValueError("dedup_selection suppress 项必须为 object")
        suppressed_id = cell_text(item.get("family_id"))
        representative_id = cell_text(item.get("representative_id"))
        reason = cell_text(item.get("reason"))
        if suppressed_id not in allowed_family_ids or representative_id not in allowed_family_ids:
            raise ValueError("dedup_selection suppress 包含非法 family_id")
        if not reason:
            raise ValueError("dedup_selection suppress 缺少 reason")
        if suppressed_id == representative_id:
            raise ValueError("dedup_selection suppress 不能自引用")
        if suppressed_id in suppress_map and suppress_map[suppressed_id] != representative_id:
            raise ValueError("dedup_selection suppress 存在冲突")
        suppress_map[suppressed_id] = representative_id
        suppress_detail.append(
            {
                "family_id": suppressed_id,
                "representative_id": representative_id,
                "reason": reason,
            }
        )

    keep_reasons: list[dict[str, str]] = []
    for item in keep_reasons_raw:
        if not isinstance(item, dict):
            raise ValueError("dedup_selection keep_reasons 项必须为 object")
        family_id = cell_text(item.get("family_id"))
        if family_id not in allowed_family_ids:
            raise ValueError("dedup_selection keep_reasons 包含非法 family_id")
        keep_reasons.append({"family_id": family_id, "reason": cell_text(item.get("reason"))})

    if keep_ids & set(suppress_map):
        raise ValueError("dedup_selection family 不能同时 keep 和 suppress")
    if has_suppression_cycle(suppress_map):
        raise ValueError("dedup_selection suppress 形成环")

    return keep_ids, suppress_map, suppress_detail, keep_reasons


def generate_dedup_selection(
    rewrite: QueryRewrite,
    selected_families: pd.DataFrame,
    candidate_families: pd.DataFrame,
    warnings: list[str] | None = None,
) -> tuple[pd.DataFrame, bool, bool, str, str, dict[str, Any], dict[str, Any]]:
    auto_selected, auto_suppressed = deterministic_dedup_selection(selected_families, candidate_families)
    family_ids = suspicious_dedup_family_ids(auto_selected, candidate_families)
    records = selected_family_payload(auto_selected, candidate_families, family_ids)
    selected_before_ids = [cell_text(row.get("family_id")) for _index, row in selected_families.iterrows() if cell_text(row.get("family_id"))]
    input_ids = [record["id"] for record in records]
    auto_keep_ids = [cell_text(row.get("family_id")) for _index, row in auto_selected.iterrows() if cell_text(row.get("family_id"))]
    meta: dict[str, Any] = {
        "auto_suppressed": [f"{left}->{right}" for left, right in auto_suppressed.items()],
        "llm_suppressed": [],
        "input_ids": input_ids,
        "keep_ids": auto_keep_ids,
        "suppress_detail": [],
        "keep_reasons": [],
        "invalid": [],
    }

    if len(records) <= 1:
        trace = trace_row(
            "dedup_selection",
            "抑制已选 family 中的重复展示项",
            True,
            prompt="",
            max_tokens=0,
            input_summary=json_text(
                {
                    "selected_before_dedup": len(selected_families),
                    "input_ids": trace_id_summary(input_ids),
                    "kept_ids": trace_id_summary(auto_keep_ids),
                    "suppressed_ids": trace_id_summary(list(auto_suppressed)),
                }
            ),
        )
        return auto_selected, True, False, "", "", trace, meta

    prompt = build_dedup_selection_prompt(rewrite.raw_query, records)
    allowed_family_ids = {record["id"] for record in records}
    max_tokens = 512
    try:
        response = request_llm_json_with_usage(
            prompt,
            max_tokens=max_tokens,
            system_prompt="你只输出一个 JSON object，不输出解释、Markdown 或思考过程。",
        )
        keep_ids, llm_suppressed, suppress_detail, keep_reasons = parse_dedup_selection_result(response.content, allowed_family_ids)
        deduped = apply_dedup_suppression(auto_selected, llm_suppressed)
        kept_ids = [cell_text(row.get("family_id")) for _index, row in deduped.iterrows() if cell_text(row.get("family_id"))]
        meta["llm_suppressed"] = [f"{left}->{right}" for left, right in llm_suppressed.items()]
        meta["keep_ids"] = kept_ids
        meta["suppress_detail"] = suppress_detail
        meta["keep_reasons"] = keep_reasons
        meta["llm_keep_ids"] = sorted(keep_ids)
        trace = trace_row(
            "dedup_selection",
            "抑制已选 family 中的重复展示项",
            True,
            prompt=prompt,
            max_tokens=max_tokens,
            input_summary=json_text(
                {
                    "selected_before_dedup": len(selected_families),
                    "input_ids": trace_id_summary(input_ids),
                    "kept_ids": trace_id_summary(kept_ids),
                    "suppressed_ids": trace_id_summary(list(llm_suppressed)),
                }
            ),
            usage=response.usage,
        )
        return deduped, True, False, "", prompt, trace, meta
    except (LLMServiceError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        append_warning(warnings, "dedup_selection_failed")
        meta["invalid"] = [str(exc)]
        trace = trace_row(
            "dedup_selection",
            "抑制已选 family 中的重复展示项",
            False,
            error=str(exc),
            prompt=prompt,
            max_tokens=max_tokens,
            input_summary=json_text(
                {
                    "selected_before_dedup": len(selected_families),
                    "input_ids": trace_id_summary(input_ids),
                    "kept_ids": trace_id_summary(auto_keep_ids),
                    "suppressed_ids": trace_id_summary(list(auto_suppressed)),
                }
            ),
        )
        return auto_selected, False, True, str(exc), prompt, trace, meta


def normalized_unit(value: Any) -> str:
    text = cell_text(value).lower()
    text = text.replace("㎡", "m²").replace("平方米", "m²").replace("平方", "m²")
    text = re.sub(r"m\s*2|m\^2", "m²", text)
    text = text.replace("毫米", "mm")
    return text.strip()


def matched_user_quantity(parsed_quantities: list[dict[str, Any]], unit: Any) -> float | None:
    target_unit = normalized_unit(unit)
    if not target_unit:
        return None
    for item in parsed_quantities:
        if not isinstance(item, dict):
            continue
        value = numeric_or_none(item.get("value"))
        if value is None:
            continue
        if normalized_unit(item.get("unit")) == target_unit:
            return value
    return None


def build_historical_quantity_context(
    selected_families: pd.DataFrame,
    candidate_families: pd.DataFrame,
    candidates: pd.DataFrame,
    parsed_query: QueryRewrite,
    max_relations_per_family: int = 5,
) -> list[dict[str, Any]]:
    if selected_families.empty:
        return []
    family_map = {cell_text(row.get("family_id")): row for _index, row in candidate_families.iterrows()}
    signature_by_family = {
        family_id: cell_text(row.get("fine_signature"))
        for family_id, row in family_map.items()
    }
    selected_ids = {
        cell_text(row.get("family_id"))
        for _index, row in selected_families.iterrows()
        if cell_text(row.get("family_id"))
    }

    contexts: list[dict[str, Any]] = []
    for _index, selected in selected_families.iterrows():
        family_id = cell_text(selected.get("family_id"))
        family = family_map.get(family_id)
        if family is None:
            continue
        unit = cell_text(family.get("unit_normalized")) or cell_text(family.get("unit"))
        user_quantity = matched_user_quantity(parsed_query.parsed_quantities, unit)
        relations: list[dict[str, Any]] = []
        target_signature = signature_by_family.get(family_id, "")
        target_rows = candidates[candidates["fine_signature"].map(cell_text).eq(target_signature)].copy()
        if user_quantity is None and not target_rows.empty:
            for _target_index, target_row in target_rows.sort_values("final_score", ascending=False).iterrows():
                project_key = cell_text(target_row.get("project_key")) or cell_text(target_row.get("project_package_id"))
                if not project_key:
                    continue
                related_ids = [item for item in selected_ids if item != family_id]
                for related_family_id in related_ids:
                    if related_family_id == family_id:
                        continue
                    related_signature = signature_by_family.get(related_family_id, "")
                    related_rows = candidates[
                        candidates["fine_signature"].map(cell_text).eq(related_signature)
                        & (
                            candidates.get("project_key", pd.Series(dtype=object)).map(cell_text).eq(project_key)
                            | candidates.get("project_package_id", pd.Series(dtype=object)).map(cell_text).eq(project_key)
                        )
                    ]
                    for _related_index, related_row in related_rows.iterrows():
                        related_quantity = numeric_or_none(related_row.get("quantity"))
                        target_quantity = numeric_or_none(target_row.get("quantity"))
                        if related_quantity is None or target_quantity is None:
                            continue
                        relations.append(
                            {
                                "related_family_id": related_family_id,
                                "related_quantity": related_quantity,
                                "related_unit": cell_text(related_row.get("unit_normalized")) or cell_text(related_row.get("unit")),
                                "target_quantity": target_quantity,
                                "target_unit": unit,
                                "_score": float(target_row.get("final_score") or 0.0),
                            }
                        )
        seen_relation_keys: set[tuple[Any, ...]] = set()
        compact_relations: list[dict[str, Any]] = []
        for relation in sorted(relations, key=lambda item: item.get("_score", 0.0), reverse=True):
            key = (
                relation["related_family_id"],
                relation["related_quantity"],
                relation["related_unit"],
                relation["target_quantity"],
                relation["target_unit"],
            )
            if key in seen_relation_keys:
                continue
            seen_relation_keys.add(key)
            relation.pop("_score", None)
            compact_relations.append(relation)
            if len(compact_relations) >= max_relations_per_family:
                break
        contexts.append(
            {
                "family_id": family_id,
                "cost_item_name": cell_text(family.get("representative_cost_item_name")),
                "project_description": cell_text(family.get("representative_project_description")),
                "unit": unit,
                "user_quantity_match": user_quantity,
                "historical_relations": compact_relations,
            }
        )
    return contexts


def build_quantity_decision_prompt(
    rewrite: QueryRewrite,
    selected_families_with_history: list[dict[str, Any]],
) -> str:
    payload = {
        "raw_query": rewrite.raw_query,
        "parsed_quantities": rewrite.parsed_quantities,
        "selected_families": selected_families_with_history,
    }
    return f"""
你是物业维修工程量与金额口径判断助手。

根据用户需求、已选施工项和历史数量关系，为每个 family 判断工程量范围及是否计入参考金额。

quantity_source 只能取：
- 用户明确给定
- 历史样本估算
- 需现场确认

规则：
1. 用户数量能直接对应施工项单位时，使用“用户明确给定”。
2. 历史相似工程存在稳定、可解释的规模关系时，可使用“历史样本估算”。
3. 不得机械复制单条历史数量；历史关系不稳定或现场条件影响较大时，使用“需现场确认”，数量填 null。
4. suggested_quantity_low、suggested_quantity_mid、suggested_quantity_high 分别表示最低、最可能、最高估计；用户明确数量时三者相同。
5. include_in_amount=true 仅当该项属于实际建议方案、数量有合理依据且不会与替代方案重复计算。
6. 数量全部为 null 时必须不计入。
7. 不得新增 family，不得修改名称、特征和单位。
8. confirmation_note 只写该项仍需确认的关键因素，没有则填空字符串。
9. 只输出 JSON。

输出：
{{
  "family_quantities": [
    {{
      "family_id": "F001",
      "quantity_source": "用户明确给定",
      "suggested_quantity_low": 500,
      "suggested_quantity_mid": 500,
      "suggested_quantity_high": 500,
      "include_in_amount": true,
      "quantity_reason": "用户明确给出约500m²",
      "confirmation_note": "需确认实际施工边界"
    }}
  ]
}}

输入：
{json_text(payload)}
""".strip()


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = cell_text(value).lower()
    return text in {"true", "1", "yes", "y", "是", "计入"}


def parse_quantity_decision_result(
    result: dict[str, Any],
    selected_families: pd.DataFrame,
    warnings: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    allowed_ids = [cell_text(value) for value in selected_families.get("family_id", pd.Series(dtype=object)).tolist()]
    raw_rows = result.get("family_quantities")
    if not isinstance(raw_rows, list):
        raise ValueError("LLM 输出缺少 family_quantities list")
    rows_by_id: dict[str, dict[str, Any]] = {}
    meta = {"invalid_family_ids": [], "duplicate_family_ids": [], "invalid_quantity_sources": [], "invalid_quantity_ranges": []}
    for item in raw_rows:
        if not isinstance(item, dict):
            continue
        family_id = cell_text(item.get("family_id"))
        if family_id not in allowed_ids:
            if family_id:
                meta["invalid_family_ids"].append(family_id)
                append_warning(warnings, "invalid_quantity_family_ids")
            continue
        if family_id in rows_by_id:
            meta["duplicate_family_ids"].append(family_id)
            append_warning(warnings, "duplicate_quantity_family_ids")
            continue
        quantity_source = cell_text(item.get("quantity_source"))
        if quantity_source not in ALLOWED_QUANTITY_SOURCES:
            meta["invalid_quantity_sources"].append(quantity_source or family_id)
            append_warning(warnings, "invalid_quantity_sources")
            continue
        low = numeric_or_none(item.get("suggested_quantity_low"))
        mid = numeric_or_none(item.get("suggested_quantity_mid"))
        high = numeric_or_none(item.get("suggested_quantity_high"))
        quantities = [value for value in [low, mid, high] if value is not None]
        if quantities and (low is None or mid is None or high is None or low > mid or mid > high):
            low = mid = high = None
            meta["invalid_quantity_ranges"].append(family_id)
            append_warning(warnings, "invalid_quantity_ranges")
        include = parse_bool(item.get("include_in_amount"))
        if low is None and mid is None and high is None:
            include = False
        rows_by_id[family_id] = {
            "family_id": family_id,
            "quantity_source": quantity_source,
            "suggested_quantity_low": low,
            "suggested_quantity_mid": mid,
            "suggested_quantity_high": high,
            "include_in_amount": include,
            "quantity_reason": cell_text(item.get("quantity_reason")),
            "confirmation_note": cell_text(item.get("confirmation_note")),
        }
    for family_id in allowed_ids:
        if family_id not in rows_by_id:
            rows_by_id[family_id] = {
                "family_id": family_id,
                "quantity_source": "需现场确认",
                "suggested_quantity_low": None,
                "suggested_quantity_mid": None,
                "suggested_quantity_high": None,
                "include_in_amount": False,
                "quantity_reason": "",
                "confirmation_note": "工程量及计价范围需确认",
            }
    rows = [rows_by_id[family_id] for family_id in allowed_ids]
    return pd.DataFrame(rows), meta


def generate_quantity_decisions(
    rewrite: QueryRewrite,
    selected_families: pd.DataFrame,
    candidate_families: pd.DataFrame,
    candidates: pd.DataFrame,
    warnings: list[str] | None = None,
) -> tuple[pd.DataFrame, bool, bool, str, str, dict[str, Any], dict[str, Any], int]:
    selected_with_history = build_historical_quantity_context(selected_families, candidate_families, candidates, rewrite)
    relation_count = sum(len(item.get("historical_relations") or []) for item in selected_with_history)
    prompt = build_quantity_decision_prompt(rewrite, selected_with_history)
    max_tokens = 3072
    if selected_families.empty:
        trace = trace_row(
            "quantity_decision",
            "判断已选 family 的工程量区间和金额口径",
            True,
            prompt=prompt,
            max_tokens=max_tokens,
            input_summary="selected=0, relations=0",
        )
        return pd.DataFrame(), True, False, "", prompt, trace, {}, relation_count
    try:
        response = request_llm_json_with_usage(
            prompt,
            max_tokens=max_tokens,
            system_prompt="你只输出一个 JSON object，不输出解释、Markdown 或思考过程。",
        )
        decisions, meta = parse_quantity_decision_result(response.content, selected_families, warnings)
        trace = trace_row(
            "quantity_decision",
            "判断已选 family 的工程量区间和金额口径",
            True,
            prompt=prompt,
            max_tokens=max_tokens,
            input_summary=f"selected={len(selected_families)}, relations={relation_count}",
            usage=response.usage,
        )
        return decisions, True, False, "", prompt, trace, meta, relation_count
    except (LLMServiceError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        append_warning(warnings, "quantity_decision_failed")
        fallback = []
        for _index, row in selected_families.iterrows():
            fallback.append(
                {
                    "family_id": cell_text(row.get("family_id")),
                    "quantity_source": "需现场确认",
                    "suggested_quantity_low": None,
                    "suggested_quantity_mid": None,
                    "suggested_quantity_high": None,
                    "include_in_amount": False,
                    "quantity_reason": "",
                    "confirmation_note": "工程量及计价范围需确认",
                }
            )
        trace = trace_row(
            "quantity_decision",
            "判断已选 family 的工程量区间和金额口径",
            False,
            error=str(exc),
            prompt=prompt,
            max_tokens=max_tokens,
            input_summary=f"selected={len(selected_families)}, relations={relation_count}",
        )
        meta = {"invalid_family_ids": [], "duplicate_family_ids": [], "invalid_quantity_sources": [], "invalid_quantity_ranges": []}
        return pd.DataFrame(fallback), False, True, str(exc), prompt, trace, meta, relation_count


def build_display_historical_quantity_context(
    selected_displays: pd.DataFrame,
    candidate_families: pd.DataFrame,
    candidates: pd.DataFrame,
    parsed_query: QueryRewrite,
    max_relations_per_display: int = 5,
) -> list[dict[str, Any]]:
    if selected_displays.empty:
        return []
    family_map = {cell_text(row.get("family_id")): row for _index, row in candidate_families.iterrows()}
    signature_by_family = {family_id: cell_text(row.get("fine_signature")) for family_id, row in family_map.items()}
    family_by_display = {
        cell_text(row.get("display_id")): cell_text(row.get("selected_family_id"))
        for _index, row in selected_displays.iterrows()
    }
    selected_display_ids = set(family_by_display)

    contexts: list[dict[str, Any]] = []
    for _index, selected in selected_displays.iterrows():
        display_id = cell_text(selected.get("display_id"))
        family_id = cell_text(selected.get("selected_family_id"))
        family = family_map.get(family_id)
        if family is None:
            continue
        unit = cell_text(family.get("unit_normalized")) or cell_text(family.get("unit"))
        user_quantity = matched_user_quantity(parsed_query.parsed_quantities, unit)
        relations: list[dict[str, Any]] = []
        target_signature = signature_by_family.get(family_id, "")
        target_rows = candidates[candidates["fine_signature"].map(cell_text).eq(target_signature)].copy()
        if user_quantity is None and not target_rows.empty:
            for _target_index, target_row in target_rows.sort_values("final_score", ascending=False).iterrows():
                project_key = cell_text(target_row.get("project_key")) or cell_text(target_row.get("project_package_id"))
                if not project_key:
                    continue
                related_display_ids = [item for item in selected_display_ids if item != display_id]
                for related_display_id in related_display_ids:
                    if related_display_id == display_id:
                        continue
                    related_family_id = family_by_display.get(related_display_id, "")
                    related_signature = signature_by_family.get(related_family_id, "")
                    related_rows = candidates[
                        candidates["fine_signature"].map(cell_text).eq(related_signature)
                        & (
                            candidates.get("project_key", pd.Series(dtype=object)).map(cell_text).eq(project_key)
                            | candidates.get("project_package_id", pd.Series(dtype=object)).map(cell_text).eq(project_key)
                        )
                    ]
                    for _related_index, related_row in related_rows.iterrows():
                        related_quantity = numeric_or_none(related_row.get("quantity"))
                        target_quantity = numeric_or_none(target_row.get("quantity"))
                        if related_quantity is None or target_quantity is None:
                            continue
                        relations.append(
                            {
                                "related_display_id": related_display_id,
                                "related_quantity": related_quantity,
                                "related_unit": cell_text(related_row.get("unit_normalized")) or cell_text(related_row.get("unit")),
                                "target_quantity": target_quantity,
                                "target_unit": unit,
                                "_score": float(target_row.get("final_score") or 0.0),
                            }
                        )
        seen_relation_keys: set[tuple[Any, ...]] = set()
        compact_relations: list[dict[str, Any]] = []
        for relation in sorted(relations, key=lambda item: item.get("_score", 0.0), reverse=True):
            key = (
                relation["related_display_id"],
                relation["related_quantity"],
                relation["related_unit"],
                relation["target_quantity"],
                relation["target_unit"],
            )
            if key in seen_relation_keys:
                continue
            seen_relation_keys.add(key)
            relation.pop("_score", None)
            compact_relations.append(relation)
            if len(compact_relations) >= max_relations_per_display:
                break
        contexts.append(
            {
                "display_id": display_id,
                "display_name": cell_text(selected.get("display_name")),
                "selected_family_id": family_id,
                "default_practice": cell_text(selected.get("default_practice")),
                "unit": unit,
                "user_quantity_match": user_quantity,
                "historical_relations": compact_relations,
            }
        )
    return contexts


def build_display_quantity_decision_prompt(
    rewrite: QueryRewrite,
    selected_displays_with_history: list[dict[str, Any]],
) -> str:
    payload = {
        "raw_query": rewrite.raw_query,
        "parsed_quantities": rewrite.parsed_quantities,
        "selected_displays": selected_displays_with_history,
    }
    return f"""
你是物业维修工程量与金额口径判断助手。

根据用户需求、已选 display 和历史数量关系，为每个 display 判断工程量范围及是否计入参考金额。
价格和单位来自 selected_family_id，但本阶段只输出 display_id。

quantity_source 只能取：
- 用户明确给定
- 历史样本估算
- 需现场确认

规则：
1. 用户数量能直接对应施工项单位时，使用“用户明确给定”。
2. 历史相似工程存在稳定、可解释的规模关系时，可使用“历史样本估算”。
3. 不得机械复制单条历史数量；历史关系不稳定或现场条件影响较大时，使用“需现场确认”，数量填 null。
4. suggested_quantity_low、suggested_quantity_mid、suggested_quantity_high 分别表示最低、最可能、最高估计；用户明确数量时三者相同。
5. include_in_amount=true 仅当该项属于实际建议方案、数量有合理依据且不会与替代方案重复计算。
6. 数量全部为 null 时必须不计入。
7. 不得新增 display，不得修改名称、默认做法和单位。
8. confirmation_note 只写该项仍需确认的关键因素，没有则填空字符串。
9. 只输出 JSON。

输出：
{{
  "display_quantities": [
    {{
      "display_id": "D001",
      "quantity_source": "用户明确给定",
      "suggested_quantity_low": 500,
      "suggested_quantity_mid": 500,
      "suggested_quantity_high": 500,
      "include_in_amount": true,
      "quantity_reason": "用户明确给出约500m²",
      "confirmation_note": "需确认实际施工边界"
    }}
  ]
}}

输入：
{json_text(payload)}
""".strip()


def parse_display_quantity_decision_result(
    result: dict[str, Any],
    selected_displays: pd.DataFrame,
    warnings: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    allowed_ids = [cell_text(value) for value in selected_displays.get("display_id", pd.Series(dtype=object)).tolist()]
    raw_rows = result.get("display_quantities")
    if not isinstance(raw_rows, list):
        raise ValueError("LLM 输出缺少 display_quantities list")
    rows_by_id: dict[str, dict[str, Any]] = {}
    meta = {"invalid_display_ids": [], "duplicate_display_ids": [], "invalid_quantity_sources": [], "invalid_quantity_ranges": []}
    for item in raw_rows:
        if not isinstance(item, dict):
            continue
        display_id = cell_text(item.get("display_id"))
        if display_id not in allowed_ids:
            if display_id:
                meta["invalid_display_ids"].append(display_id)
                append_warning(warnings, "invalid_quantity_display_ids")
            continue
        if display_id in rows_by_id:
            meta["duplicate_display_ids"].append(display_id)
            append_warning(warnings, "duplicate_quantity_display_ids")
            continue
        quantity_source = cell_text(item.get("quantity_source"))
        if quantity_source not in ALLOWED_QUANTITY_SOURCES:
            meta["invalid_quantity_sources"].append(quantity_source or display_id)
            append_warning(warnings, "invalid_quantity_sources")
            continue
        low = numeric_or_none(item.get("suggested_quantity_low"))
        mid = numeric_or_none(item.get("suggested_quantity_mid"))
        high = numeric_or_none(item.get("suggested_quantity_high"))
        quantities = [value for value in [low, mid, high] if value is not None]
        if quantities and (low is None or mid is None or high is None or low > mid or mid > high):
            low = mid = high = None
            meta["invalid_quantity_ranges"].append(display_id)
            append_warning(warnings, "invalid_quantity_ranges")
        include = parse_bool(item.get("include_in_amount"))
        if low is None and mid is None and high is None:
            include = False
        rows_by_id[display_id] = {
            "display_id": display_id,
            "quantity_source": quantity_source,
            "suggested_quantity_low": low,
            "suggested_quantity_mid": mid,
            "suggested_quantity_high": high,
            "include_in_amount": include,
            "quantity_reason": cell_text(item.get("quantity_reason")),
            "confirmation_note": cell_text(item.get("confirmation_note")),
        }
    for display_id in allowed_ids:
        if display_id not in rows_by_id:
            rows_by_id[display_id] = {
                "display_id": display_id,
                "quantity_source": "需现场确认",
                "suggested_quantity_low": None,
                "suggested_quantity_mid": None,
                "suggested_quantity_high": None,
                "include_in_amount": False,
                "quantity_reason": "",
                "confirmation_note": "工程量及计价范围需确认",
            }
    rows = [rows_by_id[display_id] for display_id in allowed_ids]
    return pd.DataFrame(rows), meta


def generate_display_quantity_decisions(
    rewrite: QueryRewrite,
    selected_displays: pd.DataFrame,
    candidate_families: pd.DataFrame,
    candidates: pd.DataFrame,
    warnings: list[str] | None = None,
) -> tuple[pd.DataFrame, bool, bool, str, str, dict[str, Any], dict[str, Any], int]:
    selected_with_history = build_display_historical_quantity_context(selected_displays, candidate_families, candidates, rewrite)
    relation_count = sum(len(item.get("historical_relations") or []) for item in selected_with_history)
    prompt = build_display_quantity_decision_prompt(rewrite, selected_with_history)
    max_tokens = 3072
    if selected_displays.empty:
        trace = trace_row(
            "quantity_decision",
            "判断已选 display 的工程量区间和金额口径",
            True,
            prompt=prompt,
            max_tokens=max_tokens,
            input_summary="selected=0, relations=0",
        )
        return pd.DataFrame(), True, False, "", prompt, trace, {}, relation_count
    try:
        response = request_llm_json_with_usage(
            prompt,
            max_tokens=max_tokens,
            system_prompt="你只输出一个 JSON object，不输出解释、Markdown 或思考过程。",
        )
        decisions, meta = parse_display_quantity_decision_result(response.content, selected_displays, warnings)
        trace = trace_row(
            "quantity_decision",
            "判断已选 display 的工程量区间和金额口径",
            True,
            prompt=prompt,
            max_tokens=max_tokens,
            input_summary=f"selected={len(selected_displays)}, relations={relation_count}",
            usage=response.usage,
        )
        return decisions, True, False, "", prompt, trace, meta, relation_count
    except (LLMServiceError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        append_warning(warnings, "quantity_decision_failed")
        fallback = []
        for _index, row in selected_displays.iterrows():
            fallback.append(
                {
                    "display_id": cell_text(row.get("display_id")),
                    "quantity_source": "需现场确认",
                    "suggested_quantity_low": None,
                    "suggested_quantity_mid": None,
                    "suggested_quantity_high": None,
                    "include_in_amount": False,
                    "quantity_reason": "",
                    "confirmation_note": "工程量及计价范围需确认",
                }
            )
        trace = trace_row(
            "quantity_decision",
            "判断已选 display 的工程量区间和金额口径",
            False,
            error=str(exc),
            prompt=prompt,
            max_tokens=max_tokens,
            input_summary=f"selected={len(selected_displays)}, relations={relation_count}",
        )
        meta = {"invalid_display_ids": [], "duplicate_display_ids": [], "invalid_quantity_sources": [], "invalid_quantity_ranges": []}
        return pd.DataFrame(fallback), False, True, str(exc), prompt, trace, meta, relation_count


def amount_calc_note(row: pd.Series, family_id: str) -> str:
    quantities = [
        numeric_or_none(row.get("建议工程量最低值")),
        numeric_or_none(row.get("建议工程量中位数")),
        numeric_or_none(row.get("建议工程量最高值")),
    ]
    prices = [
        numeric_or_none(row.get("综合单价最低值")),
        numeric_or_none(row.get("综合单价中位数")),
        numeric_or_none(row.get("综合单价最高值")),
    ]
    if not any(value is not None for value in quantities):
        note = "缺少可计算工程量，暂不计算金额，仅保留历史单价参考"
    elif not any(value is not None for value in prices):
        note = "缺少历史单价，暂不计算金额"
    else:
        note = f"按建议工程量低/中/高 × 历史综合单价低/中/高计算；单价来自 {family_id} 的 fine_signature 历史样本统计"
    if cell_text(row.get("是否计入参考金额区间")) == "否":
        note = f"{note}；本行不参与参考金额区间汇总"
    return note


def build_final_suggested_bill(
    selected_displays: pd.DataFrame,
    quantity_decisions: pd.DataFrame,
    candidate_families: pd.DataFrame,
) -> pd.DataFrame:
    if selected_displays.empty:
        return pd.DataFrame(columns=SUGGESTED_BILL_COLUMNS)
    family_map = {cell_text(row.get("family_id")): row for _index, row in candidate_families.iterrows()}
    quantity_map = {cell_text(row.get("display_id")): row for _index, row in quantity_decisions.iterrows()}
    rows: list[dict[str, Any]] = []
    for _index, selected in selected_displays.iterrows():
        display_id = cell_text(selected.get("display_id"))
        family_id = cell_text(selected.get("selected_family_id"))
        family = family_map.get(family_id)
        decision = quantity_map.get(display_id)
        if family is None or decision is None:
            continue
        low_quantity = numeric_or_none(decision.get("suggested_quantity_low"))
        mid_quantity = numeric_or_none(decision.get("suggested_quantity_mid"))
        high_quantity = numeric_or_none(decision.get("suggested_quantity_high"))
        other_practices = selected.get("other_practices") if isinstance(selected.get("other_practices"), list) else []
        other_text = "\n".join(
            f"{cell_text(item.get('family_id'))}：{cell_text(item.get('difference'))}"
            for item in other_practices
            if isinstance(item, dict) and cell_text(item.get("family_id"))
        )
        row = {
            "序号": len(rows) + 1,
            "display_id": display_id,
            "清单名称": cell_text(selected.get("display_name")),
            "单位": cell_text(family.get("unit_normalized")) or cell_text(family.get("unit")),
            "默认参考做法": cell_text(selected.get("default_practice")) or normalize_display_description(family.get("representative_project_description")),
            "selected_family_id": family_id,
            "其他历史做法": other_text,
            "历史family数量": selected.get("family_count", ""),
            "默认family历史样本数": family.get("历史样本数"),
            "默认family来源工程包数": family.get("来源工程包数"),
            "工程量来源": cell_text(decision.get("quantity_source")),
            "建议工程量最低值": low_quantity,
            "建议工程量中位数": mid_quantity,
            "建议工程量最高值": high_quantity,
            "工程量依据": cell_text(decision.get("quantity_reason")),
            "是否计入参考金额区间": "是" if bool(decision.get("include_in_amount")) else "否",
            "综合单价最低值": family.get("历史综合单价最低值"),
            "综合单价中位数": family.get("历史综合单价中位数"),
            "综合单价最高值": family.get("历史综合单价最高值"),
            "其中包含人工费单价最低值": family.get("历史人工单价最低值"),
            "其中包含人工费单价中位数": family.get("历史人工单价中位数"),
            "其中包含人工费单价最高值": family.get("历史人工单价最高值"),
            "其中包含机械费单价最低值": family.get("历史机械单价最低值"),
            "其中包含机械费单价中位数": family.get("历史机械单价中位数"),
            "其中包含机械费单价最高值": family.get("历史机械单价最高值"),
            "推荐依据": join_non_empty([selected.get("selection_reason"), selected.get("family_selection_reason")]),
            "需确认事项": cell_text(decision.get("confirmation_note")),
            "来源样本": ", ".join(split_refs(family.get("source_refs"), 10)),
        }
        amount_pairs = [
            ("估算金额最低值", low_quantity, row["综合单价最低值"]),
            ("估算金额中位数", mid_quantity, row["综合单价中位数"]),
            ("估算金额最高值", high_quantity, row["综合单价最高值"]),
            ("估算金额中包含人工费最低值", low_quantity, row["其中包含人工费单价最低值"]),
            ("估算金额中包含人工费中位数", mid_quantity, row["其中包含人工费单价中位数"]),
            ("估算金额中包含人工费最高值", high_quantity, row["其中包含人工费单价最高值"]),
            ("估算金额中包含机械费最低值", low_quantity, row["其中包含机械费单价最低值"]),
            ("估算金额中包含机械费中位数", mid_quantity, row["其中包含机械费单价中位数"]),
            ("估算金额中包含机械费最高值", high_quantity, row["其中包含机械费单价最高值"]),
        ]
        for column, quantity, price in amount_pairs:
            row[column] = calc_amount(quantity, price)
        row["金额计算口径"] = amount_calc_note(pd.Series(row), family_id)
        rows.append(row)
    output = pd.DataFrame(rows, columns=SUGGESTED_BILL_COLUMNS)
    return output.fillna("")


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


def calc_amount(quantity: Any, unit_price: Any) -> float | None:
    quantity_number = numeric_or_none(quantity)
    price_number = numeric_or_none(unit_price)
    if quantity_number is None or price_number is None:
        return None
    return round(quantity_number * price_number, 2)


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


def short_description(value: Any, limit: int = 28) -> str:
    text = re.sub(r"\s+", " ", cell_text(value)).strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def display_item_label(row: pd.Series) -> str:
    name = cell_text(row.get("清单名称")) or cell_text(row.get("清单项名称"))
    description = short_description(row.get("默认参考做法") or row.get("项目特征/施工工艺"))
    if name and description:
        return f"{name}（{description}）"
    return name or description


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

    demand_parts = [f"用户需求为“{rewrite.raw_query}”"]
    if rewrite.repair_object:
        demand_parts.append(f"维修对象为{rewrite.repair_object}")
    if rewrite.materials_or_specs:
        demand_parts.append(f"涉及{join_non_empty(rewrite.materials_or_specs)}")
    if quantity_texts:
        demand_parts.append(f"明确{join_non_empty(quantity_texts)}")
    demand_understanding = "，".join(demand_parts) + "。"

    catalog_parts = [
        f"catalog_id={query_catalog.catalog_id or '未匹配'}",
        f"一级分类={query_catalog.一级分类 or '未匹配'}",
        f"二级分类={query_catalog.二级分类 or '未匹配'}",
        f"维修状态={query_catalog.维修状态 or '未匹配'}",
        f"标准对象={query_catalog.标准对象 or '未匹配'}",
    ]
    catalog_summary = "；".join(catalog_parts) + "。"

    if "清单名称" in suggested_bill.columns or "清单项名称" in suggested_bill.columns:
        overview = join_non_empty([display_item_label(row) for _index, row in suggested_bill.iterrows()], limit=10)
        suggested_overview = f"建议清单包括：{overview}。" if overview else "当前未形成可展示的建议方案概览。"
    else:
        suggested_overview = "当前未形成可展示的建议方案概览。"

    amount_bill = suggested_bill.copy()
    if "是否计入参考金额区间" not in amount_bill.columns:
        amount_bill["是否计入参考金额区间"] = ""
    amount_bill = amount_bill[amount_bill["是否计入参考金额区间"].map(cell_text).eq("是")]

    included_items = join_non_empty([display_item_label(row) for _index, row in amount_bill.iterrows()], limit=20)
    if included_items:
        included_items_text = included_items + "。"
    else:
        included_items_text = "当前没有清单项计入参考金额区间。"

    low_amount = amount_sum(amount_bill, "估算金额最低值")
    mid_amount = amount_sum(amount_bill, "估算金额中位数")
    high_amount = amount_sum(amount_bill, "估算金额最高值")
    if low_amount is not None and mid_amount is not None and high_amount is not None:
        amount_range = (
            f"当前计入参考金额区间项目的参考金额约为 {low_amount:,.2f} - {high_amount:,.2f} 元，"
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
            f"当前计入参考金额区间项目已有部分金额参考：{join_non_empty(amount_parts)}；"
            "因部分项目缺少可计算工程量，区间可能不完整。"
        )
    else:
        amount_range = "当前没有可计入参考金额区间的可计算项目，暂不汇总总价，仅提供历史单价参考。"

    confirmation_values = []
    confirmation_column = "需确认事项" if "需确认事项" in suggested_bill.columns else "不确定性说明"
    suggested_confirmations = suggested_bill[confirmation_column].tolist() if confirmation_column in suggested_bill.columns else []
    for value in [*rewrite.uncertainties, *suggested_confirmations]:
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
        ("匹配分类", catalog_summary),
        ("建议方案概览", suggested_overview),
        ("计入金额项目", included_items_text),
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
    candidate_pool_row_count: int,
    evidence_item_row_count: int,
    candidate_family_count: int,
    candidate_display_group_count: int,
    display_selection_input_count: int,
    display_selection_selected_count: int,
    display_selection_trace: dict[str, Any],
    display_selection_fallback: bool,
    display_selection_error: str,
    display_selection_meta: dict[str, Any],
    display_family_selection_display_count: int,
    display_family_selection_trace: dict[str, Any],
    display_family_selection_fallback: bool,
    display_family_selection_error: str,
    display_family_selection_meta: dict[str, Any],
    dedup_selection_input_count: int,
    dedup_selection_output_count: int,
    dedup_selection_trace: dict[str, Any],
    dedup_selection_fallback: bool,
    dedup_selection_error: str,
    dedup_selection_meta: dict[str, Any],
    quantity_decision_input_count: int,
    quantity_relation_count: int,
    quantity_decision_trace: dict[str, Any],
    quantity_decision_fallback: bool,
    quantity_decision_error: str,
    quantity_decision_meta: dict[str, Any],
    output_path: Path | None,
    started_at: datetime,
    index_dir: Path,
    include_debug_text: bool,
    display_selection_prompt: str,
    display_family_selection_prompt: str,
    dedup_selection_prompt: str,
    quantity_decision_prompt: str,
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
        ("candidate_pool_row_count", candidate_pool_row_count),
        ("evidence_item_row_count", evidence_item_row_count),
        ("candidate_family_count", candidate_family_count),
        ("candidate_display_group_count", candidate_display_group_count),
        ("display_selection_input_count", display_selection_input_count),
        ("display_selection_candidate_count", len(display_selection_meta.get("candidate_ids") or [])),
        ("display_selection_candidate_ids", json_text(display_selection_meta.get("candidate_ids") or [])),
        ("display_selection_selected_count", display_selection_selected_count),
        ("display_selection_selected_ids", json_text(display_selection_meta.get("selected_ids") or [])),
        ("display_selection_selected_detail", json_text(display_selection_meta.get("selected_detail") or [])),
        ("display_selection_candidate_source_counts", json_text(display_selection_meta.get("candidate_source_counts") or {})),
        ("display_selection_exploration_count", display_selection_meta.get("exploration_count", "")),
        ("display_selection_prompt_chars", display_selection_trace.get("prompt_chars", "")),
        ("display_selection_prompt_tokens", display_selection_trace.get("prompt_tokens") or display_selection_trace.get("estimated_tokens", "")),
        ("display_selection_completion_tokens", display_selection_trace.get("completion_tokens", "")),
        ("display_family_selection_display_count", display_family_selection_display_count),
        ("display_family_selection_selected_family_ids", json_text(display_family_selection_meta.get("selected_family_ids") or [])),
        ("display_family_selection_other_family_count", display_family_selection_meta.get("other_family_count", "")),
        ("display_family_selection_other_family_ids", json_text(display_family_selection_meta.get("other_family_ids") or [])),
        ("display_family_selection_prompt_chars", display_family_selection_trace.get("prompt_chars", "")),
        ("display_family_selection_prompt_tokens", display_family_selection_trace.get("prompt_tokens") or display_family_selection_trace.get("estimated_tokens", "")),
        ("display_family_selection_completion_tokens", display_family_selection_trace.get("completion_tokens", "")),
        ("dedup_selection_input_count", dedup_selection_input_count),
        ("dedup_selection_output_count", dedup_selection_output_count),
        ("dedup_input_ids", json_text(dedup_selection_meta.get("input_ids") or [])),
        ("dedup_keep_ids", json_text(dedup_selection_meta.get("keep_ids") or [])),
        ("dedup_suppress_detail", json_text(dedup_selection_meta.get("suppress_detail") or [])),
        ("dedup_keep_reasons", json_text(dedup_selection_meta.get("keep_reasons") or [])),
        ("dedup_selection_prompt_chars", dedup_selection_trace.get("prompt_chars", "")),
        ("dedup_selection_prompt_tokens", dedup_selection_trace.get("prompt_tokens") or dedup_selection_trace.get("estimated_tokens", "")),
        ("dedup_selection_completion_tokens", dedup_selection_trace.get("completion_tokens", "")),
        ("quantity_decision_input_count", quantity_decision_input_count),
        ("quantity_relation_count", quantity_relation_count),
        ("quantity_decision_prompt_chars", quantity_decision_trace.get("prompt_chars", "")),
        ("quantity_decision_prompt_tokens", quantity_decision_trace.get("prompt_tokens") or quantity_decision_trace.get("estimated_tokens", "")),
        ("quantity_decision_completion_tokens", quantity_decision_trace.get("completion_tokens", "")),
        ("invalid_display_ids", join_non_empty([*(display_selection_meta.get("invalid_display_ids") or []), *(display_family_selection_meta.get("invalid_display_ids") or []), *(quantity_decision_meta.get("invalid_display_ids") or [])])),
        ("duplicate_display_ids", join_non_empty([*(display_selection_meta.get("duplicate_display_ids") or []), *(quantity_decision_meta.get("duplicate_display_ids") or [])])),
        ("invalid_family_ids", join_non_empty(display_family_selection_meta.get("invalid_family_ids") or [])),
        ("invalid_quantity_sources", join_non_empty(quantity_decision_meta.get("invalid_quantity_sources") or [])),
        ("invalid_quantity_ranges", join_non_empty(quantity_decision_meta.get("invalid_quantity_ranges") or [])),
        ("dedup_auto_suppressed", join_non_empty(dedup_selection_meta.get("auto_suppressed") or [])),
        ("dedup_llm_suppressed", join_non_empty(dedup_selection_meta.get("llm_suppressed") or [])),
        ("是否 display_selection fallback", "是" if display_selection_fallback else "否"),
        ("是否 display_family_selection fallback", "是" if display_family_selection_fallback else "否"),
        ("是否 dedup_selection fallback", "是" if dedup_selection_fallback else "否"),
        ("是否 quantity_decision fallback", "是" if quantity_decision_fallback else "否"),
        ("display_selection LLM error", display_selection_error),
        ("display_family_selection LLM error", display_family_selection_error),
        ("dedup_selection LLM error", dedup_selection_error),
        ("quantity_decision LLM error", quantity_decision_error),
        ("output_path", str(output_path or "")),
        ("运行时间", f"{(datetime.now() - started_at).total_seconds():.2f}s"),
        ("index_dir", str(index_dir)),
        ("主要文件路径", json_text((meta.get("files") or {}))),
        ("rewrite_notes", "；".join(rewrite.notes)),
        ("catalog_notes", "；".join(query_catalog.notes)),
        ("warnings", "；".join(warnings or [])),
    ]
    if include_debug_text:
        rows.append(("display_selection_prompt_preview", display_selection_prompt[:3000]))
        rows.append(("display_family_selection_prompt_preview", display_family_selection_prompt[:3000]))
        rows.append(("dedup_selection_prompt_preview", dedup_selection_prompt[:3000]))
        rows.append(("quantity_decision_prompt_preview", quantity_decision_prompt[:3000]))
    return pd.DataFrame(rows, columns=["字段", "值"])


def write_query_result_workbook(output_path: Path, result: QueryResult, display: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        display_frame(result.estimate_summary, display).to_excel(writer, sheet_name="estimate_summary", index=False)
        display_frame(result.suggested_bill, display).to_excel(writer, sheet_name="suggested_bill", index=False)
        display_frame(result.candidate_display_groups, display).to_excel(
            writer,
            sheet_name="candidate_display_groups",
            index=False,
        )
        display_frame(result.candidate_families, display).to_excel(
            writer,
            sheet_name="candidate_families",
            index=False,
        )
        display_frame(result.display_selection_trace, display).to_excel(
            writer,
            sheet_name="display_selection_trace",
            index=False,
        )
        display_frame(result.display_family_selection_trace, display).to_excel(
            writer,
            sheet_name="display_family_selection_trace",
            index=False,
        )
        display_frame(result.matched_project_packages, display).to_excel(
            writer,
            sheet_name="matched_project_packages",
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
    family_selection_limit: int = 50,
    family_exploration_limit: int = 5,
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
    candidate_families = build_candidate_families(candidates)
    evidence_items = build_evidence_items(candidates, candidate_families)
    candidate_display_groups, display_group_families = build_candidate_display_groups(candidate_families, evidence_items)
    matched_project_packages = matched_project_packages_for_output(matched_raw)

    (
        selected_displays,
        display_selection_success,
        display_selection_fallback,
        display_selection_error,
        display_selection_prompt,
        display_selection_trace,
        display_selection_meta,
        display_selection_trace_frame,
    ) = generate_display_selection(
        rewrite,
        query_catalog,
        candidate_display_groups,
        display_selection_limit=family_selection_limit,
        exploration_limit=family_exploration_limit,
        warnings=warnings,
    )
    (
        selected_display_practices,
        display_family_selection_success,
        display_family_selection_fallback,
        display_family_selection_error,
        display_family_selection_prompt,
        display_family_selection_trace,
        display_family_selection_meta,
        display_family_selection_trace_frame,
    ) = generate_display_family_selection(
        rewrite,
        selected_displays,
        candidate_display_groups,
        display_group_families,
        candidate_families,
        warnings=warnings,
    )
    (
        deduped_displays,
        dedup_selection_success,
        dedup_selection_fallback,
        dedup_selection_error,
        dedup_selection_prompt,
        dedup_selection_trace,
        dedup_selection_meta,
    ) = generate_display_dedup_selection(
        rewrite,
        selected_display_practices,
        warnings=warnings,
    )
    (
        quantity_decisions,
        quantity_decision_success,
        quantity_decision_fallback,
        quantity_decision_error,
        quantity_decision_prompt,
        quantity_decision_trace,
        quantity_decision_meta,
        quantity_relation_count,
    ) = generate_display_quantity_decisions(
        rewrite,
        deduped_displays,
        candidate_families,
        candidates,
        warnings=warnings,
    )
    suggested_bill = build_final_suggested_bill(deduped_displays, quantity_decisions, candidate_families)
    estimate_summary = build_estimate_summary(rewrite, query_catalog, suggested_bill)
    if warnings:
        append_trace_warnings(display_selection_trace, warnings)
        append_trace_warnings(display_family_selection_trace, warnings)
        append_trace_warnings(dedup_selection_trace, warnings)
        append_trace_warnings(quantity_decision_trace, warnings)
    displays_for_llm_count = len(display_selection_meta.get("candidate_ids") or [])
    parse_info = build_parse_info(
        rewrite=rewrite,
        query_catalog=query_catalog,
        top_packages=top_packages,
        top_items=top_items,
        max_packages_per_cache_subject=max_packages_per_cache_subject,
        meta=meta,
        sample_count=len(samples),
        package_count=len(project_packages),
        candidate_pool_row_count=len(candidates),
        evidence_item_row_count=len(evidence_items),
        candidate_family_count=len(candidate_families),
        candidate_display_group_count=len(candidate_display_groups),
        display_selection_input_count=displays_for_llm_count,
        display_selection_selected_count=len(selected_displays),
        display_selection_trace=display_selection_trace,
        display_selection_fallback=display_selection_fallback,
        display_selection_error=display_selection_error,
        display_selection_meta=display_selection_meta,
        display_family_selection_display_count=len(selected_displays),
        display_family_selection_trace=display_family_selection_trace,
        display_family_selection_fallback=display_family_selection_fallback,
        display_family_selection_error=display_family_selection_error,
        display_family_selection_meta=display_family_selection_meta,
        dedup_selection_input_count=len(selected_display_practices),
        dedup_selection_output_count=len(deduped_displays),
        dedup_selection_trace=dedup_selection_trace,
        dedup_selection_fallback=dedup_selection_fallback,
        dedup_selection_error=dedup_selection_error,
        dedup_selection_meta=dedup_selection_meta,
        quantity_decision_input_count=len(deduped_displays),
        quantity_relation_count=quantity_relation_count,
        quantity_decision_trace=quantity_decision_trace,
        quantity_decision_fallback=quantity_decision_fallback,
        quantity_decision_error=quantity_decision_error,
        quantity_decision_meta=quantity_decision_meta,
        output_path=output,
        started_at=started_at,
        index_dir=index_dir,
        include_debug_text=include_debug_text,
        display_selection_prompt=display_selection_prompt,
        display_family_selection_prompt=display_family_selection_prompt,
        dedup_selection_prompt=dedup_selection_prompt,
        quantity_decision_prompt=quantity_decision_prompt,
        warnings=warnings,
    )
    llm_trace = pd.DataFrame(
        [rewrite_trace, catalog_trace, display_selection_trace, display_family_selection_trace, dedup_selection_trace, quantity_decision_trace],
        columns=LLM_TRACE_COLUMNS,
    )

    result = QueryResult(
        rewrite=rewrite,
        query_catalog=query_catalog,
        estimate_summary=estimate_summary,
        suggested_bill=suggested_bill,
        matched_project_packages=matched_project_packages,
        candidate_families=candidate_families,
        candidate_display_groups=candidate_display_groups,
        display_group_families=display_group_families,
        display_selection_trace=display_selection_trace_frame,
        display_family_selection_trace=display_family_selection_trace_frame,
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
    print(f"[DONE] candidate display groups: {len(result.candidate_display_groups)}")
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
            family_selection_limit=args.family_selection_limit,
            family_exploration_limit=args.family_exploration_limit,
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
