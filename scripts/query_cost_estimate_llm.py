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


DEFAULT_PACKAGE_WEIGHT_TEMPERATURE = 0.10

MATCHED_PROJECT_PACKAGE_COLUMNS = [
    "rank",
    "package_query_similarity",
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
    "本次召回样本数",
    "本次召回工程包数",
    "本次召回工程量最低值",
    "本次召回工程量中位数",
    "本次召回工程量最高值",
    "本次召回综合单价最低值",
    "本次召回综合单价中位数",
    "本次召回综合单价最高值",
    "本次召回合价最低值",
    "本次召回合价中位数",
    "本次召回合价最高值",
    "本次召回人工费单价最低值",
    "本次召回人工费单价中位数",
    "本次召回人工费单价最高值",
    "本次召回机械费单价最低值",
    "本次召回机械费单价中位数",
    "本次召回机械费单价最高值",
    "package_query_similarity最大值",
    "item_query_similarity最大值",
    "source_refs",
]

PACKAGE_EVIDENCE_WEIGHT_COLUMNS = [
    "project_package_id",
    "package_query_similarity",
    "package_evidence_weight",
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
    "package_query_similarity",
    "item_query_similarity",
]

CANDIDATE_DISPLAY_GROUP_COLUMNS = [
    "display_id",
    "display_key",
    "display_name",
    "unit",
    "family_count",
    "family_ids",
    "retrieval_package_support_ratio",
    "support_rank",
    "retrieval_item_count",
    "retrieval_package_count",
    "top_family_examples",
    "direct_item_similarity_max",
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
    "本次召回样本数",
    "本次召回工程包数",
    "item_query_similarity最大值",
]

DISPLAY_SELECTION_TRACE_COLUMNS = [
    "selection_rank",
    "display_id",
    "display_key",
    "display_name",
    "unit",
    "family_count",
    "family_ids",
    "retrieval_package_support_ratio",
    "support_rank",
    "retrieval_item_count",
    "retrieval_package_count",
    "direct_item_similarity_max",
    "candidate_source",
    "selected_by_llm",
    "selection_reason",
    "selection_source",
]

DISPLAY_FAMILY_SELECTION_TRACE_COLUMNS = [
    "display_id",
    "display_name",
    "practice_option_id",
    "representative_family_id",
    "practice_description",
    "family_id",
    "family_role",
    "representative_cost_item_name",
    "representative_project_description",
    "unit",
    "本次召回样本数",
    "本次召回工程包数",
    "item_query_similarity最大值",
    "unit_price_min",
    "unit_price_median",
    "unit_price_max",
    "group_reason",
]

ESTIMATE_SCENARIO_COLUMNS = [
    "方案顺序",
    "方案编号",
    "方案名称",
    "display_id",
    "清单名称",
    "单位",
    "选用工艺",
    "其他可选工艺",
    "项目说明",
    "工程量类型",
    "工程量最低值",
    "工程量中位数",
    "工程量最高值",
    "工程量依据",
    "合价最低值",
    "合价中位数",
    "合价最高值",
    "综合单价最低值",
    "综合单价中位数",
    "综合单价最高值",
    "其中包含人工费单价最低值",
    "其中包含人工费单价中位数",
    "其中包含人工费单价最高值",
    "其中包含机械费单价最低值",
    "其中包含机械费单价中位数",
    "其中包含机械费单价最高值",
    "价格证据样本数",
    "来源样本",
    "practice_option_id",
    "价格证据family",
]

ESTIMATE_SUMMARY_COLUMNS = [
    "方案顺序",
    "方案编号",
    "方案名称",
    "是否推荐方案",
    "方案说明",
    "主要施工内容",
    "与其他方案的核心差异",
    "计价项目数",
    "合价最低值",
    "合价中位数",
    "合价最高值",
    "待现场确认事项",
]

LLM_TRACE_COLUMNS = [
    "stage",
    "purpose",
    "prompt",
    "raw_response",
    "parsed_status",
    "error_message",
    "scenario_count",
    "scenario_item_count",
    "prompt_chars",
    "estimated_tokens",
    "max_tokens",
    "input_summary",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
]


@dataclass(frozen=True)
class QueryRewrite:
    raw_query: str
    project_package_query_text: str
    item_query_text: str
    notes: list[str]
    success: bool


@dataclass(frozen=True)
class ScenarioItem:
    display_id: str
    practice_option_id: str
    selection_reason: str
    quantity: dict[str, Any]


@dataclass(frozen=True)
class EstimateScenario:
    scenario_id: str
    scenario_order: int
    scenario_name: str
    scenario_summary: str
    items: list[ScenarioItem]


@dataclass(frozen=True)
class QueryResult:
    rewrite: QueryRewrite
    estimate_summary: pd.DataFrame
    estimate_scenarios: pd.DataFrame
    matched_project_packages: pd.DataFrame
    candidate_families: pd.DataFrame
    candidate_display_groups: pd.DataFrame
    package_evidence_weights: pd.DataFrame
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
    parser.add_argument("--display-selection-limit", type=int, default=50, help="发送给 display_selection LLM 的 display 上限，默认 50")
    parser.add_argument("--display-exploration-limit", type=int, default=5, help="display_selection 等距探索候选上限，默认 5")
    parser.add_argument(
        "--package-weight-temperature",
        type=float,
        default=DEFAULT_PACKAGE_WEIGHT_TEMPERATURE,
        help=f"工程包证据权重 softmax temperature，必须大于 0，默认 {DEFAULT_PACKAGE_WEIGHT_TEMPERATURE}",
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
    stage: str,
    purpose: str,
    success: bool,
    error: str = "",
    prompt: str = "",
    max_tokens: int | str = "",
    input_summary: str = "",
    usage: dict[str, Any] | None = None,
    raw_response: str = "",
    scenario_count: int | str = "",
    scenario_item_count: int | str = "",
) -> dict[str, Any]:
    usage = usage or {}
    return {
        "stage": stage,
        "purpose": purpose,
        "prompt": prompt,
        "raw_response": raw_response,
        "parsed_status": "success" if success else "failed",
        "error_message": error,
        "scenario_count": scenario_count,
        "scenario_item_count": scenario_item_count,
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
  "item_query_text": ""
}}

当前 embedding 结构：
1. project_package_text 由“工程名称、project_name_text、cost_item_name 去重列表”组成。
   project_package_query_text 用于匹配相似历史工程包，应描述用户明确表达或直接相关的维修工程场景，保持短检索 query，不要预设建议清单、前置项、措施项或替代工艺。
2. item_retrieval_text 由“cost_item_name、project_description、unit_normalized”组成。
   item_query_text 用于匹配相似清单行，应贴近用户明确表达的维修对象、材料规格和做法，不要扩展未明确发生的清单项。
3. item_query_text 必须非空。如果用户问得很粗，也输出宽泛 item query，不要留空。
4. 不扩展用户未明确提出的清单项。
5. 不输出数量分析、材料列表、不确定性、方案建议、价格或施工清单。

示例：
用户：屋面漏水，想做3mm SBS防水，面积大概500平
输出：{{"project_package_query_text":"屋面漏水维修工程 屋面防水维修 3mm SBS防水","item_query_text":"屋面卷材防水 3mm SBS防水卷材"}}

用户：屋面漏水帮我估价
输出：{{"project_package_query_text":"屋面漏水维修工程 屋面防水维修","item_query_text":"屋面防水 防水层维修"}}

用户：地下室渗水维修
输出：{{"project_package_query_text":"地下室渗水维修工程 地下室防水维修","item_query_text":"地下室防水 渗水维修 防水层维修"}}

用户需求：{query}
""".strip()


def fallback_query_rewrite(query: str, note: str) -> QueryRewrite:
    return QueryRewrite(
        raw_query=query,
        project_package_query_text=query,
        item_query_text=query,
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
    rows["package_query_similarity"] = scores[indices].astype(float)
    rows["package_dedupe_key"] = rows.apply(package_dedupe_key, axis=1)
    rows = rows.sort_values("package_query_similarity", ascending=False)
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


def score_direct_items(samples: pd.DataFrame, item_query_similarities: np.ndarray, top_items: int) -> pd.DataFrame:
    indices = top_score_indices(item_query_similarities, top_items)
    rows = samples.iloc[indices].copy()
    rows["item_query_similarity"] = item_query_similarities[indices].astype(float)
    return rows


def project_package_similarity_map(project_packages: pd.DataFrame, package_query_similarities: np.ndarray) -> dict[str, float]:
    if len(project_packages) != len(package_query_similarities):
        raise ValueError("工程包数量与 package_query_similarities 数量不一致")
    return {
        cell_text(row.get("project_package_id")): float(package_query_similarities[index])
        for index, (_row_index, row) in enumerate(project_packages.iterrows())
        if cell_text(row.get("project_package_id"))
    }


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
        score_map[package_id] = float(row.get("package_query_similarity") or 0.0)
        rank_map[package_id] = int(row.get("rank") or 0)
    return score_map, rank_map


def evidence_package_universe(matched_project_packages: pd.DataFrame, direct_item_hits: pd.DataFrame) -> list[str]:
    package_ids: list[str] = []
    seen: set[str] = set()
    for frame in [matched_project_packages, direct_item_hits]:
        if frame.empty or "project_package_id" not in frame.columns:
            continue
        for value in frame["project_package_id"].tolist():
            package_id = cell_text(value)
            if package_id and package_id not in seen:
                package_ids.append(package_id)
                seen.add(package_id)
    return package_ids


def build_package_evidence_weights(
    evidence_package_ids: list[str],
    package_query_similarity_by_id: dict[str, float],
    temperature: float,
) -> pd.DataFrame:
    if temperature <= 0:
        raise ValueError("package weight temperature 必须大于 0")
    package_ids = [package_id for package_id in evidence_package_ids if cell_text(package_id)]
    if not package_ids:
        return pd.DataFrame(columns=PACKAGE_EVIDENCE_WEIGHT_COLUMNS)

    similarities = np.array(
        [float(package_query_similarity_by_id.get(package_id, 0.0)) for package_id in package_ids],
        dtype=np.float64,
    )
    max_similarity = float(np.max(similarities))
    raw_weights = np.exp((similarities - max_similarity) / float(temperature))
    denominator = float(raw_weights.sum())
    weights = raw_weights / denominator if denominator > 0 else np.zeros_like(raw_weights)
    return pd.DataFrame(
        {
            "project_package_id": package_ids,
            "package_query_similarity": similarities.astype(float),
            "package_evidence_weight": weights.astype(float),
        },
        columns=PACKAGE_EVIDENCE_WEIGHT_COLUMNS,
    )


def build_retrieved_evidence_items(
    samples: pd.DataFrame,
    matched_project_packages: pd.DataFrame,
    direct_item_hits: pd.DataFrame,
    item_query_similarities: np.ndarray,
    package_query_similarity_by_id: dict[str, float] | None = None,
    warnings: list[str] | None = None,
) -> pd.DataFrame:
    package_query_similarity_map, package_rank_map = matched_package_maps(matched_project_packages)
    package_query_similarity_by_id = package_query_similarity_by_id or package_query_similarity_map
    matched_package_ids = list(package_query_similarity_map.keys())
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
    if rows["sample_index"].min() < 0 or rows["sample_index"].max() >= len(item_query_similarities):
        raise ValueError("sample_index 超出 item_embeddings 范围")

    rows["package_query_similarity"] = rows["project_package_id"].map(package_query_similarity_by_id).fillna(0.0).astype(float)
    rows["package_rank"] = rows["project_package_id"].map(package_rank_map)
    rows["item_query_similarity"] = rows["sample_index"].map(lambda sample_index: float(item_query_similarities[int(sample_index)]))
    rows["direct_hit"] = rows["sample_index"].isin(direct_indices)
    rows = attach_source_refs(rows, warnings)
    sort_columns = ["item_query_similarity", "package_query_similarity"]
    return rows.sort_values(sort_columns, ascending=[False, False]).reset_index(drop=True)


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
        group = group.sort_values(["item_query_similarity", "package_query_similarity"], ascending=[False, False])
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
                "本次召回样本数": int(len(group)),
                "本次召回工程包数": source_package_count(group),
                "本次召回工程量最低值": quantity_min,
                "本次召回工程量中位数": quantity_median,
                "本次召回工程量最高值": quantity_max,
                "本次召回综合单价最低值": unit_price_min,
                "本次召回综合单价中位数": unit_price_median,
                "本次召回综合单价最高值": unit_price_max,
                "本次召回合价最低值": total_price_min,
                "本次召回合价中位数": total_price_median,
                "本次召回合价最高值": total_price_max,
                "本次召回人工费单价最低值": labor_min,
                "本次召回人工费单价中位数": labor_median,
                "本次召回人工费单价最高值": labor_max,
                "本次召回机械费单价最低值": machinery_min,
                "本次召回机械费单价中位数": machinery_median,
                "本次召回机械费单价最高值": machinery_max,
                "package_query_similarity最大值": max_numeric_or_zero(group, "package_query_similarity"),
                "item_query_similarity最大值": max_numeric_or_zero(group, "item_query_similarity"),
                "source_refs": ordered_refs(group.get("source_ref", pd.Series(dtype=object)), limit=10),
            }
        )

    output = pd.DataFrame(rows)
    output = output.sort_values(["item_query_similarity最大值", "本次召回样本数", "本次召回工程包数"], ascending=[False, False, False])
    output.insert(0, "family_id", [f"F{index:03d}" for index in range(1, len(output) + 1)])
    for column in CANDIDATE_FAMILY_COLUMNS:
        if column not in output.columns:
            output[column] = None
    return output[CANDIDATE_FAMILY_COLUMNS].reset_index(drop=True)


def attach_family_ids_to_evidence_items(candidates: pd.DataFrame, candidate_families: pd.DataFrame) -> pd.DataFrame:
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
            "package_query_similarity": candidates.get("package_query_similarity", ""),
            "item_query_similarity": candidates.get("item_query_similarity", ""),
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
        ["item_query_similarity最大值", "本次召回样本数", "本次召回工程包数"],
        ascending=[False, False, False],
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
                "samples": int(row.get("本次召回样本数") or 0),
                "packages": int(row.get("本次召回工程包数") or 0),
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
        group = group.sort_values(["item_query_similarity最大值", "本次召回样本数", "本次召回工程包数"], ascending=[False, False, False])
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
                    "retrieval_package_support_ratio": 0.0,
                    "retrieval_item_count": 0,
                    "retrieval_package_count": int(package_values.nunique()),
                    "top_family_examples": json_text(top_family_examples(group)),
                    "direct_item_similarity_max": max_numeric_or_zero(group, "item_query_similarity最大值"),
                },
                group,
            )
        )

    raw_groups.sort(
        key=lambda item: (
            -float(item[0].get("direct_item_similarity_max") or 0.0),
            -int(item[0].get("retrieval_package_count") or 0),
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
                    "本次召回样本数": family.get("本次召回样本数", ""),
                    "本次召回工程包数": family.get("本次召回工程包数", ""),
                    "item_query_similarity最大值": family.get("item_query_similarity最大值", ""),
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


def attach_display_support_ratios(
    candidate_display_groups: pd.DataFrame,
    display_group_families: pd.DataFrame,
    evidence_items: pd.DataFrame,
    package_evidence_weights: pd.DataFrame,
) -> pd.DataFrame:
    if candidate_display_groups.empty:
        return candidate_display_groups.copy()

    weight_map = {
        cell_text(row.get("project_package_id")): float(row.get("package_evidence_weight") or 0.0)
        for _index, row in package_evidence_weights.iterrows()
        if cell_text(row.get("project_package_id"))
    }
    family_ids_by_display = {
        display_id: set(group["family_id"].map(cell_text).tolist())
        for display_id, group in display_group_families.groupby("display_id", sort=False, dropna=False)
    }
    evidence_family_ids = evidence_items.get("family_id", pd.Series(dtype=object)).map(cell_text)
    evidence_package_ids = evidence_items.get("project_package_id", pd.Series(dtype=object)).map(cell_text)

    output = candidate_display_groups.copy()
    support_values: list[float] = []
    item_counts: list[int] = []
    package_counts: list[int] = []
    direct_item_similarity_values: list[float] = []
    for _index, row in output.iterrows():
        display_id = cell_text(row.get("display_id"))
        family_ids = family_ids_by_display.get(display_id, set())
        if not family_ids:
            support_values.append(0.0)
            item_counts.append(0)
            package_counts.append(0)
            direct_item_similarity_values.append(0.0)
            continue
        evidence = evidence_items[evidence_family_ids.isin(family_ids)].copy()
        package_ids = {
            package_id
            for package_id in evidence_package_ids.loc[evidence.index].tolist()
            if package_id
        }
        support_values.append(float(sum(weight_map.get(package_id, 0.0) for package_id in package_ids)))
        item_counts.append(int(len(evidence)))
        package_counts.append(int(len(package_ids)))
        direct_item_similarity_values.append(max_numeric_or_zero(evidence, "item_query_similarity"))

    output["retrieval_package_support_ratio"] = support_values
    output["retrieval_item_count"] = item_counts
    output["retrieval_package_count"] = package_counts
    output["direct_item_similarity_max"] = direct_item_similarity_values
    ranked = output.sort_values(
        ["retrieval_package_support_ratio", "retrieval_package_count", "retrieval_item_count", "display_id"],
        ascending=[False, False, False, True],
        kind="mergesort",
    )
    ranks = pd.Series(range(1, len(ranked) + 1), index=ranked.index)
    output["support_rank"] = ranks.reindex(output.index).astype(int)
    return output[CANDIDATE_DISPLAY_GROUP_COLUMNS].reset_index(drop=True)


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
            "retrieval_package_support_ratio",
            candidate_display_groups.sort_values(
                ["retrieval_package_support_ratio", "retrieval_package_count", "retrieval_item_count"],
                ascending=[False, False, False],
            ),
        ),
        (
            "direct_item_similarity",
            candidate_display_groups.sort_values(
                ["direct_item_similarity_max", "retrieval_package_support_ratio", "retrieval_item_count"],
                ascending=[False, False, False],
            ),
        ),
        (
            "retrieval_package_count",
            candidate_display_groups.sort_values(
                ["retrieval_package_count", "retrieval_item_count", "retrieval_package_support_ratio"],
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
            for item in raw_examples[:2]:
                if not isinstance(item, dict):
                    continue
                text = truncate_text(item.get("项目特征简述") or item.get("spec"), 60)
                if text:
                    examples.append(text)
        rows.append(
            {
                "id": cell_text(row.get("display_id")),
                "name": truncate_text(row.get("display_name"), 40),
                "unit": cell_text(row.get("unit")),
                "retrieval_package_support_ratio": round(float(row.get("retrieval_package_support_ratio") or 0.0), 3),
                "support_rank": int(row.get("support_rank") or 0),
                "examples": examples,
            }
        )
    return rows


def build_display_selection_prompt(
    rewrite: QueryRewrite,
    candidate_displays: pd.DataFrame,
) -> tuple[str, list[dict[str, Any]]]:
    records = display_selection_records(candidate_displays)
    payload = {
        "raw_query": rewrite.raw_query,
        "project_package_query_text": rewrite.project_package_query_text,
        "item_query_text": rewrite.item_query_text,
        "candidate_displays": records,
    }
    prompt = f"""
你是物业维修工程建议清单选择器。

【任务】

根据用户需求和本次召回候选证据，从 candidate_displays 中选择应进入本次建议清单的工作项。

每个 display 代表一个清单工作项，内部可能包含多个不同参考做法。
本阶段只判断哪些 display 应进入建议清单：

- 不选择具体 family；
- 不判断价格和工程量；
- 不撰写最终方案总结。

【retrieval_package_support_ratio】

retrieval_package_support_ratio 表示该工作项获得的本次召回工程包支持比例。

计算方法：
1. 对本次召回涉及的工程包，按照其与用户需求的整体语义相似度计算并归一化权重；
2. 将包含当前工作项的工程包权重相加；
3. 所得结果范围为 0 到 1。

例如 0.62 表示：
在本次召回工程包证据中，约 62% 的相关性权重支持该工作项。

该比例已经同时考虑：
- 该工作项出现于哪些本次召回工程包；
- 这些工程包与当前需求有多相似。

判断时可以比较不同候选的 retrieval_package_support_ratio，
但不能只按比例机械选择，还应结合工作项名称、做法示例及其与用户需求的实际关系。

【support_rank】

support_rank 表示该工作项按 retrieval_package_support_ratio
在本次全部候选 display 中的排名，1 表示本次召回工程包支持最高。

当某个工作项的 retrieval_package_support_ratio 绝对值不高，
但 support_rank 较靠前时，仍说明它相对于本次其他候选
具有较强的本次召回工程包支持。

support_rank 只表示相对支持强度，
仍需结合用户需求、工作项名称、做法示例、
施工关系、替代关系和现场条件综合判断。

请综合判断：

1. 用户明确描述的维修对象、问题、材料、规格和工程量；
2. display 与用户维修目标的直接相关性；
3. display 在本次召回相似工程包中的出现情况；
4. display 与其他拟选工作项之间是否存在合理的施工或配套关系；
5. 该工作项是否可能因现场条件、原有构造、施工组织或实施方案而需要。

用户通常只描述维修目标，不会完整列出实际工程中的全部清单工作项。
因此，不要只选择与用户原文措辞最相似的项目。

但本次召回工程包中出现过，也不代表当前工程一定需要。
只有当候选与本次需求存在清楚、可解释的关系时，才应选择。

【需求与检索文本使用规则】

1. display 业务选择必须优先依据 raw_query。
2. project_package_query_text 和 item_query_text 只用于理解本次召回来源和追溯检索语义。
3. 不得把检索文本当作用户新增需求，不得据此扩展用户未明确提出的清单项。

【选择规则】

1. 只能选择 candidate_displays 中存在的 display_id。
2. 同一 display_id 最多选择一次。
3. 不设置固定选择数量。
4. 不得仅因为现场条件尚未明确，就自动排除一个与当前工程有较强关系的候选。
5. 如果某个工作项是否实施依赖现场条件，可以选择，但 selection_reason 必须说明需要确认的条件。
6. 对于 retrieval_package_support_ratio 排名靠前，且与已选主要工作项存在清楚、合理施工关系或实施关系的候选，即使其是否实施依赖现场条件、原有构造、楼层高度、运输条件、施工组织、作业方式或实施方案，也不要仅因为这些条件尚未明确而直接排除。
7. 可以将上述候选条件性选入建议清单，并在 selection_reason 中说明它与当前主要工作项的关系，以及是否实施需要确认的具体条件。
8. 对实质重复、相互包含或通常互为替代的 display，不要同时选择，除非它们确实代表可以独立计价且同时实施的不同工作内容。

【selection_reason】

每个已选 display 的 selection_reason 必须说明：

1. 它为什么与用户需求相关；
2. 如果是否实施依赖现场条件，需要确认什么条件。

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
                "retrieval_package_support_ratio": row.get("retrieval_package_support_ratio", ""),
                "support_rank": row.get("support_rank", ""),
                "retrieval_item_count": row.get("retrieval_item_count", ""),
                "retrieval_package_count": row.get("retrieval_package_count", ""),
                "direct_item_similarity_max": row.get("direct_item_similarity_max", ""),
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
    prompt, records = build_display_selection_prompt(rewrite, displays_for_llm)
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


def family_selection_payload_for_display(
    display_id: str,
    display_group_families: pd.DataFrame,
    candidate_families: pd.DataFrame,
) -> list[dict[str, Any]]:
    family_map = {cell_text(row.get("family_id")): row for _index, row in candidate_families.iterrows()}
    rows = display_group_families[display_group_families["display_id"].map(cell_text).eq(display_id)].copy()
    rows = rows.sort_values(["item_query_similarity最大值", "本次召回样本数", "本次召回工程包数"], ascending=[False, False, False])
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
                "samples": int(row.get("本次召回样本数") or 0),
                "packages": int(row.get("本次召回工程包数") or 0),
                "item_query_similarity": round(float(row.get("item_query_similarity最大值") or 0.0), 4),
                "unit_price_min": family.get("本次召回综合单价最低值"),
                "unit_price_median": family.get("本次召回综合单价中位数"),
                "unit_price_max": family.get("本次召回综合单价最高值"),
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
        candidate_families_payload = family_selection_payload_for_display(display_id, display_group_families, candidate_families)
        candidate_family_ids = [cell_text(family.get("family_id")) for family in candidate_families_payload if cell_text(family.get("family_id"))]
        records.append(
            {
                "display_id": display_id,
                "display_name": truncate_text(display.get("display_name"), 40),
                "selection_reason": cell_text(selected.get("selection_reason")),
                "candidate_family_ids": candidate_family_ids,
                "candidate_family_count": len(candidate_family_ids),
                "candidate_families": candidate_families_payload,
            }
        )
    payload = {
        "raw_query": rewrite.raw_query,
        "project_package_query_text": rewrite.project_package_query_text,
        "item_query_text": rewrite.item_query_text,
        "selected_displays": records,
    }
    prompt = f"""
你是物业维修工程参考做法整理器。

【任务】

candidate_families 已经属于同一个 selected display。

本步骤不重新创建、拆分或选择 display，也不判断这些 family 是否应属于其他 display。

对于每个 selected display：

1. 将其内部全部 candidate_families 完整划分为一个或多个 practice_options；
2. 每个 practice_option 表示一种可以共同作为价格证据范围的明确参考做法；
3. 为每个 option 选择一个 representative_family_id；
4. 根据用户原始需求，从所有 options 中选择一个当前默认做法，使用 default_representative_family_id 表示；
5. 每个 candidate family 必须且只能属于一个 option。
6. 每个 selected display 输入中的 candidate_family_ids 是必须完整覆盖的 family_id 清单。

【practice option 分组原则】

1. 同一 option 内 family 的主要材料或设备类型、关键规格、主要施工或维修方法、施工对象、主要部位和核心工作内容应基本一致。
2. 如果 family 之间仅在 OCR、标点、文字表达，或基层清理、垃圾清运、现场保洁、运距、结算口径等附属范围存在差异，且这些差异不改变主要施工内容和价格口径，可以归入同一 option。
3. 如果材料种类、设备类型、关键规格、厚度、层数、主要施工或维修工艺、施工对象、主要部位或核心工作内容明显不同，应拆成不同 option。
4. 不要仅因为 cost_item_name 相同就放入同一 option。必须比较具体项目特征、材料或设备、规格、施工方法、施工对象、主要部位和核心工作内容。
5. 不要因为来源工程主目录与查询目录不一致而拆分，主要依据具体清单名称、项目特征和施工内容判断。
6. 每个 candidate family 必须且只能属于一个 option，不得遗漏、重复、修改或新增 family_id。
7. 即使某个 family 与当前默认做法不同，也必须放入某个其他 option；不得通过省略 family_id 表示排除。

【当前默认参考做法选择原则】

1. 优先匹配用户明确提出的材料、规格、部位和工艺。
2. 优先选择描述清晰、没有明显 OCR 歧义的 option。
3. 优先选择施工边界适合作为当前参考口径的 option。
4. 本次召回样本数和工程包数充分的 option 优先。
5. 样本数量不能覆盖材料、规格或施工边界不匹配。
6. default_representative_family_id 必须等于某个 practice_option 的 representative_family_id。
7. 当前默认 option 只用于现有后续链路继续运行，不表示其他 option 无效，也不限制后续 scenario 选择。

【需求与检索文本使用规则】

1. 默认参考做法选择必须优先依据 raw_query。
2. project_package_query_text 和 item_query_text 只用于理解本次召回来源和追溯检索语义。
3. 不得把检索文本当作用户新增需求，不得据此覆盖用户原始需求。

【覆盖校验】

对每个 selected display，必须满足：

1. flatten(practice_options[*].family_ids) 与输入的 candidate_family_ids 完全一致。
2. candidate_family_ids 中的每个 family_id 都必须出现且只能出现一次。
3. 不得输出 candidate_family_ids 之外的 family_id。
4. representative_family_id 也必须同时出现在自己 option 的 family_ids 中。
5. 不输出 exclude，不允许通过省略 family_id 表达排除或不采用。
6. 输出 JSON 前必须按 candidate_family_ids 顺序逐个检查：每个 family_id 已出现一次，且没有任何 family_id 出现两次。
7. 如果某个 family_id 在分组判断中发生调整，只输出最终归属；不得在 group_reason 中写“修正”“重新分组”“应归入”等自我修改过程。
8. group_reason 只说明最终分组理由，保持简短，不讨论未采用的分组方案。
9. family_assignments 必须按 candidate_family_ids 原顺序逐个列出每个 family_id 的最终 option 序号，用于自检覆盖完整性。

【输出文字限制】

1. practice_description 用一句短语描述该 option。
2. group_reason 只写最终理由，不超过 80 个汉字。
3. 不输出推理过程、检查过程、草稿、纠错说明或 Markdown。

【输出 JSON】

{{
  "display_results": [
    {{
      "display_id": "D001",
      "default_representative_family_id": "F004",
      "selection_reason": "与用户明确要求一致，描述清晰，本次召回样本可追溯",
      "family_assignments": [
        {{"family_id": "F004", "practice_option_index": 1}},
        {{"family_id": "F007", "practice_option_index": 1}},
        {{"family_id": "F010", "practice_option_index": 2}},
        {{"family_id": "F012", "practice_option_index": 2}}
      ],
      "practice_options": [
        {{
          "representative_family_id": "F004",
          "practice_description": "与用户需求匹配的参考做法",
          "family_ids": ["F004", "F007"],
          "group_reason": "主要材料、规格、施工方法和施工对象一致，仅附属施工范围存在差异"
        }},
        {{
          "representative_family_id": "F010",
          "practice_description": "另一种可选参考做法",
          "family_ids": ["F010", "F012"],
          "group_reason": "主要材料和施工方法一致，但与默认工艺不同"
        }}
      ]
    }}
  ]
}}

【输入数据】

{json_text(payload)}
""".strip()
    return prompt, records


def display_family_unit(row: pd.Series) -> str:
    return cell_text(row.get("unit_normalized")) or normalized_unit(row.get("unit"))


def empty_display_family_selection_meta() -> dict[str, Any]:
    return {
        "selected_display_ids": [],
        "practice_option_count": 0,
        "families_grouped_count": 0,
        "option_count_by_display": {},
        "max_options_per_display": 0,
    }


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
    allowed_order_by_display = {
        display_id: [
            family_id
            for family_id in display_group_families[
                display_group_families["display_id"].map(cell_text).eq(display_id)
            ]["family_id"].map(cell_text).tolist()
            if family_id
        ]
        for display_id in display_ids
    }
    allowed_by_display = {
        display_id: set(family_ids)
        for display_id, family_ids in allowed_order_by_display.items()
    }
    selected_by_display = {cell_text(row.get("display_id")): row for _index, row in selected_displays.iterrows()}
    display_map = {cell_text(row.get("display_id")): row for _index, row in candidate_display_groups.iterrows()}
    family_row_map = {
        (cell_text(row.get("display_id")), cell_text(row.get("family_id"))): row
        for _index, row in display_group_families.iterrows()
    }
    result_by_display: dict[str, dict[str, Any]] = {}
    meta = empty_display_family_selection_meta()

    for item in raw_rows:
        if not isinstance(item, dict):
            raise ValueError("display_family_selection display_result 必须为 object")
        display_id = cell_text(item.get("display_id"))
        if display_id not in allowed_by_display:
            raise ValueError(f"display_family_selection 返回未选中的 display: {display_id}")
        if display_id in result_by_display:
            raise ValueError(f"display_family_selection 重复返回 display: {display_id}")
        allowed_family_ids = allowed_by_display[display_id]
        if not allowed_family_ids:
            raise ValueError(f"display 缺少 candidate families: {display_id}")
        raw_options = item.get("practice_options")
        if not isinstance(raw_options, list) or not raw_options:
            raise ValueError(f"practice_options 必须为非空 list: {display_id}")

        practice_options: list[dict[str, Any]] = []
        seen_family_ids: set[str] = set()
        representative_family_ids: set[str] = set()
        for option_index, raw_option in enumerate(raw_options, start=1):
            if not isinstance(raw_option, dict):
                raise ValueError(f"practice_option 必须为 object: {display_id}/O{option_index:02d}")
            representative_family_id = cell_text(raw_option.get("representative_family_id"))
            practice_description = truncate_text(normalize_display_description(raw_option.get("practice_description")), 120)
            group_reason = cell_text(raw_option.get("group_reason"))
            raw_family_ids = raw_option.get("family_ids")
            if not representative_family_id:
                raise ValueError(f"practice_option 缺少 representative_family_id: {display_id}/O{option_index:02d}")
            if not practice_description:
                raise ValueError(f"practice_description 不得为空: {display_id}/{representative_family_id}")
            if not group_reason:
                raise ValueError(f"group_reason 不得为空: {display_id}/{representative_family_id}")
            if not isinstance(raw_family_ids, list) or not raw_family_ids:
                raise ValueError(f"family_ids 必须为非空 list: {display_id}/{representative_family_id}")
            family_ids = [cell_text(family_id) for family_id in raw_family_ids]
            if any(not family_id for family_id in family_ids):
                raise ValueError(f"family_ids 不得为空: {display_id}/{representative_family_id}")
            duplicate_in_option = [family_id for family_id in family_ids if family_ids.count(family_id) > 1]
            if duplicate_in_option:
                raise ValueError(f"同一 practice_option 内 family 重复: {display_id}/{join_non_empty(duplicate_in_option)}")
            unknown_family_ids = [family_id for family_id in family_ids if family_id not in allowed_family_ids]
            if unknown_family_ids:
                raise ValueError(f"practice_option 包含不属于当前 display 的 family: {display_id}/{join_non_empty(unknown_family_ids)}")
            duplicate_across_options = [family_id for family_id in family_ids if family_id in seen_family_ids]
            if duplicate_across_options:
                raise ValueError(f"family 不得出现在多个 practice_options: {display_id}/{join_non_empty(duplicate_across_options)}")
            if representative_family_id not in family_ids:
                raise ValueError(f"representative_family_id 必须位于自己的 family_ids 中: {display_id}/{representative_family_id}")
            if representative_family_id in representative_family_ids:
                raise ValueError(f"不同 option 的 representative family 不得重复: {display_id}/{representative_family_id}")
            option_units = [
                display_family_unit(family_row_map.get((display_id, family_id), pd.Series(dtype=object)))
                for family_id in family_ids
            ]
            if len(set(option_units)) > 1:
                raise ValueError(f"同一 practice_option 中 family 单位必须一致: {display_id}/{representative_family_id}")
            sample_count = 0
            for family_id in family_ids:
                family_row = family_row_map.get((display_id, family_id), pd.Series(dtype=object))
                sample_count += int(numeric_or_none(family_row.get("本次召回样本数")) or 0)

            representative_family_ids.add(representative_family_id)
            seen_family_ids.update(family_ids)
            practice_option_id = f"{display_id}-O{option_index:02d}"
            practice_options.append(
                {
                    "practice_option_id": practice_option_id,
                    "representative_family_id": representative_family_id,
                    "practice_description": practice_description,
                    "sample_count": sample_count,
                    "family_ids": family_ids,
                    "group_reason": group_reason,
                }
            )

        missing_family_ids = [family_id for family_id in allowed_order_by_display[display_id] if family_id not in seen_family_ids]
        if missing_family_ids:
            raise ValueError(f"practice_options 遗漏 candidate family: {display_id}/{join_non_empty(missing_family_ids)}")
        extra_family_ids = [family_id for family_id in seen_family_ids if family_id not in allowed_family_ids]
        if extra_family_ids:
            raise ValueError(f"practice_options 新增非法 family: {display_id}/{join_non_empty(extra_family_ids)}")
        raw_assignments = item.get("family_assignments")
        if isinstance(raw_assignments, list):
            assignment_ids: list[str] = []
            option_by_family = {
                family_id: index
                for index, option in enumerate(practice_options, start=1)
                for family_id in option.get("family_ids", [])
            }
            for assignment in raw_assignments:
                if not isinstance(assignment, dict):
                    raise ValueError(f"family_assignments 中每项必须为 object: {display_id}")
                family_id = cell_text(assignment.get("family_id"))
                assignment_ids.append(family_id)
                option_index = int(numeric_or_none(assignment.get("practice_option_index")) or 0)
                if family_id not in allowed_family_ids:
                    raise ValueError(f"family_assignments 包含不属于当前 display 的 family: {display_id}/{family_id}")
                if option_by_family.get(family_id) != option_index:
                    raise ValueError(f"family_assignments 与 practice_options 不一致: {display_id}/{family_id}")
            if assignment_ids != allowed_order_by_display[display_id]:
                raise ValueError(f"family_assignments 必须按 candidate_family_ids 完整列出: {display_id}")

        selected_row = selected_by_display[display_id]
        display_row = display_map.get(display_id, pd.Series(dtype=object))
        result_by_display[display_id] = {
            "display_id": display_id,
            "display_name": cell_text(display_row.get("display_name")),
            "unit": cell_text(display_row.get("unit")),
            "family_count": display_row.get("family_count", ""),
            "selection_reason": cell_text(selected_row.get("selection_reason")),
            "practice_options": practice_options,
        }
        meta["selected_display_ids"].append(display_id)
        meta["practice_option_count"] += len(practice_options)
        meta["families_grouped_count"] += len(seen_family_ids)
        meta["option_count_by_display"][display_id] = len(practice_options)
        meta["max_options_per_display"] = max(meta["max_options_per_display"], len(practice_options))

    missing = [display_id for display_id in display_ids if display_id not in result_by_display]
    if missing:
        raise ValueError(f"每个 selected_display 必须返回 practice_options: {join_non_empty(missing)}")
    return pd.DataFrame([result_by_display[display_id] for display_id in display_ids]), meta


def build_display_family_selection_trace_frame(
    selected_display_practices: pd.DataFrame,
    display_group_families: pd.DataFrame,
    candidate_families: pd.DataFrame,
) -> pd.DataFrame:
    family_map = {cell_text(row.get("family_id")): row for _index, row in candidate_families.iterrows()}
    display_family_map = {
        (cell_text(row.get("display_id")), cell_text(row.get("family_id"))): row
        for _index, row in display_group_families.iterrows()
    }
    selected_map = {cell_text(row.get("display_id")): row for _index, row in selected_display_practices.iterrows()}
    rows: list[dict[str, Any]] = []
    for display_id, selected in selected_map.items():
        practice_options = selected.get("practice_options") if isinstance(selected.get("practice_options"), list) else []
        for option in practice_options:
            if not isinstance(option, dict):
                continue
            practice_option_id = cell_text(option.get("practice_option_id"))
            representative_family_id = cell_text(option.get("representative_family_id"))
            practice_description = cell_text(option.get("practice_description"))
            group_reason = cell_text(option.get("group_reason"))
            family_ids = option.get("family_ids") if isinstance(option.get("family_ids"), list) else []
            for family_id_value in family_ids:
                family_id = cell_text(family_id_value)
                row = display_family_map.get((display_id, family_id), pd.Series(dtype=object))
                candidate = family_map.get(family_id, pd.Series(dtype=object))
                rows.append(
                    {
                        "display_id": display_id,
                        "display_name": cell_text(row.get("display_name")) or cell_text(selected.get("display_name")),
                        "practice_option_id": practice_option_id,
                        "representative_family_id": representative_family_id,
                        "practice_description": practice_description,
                        "family_id": family_id,
                        "family_role": "representative" if family_id == representative_family_id else "member",
                        "representative_cost_item_name": cell_text(row.get("representative_cost_item_name")),
                        "representative_project_description": cell_text(row.get("representative_project_description")),
                        "unit": cell_text(candidate.get("unit_normalized")) or cell_text(row.get("unit")) or cell_text(candidate.get("unit")),
                        "本次召回样本数": row.get("本次召回样本数", ""),
                        "本次召回工程包数": row.get("本次召回工程包数", ""),
                        "item_query_similarity最大值": row.get("item_query_similarity最大值", ""),
                        "unit_price_min": candidate.get("本次召回综合单价最低值", ""),
                        "unit_price_median": candidate.get("本次召回综合单价中位数", ""),
                        "unit_price_max": candidate.get("本次召回综合单价最高值", ""),
                        "group_reason": group_reason,
                    }
                )
    frame = pd.DataFrame(rows, columns=DISPLAY_FAMILY_SELECTION_TRACE_COLUMNS)
    if frame.empty:
        return frame
    frame["_family_role_sort"] = frame["family_role"].map(lambda value: 0 if cell_text(value) == "representative" else 1)
    frame["_item_similarity_sort"] = pd.to_numeric(frame["item_query_similarity最大值"], errors="coerce").fillna(-1)
    frame = frame.sort_values(
        ["display_id", "practice_option_id", "_family_role_sort", "_item_similarity_sort"],
        ascending=[True, True, True, False],
    ).drop(columns=["_family_role_sort", "_item_similarity_sort"])
    return frame[DISPLAY_FAMILY_SELECTION_TRACE_COLUMNS]


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
    family_count = sum(len(item.get("candidate_families") or []) for item in records)
    max_tokens = min(8192, max(3072, 3072 + family_count * 160 + len(records) * 256))
    if selected_displays.empty:
        trace = trace_row(
            "display_family_selection",
            "将已选 display 内的 family 整理为 practice options，并选择当前默认参考做法",
            True,
            prompt=prompt,
            max_tokens=max_tokens,
            input_summary=json_text({"display_count": 0, "family_count": 0, "practice_option_count": 0}),
        )
        return pd.DataFrame(), True, False, "", prompt, trace, empty_display_family_selection_meta(), pd.DataFrame(columns=DISPLAY_FAMILY_SELECTION_TRACE_COLUMNS)
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
    trace = trace_row(
        "display_family_selection",
        "将已选 display 内的 family 整理为 practice options，并选择当前默认参考做法",
        True,
        prompt=prompt,
        max_tokens=max_tokens,
        input_summary=json_text(
            {
                "display_count": len(records),
                "family_count": family_count,
                "practice_option_count": meta.get("practice_option_count", 0),
            }
        ),
        usage=response.usage,
        raw_response=getattr(response, "raw_content", ""),
    )
    return selected, True, False, "", prompt, trace, meta, trace_frame


def normalized_unit(value: Any) -> str:
    text = cell_text(value).lower()
    text = text.replace("㎡", "m²").replace("平方米", "m²").replace("平方", "m²")
    text = re.sub(r"m\s*2|m\^2", "m²", text)
    text = text.replace("毫米", "mm")
    return text.strip()



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


def format_number_cell(value: Any) -> str:
    number = numeric_or_none(value)
    if number is None:
        return ""
    if float(number).is_integer():
        return str(int(number))
    return str(number)


def build_scenario_generation_prompt(
    raw_text: str,
    selected_displays: pd.DataFrame,
) -> tuple[str, list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    for _index, selected in selected_displays.iterrows():
        practice_options = selected.get("practice_options") if isinstance(selected.get("practice_options"), list) else []
        records.append(
            {
                "display_id": cell_text(selected.get("display_id")),
                "display_name": cell_text(selected.get("display_name")),
                "unit": cell_text(selected.get("unit")),
                "practice_options": [
                    {
                        "practice_option_id": cell_text(option.get("practice_option_id")),
                        "practice_description": cell_text(option.get("practice_description")),
                        "sample_count": int(numeric_or_none(option.get("sample_count")) or 0),
                    }
                    for option in practice_options
                    if isinstance(option, dict)
                ],
            }
        )
    prompt = f"""
你负责根据用户原始需求、已选 display 以及每个 display 下的全部 practice options，生成一个或多个完整的估价 scenario。

上游已经完成：

1. 识别与用户需求相关的 display；
2. 将每个 display 下的历史做法整理为若干 practice options。

上游没有决定最终 scenario，也没有预先选定任何 practice option。

输入中每个 practice option 包含：

- practice_option_id
- practice_description
- sample_count

sample_count 表示该 option 在当前历史样本中的支持数量，只作为判断信息之一。它不是价格、工程量或最终选择概率。

你的任务：

1. 判断应生成一个 scenario，还是多个存在实质差异的 scenario。
2. 决定每个 scenario 应包含哪些 display。
3. 为每个被纳入的 display 选择对应的 practice_option_id。
4. 为每个 scenario 生成 scenario_name。
5. 为每个 scenario 生成 scenario_summary，说明该 scenario 的整体施工范围、施工链、适用现场条件和与其他 scenario 的实质差异。
6. 为每个 scenario item 生成 item_explanation，说明当前 option 在本方案中的用途、施工环节、选用原因、工程量来源或继承逻辑。
7. 为每个 scenario item 判断 quantity：
   - 可以可靠确定单一数值时，输出 exact；
   - 可以可靠确定一个范围时，输出 range。
8. 按与用户原始需求的符合程度排列 scenario。

判断原则：

1. 优先依据用户原始输入中明确表达的维修内容、材料、规格、型号、性能、施工方法和工程量。
2. 只有当不同 practice options 会形成实际不同的实施方案时，才需要生成不同 scenario。
3. 不要为了覆盖所有 practice options 而机械生成大量 scenario。
4. 不同 display 是否放入同一个 scenario，应根据用户需求和各 option 的实际描述判断。
5. sample_count 只能作为历史支持程度的辅助信息，不能直接决定 option 必须被选择。
6. 不得选择输入中不存在的 display_id 或 practice_option_id。
7. 不得输出单价或合价。
8. 不得根据 sample_count 推算工程量。
9. 不得根据其他 item 的工程量推算当前 item 的工程量。
10. scenario_summary 和 item_explanation 必须针对当前 scenario 的实际内容撰写，避免通用模板。
11. practice_description 只描述工艺本身，不承担方案解释；不得用 scenario_summary 或 item_explanation 简单复述 practice_description。

scenario_summary 必须同时覆盖：

1. 整体施工内容：说明该方案包含的主要施工范围和施工链。
2. 适用条件：说明适合什么现场条件、基层状态或维修目标。
3. 与其他方案的差异：如果存在多个 scenario，必须明确指出与其他方案不同的前置工序、拆除范围、基层处理或材料做法；如果只有一个 scenario，也要说明该方案的关键取舍。

item_explanation 必须同时覆盖：

1. 当前选用 option 的具体用途，不能只写“符合用户要求”“样本支持较多”或类似空泛理由。
2. 该 item 在当前施工方案中的施工环节。
3. 为什么选用该 practice option。
4. 工程量来源或继承逻辑；如果工程量从同一施工范围继承，必须说明“按用户给出的同一施工面积暂估，最终以现场核定为准”。

quantity 格式：

精确值：

{{
  "type": "exact",
  "value": 100
}}

区间：

{{
  "type": "range",
  "min": 80,
  "max": 120
}}

输出必须严格符合：

{{
  "scenarios": [
    {{
      "scenario_id": "S001",
      "scenario_order": 1,
      "scenario_name": "方案名称",
      "scenario_summary": "方案整体说明",
      "items": [
        {{
          "display_id": "D001",
          "practice_option_id": "D001-O01",
          "item_explanation": "当前项目说明，覆盖用途、施工环节、选用原因和工程量来源",
          "quantity": {{
            "type": "exact",
            "value": 100
          }}
        }}
      ]
    }}
  ]
}}

严格约束：

1. scenarios 必须为非空数组。
2. scenario_id 必须唯一。
3. scenario_order 必须从 1 开始连续递增。
4. 每个 scenario 至少包含一个 item。
5. display_id 必须存在于输入中。
6. practice_option_id 必须属于对应 display。
7. 同一 scenario 中不得重复相同的 display_id 和 practice_option_id。
8. scenario_name 不得为空。
9. scenario_summary 不得为空。
10. item_explanation 不得为空。
11. quantity.type 只能是 exact 或 range。
12. exact 必须包含非负 value。
13. range 必须包含非负 min 和 max，且 min 不得大于 max。
14. 不得输出单价、合价或其他未要求字段。

输入：
{json_text({"user_query": raw_text, "displays": records})}
""".strip()
    return prompt, records


def selected_display_maps(selected_displays: pd.DataFrame) -> tuple[dict[str, pd.Series], dict[tuple[str, str], dict[str, Any]]]:
    display_map = {cell_text(row.get("display_id")): row for _index, row in selected_displays.iterrows()}
    option_map: dict[tuple[str, str], dict[str, Any]] = {}
    for display_id, row in display_map.items():
        practice_options = row.get("practice_options") if isinstance(row.get("practice_options"), list) else []
        for option in practice_options:
            if isinstance(option, dict):
                option_map[(display_id, cell_text(option.get("practice_option_id")))] = option
    return display_map, option_map


def validate_quantity(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("quantity 必须为 object")
    quantity_type = cell_text(value.get("type"))
    if quantity_type == "exact":
        if set(value) != {"type", "value"}:
            raise ValueError("exact quantity 只能包含 type 和 value")
        number = numeric_or_none(value.get("value"))
        if number is None or number < 0:
            raise ValueError("exact quantity value 必须为非负数")
        return {"type": "exact", "value": number}
    if quantity_type == "range":
        if set(value) != {"type", "min", "max"}:
            raise ValueError("range quantity 只能包含 type、min 和 max")
        minimum = numeric_or_none(value.get("min"))
        maximum = numeric_or_none(value.get("max"))
        if minimum is None or maximum is None:
            raise ValueError("range quantity 必须包含 min 和 max")
        if minimum < 0 or maximum < 0:
            raise ValueError("range quantity min/max 必须为非负数")
        if minimum > maximum:
            raise ValueError("range quantity min 不得大于 max")
        return {"type": "range", "min": minimum, "max": maximum}
    raise ValueError("quantity.type 只能是 exact 或 range")


def parse_scenario_generation_result(
    result: dict[str, Any],
    selected_displays: pd.DataFrame,
    warnings: list[str] | None = None,
) -> list[EstimateScenario]:
    _ = warnings
    allowed_scenario_keys = {"scenario_id", "scenario_order", "scenario_name", "scenario_summary", "items"}
    allowed_item_keys = {"display_id", "practice_option_id", "item_explanation", "selection_reason", "quantity"}
    raw_scenarios = result.get("scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise ValueError("scenario_generation 输出缺少非空 scenarios list")
    display_map, option_map = selected_display_maps(selected_displays)
    selected_id_set = set(display_map)
    scenarios: list[EstimateScenario] = []
    scenario_ids: set[str] = set()
    for expected_order, raw_scenario in enumerate(raw_scenarios, start=1):
        if not isinstance(raw_scenario, dict):
            raise ValueError("scenario 必须为 object")
        extra_keys = set(raw_scenario) - allowed_scenario_keys
        if extra_keys:
            raise ValueError(f"scenario 包含未要求字段: {join_non_empty(sorted(extra_keys))}")
        scenario_id = cell_text(raw_scenario.get("scenario_id"))
        if not scenario_id:
            raise ValueError("scenario_id 不得为空")
        if scenario_id in scenario_ids:
            raise ValueError(f"scenario_id 重复: {scenario_id}")
        scenario_ids.add(scenario_id)
        scenario_order = int(numeric_or_none(raw_scenario.get("scenario_order")) or 0)
        if scenario_order != expected_order:
            raise ValueError("scenario_order 必须从 1 开始连续递增")
        scenario_name = cell_text(raw_scenario.get("scenario_name"))
        scenario_summary = cell_text(raw_scenario.get("scenario_summary"))
        if not scenario_name:
            raise ValueError("scenario_name 不得为空")
        if not scenario_summary:
            raise ValueError("scenario_summary 不得为空")
        raw_items = raw_scenario.get("items")
        if not isinstance(raw_items, list) or not raw_items:
            raise ValueError("每个 scenario 至少包含一个 item")
        items: list[ScenarioItem] = []
        seen_items: set[tuple[str, str]] = set()
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                raise ValueError("scenario item 必须为 object")
            extra_item_keys = set(raw_item) - allowed_item_keys
            if extra_item_keys:
                raise ValueError(f"scenario item 包含未要求字段: {join_non_empty(sorted(extra_item_keys))}")
            display_id = cell_text(raw_item.get("display_id"))
            practice_option_id = cell_text(raw_item.get("practice_option_id"))
            if display_id not in selected_id_set:
                raise ValueError(f"scenario item 引用了无效 display_id: {display_id}")
            if (display_id, practice_option_id) not in option_map:
                raise ValueError(f"practice_option_id 不属于对应 display: {display_id}/{practice_option_id}")
            item_key = (display_id, practice_option_id)
            if item_key in seen_items:
                raise ValueError(f"同一 scenario 中重复 display_id 和 practice_option_id: {display_id}/{practice_option_id}")
            seen_items.add(item_key)
            selection_reason = cell_text(raw_item.get("item_explanation")) or cell_text(raw_item.get("selection_reason"))
            if not selection_reason:
                raise ValueError("item_explanation 不得为空")
            items.append(
                ScenarioItem(
                    display_id=display_id,
                    practice_option_id=practice_option_id,
                    selection_reason=selection_reason,
                    quantity=validate_quantity(raw_item.get("quantity")),
                )
            )
        scenarios.append(
            EstimateScenario(
                scenario_id=scenario_id,
                scenario_order=scenario_order,
                scenario_name=scenario_name,
                scenario_summary=scenario_summary,
                items=items,
            )
        )
    return scenarios


def generate_estimate_scenarios(
    raw_text: str,
    selected_displays: pd.DataFrame,
    warnings: list[str] | None = None,
) -> tuple[list[EstimateScenario], bool, bool, str, str, dict[str, Any]]:
    prompt, records = build_scenario_generation_prompt(raw_text, selected_displays)
    max_tokens = 4096
    if selected_displays.empty:
        trace = trace_row(
            "scenario_generation",
            "根据 practice options 生成估价 scenarios",
            True,
            prompt=prompt,
            max_tokens=max_tokens,
            input_summary=json_text({"selected_display_ids": [], "scenario_count": 0}),
            scenario_count=0,
            scenario_item_count=0,
        )
        return [], True, False, "", prompt, trace
    try:
        response = request_llm_json_with_usage(
            prompt,
            max_tokens=max_tokens,
            system_prompt="你只输出一个 JSON object，不输出解释、Markdown 或思考过程。",
        )
        scenarios = parse_scenario_generation_result(response.content, selected_displays, warnings)
        scenario_item_count = sum(len(scenario.items) for scenario in scenarios)
        trace = trace_row(
            "scenario_generation",
            "根据 practice options 生成估价 scenarios",
            True,
            prompt=prompt,
            max_tokens=max_tokens,
            input_summary=json_text(
                {
                    "selected_display_ids": [record["display_id"] for record in records],
                    "scenario_count": len(scenarios),
                    "scenario_item_count": scenario_item_count,
                }
            ),
            usage=response.usage,
            raw_response=getattr(response, "raw_content", ""),
            scenario_count=len(scenarios),
            scenario_item_count=scenario_item_count,
        )
        return scenarios, True, False, "", prompt, trace
    except (LLMServiceError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        append_warning(warnings, "scenario_generation_failed")
        scenarios: list[EstimateScenario] = []
        scenario_item_count = 0
        trace = trace_row(
            "scenario_generation",
            "根据 practice options 生成估价 scenarios",
            False,
            error=str(exc),
            prompt=prompt,
            max_tokens=max_tokens,
            input_summary=json_text(
                {
                    "selected_display_ids": [record["display_id"] for record in records],
                    "scenario_count": len(scenarios),
                    "scenario_item_count": scenario_item_count,
                    "fallback": False,
                }
            ),
            scenario_count=len(scenarios),
            scenario_item_count=scenario_item_count,
        )
        return scenarios, False, False, str(exc), prompt, trace


def aggregate_price_from_evidence_items(evidence_items: pd.DataFrame, family_ids: list[str]) -> dict[str, Any]:
    evidence_family_ids = evidence_items.get("family_id", pd.Series(dtype=object)).map(cell_text)
    evidence = evidence_items[evidence_family_ids.isin(family_ids)].copy()
    unit_price = min_median_max(evidence, "unit_price")
    labor_price = min_median_max(evidence, "labor_unit_price")
    machinery_price = min_median_max(evidence, "machinery_unit_price")
    return {
        "unit_price_min": unit_price[0],
        "unit_price_median": unit_price[1],
        "unit_price_max": unit_price[2],
        "labor_unit_price_min": labor_price[0],
        "labor_unit_price_median": labor_price[1],
        "labor_unit_price_max": labor_price[2],
        "machinery_unit_price_min": machinery_price[0],
        "machinery_unit_price_median": machinery_price[1],
        "machinery_unit_price_max": machinery_price[2],
        "source_refs": ordered_refs(evidence.get("source_ref", pd.Series(dtype=object)), limit=50),
        "evidence_count": int(len(evidence)),
    }


def median_or_none(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.median())


def min_or_none(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.min())


def max_or_none(values: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return None
    return float(numeric.max())


def fallback_price_stats_from_families(candidate_families: pd.DataFrame, family_ids: list[str]) -> dict[str, Any]:
    family_rows = candidate_families[
        candidate_families.get("family_id", pd.Series(dtype=object)).map(cell_text).isin(family_ids)
    ].copy()
    return {
        "unit_price_min": min_or_none(family_rows.get("本次召回综合单价最低值", pd.Series(dtype=float))),
        "unit_price_median": median_or_none(family_rows.get("本次召回综合单价中位数", pd.Series(dtype=float))),
        "unit_price_max": max_or_none(family_rows.get("本次召回综合单价最高值", pd.Series(dtype=float))),
        "labor_unit_price_min": min_or_none(family_rows.get("本次召回人工费单价最低值", pd.Series(dtype=float))),
        "labor_unit_price_median": median_or_none(family_rows.get("本次召回人工费单价中位数", pd.Series(dtype=float))),
        "labor_unit_price_max": max_or_none(family_rows.get("本次召回人工费单价最高值", pd.Series(dtype=float))),
        "machinery_unit_price_min": min_or_none(family_rows.get("本次召回机械费单价最低值", pd.Series(dtype=float))),
        "machinery_unit_price_median": median_or_none(family_rows.get("本次召回机械费单价中位数", pd.Series(dtype=float))),
        "machinery_unit_price_max": max_or_none(family_rows.get("本次召回机械费单价最高值", pd.Series(dtype=float))),
        "source_refs": ordered_refs(family_rows.get("source_refs", pd.Series(dtype=object)), limit=50),
        "evidence_count": int(pd.to_numeric(family_rows.get("本次召回样本数", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if not family_rows.empty else 0,
    }


def price_stats_for_option(option: dict[str, Any], candidate_families: pd.DataFrame, evidence_items: pd.DataFrame) -> dict[str, Any]:
    family_ids = [cell_text(value) for value in option.get("family_ids", []) if cell_text(value)]
    stats = aggregate_price_from_evidence_items(evidence_items, family_ids)
    fallback_stats = fallback_price_stats_from_families(candidate_families, family_ids)
    for key, value in fallback_stats.items():
        if stats.get(key) in (None, ""):
            stats[key] = value
    return stats


def quantity_display(quantity: dict[str, Any]) -> Any:
    quantity_type = cell_text(quantity.get("type"))
    if quantity_type == "exact":
        return quantity.get("value", "")
    if quantity_type == "range":
        return f"{format_number_cell(quantity.get('min'))}～{format_number_cell(quantity.get('max'))}"
    raise ValueError("quantity.type 只能是 exact 或 range")


def quantity_values(quantity: dict[str, Any]) -> tuple[Any, Any, Any]:
    quantity_type = cell_text(quantity.get("type"))
    if quantity_type == "exact":
        value = quantity.get("value", "")
        return value, value, value
    if quantity_type == "range":
        minimum = numeric_or_none(quantity.get("min"))
        maximum = numeric_or_none(quantity.get("max"))
        midpoint = None if minimum is None or maximum is None else (minimum + maximum) / 2
        return minimum if minimum is not None else "", midpoint if midpoint is not None else "", maximum if maximum is not None else ""
    raise ValueError("quantity.type 只能是 exact 或 range")


def quantity_basis(quantity: dict[str, Any]) -> str:
    quantity_type = cell_text(quantity.get("type"))
    if quantity_type == "exact":
        return "按 scenario item 判断为精确工程量，最终以现场核定为准"
    if quantity_type == "range":
        return "按 scenario item 判断为工程量区间，中位数按区间中点暂估，最终以现场核定为准"
    raise ValueError("quantity.type 只能是 exact 或 range")


def quantity_amounts(quantity: dict[str, Any], price_stats: dict[str, Any]) -> tuple[Any, Any, Any]:
    quantity_type = cell_text(quantity.get("type"))
    if quantity_type == "exact":
        value = quantity.get("value")
        return (
            calc_amount(value, price_stats.get("unit_price_min")),
            calc_amount(value, price_stats.get("unit_price_median")),
            calc_amount(value, price_stats.get("unit_price_max")),
        )
    if quantity_type == "range":
        minimum = numeric_or_none(quantity.get("min"))
        maximum = numeric_or_none(quantity.get("max"))
        midpoint = None if minimum is None or maximum is None else (minimum + maximum) / 2
        return (
            calc_amount(minimum, price_stats.get("unit_price_min")),
            calc_amount(midpoint, price_stats.get("unit_price_median")),
            calc_amount(maximum, price_stats.get("unit_price_max")),
        )
    raise ValueError("quantity.type 只能是 exact 或 range")


def build_scenario_outputs(
    scenarios: list[EstimateScenario],
    selected_displays: pd.DataFrame,
    candidate_families: pd.DataFrame,
    evidence_items: pd.DataFrame,
) -> pd.DataFrame:
    display_map, option_map = selected_display_maps(selected_displays)
    scenario_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for item in scenario.items:
            display_row = display_map.get(item.display_id)
            option = option_map.get((item.display_id, item.practice_option_id))
            if display_row is None or option is None:
                continue
            practice_options = display_row.get("practice_options") if isinstance(display_row.get("practice_options"), list) else []
            other_options = [
                cell_text(other.get("practice_description"))
                for other in practice_options
                if isinstance(other, dict) and cell_text(other.get("practice_option_id")) != item.practice_option_id
            ]
            price_stats = price_stats_for_option(option, candidate_families, evidence_items)
            amount_low, amount_mid, amount_high = quantity_amounts(item.quantity, price_stats)
            quantity_low, quantity_mid, quantity_high = quantity_values(item.quantity)
            family_ids = [cell_text(value) for value in option.get("family_ids", []) if cell_text(value)]
            scenario_rows.append(
                {
                    "方案顺序": scenario.scenario_order,
                    "方案编号": scenario.scenario_id,
                    "方案名称": scenario.scenario_name,
                    "display_id": item.display_id,
                    "清单名称": cell_text(display_row.get("display_name")),
                    "单位": cell_text(display_row.get("unit")),
                    "选用工艺": cell_text(option.get("practice_description")),
                    "其他可选工艺": "；".join([text for text in other_options if text]),
                    "项目说明": item.selection_reason,
                    "工程量类型": cell_text(item.quantity.get("type")),
                    "工程量最低值": quantity_low,
                    "工程量中位数": quantity_mid,
                    "工程量最高值": quantity_high,
                    "工程量依据": quantity_basis(item.quantity),
                    "合价最低值": amount_low,
                    "合价中位数": amount_mid,
                    "合价最高值": amount_high,
                    "综合单价最低值": price_stats.get("unit_price_min"),
                    "综合单价中位数": price_stats.get("unit_price_median"),
                    "综合单价最高值": price_stats.get("unit_price_max"),
                    "其中包含人工费单价最低值": price_stats.get("labor_unit_price_min"),
                    "其中包含人工费单价中位数": price_stats.get("labor_unit_price_median"),
                    "其中包含人工费单价最高值": price_stats.get("labor_unit_price_max"),
                    "其中包含机械费单价最低值": price_stats.get("machinery_unit_price_min"),
                    "其中包含机械费单价中位数": price_stats.get("machinery_unit_price_median"),
                    "其中包含机械费单价最高值": price_stats.get("machinery_unit_price_max"),
                    "价格证据样本数": price_stats.get("evidence_count"),
                    "来源样本": cell_text(price_stats.get("source_refs")),
                    "practice_option_id": item.practice_option_id,
                    "价格证据family": ",".join(family_ids),
                }
            )
    return pd.DataFrame(scenario_rows, columns=ESTIMATE_SCENARIO_COLUMNS).fillna("")


def display_frame(frame: pd.DataFrame, display: bool) -> pd.DataFrame:
    output = frame.copy()
    _ = display  # 保留 CLI 参数兼容；显示精度由 Excel number_format 控制。
    for column in output.columns:
        if is_text_identifier_column(column):
            output[column] = output[column].map(cell_text)
    return output.rename(columns=EXCEL_DISPLAY_COLUMN_LABELS)


EXCEL_DISPLAY_COLUMN_LABELS = {
    "retrieval_package_support_ratio": "本次召回工程包支持比例",
    "retrieval_item_count": "本次召回清单行数",
    "retrieval_package_count": "本次召回工程包数",
    "support_rank": "本次召回支持度排序",
}


TEXT_IDENTIFIER_COLUMNS = {
    "方案编号",
    "scenario_id",
    "display_id",
    "family_id",
    "project_package_id",
    "project_key",
    "item_key",
    "stable_sample_id",
    "source_ref",
    "catalog_id",
    "batch_id",
    "source_row_id",
    "item_row_id",
    "project_code",
}

TEXT_VALUE_COLUMNS = {
    "是否推荐方案",
    "工程量类型",
    "工程量依据",
    "方案说明",
    "主要施工内容",
    "与其他方案的核心差异",
    "待现场确认事项",
    "项目说明",
    "来源样本",
    "价格证据family",
}

INTEGER_COLUMNS = {
    "序号",
    "rank",
    "selection_rank",
    "support_rank",
    "family_count",
    "本次召回family数量",
    "默认family本次召回样本数",
    "默认family本次召回工程包数",
    "本次召回样本数",
    "本次召回工程包数",
    "retrieval_item_count",
    "retrieval_package_count",
    "本次召回清单行数",
    "本次召回支持度排序",
    "item_count",
    "page_no",
    "prompt_chars",
    "estimated_tokens",
    "max_tokens",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
}

DECIMAL_VALUE_COLUMNS = {
    "quantity",
    "unit_price",
    "labor_unit_price",
    "machinery_unit_price",
}

AMOUNT_VALUE_COLUMNS = {
    "total_price",
}

AMOUNT_WIDTH_COLUMNS = {
    "total_price",
    "合价最低值",
    "合价中位数",
    "合价最高值",
}


def is_text_identifier_column(column: Any) -> bool:
    name = cell_text(column)
    return (
        name in TEXT_IDENTIFIER_COLUMNS
        or name.endswith("_id")
        or name.endswith("_ids")
    )


def excel_number_format(column: Any) -> str | None:
    name = cell_text(column)
    lower_name = name.lower()
    if is_text_identifier_column(name):
        return "@"
    if name in TEXT_VALUE_COLUMNS:
        return None
    if name == "package_evidence_weight":
        return "0.000000"
    if "similarity" in lower_name or "相似度" in name or lower_name.endswith("_ratio") or "比例" in name:
        return "0.0000"
    if (
        name in INTEGER_COLUMNS
        or lower_name.endswith("_count")
        or lower_name.endswith("_rank")
        or "样本数" in name
        or "工程包数" in name
        or ("数量" in name and "工程量" not in name)
    ):
        return "0"
    if name in AMOUNT_VALUE_COLUMNS or "金额" in name or "合价" in name:
        return "#,##0.00"
    if name in DECIMAL_VALUE_COLUMNS or "工程量" in name or "单价" in name:
        return "0.00"
    return None


def excel_min_column_width(column: Any) -> float | None:
    name = cell_text(column)
    if name in AMOUNT_WIDTH_COLUMNS:
        return 15.0
    if name in DECIMAL_VALUE_COLUMNS or "工程量" in name or "单价" in name:
        return 12.0
    return None


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
    name = cell_text(row.get("清单名称")) or cell_text(row.get("项目名称")) or cell_text(row.get("清单项名称"))
    description = short_description(row.get("项目特征") or row.get("选用工艺") or row.get("项目特征/施工工艺"))
    if name and description:
        return f"{name}（{description}）"
    return name or description


def build_estimate_summary(
    scenarios: list[EstimateScenario],
    estimate_scenarios: pd.DataFrame,
) -> pd.DataFrame:
    if estimate_scenarios.empty:
        return pd.DataFrame(columns=ESTIMATE_SUMMARY_COLUMNS)
    scenario_summary_by_id = {scenario.scenario_id: scenario.scenario_summary for scenario in scenarios}
    rows: list[dict[str, Any]] = []
    first_order = numeric_or_none(estimate_scenarios.iloc[0].get("方案顺序"))
    for (_scenario_order, scenario_id), frame in estimate_scenarios.groupby(["方案顺序", "方案编号"], sort=True):
        scenario = frame.iloc[0]
        conditional_frame = frame[
            frame["工程量类型"].map(cell_text).eq("range")
            & pd.to_numeric(frame["工程量最低值"], errors="coerce").fillna(-1).eq(0)
        ]
        scenario_summary = scenario_summary_by_id.get(cell_text(scenario_id), "")
        rows.append(
            {
                "方案顺序": scenario.get("方案顺序", ""),
                "方案编号": scenario_id,
                "方案名称": scenario.get("方案名称", ""),
                "是否推荐方案": "是" if numeric_or_none(scenario.get("方案顺序")) == first_order else "否",
                "方案说明": scenario_summary,
                "主要施工内容": join_non_empty([display_item_label(row) for _index, row in frame.iterrows()]),
                "与其他方案的核心差异": scenario_summary,
                "计价项目数": int(len(frame)),
                "合价最低值": amount_sum(frame, "合价最低值"),
                "合价中位数": amount_sum(frame, "合价中位数"),
                "合价最高值": amount_sum(frame, "合价最高值"),
                "待现场确认事项": join_non_empty(conditional_frame.get("项目说明", pd.Series(dtype=object)).tolist()),
            }
        )
    return pd.DataFrame(rows, columns=ESTIMATE_SUMMARY_COLUMNS).fillna("")


def build_parse_info(
    rewrite: QueryRewrite,
    top_packages: int,
    top_items: int,
    max_packages_per_cache_subject: int,
    package_weight_temperature: float,
    evidence_package_universe_count: int,
    package_evidence_weight_count: int,
    package_evidence_weight_sum: float,
    meta: dict[str, Any],
    sample_count: int,
    package_count: int,
    retrieved_evidence_item_row_count: int,
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
    scenario_count: int,
    scenario_item_count: int,
    scenario_exact_quantity_count: int,
    scenario_range_quantity_count: int,
    scenario_generation_trace: dict[str, Any],
    scenario_generation_fallback: bool,
    scenario_generation_error: str,
    output_path: Path | None,
    started_at: datetime,
    index_dir: Path,
    include_debug_text: bool,
    display_selection_prompt: str,
    display_family_selection_prompt: str,
    scenario_generation_prompt: str,
    warnings: list[str] | None = None,
) -> pd.DataFrame:
    rows = [
        ("原始用户需求", rewrite.raw_query),
        ("project_package_query_text", rewrite.project_package_query_text),
        ("item_query_text", rewrite.item_query_text),
        ("item_retrieval_text_fields", "cost_item_name + project_description + unit_normalized"),
        ("package_retrieval_text_fields", "工程名称 + project_name_text + cost_item_names_summary"),
        ("top_packages", top_packages),
        ("top_items", top_items),
        ("max_packages_per_cache_subject", max_packages_per_cache_subject),
        ("package_weight_temperature", package_weight_temperature),
        ("evidence_package_universe_count", evidence_package_universe_count),
        ("package_evidence_weight_count", package_evidence_weight_count),
        ("package_evidence_weight_sum", f"{package_evidence_weight_sum:.12f}"),
        ("embedding_model", meta.get("model", "")),
        ("sample_count", sample_count),
        ("package_count", package_count),
        ("LLM query rewrite 是否成功", "是" if rewrite.success else "否"),
        ("retrieved_evidence_item_row_count", retrieved_evidence_item_row_count),
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
        ("display_family_selection_selected_display_ids", json_text(display_family_selection_meta.get("selected_display_ids") or [])),
        ("display_family_selection_practice_option_count", display_family_selection_meta.get("practice_option_count", "")),
        ("display_family_selection_families_grouped_count", display_family_selection_meta.get("families_grouped_count", "")),
        ("display_family_selection_option_count_by_display", json_text(display_family_selection_meta.get("option_count_by_display") or {})),
        ("display_family_selection_max_options_per_display", display_family_selection_meta.get("max_options_per_display", "")),
        ("display_family_selection_prompt_chars", display_family_selection_trace.get("prompt_chars", "")),
        ("display_family_selection_prompt_tokens", display_family_selection_trace.get("prompt_tokens") or display_family_selection_trace.get("estimated_tokens", "")),
        ("display_family_selection_completion_tokens", display_family_selection_trace.get("completion_tokens", "")),
        ("scenario_count", scenario_count),
        ("scenario_item_count", scenario_item_count),
        ("scenario_exact_quantity_count", scenario_exact_quantity_count),
        ("scenario_range_quantity_count", scenario_range_quantity_count),
        ("scenario_generation_status", "fallback" if scenario_generation_fallback else ("failed" if scenario_generation_error else "success")),
        ("scenario_generation_prompt_chars", scenario_generation_trace.get("prompt_chars", "")),
        ("scenario_generation_prompt_tokens", scenario_generation_trace.get("prompt_tokens") or scenario_generation_trace.get("estimated_tokens", "")),
        ("scenario_generation_completion_tokens", scenario_generation_trace.get("completion_tokens", "")),
        ("invalid_display_ids", join_non_empty(display_selection_meta.get("invalid_display_ids") or [])),
        ("duplicate_display_ids", join_non_empty(display_selection_meta.get("duplicate_display_ids") or [])),
        ("是否 display_selection fallback", "是" if display_selection_fallback else "否"),
        ("是否 display_family_selection fallback", "是" if display_family_selection_fallback else "否"),
        ("是否 scenario_generation fallback", "是" if scenario_generation_fallback else "否"),
        ("display_selection LLM error", display_selection_error),
        ("display_family_selection LLM error", display_family_selection_error),
        ("scenario_generation LLM error", scenario_generation_error),
        ("output_path", str(output_path or "")),
        ("运行时间", f"{(datetime.now() - started_at).total_seconds():.2f}s"),
        ("index_dir", str(index_dir)),
        ("主要文件路径", json_text((meta.get("files") or {}))),
        ("rewrite_notes", "；".join(rewrite.notes)),
        ("warnings", "；".join(warnings or [])),
    ]
    if include_debug_text:
        rows.append(("display_selection_prompt_preview", display_selection_prompt[:3000]))
        rows.append(("display_family_selection_prompt_preview", display_family_selection_prompt[:3000]))
        rows.append(("scenario_generation_prompt_preview", scenario_generation_prompt[:3000]))
    return pd.DataFrame(rows, columns=["字段", "值"])


def write_query_result_workbook(output_path: Path, result: QueryResult, display: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        display_frame(result.estimate_summary, display).to_excel(writer, sheet_name="estimate_summary", index=False)
        display_frame(result.estimate_scenarios, display).to_excel(writer, sheet_name="estimate_scenarios", index=False)
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
        display_frame(result.package_evidence_weights, display).to_excel(
            writer,
            sheet_name="package_evidence_weights",
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
    scenario_worksheet = workbook["estimate_scenarios"] if "estimate_scenarios" in workbook.sheetnames else None
    if scenario_worksheet is not None:
        header_by_name = {cell_text(cell.value): cell.column for cell in scenario_worksheet[1]}
        scenario_id_column = header_by_name.get("方案编号")
        if scenario_id_column is not None:
            separator_rows: list[int] = []
            previous_scenario_id = ""
            for row_index in range(2, scenario_worksheet.max_row + 1):
                scenario_id = cell_text(scenario_worksheet.cell(row_index, scenario_id_column).value)
                if previous_scenario_id and scenario_id and scenario_id != previous_scenario_id:
                    separator_rows.append(row_index)
                if scenario_id:
                    previous_scenario_id = scenario_id
            for row_index in reversed(separator_rows):
                scenario_worksheet.insert_rows(row_index)
                scenario_worksheet.row_dimensions[row_index].height = 16

    for worksheet in workbook.worksheets:
        worksheet.freeze_panes = None
        column_formats = {
            cell.column: excel_number_format(cell.value)
            for cell in worksheet[1]
        }
        for cell in worksheet[1]:
            min_width = excel_min_column_width(cell.value)
            if min_width is None:
                continue
            column_letter = cell.column_letter
            current_width = worksheet.column_dimensions[column_letter].width or 0
            worksheet.column_dimensions[column_letter].width = max(current_width, min_width)
        for cell in worksheet[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(wrap_text=False, vertical="top")
        for row in worksheet.iter_rows():
            for cell in row:
                cell.alignment = Alignment(wrap_text=False, vertical="top")
                number_format = column_formats.get(cell.column)
                if number_format:
                    cell.number_format = number_format
    workbook.save(path)
    workbook.close()


def run_query(
    index_dir: Path,
    raw_text: str,
    top_packages: int,
    top_items: int,
    output: Path | None,
    max_packages_per_cache_subject: int = 1,
    display_selection_limit: int = 50,
    display_exploration_limit: int = 5,
    package_weight_temperature: float = DEFAULT_PACKAGE_WEIGHT_TEMPERATURE,
    include_debug_text: bool = False,
    display: bool = False,
) -> QueryResult:
    if package_weight_temperature <= 0:
        raise ValueError("package weight temperature 必须大于 0")
    started_at = datetime.now()
    warnings: list[str] = []
    samples, project_packages, project_package_embeddings, item_embeddings, meta = load_index(index_dir)
    rewrite, rewrite_trace = query_rewrite_for_embedding(raw_text)

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

    package_query_similarities = project_package_embeddings @ package_query_embedding
    package_query_similarity_by_id = project_package_similarity_map(project_packages, package_query_similarities)
    matched_raw = score_project_packages(
        project_packages,
        project_package_embeddings,
        package_query_embedding,
        top_packages,
        max_packages_per_cache_subject=max_packages_per_cache_subject,
    )
    item_query_similarities = item_embeddings @ item_query_embedding
    direct_item_hits = score_direct_items(samples, item_query_similarities, top_items)
    evidence_package_ids = evidence_package_universe(matched_raw, direct_item_hits)
    package_evidence_weights = build_package_evidence_weights(
        evidence_package_ids,
        package_query_similarity_by_id,
        package_weight_temperature,
    )
    retrieved_evidence_items = build_retrieved_evidence_items(
        samples,
        matched_raw,
        direct_item_hits,
        item_query_similarities,
        package_query_similarity_by_id=package_query_similarity_by_id,
        warnings=warnings,
    )
    candidate_families = build_candidate_families(retrieved_evidence_items)
    evidence_items = attach_family_ids_to_evidence_items(retrieved_evidence_items, candidate_families)
    candidate_display_groups, display_group_families = build_candidate_display_groups(candidate_families, evidence_items)
    candidate_display_groups = attach_display_support_ratios(
        candidate_display_groups,
        display_group_families,
        evidence_items,
        package_evidence_weights,
    )
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
        candidate_display_groups,
        display_selection_limit=display_selection_limit,
        exploration_limit=display_exploration_limit,
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
        scenarios,
        scenario_generation_success,
        scenario_generation_fallback,
        scenario_generation_error,
        scenario_generation_prompt,
        scenario_generation_trace,
    ) = generate_estimate_scenarios(
        raw_text,
        selected_display_practices,
        warnings=warnings,
    )
    estimate_scenarios = build_scenario_outputs(scenarios, selected_display_practices, candidate_families, evidence_items)
    estimate_summary = build_estimate_summary(scenarios, estimate_scenarios)
    if warnings:
        append_trace_warnings(display_selection_trace, warnings)
        append_trace_warnings(display_family_selection_trace, warnings)
        append_trace_warnings(scenario_generation_trace, warnings)
    displays_for_llm_count = len(display_selection_meta.get("candidate_ids") or [])
    scenario_item_count = sum(len(scenario.items) for scenario in scenarios)
    scenario_exact_quantity_count = sum(1 for scenario in scenarios for item in scenario.items if cell_text(item.quantity.get("type")) == "exact")
    scenario_range_quantity_count = sum(1 for scenario in scenarios for item in scenario.items if cell_text(item.quantity.get("type")) == "range")
    parse_info = build_parse_info(
        rewrite=rewrite,
        top_packages=top_packages,
        top_items=top_items,
        max_packages_per_cache_subject=max_packages_per_cache_subject,
        package_weight_temperature=package_weight_temperature,
        evidence_package_universe_count=len(evidence_package_ids),
        package_evidence_weight_count=len(package_evidence_weights),
        package_evidence_weight_sum=float(package_evidence_weights.get("package_evidence_weight", pd.Series(dtype=float)).sum()),
        meta=meta,
        sample_count=len(samples),
        package_count=len(project_packages),
        retrieved_evidence_item_row_count=len(retrieved_evidence_items),
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
        scenario_count=len(scenarios),
        scenario_item_count=scenario_item_count,
        scenario_exact_quantity_count=scenario_exact_quantity_count,
        scenario_range_quantity_count=scenario_range_quantity_count,
        scenario_generation_trace=scenario_generation_trace,
        scenario_generation_fallback=scenario_generation_fallback,
        scenario_generation_error=scenario_generation_error,
        output_path=output,
        started_at=started_at,
        index_dir=index_dir,
        include_debug_text=include_debug_text,
        display_selection_prompt=display_selection_prompt,
        display_family_selection_prompt=display_family_selection_prompt,
        scenario_generation_prompt=scenario_generation_prompt,
        warnings=warnings,
    )
    llm_trace = pd.DataFrame(
        [
            rewrite_trace,
            display_selection_trace,
            display_family_selection_trace,
            scenario_generation_trace,
        ],
        columns=LLM_TRACE_COLUMNS,
    )

    result = QueryResult(
        rewrite=rewrite,
        estimate_summary=estimate_summary,
        estimate_scenarios=estimate_scenarios,
        matched_project_packages=matched_project_packages,
        candidate_families=candidate_families,
        candidate_display_groups=candidate_display_groups,
        package_evidence_weights=package_evidence_weights,
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
    print(f"[DONE] matched project packages: {len(result.matched_project_packages)}")
    print(f"[DONE] candidate families: {len(result.candidate_families)}")
    print(f"[DONE] candidate display groups: {len(result.candidate_display_groups)}")
    print(f"[DONE] evidence items: {len(result.evidence_items)}")
    print(f"[DONE] estimate scenarios: {len(result.estimate_scenarios)}")
    if result.rewrite.notes:
        print(f"rewrite notes: {'；'.join(result.rewrite.notes)}")
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
            display_selection_limit=args.display_selection_limit,
            display_exploration_limit=args.display_exploration_limit,
            package_weight_temperature=args.package_weight_temperature,
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
