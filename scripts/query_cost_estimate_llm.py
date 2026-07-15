#!/usr/bin/env python3
from __future__ import annotations

import argparse
import calendar
import gc
import json
import re
import sys
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
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
MUNICIPALITIES = {"北京市", "上海市", "天津市", "重庆市"}
PREFECTURE_LOCATION_PATTERN = re.compile(
    r"^(?:[^,，/、]+省|[^,，/、]+自治区)[^,，/、省市]+市$"
)


def shift_months(value: date, months: int) -> date:
    month_index = value.year * 12 + value.month - 1 + months
    year, zero_based_month = divmod(month_index, 12)
    month = zero_based_month + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)

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
    "item_count_average",
    "is_top5_similarity",
    "item_count_distance_to_average",
    "project_selection_rank",
    "is_selected_package",
]

MATCHED_PROJECT_EXAMPLE_COLUMNS = [
    "rank",
    "project_package_id",
    "工程名称",
    "project_name_text",
    "consultation_time",
    "location",
    "item_order",
    "stable_sample_id",
    "source_ref",
    "cost_item_name",
    "project_description",
    "unit",
    "quantity",
    "unit_price",
    "total_price",
]

CANDIDATE_FAMILY_COLUMNS = [
    "family_id",
    "normalized_signature",
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
    "normalized_signature",
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
    "normalized_signature",
    "representative_cost_item_name",
    "representative_project_description",
    "unit",
    "本次召回样本数",
    "本次召回工程包数",
    "item_query_similarity最大值",
]

DISPLAY_OPTION_GROUPING_TRACE_COLUMNS = [
    "display_id",
    "display_name",
    "practice_option_id",
    "family_id",
    "representative_cost_item_name",
    "representative_project_description",
    "unit",
    "本次召回样本数",
    "本次召回工程包数",
    "item_query_similarity最大值",
    "unit_price_min",
    "unit_price_median",
    "unit_price_max",
    "option_sample_count",
    "option_package_count",
    "is_original_option",
    "is_selected_option",
    "option_selection_decision",
    "option_selection_reason",
    "is_original_family",
    "is_representative_family",
    "representative_selection_reason",
]

OPTION_SELECTION_TRACE_COLUMNS = [
    "final_item_position", "display_id", "original_option_id", "selected_option_id",
    "original_family_id", "representative_family_id", "candidate_option_count",
    "whether_replaced", "selection_reason", "option_selection_decision",
    "representative_selection_reason",
]

ESTIMATE_SCENARIO_COLUMNS = [
    "清单名称",
    "项目特征",
    "单位",
    "工程量",
    "工程量来源",
    "工程量说明",
    "工程量样本数",
    "工程量最低值",
    "工程量中位数",
    "工程量最高值",
    "综合单价",
    "暂估合价",
    "合价P10",
    "合价中位数",
    "合价P90",
    "综合单价P10",
    "综合单价中位数",
    "综合单价P90",
    "其中包含人工费单价P10",
    "其中包含人工费单价中位数",
    "其中包含人工费单价P90",
    "其中包含机械费单价P10",
    "其中包含机械费单价中位数",
    "其中包含机械费单价P90",
    "价格证据样本数",
    "来源样本",
    "practice_option_id",
    "original_option_id",
    "original_family_id",
    "representative_family_id",
    "价格证据family",
    "display_id",
]

ESTIMATE_SUMMARY_COLUMNS = [
    "方案名称",
    "方案说明",
    "主要施工内容",
    "计价项目数",
    "合价P10",
    "合价中位数",
    "合价P90",
    "待现场确认事项",
]

PRICE_EVIDENCE_ITEM_COLUMNS = [
    "final_item_position", "清单名称", "display_id", "practice_option_id",
    "family_id", "normalized_signature", "stable_sample_id", "project_key",
    "source_ref", "工程名称", "location", "consultation_time", "cost_item_name",
    "project_description", "unit", "quantity", "unit_price", "labor_unit_price",
    "machinery_unit_price",
]

OPTION_EVIDENCE_EXPANSION_COLUMNS = [
    "final_item_position",
    "清单名称",
    "practice_option_id",
    "原family数",
    "候选family数",
    "新增family数",
    "扩展后family数",
    "原价格证据样本数",
    "扩展后价格证据样本数",
    "新增family_ids",
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
    "fallback",
    "quantity_items",
]


@dataclass(frozen=True)
class QueryRewrite:
    raw_query: str
    project_package_query_text: str
    item_query_text: str
    location: str
    start_date: str
    end_date: str
    notes: list[str]
    success: bool


@dataclass(frozen=True)
class ScenarioItem:
    project_package_id: str
    stable_sample_id: str
    source_ref: str
    display_id: str
    practice_option_id: str
    original_option_id: str
    original_family_id: str
    representative_family_id: str
    selection_reason: str
    quantity: dict[str, Any]
    quantity_reason: str
    item_position: int = 0
    quantity_source: str = ""
    quantity_explanation: str = ""
    quantity_sample_count: int | None = None
    quantity_minimum: float | None = None
    quantity_median: float | None = None
    quantity_maximum: float | None = None
    quantity_fallback_used: bool = False
    quantity_fallback_reason: str = ""


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
    display_option_grouping_trace: pd.DataFrame
    option_selection_trace: pd.DataFrame
    matched_project_examples: pd.DataFrame
    evidence_items: pd.DataFrame
    option_evidence_expansion: pd.DataFrame
    price_evidence_items: pd.DataFrame
    parse_info: pd.DataFrame
    llm_trace: pd.DataFrame
    success: bool = True
    error_message: str = ""


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
        "--with-explanations",
        action="store_true",
        help="生成项目级和清单级 LLM 解释，默认关闭",
    )
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


EVIDENCE_EXPANSION_SCOPE_PATTERN = re.compile(
    r"^(?:含)?(?:人工费?|安装|拆除及安装(?:人工费?)?|拆机及安装(?:人工费?)?|运输|运费|起吊费)$"
)


def normalize_evidence_expansion_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", cell_text(value))
    text = text.replace("（", "(").replace("）", ")")

    def remove_scope_parentheses(match: re.Match[str]) -> str:
        content = re.sub(r"\s+", "", match.group(1))
        if EVIDENCE_EXPANSION_SCOPE_PATTERN.fullmatch(content):
            return ""
        return match.group(0)

    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\(([^()]*)\)", remove_scope_parentheses, text)
    return re.sub(r"\s+", " ", text).strip()


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


def encode_texts(model: Any, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    embeddings = model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return normalize_embeddings(embeddings)


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


def build_query_rewrite_prompt(query: str, current_date: date | None = None) -> str:
    current_date = current_date or date.today()
    current_date_text = current_date.isoformat()
    return f"""
你是维修工程需求解析和 embedding query rewrite 助手。请把用户原始需求解析为严格 JSON object。

只能输出 JSON object，不要 Markdown，不要解释，不要建议清单，不要计算价格。
当前日期：{current_date_text}

输出格式：
{{
  "project_package_query_text": "",
  "item_query_text": "",
  "location": "",
  "start_date": "",
  "end_date": ""
}}
不要增加其他字段。

当前 embedding 结构：
1. project_package_text 由“工程名称、project_name_text、cost_item_name 去重列表”组成。
   project_package_query_text 用于匹配相似历史工程包，应描述用户明确表达或直接相关的维修工程场景，保持短检索 query，不要预设建议清单、前置项、措施项或替代工艺。
2. item_retrieval_text 由“cost_item_name、project_description、unit_normalized”组成。
   item_query_text 用于匹配相似清单行，应贴近用户明确表达的维修对象、材料规格和做法，不要扩展未明确发生的清单项。
3. item_query_text 必须非空。如果用户问得很粗，也输出宽泛 item query，不要留空。
4. location 表示项目所属的标准地级行政区域。普通地级市必须输出“省级行政区 + 地级市”，例如“浙江省嘉兴市”“江苏省苏州市”“四川省成都市”；直辖市只输出“北京市”“上海市”“天津市”“重庆市”。
5. 用户未提出地域限制时 location 输出空字符串。不要输出简称、县、区或镇；县级行政区所属地级行政区明确时可输出标准地级区域，不确定时不要猜测。
6. start_date 和 end_date 只能是 YYYY-MM-DD 或空字符串。所有相对时间以当前日期 {current_date_text} 为基准转换为绝对日期。
7. “最近一年”“一年内”向前推 12 个月；“最近半年”向前推 6 个月；“最近三个月”向前推 3 个月，end_date 均为当前日期。
8. 整年使用当年 01-01 至 12-31；整月使用当月首日至末日；月份区间使用首月首日至末月末日；某日以后截至当前日期；“截至某日”只填写 end_date。
9. 用户未提出时间约束时 start_date 和 end_date 均输出空字符串，不输出相对时间自然语言。
10. 不扩展用户未明确提出的清单项，不输出数量分析、材料列表、不确定性、方案建议、价格或施工清单。

示例：
用户：屋面漏水，想做3mm SBS防水，面积大概500平
输出：{{"project_package_query_text":"屋面漏水维修工程 屋面防水维修 3mm SBS防水","item_query_text":"屋面卷材防水 3mm SBS防水卷材","location":"","start_date":"","end_date":""}}

用户：屋面漏水帮我估价
输出：{{"project_package_query_text":"屋面漏水维修工程 屋面防水维修","item_query_text":"屋面防水 防水层维修","location":"","start_date":"","end_date":""}}

用户：屋面漏水，参考嘉兴一年内的造价
输出：{{"project_package_query_text":"屋面漏水维修工程 屋面防水维修","item_query_text":"屋面防水 防水层维修","location":"浙江省嘉兴市","start_date":"{shift_months(current_date, -12).isoformat()}","end_date":"{current_date_text}"}}

用户：参考上海市2025年3月的消防报警主机更换造价
输出：{{"project_package_query_text":"消防报警主机更换工程","item_query_text":"消防报警主机更换","location":"上海市","start_date":"2025-03-01","end_date":"2025-03-31"}}

用户需求：{query}
""".strip()


def fallback_query_rewrite(query: str, note: str) -> QueryRewrite:
    return QueryRewrite(
        raw_query=query,
        project_package_query_text=query,
        item_query_text=query,
        location="",
        start_date="",
        end_date="",
        notes=[note],
        success=False,
    )


def normalize_location(value: Any) -> str:
    text = unicodedata.normalize("NFKC", cell_text(value)).strip()
    return re.sub(r"\s+", " ", text)


def validate_query_constraints(
    location: Any,
    start_date: Any,
    end_date: Any,
) -> tuple[str, str, str, list[str]]:
    notes: list[str] = []
    normalized_location = normalize_location(location)
    if normalized_location and normalized_location not in MUNICIPALITIES and not PREFECTURE_LOCATION_PATTERN.fullmatch(normalized_location):
        notes.append(f"location 格式非法，已清空: {normalized_location}")
        normalized_location = ""

    parsed_dates: dict[str, str] = {}
    for field_name, value in (("start_date", start_date), ("end_date", end_date)):
        text = cell_text(value)
        if not text:
            parsed_dates[field_name] = ""
            continue
        try:
            parsed_dates[field_name] = datetime.strptime(text, "%Y-%m-%d").date().isoformat()
        except ValueError:
            parsed_dates[field_name] = ""
            notes.append(f"{field_name} 格式非法，已清空: {text}")

    start_text = parsed_dates["start_date"]
    end_text = parsed_dates["end_date"]
    if start_text and end_text and start_text > end_text:
        notes.append("start_date 晚于 end_date，两个日期约束均已清空")
        start_text = ""
        end_text = ""
    return normalized_location, start_text, end_text, notes


def parse_consultation_dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, format="%Y-%m-%d", errors="coerce").dt.date


def build_constraint_mask(
    rows: pd.DataFrame,
    location: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    mask = pd.Series(True, index=rows.index, dtype=bool)
    if location:
        if "location" not in rows.columns:
            return pd.Series(False, index=rows.index, dtype=bool)
        mask &= rows["location"].map(normalize_location).eq(location)
    if start_date or end_date:
        if "consultation_time" not in rows.columns:
            return pd.Series(False, index=rows.index, dtype=bool)
        dates = parse_consultation_dates(rows["consultation_time"])
        if start_date:
            mask &= dates.notna() & dates.ge(datetime.strptime(start_date, "%Y-%m-%d").date())
        if end_date:
            mask &= dates.notna() & dates.le(datetime.strptime(end_date, "%Y-%m-%d").date())
    return mask


def filter_rows_and_embeddings(
    rows: pd.DataFrame,
    embeddings: np.ndarray,
    mask: pd.Series,
    label: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    if len(rows) != len(embeddings) or len(rows) != len(mask):
        raise ValueError(f"{label} 约束过滤前 DataFrame、embedding 与 mask 行数不一致")
    filtered_rows = rows.loc[mask].copy()
    filtered_embeddings = embeddings[mask.to_numpy()]
    if len(filtered_rows) != len(filtered_embeddings):
        raise ValueError(f"约束过滤后的{label}与 embedding 行数不一致")
    return filtered_rows, filtered_embeddings


def ensure_project_package_candidates(rows: pd.DataFrame, rewrite: QueryRewrite) -> None:
    if not rows.empty:
        return
    if rewrite.location or rewrite.start_date or rewrite.end_date:
        raise ValueError("没有找到同时满足地域和时间约束的历史工程包。")
    raise ValueError("没有找到历史工程包。")


def query_rewrite_for_embedding(query: str, current_date: date | None = None) -> tuple[QueryRewrite, dict[str, Any]]:
    prompt = build_query_rewrite_prompt(query, current_date=current_date)
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
    location, start_date, end_date, constraint_notes = validate_query_constraints(
        result.get("location") if isinstance(result, dict) else "",
        result.get("start_date") if isinstance(result, dict) else "",
        result.get("end_date") if isinstance(result, dict) else "",
    )
    notes.extend(constraint_notes)
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
        location=location,
        start_date=start_date,
        end_date=end_date,
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


def select_representative_project_package(
    matched_project_packages: pd.DataFrame,
) -> tuple[pd.Series, pd.DataFrame]:
    if matched_project_packages.empty:
        raise ValueError("matched_project_packages 为空，无法选择历史工程包")
    required = {"package_query_similarity", "item_count", "project_package_id"}
    missing = sorted(required - set(matched_project_packages.columns))
    if missing:
        raise ValueError(f"matched_project_packages 缺少字段: {', '.join(missing)}")

    ranked = matched_project_packages.copy()
    ranked["_recall_order"] = range(len(ranked))
    ranked["package_query_similarity"] = pd.to_numeric(
        ranked["package_query_similarity"], errors="raise"
    )
    ranked["item_count"] = pd.to_numeric(ranked["item_count"], errors="raise")
    if ranked["item_count"].isna().any():
        raise ValueError("matched_project_packages.item_count 不得为空")
    ranked = ranked.sort_values(
        ["package_query_similarity", "_recall_order"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    average_item_count = float(ranked["item_count"].mean())
    ranked["item_count_average"] = average_item_count
    ranked["project_selection_rank"] = range(1, len(ranked) + 1)
    ranked["is_top5_similarity"] = ranked.index < 5
    ranked["item_count_distance_to_average"] = (
        ranked["item_count"].astype(float) - average_item_count
    ).abs()
    top_five = ranked.head(5)
    selected_index = min(
        range(len(top_five)),
        key=lambda index: (
            float(top_five.iloc[index]["item_count_distance_to_average"]),
            index,
        ),
    )
    ranked["is_selected_package"] = False
    ranked.loc[selected_index, "is_selected_package"] = True
    selected_package = ranked.iloc[selected_index].copy()
    return selected_package, ranked.drop(columns=["_recall_order"])


def item_row_numeric_order(value: Any) -> float | None:
    text = cell_text(value)
    if not text:
        return None
    if re.fullmatch(r"\d+(?:\.0+)?", text):
        return float(text)
    match = re.search(r"(?:^|[-_])(\d+)$", text)
    return float(match.group(1)) if match else None


def ordered_project_items(samples: pd.DataFrame, project_package_id: str) -> pd.DataFrame:
    rows = samples[samples["project_package_id"].map(cell_text).eq(project_package_id)].copy()
    if rows.empty:
        return rows
    rows["_source_order"] = range(len(rows))
    rows["_item_row_order"] = rows.get("item_row_id", pd.Series(index=rows.index, dtype=object)).map(item_row_numeric_order)
    rows["_seq_order"] = pd.to_numeric(rows.get("seq", pd.Series(index=rows.index, dtype=object)), errors="coerce")
    rows["_preferred_order"] = rows["_item_row_order"].where(rows["_item_row_order"].notna(), rows["_seq_order"])
    rows["_missing_order"] = rows["_preferred_order"].isna()
    return rows.sort_values(
        ["_missing_order", "_preferred_order", "_source_order"],
        ascending=[True, True, True],
        kind="stable",
    ).drop(columns=["_source_order", "_item_row_order", "_seq_order", "_preferred_order", "_missing_order"])


def expand_selected_project_items(
    samples: pd.DataFrame,
    selected_package: pd.Series,
) -> pd.DataFrame:
    project_package_id = cell_text(selected_package.get("project_package_id"))
    selected_items = ordered_project_items(samples, project_package_id)
    if selected_items.empty:
        raise ValueError(f"所选工程包没有清单: {project_package_id}")
    expected_count = numeric_or_none(selected_package.get("item_count"))
    if expected_count is None or not float(expected_count).is_integer():
        raise ValueError(f"所选工程包 item_count 非法: {selected_package.get('item_count')!r}")
    if len(selected_items) != int(expected_count):
        raise ValueError(
            f"所选工程包完整清单数量与 item_count 不一致: "
            f"project_package_id={project_package_id}, expected={int(expected_count)}, "
            f"actual={len(selected_items)}"
        )
    selected_items = selected_items.reset_index(drop=True)
    selected_items["item_position"] = range(len(selected_items))
    return selected_items


def attach_family_and_display_ids_to_selected_items(
    selected_items: pd.DataFrame,
    evidence_items: pd.DataFrame,
    display_group_families: pd.DataFrame,
) -> pd.DataFrame:
    selected = selected_items.copy()
    selected_ids = selected.get("stable_sample_id", pd.Series(index=selected.index, dtype=object)).map(cell_text)
    if selected_ids.eq("").any():
        raise ValueError("selected_items 中 stable_sample_id 不得为空")
    duplicate_selected_ids = selected_ids[selected_ids.duplicated(keep=False)]
    if not duplicate_selected_ids.empty:
        raise ValueError(
            f"selected_items 中 stable_sample_id 必须唯一: "
            f"{join_non_empty(duplicate_selected_ids.tolist(), limit=10)}"
        )

    evidence_mapping = evidence_items[["stable_sample_id", "family_id"]].copy()
    evidence_mapping["stable_sample_id"] = evidence_mapping["stable_sample_id"].map(cell_text)
    evidence_mapping["family_id"] = evidence_mapping["family_id"].map(cell_text)
    if evidence_mapping["stable_sample_id"].eq("").any() or evidence_mapping["family_id"].eq("").any():
        raise ValueError("evidence_items 中 stable_sample_id/family_id 不得为空")
    duplicate_evidence_ids = evidence_mapping.loc[
        evidence_mapping["stable_sample_id"].duplicated(keep=False), "stable_sample_id"
    ]
    if not duplicate_evidence_ids.empty:
        raise ValueError(
            f"evidence_items 中 stable_sample_id 必须唯一: "
            f"{join_non_empty(duplicate_evidence_ids.tolist(), limit=10)}"
        )

    display_mapping = display_group_families[["family_id", "display_id"]].copy()
    display_mapping["family_id"] = display_mapping["family_id"].map(cell_text)
    display_mapping["display_id"] = display_mapping["display_id"].map(cell_text)
    if display_mapping["family_id"].eq("").any() or display_mapping["display_id"].eq("").any():
        raise ValueError("display_group_families 中 family_id/display_id 不得为空")
    duplicate_family_ids = display_mapping.loc[
        display_mapping["family_id"].duplicated(keep=False), "family_id"
    ]
    if not duplicate_family_ids.empty:
        raise ValueError(
            f"family_id 必须唯一映射到一个 display_id: "
            f"{join_non_empty(duplicate_family_ids.tolist(), limit=10)}"
        )

    selected["stable_sample_id"] = selected_ids
    original_count = len(selected)
    selected = selected.merge(evidence_mapping, on="stable_sample_id", how="left", validate="one_to_one")
    selected = selected.merge(display_mapping, on="family_id", how="left", validate="many_to_one")
    if len(selected) != original_count:
        raise ValueError("selected_items 合并 family/display 后行数发生变化")
    missing_family = selected["family_id"].map(cell_text).eq("")
    missing_display = selected["display_id"].map(cell_text).eq("")
    if missing_family.any() or missing_display.any():
        missing_ids = selected.loc[missing_family | missing_display, "stable_sample_id"].tolist()
        raise ValueError(
            f"selected_items 每行必须唯一映射到 family_id 和 display_id: "
            f"{join_non_empty(missing_ids, limit=10)}"
        )
    return selected


def range_selection_item_records(selected_items: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "item_position": int(row["item_position"]),
            "cost_item_name": cell_text(row.get("cost_item_name")),
            "project_description": cell_text(row.get("project_description")),
        }
        for _index, row in selected_items.iterrows()
    ]


def build_contiguous_item_range_prompt(
    raw_query: str,
    selected_project_name: str,
    selected_items: pd.DataFrame,
) -> str:
    return f"""
你会收到一个已经由程序选定的真实历史工程，以及该工程按原始顺序排列的全部清单。

请根据用户原始需求，在该工程中选择一个最适合作为当前方案骨架的连续清单区间。

规则：
1. start_item_position 和 end_item_position 均为 0-based，并且包含边界。
2. 必须选择一个连续区间，不得返回多个区间。
3. 区间应尽量保留完整的相关施工内容，避免只选择用户直接提到的单个主体项。
4. 工程名称和清单原始顺序是重要参考。
5. 无法明确缩小时，选择完整工程。
6. 本阶段不判断工程量、不修改工艺、不生成价格。

只输出：
{{
  "start_item_position": 0,
  "end_item_position": 5
}}

输入：
{json_text({"user_query": raw_query, "selected_project_name": selected_project_name, "items": range_selection_item_records(selected_items)})}
""".strip()


def validate_contiguous_range(result: Any, item_count: int) -> tuple[int, int]:
    if not isinstance(result, dict):
        raise ValueError("区间结果必须是 object")
    if set(result) != {"start_item_position", "end_item_position"}:
        raise ValueError("区间结果必须且只能包含 start_item_position 和 end_item_position")
    start = result["start_item_position"]
    end = result["end_item_position"]
    if isinstance(start, bool) or not isinstance(start, int):
        raise ValueError("start_item_position 必须是整数")
    if isinstance(end, bool) or not isinstance(end, int):
        raise ValueError("end_item_position 必须是整数")
    if start < 0 or end >= item_count:
        raise ValueError("区间位置越界")
    if start > end:
        raise ValueError("区间起点不得大于终点")
    return start, end


def select_contiguous_item_range(
    raw_query: str,
    selected_project_name: str,
    selected_items: pd.DataFrame,
) -> tuple[int, int, dict[str, Any]]:
    if selected_items.empty:
        raise ValueError("所选工程包没有清单，不执行连续区间选择")
    prompt = build_contiguous_item_range_prompt(raw_query, selected_project_name, selected_items)
    fallback_start, fallback_end = 0, len(selected_items) - 1
    raw_response = ""
    usage: dict[str, Any] = {}
    try:
        response = request_llm_json_with_usage(
            prompt,
            max_tokens=256,
            system_prompt="你只输出一个 JSON object，不输出解释、Markdown 或思考过程。",
        )
        raw_response = cell_text(getattr(response, "raw_content", ""))
        usage = getattr(response, "usage", {}) or {}
        start, end = validate_contiguous_range(response.content, len(selected_items))
        return start, end, {
            "range_selection_status": "ok",
            "fallback": False,
            "error_message": "",
            "prompt": prompt,
            "raw_response": raw_response,
            "usage": usage,
        }
    except (LLMServiceError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        return fallback_start, fallback_end, {
            "range_selection_status": "fallback_full_project",
            "fallback": True,
            "error_message": str(exc),
            "prompt": prompt,
            "raw_response": raw_response,
            "usage": usage,
        }


def build_matched_project_examples(
    matched_project_packages: pd.DataFrame,
    samples: pd.DataFrame,
    limit: int = 3,
) -> list[dict[str, Any]]:
    if matched_project_packages.empty or limit <= 0:
        return []
    ranked = matched_project_packages.copy()
    ranked["_rank_order"] = pd.to_numeric(ranked.get("rank"), errors="coerce")
    ranked["_source_order"] = range(len(ranked))
    ranked = ranked.sort_values(["_rank_order", "_source_order"], kind="stable").head(limit)
    examples: list[dict[str, Any]] = []
    for fallback_rank, (_index, project) in enumerate(ranked.iterrows(), start=1):
        project_package_id = cell_text(project.get("project_package_id"))
        project_items = ordered_project_items(samples, project_package_id)
        items: list[dict[str, Any]] = []
        for _item_index, item in project_items.iterrows():
            source_ref = cell_text(item.get("source_ref")) or source_identity_for_row(item)[2]
            items.append(
                {
                    "stable_sample_id": cell_text(item.get("stable_sample_id")),
                    "source_ref": source_ref,
                    "cost_item_name": cell_text(item.get("cost_item_name")),
                    "project_description": cell_text(item.get("project_description")),
                    "unit": cell_text(item.get("unit")) or cell_text(item.get("unit_normalized")),
                    "quantity": numeric_or_none(item.get("quantity")),
                    "unit_price": numeric_or_none(item.get("unit_price")),
                    "total_price": numeric_or_none(item.get("total_price")),
                }
            )
        examples.append(
            {
                "rank": int(numeric_or_none(project.get("rank")) or fallback_rank),
                "project_package_id": project_package_id,
                "package_query_similarity": numeric_or_none(project.get("package_query_similarity")),
                "project_name": cell_text(project.get("工程名称")),
                "project_name_text": cell_text(project.get("project_name_text")),
                "consultation_time": cell_text(project.get("consultation_time")),
                "location": cell_text(project.get("location")),
                "items": items,
            }
        )
    return examples


def matched_project_examples_frame(examples: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for example in examples:
        items = example.get("items") if isinstance(example.get("items"), list) else []
        for item_order, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            rows.append(
                {
                    "rank": example.get("rank", ""),
                    "project_package_id": cell_text(example.get("project_package_id")),
                    "工程名称": cell_text(example.get("project_name")),
                    "project_name_text": cell_text(example.get("project_name_text")),
                    "consultation_time": cell_text(example.get("consultation_time")),
                    "location": cell_text(example.get("location")),
                    "item_order": item_order,
                    "stable_sample_id": cell_text(item.get("stable_sample_id")),
                    "source_ref": cell_text(item.get("source_ref")),
                    "cost_item_name": cell_text(item.get("cost_item_name")),
                    "project_description": cell_text(item.get("project_description")),
                    "unit": cell_text(item.get("unit")),
                    "quantity": item.get("quantity"),
                    "unit_price": item.get("unit_price"),
                    "total_price": item.get("total_price"),
                }
            )
    return pd.DataFrame(rows, columns=MATCHED_PROJECT_EXAMPLE_COLUMNS)


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


def p10_median_p90(frame: pd.DataFrame, column: str) -> tuple[float | None, float | None, float | None]:
    values = numeric_values(frame, column)
    if values.empty:
        return None, None, None
    return (
        float(values.quantile(0.10, interpolation="nearest")),
        float(values.median()),
        float(values.quantile(0.90, interpolation="nearest")),
    )


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
    for normalized_signature, group in candidates.groupby("normalized_signature", sort=False, dropna=False):
        representative = group.iloc[0]
        quantity_min, quantity_median, quantity_max = min_median_max(group, "quantity")
        unit_price_min, unit_price_median, unit_price_max = min_median_max(group, "unit_price")
        total_price_min, total_price_median, total_price_max = min_median_max(group, "total_price")
        labor_min, labor_median, labor_max = min_median_max(group, "labor_unit_price")
        machinery_min, machinery_median, machinery_max = min_median_max(group, "machinery_unit_price")
        rows.append(
            {
                "normalized_signature": cell_text(normalized_signature),
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
    output.insert(0, "family_id", [f"F{index:03d}" for index in range(1, len(output) + 1)])
    for column in CANDIDATE_FAMILY_COLUMNS:
        if column not in output.columns:
            output[column] = None
    output = output[CANDIDATE_FAMILY_COLUMNS].reset_index(drop=True)
    return output


def attach_family_ids_to_evidence_items(candidates: pd.DataFrame, candidate_families: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return pd.DataFrame(columns=EVIDENCE_ITEM_COLUMNS)
    signature_to_family_id = {
        cell_text(row.get("normalized_signature")): cell_text(row.get("family_id"))
        for _index, row in candidate_families.iterrows()
        if cell_text(row.get("normalized_signature"))
    }
    normalized_signatures = candidates.get("normalized_signature", pd.Series([""] * len(candidates), index=candidates.index))
    output = pd.DataFrame(
        {
            "source_ref": candidates.get("source_ref", ""),
            "family_id": normalized_signatures.map(lambda value: signature_to_family_id.get(cell_text(value), "")),
            "normalized_signature": normalized_signatures,
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
                    "normalized_signature": cell_text(family.get("normalized_signature")),
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


def filter_required_display_groups(
    plan_items: pd.DataFrame,
    candidate_display_groups: pd.DataFrame,
    display_group_families: pd.DataFrame,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame]:
    required_display_ids = list(dict.fromkeys(plan_items["display_id"].map(cell_text).tolist()))
    if not required_display_ids or any(not display_id for display_id in required_display_ids):
        raise ValueError("plan_items 每行必须包含 display_id")

    display_order = {display_id: index for index, display_id in enumerate(required_display_ids)}
    available_display_ids = set(candidate_display_groups["display_id"].map(cell_text).tolist())
    missing_display_ids = [display_id for display_id in required_display_ids if display_id not in available_display_ids]
    if missing_display_ids:
        raise ValueError(f"required display 不存在于 candidate_display_groups: {join_non_empty(missing_display_ids)}")

    filtered_groups = candidate_display_groups[
        candidate_display_groups["display_id"].map(cell_text).isin(required_display_ids)
    ].copy()
    filtered_groups["_required_order"] = filtered_groups["display_id"].map(cell_text).map(display_order)
    filtered_groups = filtered_groups.sort_values("_required_order", kind="stable").drop(columns="_required_order")

    filtered_families = display_group_families[
        display_group_families["display_id"].map(cell_text).isin(required_display_ids)
    ].copy()
    present_family_displays = set(filtered_families["display_id"].map(cell_text).tolist())
    missing_family_displays = [display_id for display_id in required_display_ids if display_id not in present_family_displays]
    if missing_family_displays:
        raise ValueError(f"required display 缺少 family 映射: {join_non_empty(missing_family_displays)}")
    filtered_families["_required_order"] = filtered_families["display_id"].map(cell_text).map(display_order)
    filtered_families = filtered_families.sort_values("_required_order", kind="stable").drop(columns="_required_order")
    return (
        required_display_ids,
        filtered_groups.reset_index(drop=True),
        filtered_families.reset_index(drop=True),
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


def option_grouping_payload_for_display(
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
                "spec": truncate_text(normalize_display_description(row.get("representative_project_description")), 120),
                "unit": cell_text(family.get("unit_normalized")) or cell_text(row.get("unit")) or cell_text(family.get("unit")),
            }
        )
    return payload


def build_display_option_grouping_prompt(
    candidate_display_groups: pd.DataFrame,
    display_group_families: pd.DataFrame,
    candidate_families: pd.DataFrame,
) -> tuple[str, dict[str, Any]]:
    if len(candidate_display_groups) != 1:
        raise ValueError("display_option_grouping prompt 每次必须且只能包含一个 Display")
    display = candidate_display_groups.iloc[0]
    display_id = cell_text(display.get("display_id"))
    record = {
        "display_name": truncate_text(display.get("display_name"), 40),
        "candidate_families": option_grouping_payload_for_display(
            display_id, display_group_families, candidate_families
        ),
    }
    payload = {"candidate_display": record}
    prompt = f"""
你的任务是：先逐个提取当前唯一 Display 下每个 Family 的关键标签，再将 Family 划分为若干 Option。

Option 的定义是：

同一 Option 内的所有 Family，后续必须能够作为一个整体被选中，并作为一个整体展开回全库查询价格证据。

如果用户明确某个参数后，需要保留同组中的一部分 Family、排除另一部分 Family，那么这些 Family 就不能属于同一个 Option。

第一步必须为每个 Family 提取：

- family_id：必须来自输入，不得遗漏、重复或新增；
- thickness：只填写项目特征中明确出现的厚度或核心尺寸规格，没有时为 ""；3mm、3.0mm、3厚统一为 3mm，其他如 1.2mm、1.5mm、2mm、4mm；
- material：只填写明确材料类别并标准化同义表达，没有时为 ""；
- level：只填写明确楼层、层数或高度条件，例如 1层、2层、5层、高度20m以内，没有时为 ""。

material 至少区分：聚氨酯、聚合物水泥基、水泥基、水泥基渗透结晶、JS、沥青防水涂料、非固化防水涂料、SBS改性沥青、自粘卷材、高分子卷材。
“单组分”和“单组份”不影响材料类别；“SBS改性沥青”和“弹性体改性沥青”统一为 SBS改性沥青。
自粘卷材不得与普通 SBS改性沥青统一；聚氨酯不得与聚合物水泥基、水泥基、JS统一；非固化防水涂料不得与普通沥青防水涂料统一。
不要把材料、层数或施工方法写入 thickness；不要把厚度、层数、基层清理或垃圾清运写入 material。
不要增加 is_generic 或其他标签。

第二步再根据上述标签输出 groups。groups 是二维数组：
- 每个内层数组代表一个独立 Option；
- 同一内层数组中的 Family 会在后续被整体选择、整体展开；
- 不同内层数组表示后续可以分别选择的不同候选做法。

不得只按照 Display 名称、共同用途、共同关键词或文本相似度粗略聚合。
相同厚度不代表材料相同，相同材料也不代表厚度相同。
不同明确 level 不得合并，明确 level 与空 level 也不得合并。

例如：

- 3mm 与 4mm 必须拆分；
- 2mm 水泥基渗透结晶与 2mm 聚氨酯必须拆分；
- 3mm SBS改性沥青与 3mm 聚氨酯必须拆分；
- 3mm 自粘卷材与 3mm SBS改性沥青卷材必须拆分；
- 1层、2层、5层、未注明层数必须分别拆分；
- 1.5mm单组份聚氨酯与1.5mm厚单组分聚氨酯可以合并；
- 1.2mm聚合物水泥基，含基层清理，与1.2mm聚合物水泥基可以合并。

以下差异当前可以忽略，不必单独拆分：

- OCR、标点、空格、换行、编号；
- 普通文字详略；
- 基层清理；
- 垃圾清运；
- 普通修补；
- 一般性的附带施工描述；
- 不影响用户后续选择的文字详略差异。

每个输入 family_id 必须且只能出现一次，不得遗漏、重复或新增。

每个输入 family_id 必须且只能在 families 中出现一次，也必须且只能在 groups 中出现一次。

输出格式（顶层只允许 families 和 groups）：

{{
  "families": [
    {{"family_id": "F004", "thickness": "3mm", "material": "SBS改性沥青", "level": ""}},
    {{"family_id": "F007", "thickness": "4mm", "material": "SBS改性沥青", "level": ""}}
  ],
  "groups": [
    ["F004"],
    ["F007"]
  ]
}}

每个内层数组就是一个 Option。

只输出 JSON，不输出解释或 Markdown。

【输入数据】
{json_text(payload)}
""".strip()
    return prompt, record


def display_family_unit(row: pd.Series) -> str:
    return cell_text(row.get("unit_normalized")) or normalized_unit(row.get("unit"))


def empty_display_option_grouping_meta() -> dict[str, Any]:
    return {
        "display_ids": [],
        "practice_option_count": 0,
        "families_grouped_count": 0,
        "option_count_by_display": {},
        "max_options_per_display": 0,
        "llm_display_count": 0,
        "programmatic_single_family_display_count": 0,
    }


OPTION_GROUPING_TAG_FIELDS = ("thickness", "material", "level")


def validate_option_grouping_family_tags(raw_families: Any, allowed_family_ids: set[str]) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    if not isinstance(raw_families, list):
        raise ValueError("families 必须为 list")
    parsed: list[dict[str, str]] = []
    tag_map: dict[str, dict[str, str]] = {}
    for item in raw_families:
        if not isinstance(item, dict) or set(item) != {"family_id", *OPTION_GROUPING_TAG_FIELDS}:
            raise ValueError("families 中每项必须且只能包含 family_id、thickness、material、level")
        family_id = cell_text(item.get("family_id"))
        if not family_id or family_id in tag_map:
            raise ValueError(f"families 中 family_id 为空或重复: {family_id}")
        if family_id not in allowed_family_ids:
            raise ValueError(f"families 包含不属于当前 display 的 family: {family_id}")
        if any(not isinstance(item.get(field), str) for field in OPTION_GROUPING_TAG_FIELDS):
            raise ValueError(f"families 标签必须为字符串: {family_id}")
        tags = {field: cell_text(item.get(field)) for field in OPTION_GROUPING_TAG_FIELDS}
        parsed_item = {"family_id": family_id, **tags}
        parsed.append(parsed_item)
        tag_map[family_id] = tags
    if set(tag_map) != allowed_family_ids:
        raise ValueError("families 中的 family_id 集合与当前 Display 输入 Family 集合不一致")
    return parsed, tag_map


def option_group_conflict_fields(family_ids: list[str], tag_map: dict[str, dict[str, str]]) -> list[str]:
    conflicts: list[str] = []
    for field in ("thickness", "material"):
        non_empty_values = {tag_map[family_id][field] for family_id in family_ids if tag_map[family_id][field]}
        if len(non_empty_values) > 1:
            conflicts.append(field)
    if len({tag_map[family_id]["level"] for family_id in family_ids}) > 1:
        conflicts.append("level")
    return conflicts


def split_conflicting_option_groups(
    groups: list[list[str]], tag_map: dict[str, dict[str, str]],
) -> tuple[list[list[str]], bool, list[str]]:
    final_groups: list[list[str]] = []
    reasons: list[str] = []
    for group in groups:
        conflict_fields = option_group_conflict_fields(group, tag_map)
        if not conflict_fields:
            final_groups.append(list(group))
            continue
        reason = f"{conflict_fields[0]}_conflict" if len(conflict_fields) == 1 else "multiple_conflicts"
        if reason not in reasons:
            reasons.append(reason)
        subgroups: dict[tuple[str, ...], list[str]] = {}
        for family_id in group:
            key = tuple(tag_map[family_id][field] for field in conflict_fields)
            subgroups.setdefault(key, []).append(family_id)
        final_groups.extend(subgroups.values())
    return final_groups, bool(reasons), reasons


def parse_display_option_grouping_result(
    result: dict[str, Any],
    candidate_display_groups: pd.DataFrame,
    display_group_families: pd.DataFrame,
    warnings: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if len(candidate_display_groups) != 1:
        raise ValueError("display_option_grouping 每次必须且只能校验一个 Display")
    if not isinstance(result, dict) or set(result) != {"families", "groups"}:
        actual_keys = sorted(result) if isinstance(result, dict) else [type(result).__name__]
        raise ValueError(f"display_option_grouping 顶层只允许包含 families 和 groups，实际为: {actual_keys}")
    bound_display_id = cell_text(candidate_display_groups.iloc[0].get("display_id"))
    raw_rows = [{"display_id": bound_display_id, "families": result.get("families"), "groups": result.get("groups")}]

    display_ids = [cell_text(value) for value in candidate_display_groups.get("display_id", pd.Series(dtype=object)).tolist()]
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
    display_map = {cell_text(row.get("display_id")): row for _index, row in candidate_display_groups.iterrows()}
    family_row_map = {
        (cell_text(row.get("display_id")), cell_text(row.get("family_id"))): row
        for _index, row in display_group_families.iterrows()
    }
    result_by_display: dict[str, dict[str, Any]] = {}
    meta = empty_display_option_grouping_meta()

    for item in raw_rows:
        if not isinstance(item, dict):
            raise ValueError("display_option_grouping display_result 必须为 object")
        if set(item) != {"display_id", "families", "groups"}:
            raise ValueError("display_result 只允许包含 display_id、families 和 groups")
        display_id = cell_text(item.get("display_id"))
        if display_id not in allowed_by_display:
            raise ValueError(f"display_option_grouping 返回未知 display: {display_id}")
        if display_id in result_by_display:
            raise ValueError(f"display_option_grouping 重复返回 display: {display_id}")
        allowed_family_ids = allowed_by_display[display_id]
        if not allowed_family_ids:
            raise ValueError(f"display 缺少 candidate families: {display_id}")
        parsed_family_tags, family_tag_map = validate_option_grouping_family_tags(
            item.get("families"), allowed_family_ids
        )
        raw_options = item.get("groups")
        if not isinstance(raw_options, list) or not raw_options:
            raise ValueError(f"groups 必须为非空 list: {display_id}")

        original_groups: list[list[str]] = []
        seen_family_ids: set[str] = set()
        seen_family_groups: set[frozenset[str]] = set()
        for option_index, raw_option in enumerate(raw_options, start=1):
            raw_family_ids = raw_option
            if not isinstance(raw_family_ids, list) or not raw_family_ids:
                raise ValueError(f"group 必须为非空 family_id list: {display_id}/O{option_index:02d}")
            family_ids = [cell_text(family_id) for family_id in raw_family_ids]
            if any(not family_id for family_id in family_ids):
                raise ValueError(f"family_ids 不得为空: {display_id}/O{option_index:02d}")
            duplicate_in_option = [family_id for family_id in family_ids if family_ids.count(family_id) > 1]
            if duplicate_in_option:
                raise ValueError(f"同一 practice_option 内 family 重复: {display_id}/{join_non_empty(duplicate_in_option)}")
            unknown_family_ids = [family_id for family_id in family_ids if family_id not in allowed_family_ids]
            if unknown_family_ids:
                raise ValueError(f"practice_option 包含不属于当前 display 的 family: {display_id}/{join_non_empty(unknown_family_ids)}")
            family_group = frozenset(family_ids)
            if family_group in seen_family_groups:
                raise ValueError(f"同一 display 内不得存在完全相同的 family_ids 分组: {display_id}")
            duplicate_across_options = [family_id for family_id in family_ids if family_id in seen_family_ids]
            if duplicate_across_options:
                raise ValueError(f"family 不得出现在多个 practice_options: {display_id}/{join_non_empty(duplicate_across_options)}")
            option_units = [
                display_family_unit(family_row_map.get((display_id, family_id), pd.Series(dtype=object)))
                for family_id in family_ids
            ]
            if len(set(option_units)) > 1:
                raise ValueError(f"同一 practice_option 中 family 单位必须一致: {display_id}/O{option_index:02d}")
            seen_family_groups.add(family_group)
            seen_family_ids.update(family_ids)
            original_groups.append(family_ids)

        missing_family_ids = [family_id for family_id in allowed_order_by_display[display_id] if family_id not in seen_family_ids]
        if missing_family_ids:
            raise ValueError(f"practice_options 遗漏 candidate family: {display_id}/{join_non_empty(missing_family_ids)}")
        extra_family_ids = [family_id for family_id in seen_family_ids if family_id not in allowed_family_ids]
        if extra_family_ids:
            raise ValueError(f"practice_options 新增非法 family: {display_id}/{join_non_empty(extra_family_ids)}")
        final_groups, auto_split_applied, auto_split_reasons = split_conflicting_option_groups(
            original_groups, family_tag_map
        )
        practice_options: list[dict[str, Any]] = []
        for stable_index, family_ids in enumerate(final_groups, start=1):
            option_units = [
                display_family_unit(family_row_map.get((display_id, family_id), pd.Series(dtype=object)))
                for family_id in family_ids
            ]
            if len(set(option_units)) > 1:
                raise ValueError(f"同一 practice_option 中 family 单位必须一致: {display_id}/O{stable_index:02d}")
            sample_count = sum(
                int(numeric_or_none(family_row_map.get((display_id, family_id), pd.Series(dtype=object)).get("本次召回样本数")) or 0)
                for family_id in family_ids
            )
            practice_options.append({
                "practice_option_id": f"{display_id}-O{stable_index:02d}",
                "sample_count": sample_count,
                "family_ids": family_ids,
            })
        display_row = display_map.get(display_id, pd.Series(dtype=object))
        result_by_display[display_id] = {
            "display_id": display_id,
            "display_name": cell_text(display_row.get("display_name")),
            "unit": cell_text(display_row.get("unit")),
            "family_count": display_row.get("family_count", ""),
            "practice_options": practice_options,
        }
        meta.setdefault("grouping_details_by_display", {})[display_id] = {
            "parsed_family_tags": parsed_family_tags,
            "original_groups": original_groups,
            "final_groups": final_groups,
            "auto_split_applied": auto_split_applied,
            "auto_split_reasons": auto_split_reasons,
        }
        meta["display_ids"].append(display_id)
        meta["practice_option_count"] += len(practice_options)
        meta["families_grouped_count"] += len(seen_family_ids)
        meta["option_count_by_display"][display_id] = len(practice_options)
        meta["max_options_per_display"] = max(meta["max_options_per_display"], len(practice_options))

    missing = [display_id for display_id in display_ids if display_id not in result_by_display]
    if missing:
        raise ValueError(f"每个 candidate display 必须返回 practice_options: {join_non_empty(missing)}")
    return pd.DataFrame([result_by_display[display_id] for display_id in display_ids]), meta


def build_display_option_grouping_trace_frame(
    displays_with_options: pd.DataFrame,
    display_group_families: pd.DataFrame,
    candidate_families: pd.DataFrame,
) -> pd.DataFrame:
    family_map = {cell_text(row.get("family_id")): row for _index, row in candidate_families.iterrows()}
    display_family_map = {
        (cell_text(row.get("display_id")), cell_text(row.get("family_id"))): row
        for _index, row in display_group_families.iterrows()
    }
    display_map = {cell_text(row.get("display_id")): row for _index, row in displays_with_options.iterrows()}
    rows: list[dict[str, Any]] = []
    for display_id, display in display_map.items():
        practice_options = display.get("practice_options") if isinstance(display.get("practice_options"), list) else []
        for option in practice_options:
            if not isinstance(option, dict):
                continue
            practice_option_id = cell_text(option.get("practice_option_id"))
            family_ids = option.get("family_ids") if isinstance(option.get("family_ids"), list) else []
            for family_id_value in family_ids:
                family_id = cell_text(family_id_value)
                row = display_family_map.get((display_id, family_id), pd.Series(dtype=object))
                candidate = family_map.get(family_id, pd.Series(dtype=object))
                rows.append(
                    {
                        "display_id": display_id,
                        "display_name": cell_text(row.get("display_name")) or cell_text(display.get("display_name")),
                        "practice_option_id": practice_option_id,
                        "family_id": family_id,
                        "representative_cost_item_name": cell_text(row.get("representative_cost_item_name")),
                        "representative_project_description": cell_text(row.get("representative_project_description")),
                        "unit": cell_text(candidate.get("unit_normalized")) or cell_text(row.get("unit")) or cell_text(candidate.get("unit")),
                        "本次召回样本数": row.get("本次召回样本数", ""),
                        "本次召回工程包数": row.get("本次召回工程包数", ""),
                        "item_query_similarity最大值": row.get("item_query_similarity最大值", ""),
                        "unit_price_min": candidate.get("本次召回综合单价最低值", ""),
                        "unit_price_median": candidate.get("本次召回综合单价中位数", ""),
                        "unit_price_max": candidate.get("本次召回综合单价最高值", ""),
                        "option_sample_count": int(option.get("option_sample_count") or 0),
                        "option_package_count": int(option.get("option_package_count") or 0),
                        "is_original_option": False,
                        "is_selected_option": False,
                        "option_selection_decision": "",
                        "option_selection_reason": "",
                        "is_original_family": False,
                        "is_representative_family": False,
                        "representative_selection_reason": "",
                    }
                )
    frame = pd.DataFrame(rows, columns=DISPLAY_OPTION_GROUPING_TRACE_COLUMNS)
    if frame.empty:
        return frame
    frame["_item_similarity_sort"] = pd.to_numeric(frame["item_query_similarity最大值"], errors="coerce").fillna(-1)
    frame = frame.sort_values(
        ["display_id", "practice_option_id", "family_id", "_item_similarity_sort"],
        ascending=[True, True, True, False],
    ).drop(columns=["_item_similarity_sort"])
    return frame[DISPLAY_OPTION_GROUPING_TRACE_COLUMNS]


def attach_original_practice_options(
    plan_items: pd.DataFrame,
    displays_with_options: pd.DataFrame,
) -> pd.DataFrame:
    family_to_option: dict[str, str] = {}
    for _index, display in displays_with_options.iterrows():
        options = display.get("practice_options") if isinstance(display.get("practice_options"), list) else []
        for option in options:
            if not isinstance(option, dict):
                continue
            option_id = cell_text(option.get("practice_option_id"))
            if not option_id:
                raise ValueError("practice_option_id 不得为空")
            for family_value in option.get("family_ids") or []:
                family_id = cell_text(family_value)
                if not family_id:
                    raise ValueError("practice option 中 family_id 不得为空")
                if family_id in family_to_option:
                    raise ValueError(f"family_id 必须唯一映射到一个 practice_option_id: {family_id}")
                family_to_option[family_id] = option_id

    output = plan_items.copy()
    output["practice_option_id"] = output["family_id"].map(
        lambda value: family_to_option.get(cell_text(value), "")
    )
    missing = output["practice_option_id"].map(cell_text).eq("")
    if missing.any():
        raise ValueError(
            f"plan_items family_id 无法映射到 practice_option_id: "
            f"{join_non_empty(output.loc[missing, 'family_id'].map(cell_text).tolist(), limit=10)}"
        )
    return output


def generate_display_option_grouping(
    candidate_display_groups: pd.DataFrame,
    display_group_families: pd.DataFrame,
    candidate_families: pd.DataFrame,
    warnings: list[str] | None = None,
) -> tuple[pd.DataFrame, bool, bool, str, str, dict[str, Any], dict[str, Any], pd.DataFrame, list[dict[str, Any]]]:
    family_ids_by_display = {
        display_id: [family_id for family_id in group["family_id"].map(cell_text).tolist() if family_id]
        for display_id, group in display_group_families.groupby("display_id", sort=False, dropna=False)
    }
    multi_family_mask = candidate_display_groups["display_id"].map(
        lambda value: len(family_ids_by_display.get(cell_text(value), [])) > 1
    )
    multi_family_displays = candidate_display_groups[multi_family_mask].copy().reset_index(drop=True)
    family_count = sum(
        len(family_ids_by_display.get(cell_text(row.get("display_id")), []))
        for _index, row in multi_family_displays.iterrows()
    )
    prompts: list[str] = []
    responses: list[Any] = []
    errors_by_display: dict[str, str] = {}
    per_display_traces: list[dict[str, Any]] = []
    fallback = False
    grouped_frames: list[pd.DataFrame] = []
    for _index, display in multi_family_displays.iterrows():
        display_id = cell_text(display.get("display_id"))
        current_display = pd.DataFrame([display.to_dict()])
        prompt, record = build_display_option_grouping_prompt(
            current_display, display_group_families, candidate_families
        )
        prompts.append(prompt)
        current_family_count = len(record.get("candidate_families") or [])
        max_tokens = min(4096, max(512, 256 + current_family_count * 96))
        response = None
        current_error = ""
        grouping_details = {
            "parsed_family_tags": [], "original_groups": [], "final_groups": [],
            "auto_split_applied": False, "auto_split_reasons": [],
        }
        try:
            response = request_llm_json_with_usage(
                prompt, max_tokens=max_tokens,
                system_prompt="只输出顶层仅含 families 和 groups 的 JSON object，不输出 display_id 或解释。",
            )
            grouped_display, grouped_meta = parse_display_option_grouping_result(
                response.content, current_display, display_group_families, warnings,
            )
            grouping_details = grouped_meta.get("grouping_details_by_display", {}).get(display_id, grouping_details)
        except (RuntimeError, ValueError) as exc:
            fallback = True
            current_error = f"fallback_single_family_options: {exc}"
            errors_by_display[display_id] = current_error
            append_warning(warnings, f"display_option_grouping_fallback_single_family_options:{display_id}")
            fallback_family_ids = family_ids_by_display[display_id]
            fallback_result = {
                "families": [
                    {"family_id": family_id, "thickness": "", "material": "", "level": ""}
                    for family_id in fallback_family_ids
                ],
                "groups": [[family_id] for family_id in fallback_family_ids],
            }
            grouped_display, grouped_meta = parse_display_option_grouping_result(
                fallback_result, current_display, display_group_families, warnings,
            )
            grouping_details = grouped_meta.get("grouping_details_by_display", {}).get(display_id, grouping_details)
        display_name = cell_text(display.get("display_name"))
        per_display_trace = trace_row(
            "display_option_grouping",
            f"为 Display {display_id} {display_name} 划分 Option",
            not current_error,
            error=current_error,
            prompt=prompt,
            max_tokens=max_tokens,
            input_summary=json_text({
                "display_id": display_id,
                "display_name": display_name,
                "candidate_family_count": current_family_count,
            }),
            usage=response.usage if response is not None else None,
            raw_response=getattr(response, "raw_content", "") if response is not None else "",
        )
        per_display_trace.update({
            "display_id": display_id,
            "display_name": display_name,
            "candidate_family_count": current_family_count,
            "parsed_family_tags": grouping_details["parsed_family_tags"],
            "original_groups": grouping_details["original_groups"],
            "final_groups": grouping_details["final_groups"],
            "auto_split_applied": grouping_details["auto_split_applied"],
            "auto_split_reasons": grouping_details["auto_split_reasons"],
            "status": "fallback" if current_error else "success",
            "error": current_error,
        })
        per_display_traces.append(per_display_trace)
        grouped_frames.append(grouped_display)
        if response is not None:
            responses.append(response)

    grouped_displays = pd.concat(grouped_frames, ignore_index=True) if grouped_frames else pd.DataFrame()
    error_message = "；".join(f"{display_id}: {message}" for display_id, message in errors_by_display.items())
    combined_prompt = "\n\n".join(prompts)
    combined_usage = {
        key: sum(int((getattr(response, "usage", {}) or {}).get(key) or 0) for response in responses)
        for key in ["prompt_tokens", "completion_tokens", "total_tokens"]
    }
    combined_raw_response = "\n".join(
        cell_text(getattr(response, "raw_content", "")) for response in responses
    )

    family_map = {cell_text(row.get("family_id")): row for _index, row in candidate_families.iterrows()}
    grouped_map = {cell_text(row.get("display_id")): row for _index, row in grouped_displays.iterrows()}
    display_rows: list[dict[str, Any]] = []
    meta = empty_display_option_grouping_meta()
    for _index, display in candidate_display_groups.iterrows():
        display_id = cell_text(display.get("display_id"))
        family_ids = family_ids_by_display.get(display_id, [])
        if not family_ids:
            raise ValueError(f"display 缺少 candidate families: {display_id}")
        if len(family_ids) == 1:
            family_id = family_ids[0]
            family = family_map.get(family_id, pd.Series(dtype=object))
            row = {
                "display_id": display_id,
                "display_name": cell_text(display.get("display_name")),
                "unit": cell_text(display.get("unit")),
                "family_count": display.get("family_count", 1),
                "practice_options": [
                    {
                        "practice_option_id": f"{display_id}-O01",
                        "sample_count": int(numeric_or_none(family.get("本次召回样本数")) or 0),
                        "family_ids": [family_id],
                    }
                ],
            }
            meta["programmatic_single_family_display_count"] += 1
        else:
            grouped = grouped_map.get(display_id)
            if grouped is None:
                raise ValueError(f"display_option_grouping 缺少 display: {display_id}")
            row = grouped.to_dict()
        options = row["practice_options"]
        display_rows.append(row)
        meta["display_ids"].append(display_id)
        meta["practice_option_count"] += len(options)
        meta["families_grouped_count"] += len(family_ids)
        meta["option_count_by_display"][display_id] = len(options)
        meta["max_options_per_display"] = max(meta["max_options_per_display"], len(options))
    meta["llm_display_count"] = len(multi_family_displays)
    displays_with_options = pd.DataFrame(display_rows)
    trace_frame = build_display_option_grouping_trace_frame(
        displays_with_options, display_group_families, candidate_families
    )
    trace = trace_row(
        "display_option_grouping",
        "将全部 candidate display 内的 family 划分为独立价格统计口径的 practice options",
        not fallback,
        error=error_message,
        prompt=combined_prompt,
        max_tokens=4096 if len(multi_family_displays) else 0,
        input_summary=json_text(
            {
                "display_count": len(candidate_display_groups),
                "llm_display_count": len(multi_family_displays),
                "programmatic_single_family_display_count": meta["programmatic_single_family_display_count"],
                "family_count": family_count,
                "practice_option_count": meta.get("practice_option_count", 0),
            }
        ),
        usage=combined_usage if responses else None,
        raw_response=combined_raw_response,
    )
    return (
        displays_with_options, not fallback, fallback, error_message, combined_prompt,
        trace, meta, trace_frame, per_display_traces,
    )


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


def display_option_maps(
    displays_with_options: pd.DataFrame,
) -> tuple[dict[str, pd.Series], dict[tuple[str, str], dict[str, Any]]]:
    display_map = {
        cell_text(row.get("display_id")): row
        for _index, row in displays_with_options.iterrows()
    }
    option_map: dict[tuple[str, str], dict[str, Any]] = {}
    for display_id, row in display_map.items():
        options = row.get("practice_options") if isinstance(row.get("practice_options"), list) else []
        for option in options:
            if isinstance(option, dict):
                option_map[(display_id, cell_text(option.get("practice_option_id")))] = option
    return display_map, option_map


def attach_option_support_counts(
    displays_with_options: pd.DataFrame,
    evidence_items: pd.DataFrame,
) -> pd.DataFrame:
    output = displays_with_options.copy(deep=True)
    evidence_family_ids = evidence_items.get("family_id", pd.Series(dtype=object)).map(cell_text)
    for index, display in output.iterrows():
        options = display.get("practice_options") if isinstance(display.get("practice_options"), list) else []
        supported_options: list[dict[str, Any]] = []
        for raw_option in options:
            option = dict(raw_option)
            family_ids = {cell_text(value) for value in option.get("family_ids", []) if cell_text(value)}
            evidence = evidence_items[evidence_family_ids.isin(family_ids)]
            package_ids = evidence.get("project_package_id", pd.Series(dtype=object)).map(cell_text)
            option["option_sample_count"] = int(len(evidence))
            option["option_package_count"] = int(package_ids[package_ids.ne("")].nunique())
            supported_options.append(option)
        output.at[index, "practice_options"] = supported_options
    return output


def apply_option_selection_to_grouping_trace(
    trace: pd.DataFrame,
    option_selection_trace: pd.DataFrame,
) -> pd.DataFrame:
    output = trace.copy()
    if output.empty or option_selection_trace.empty:
        return output
    selection_by_display = {
        cell_text(row.get("display_id")): row
        for _index, row in option_selection_trace.iterrows()
    }
    for index, row in output.iterrows():
        selection = selection_by_display.get(cell_text(row.get("display_id")))
        if selection is None:
            continue
        option_id = cell_text(row.get("practice_option_id"))
        family_id = cell_text(row.get("family_id"))
        output.at[index, "is_original_option"] = option_id == cell_text(selection.get("original_option_id"))
        output.at[index, "is_selected_option"] = option_id == cell_text(selection.get("selected_option_id"))
        output.at[index, "option_selection_decision"] = cell_text(selection.get("option_selection_decision"))
        output.at[index, "option_selection_reason"] = cell_text(selection.get("selection_reason"))
        output.at[index, "is_original_family"] = family_id == cell_text(selection.get("original_family_id"))
        output.at[index, "is_representative_family"] = family_id == cell_text(selection.get("representative_family_id"))
        output.at[index, "representative_selection_reason"] = cell_text(selection.get("representative_selection_reason"))
    return output[DISPLAY_OPTION_GROUPING_TRACE_COLUMNS]


def family_payload(family_id: str, family_map: dict[str, pd.Series]) -> dict[str, Any]:
    row = family_map.get(family_id, pd.Series(dtype=object))
    return {
        "family_id": family_id,
        "cost_item_name": cell_text(row.get("representative_cost_item_name")),
        "project_description": cell_text(row.get("representative_project_description")),
        "unit": cell_text(row.get("unit_normalized")) or cell_text(row.get("unit")),
        "normalized_signature": cell_text(row.get("normalized_signature")),
    }


def option_selection_family_payload(family_id: str, family_map: dict[str, pd.Series]) -> dict[str, Any]:
    payload = family_payload(family_id, family_map)
    return {
        "cost_item_name": payload["cost_item_name"],
        "project_description": payload["project_description"],
        "unit": payload["unit"],
    }


def choose_representative_family(
    option: dict[str, Any], original_family_id: str, family_map: dict[str, pd.Series]
) -> tuple[str, str]:
    family_ids = [cell_text(value) for value in option.get("family_ids", []) if cell_text(value)]
    ranked = sorted(
        family_ids,
        key=lambda family_id: (
            -int(numeric_or_none(family_map.get(family_id, pd.Series(dtype=object)).get("本次召回样本数")) or 0),
            -int(numeric_or_none(family_map.get(family_id, pd.Series(dtype=object)).get("本次召回工程包数")) or 0),
            0 if family_id == original_family_id else 1,
            family_id,
        ),
    )
    if not ranked:
        raise ValueError("selected option 不得缺少 family")
    selected = ranked[0]
    rows = {family_id: family_map.get(family_id, pd.Series(dtype=object)) for family_id in family_ids}
    max_samples = max(int(numeric_or_none(row.get("本次召回样本数")) or 0) for row in rows.values())
    sample_tied = [fid for fid, row in rows.items() if int(numeric_or_none(row.get("本次召回样本数")) or 0) == max_samples]
    if len(sample_tied) == 1:
        reason = "selected_option_family_sample_count_max"
    else:
        max_packages = max(int(numeric_or_none(rows[fid].get("本次召回工程包数")) or 0) for fid in sample_tied)
        package_tied = [fid for fid in sample_tied if int(numeric_or_none(rows[fid].get("本次召回工程包数")) or 0) == max_packages]
        if len(package_tied) == 1:
            reason = "selected_option_family_package_count_max"
        elif original_family_id in package_tied:
            reason = "support_tie_original_family"
        else:
            reason = "stable_family_id_tiebreak"
    return selected, reason


def choose_most_supported_option(
    options: list[dict[str, Any]], original_option_id: str,
) -> tuple[str, str]:
    if not options:
        raise ValueError("Display 不得缺少 Option")
    ranked = sorted(options, key=lambda option: (
        -int(option.get("option_sample_count") or 0),
        -int(option.get("option_package_count") or 0),
        0 if cell_text(option.get("practice_option_id")) == original_option_id else 1,
        cell_text(option.get("practice_option_id")),
    ))
    selected = ranked[0]
    max_samples = int(selected.get("option_sample_count") or 0)
    sample_tied = [option for option in options if int(option.get("option_sample_count") or 0) == max_samples]
    if len(sample_tied) == 1:
        reason = "no_explicit_match_selected_by_sample_count"
    else:
        max_packages = int(selected.get("option_package_count") or 0)
        package_tied = [option for option in sample_tied if int(option.get("option_package_count") or 0) == max_packages]
        if len(package_tied) == 1:
            reason = "no_explicit_match_selected_by_package_count"
        elif any(cell_text(option.get("practice_option_id")) == original_option_id for option in package_tied):
            reason = "support_tie_original_option"
        else:
            reason = "stable_option_id_tiebreak"
    return cell_text(selected.get("practice_option_id")), reason


def select_final_options(
    raw_text: str,
    plan_items: pd.DataFrame,
    sample_lookup: dict[str, dict[str, Any]],
    displays_with_options: pd.DataFrame,
    candidate_families: pd.DataFrame,
    warnings: list[str] | None = None,
    evidence_items: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    displays_with_options = attach_option_support_counts(
        displays_with_options, evidence_items if evidence_items is not None else pd.DataFrame()
    )
    display_map, option_map = display_option_maps(displays_with_options)
    family_map = {cell_text(row.get("family_id")): row for _index, row in candidate_families.iterrows()}
    selected_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    llm_traces: list[dict[str, Any]] = []
    for _index, item in plan_items.iterrows():
        position = int(item["item_position"])
        sample = sample_lookup.get(cell_text(item.get("stable_sample_id")))
        if sample is None:
            raise ValueError(f"最终清单无法回查证据: item_position={position}")
        display_id = cell_text(sample.get("display_id"))
        original_family_id = cell_text(sample.get("family_id"))
        original_option_id = cell_text(sample.get("practice_option_id"))
        display = display_map.get(display_id)
        options = display.get("practice_options") if display is not None and isinstance(display.get("practice_options"), list) else []
        selected_option_id = original_option_id
        decision = "no_explicit_match"
        selection_reason = "single_option"
        prompt = ""
        response = None
        error_message = ""
        if len(options) > 1:
            payload = {
                "raw_query": raw_text,
                "display_id": display_id,
                "original_option_id": original_option_id,
                "options": [{
                    "option_id": cell_text(option.get("practice_option_id")),
                    "families": [
                        option_selection_family_payload(cell_text(fid), family_map)
                        for fid in option.get("family_ids", [])
                    ],
                } for option in options],
            }
            prompt = f"""
你只负责判断用户是否明确提出了足以区分当前 Options 的条件。

raw_query 是用户原始需求。
original_option_id 是参考历史工程中的原 Option。
options 是当前同一 Display 下可选的所有 Option。

判断规则：

1. 只有当 raw_query 明确提到能够区分 Options 的材料、工艺、规格、厚度、楼层、层数、部位或其他限定条件时，才返回 explicit_match。

2. 用户只是提到当前清单对象本身，不算明确区分条件。
   例如用户只说“需要垂直运输”“需要脚手架”，而不同 Options 的差异在楼层或具体部位时，不得据此选择某个带限定的 Option。

3. 用户没有提到楼层、层数或高度时，不得根据参考历史工程中的楼层限定推断用户需要该限定。
   例如：
   - “1层楼垂直运输费”
   - “垂直运输费”
   用户未说明楼层时，返回 no_explicit_match。

4. 用户没有提到厚度、材料、规格、部位或施工方法时，不得根据 original_option_id、original_family 或历史工程内容补充这些条件。

5. original_option_id 只是历史参考，不是用户要求。

6. 不得根据样本数、价格、相似度、Option 顺序、Option 编号、Family 数量或历史出现频率判断 explicit_match。

7. 只有当用户明确条件能够唯一对应某个现有 Option 时，才能返回 explicit_match。

8. 如果用户表达模糊、没有明确区分条件，或者多个 Options 都可能符合，必须返回 no_explicit_match。

9. selected_option_id 只能是 options 中已有的 option_id。

10. 不得新增 Option，不得修改 Option，不得解释，不得输出思考过程。

输出格式只能是以下两种之一：

{{"decision":"explicit_match","selected_option_id":"..."}}

或：

{{"decision":"no_explicit_match","selected_option_id":""}}

输入：
{json_text(payload)}
""".strip()
            try:
                response = request_llm_json_with_usage(
                    prompt, max_tokens=128,
                    system_prompt="只输出 decision 和 selected_option_id 两字段 JSON object。",
                )
                result = response.content
                allowed_ids = {cell_text(option.get("practice_option_id")) for option in options}
                if not isinstance(result, dict) or set(result) != {"decision", "selected_option_id"}:
                    raise ValueError("option selection 只允许 decision 和 selected_option_id")
                decision = cell_text(result.get("decision"))
                candidate = cell_text(result.get("selected_option_id"))
                if decision == "explicit_match":
                    if not candidate or candidate not in allowed_ids:
                        raise ValueError("explicit_match 必须返回现有非空 Option ID")
                    selected_option_id = candidate
                    selection_reason = "user_explicit_match"
                elif decision == "no_explicit_match":
                    if candidate:
                        raise ValueError("no_explicit_match 的 selected_option_id 必须为空")
                    selected_option_id, selection_reason = choose_most_supported_option(options, original_option_id)
                else:
                    raise ValueError("option selection decision 非法")
            except (RuntimeError, ValueError) as exc:
                error_message = str(exc)
                decision = "no_explicit_match"
                selected_option_id, _support_reason = choose_most_supported_option(options, original_option_id)
                selection_reason = "llm_failed_selected_by_support"
                append_warning(warnings, f"option_selection_llm_failed_selected_by_support:{position}")
            llm_traces.append(trace_row(
                "option_selection", f"为最终清单 {position} 选择现有 Option", not error_message,
                error=error_message, prompt=prompt, max_tokens=128,
                usage=response.usage if response is not None else None,
                raw_response=getattr(response, "raw_content", "") if response is not None else "",
            ))
        selected_option = option_map.get((display_id, selected_option_id))
        if selected_option is None:
            raise ValueError(f"最终 Option 回查失败: {display_id}/{selected_option_id}")
        representative_family_id, representative_selection_reason = choose_representative_family(selected_option, original_family_id, family_map)
        representative = family_payload(representative_family_id, family_map)
        selected_rows.append({
            **item.to_dict(), "display_id": display_id,
            "original_option_id": original_option_id, "selected_option_id": selected_option_id,
            "original_family_id": original_family_id, "representative_family_id": representative_family_id,
            "cost_item_name": representative["cost_item_name"],
            "project_description": representative["project_description"], "unit": representative["unit"],
            "normalized_signature": representative["normalized_signature"],
        })
        trace_rows.append({
            "final_item_position": position, "display_id": display_id,
            "original_option_id": original_option_id, "selected_option_id": selected_option_id,
            "original_family_id": original_family_id, "representative_family_id": representative_family_id,
            "candidate_option_count": len(options), "whether_replaced": selected_option_id != original_option_id,
            "selection_reason": selection_reason,
            "option_selection_decision": decision,
            "representative_selection_reason": representative_selection_reason,
        })
    return pd.DataFrame(selected_rows), pd.DataFrame(trace_rows, columns=OPTION_SELECTION_TRACE_COLUMNS), llm_traces


def validate_quantity(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("quantity 必须为 object")
    quantity_type = cell_text(value.get("type"))
    if quantity_type == "exact":
        if set(value) != {"type", "value"}:
            raise ValueError("exact quantity 只能包含 type 和 value")
        number = numeric_or_none(value.get("value"))
        if number is None or number <= 0:
            raise ValueError("exact quantity value 必须大于 0")
        return {"type": "exact", "value": number}
    raise ValueError("quantity.type 只能是 exact")


def quantity_rule_payload(raw_text: str, plan_items: pd.DataFrame) -> dict[str, Any]:
    items = [
        {
            "item_position": int(row["item_position"]),
            "cost_item_name": cell_text(row.get("cost_item_name")),
            "project_description": cell_text(row.get("project_description")),
            "unit": cell_text(row.get("unit")),
        }
        for _index, row in plan_items.iterrows()
    ]
    if not items:
        raise ValueError("最终清单为空，无法确定工程量")
    return {"user_query": raw_text, "items": items}


def build_quantity_determination_prompt(
    raw_text: str, plan_items: pd.DataFrame,
) -> str:
    payload = quantity_rule_payload(raw_text, plan_items)
    return f"""
任务：判断每条清单的工程量来源。

工程量来源只能是：

- user_explicit
  用户明确给出数量，且该数量能根据部位、项目名称、材料、规格和单位直接对应当前清单。
  quantity 填用户给出的数量。

- historical_median
  当前清单没有可直接采用的用户数量。
  可能是用户未提供数量，也可能是用户提供的数量对应其他清单。
  quantity 必须为 null，具体工程量由程序使用全库同类历史样本中位数计算。

规则：

1. 只把用户数量绑定到直接匹配的清单。
2. 不得把同一个数量无依据地用于多个清单。
3. 不得根据不同清单之间的历史工程量关系推导数量。
4. 材料、规格、部位和单位的匹配优先于清单顺序。
5. 模糊的总体数量无法对应具体清单时，使用 historical_median。
6. 必须完整覆盖全部 item_position，不得遗漏或重复。
7. explanation 必须结合当前清单说明原因，不得所有项目重复同一句模板。
8. 只输出合法 JSON，不得增加其他字段。

输出：

{{
  "items": [
    {{
      "item_position": 0,
      "quantity_source": "user_explicit",
      "quantity": 100,
      "explanation": "用户给出的数量与当前清单的部位、材料和规格直接对应"
    }},
    {{
      "item_position": 1,
      "quantity_source": "historical_median",
      "quantity": null,
      "explanation": "用户给出的数量对应其他施工内容，当前清单缺少可直接采用的数量依据"
    }}
  ]
}}

输入：
{json_text(payload)}
""".strip()


def parse_quantity_determination_result(
    result: Any,
    plan_items: pd.DataFrame,
) -> dict[int, dict[str, Any]]:
    return validate_quantity_determination_result(result, plan_items)


def validate_quantity_determination_result(
    result: Any, plan_items: pd.DataFrame,
) -> dict[int, dict[str, Any]]:
    if plan_items.empty:
        raise ValueError("最终清单为空，无法确定工程量")
    if not isinstance(result, dict) or set(result) != {"items"}:
        raise ValueError("quantity determination 顶层字段非法")
    raw_entries = result.get("items")
    if not isinstance(raw_entries, list):
        raise ValueError("quantity determination items 必须是数组")
    expected_positions = {int(value) for value in plan_items["item_position"].tolist()}
    parsed: dict[int, dict[str, Any]] = {}
    for entry in raw_entries:
        if not isinstance(entry, dict) or set(entry) != {
            "item_position", "quantity_source", "quantity", "explanation",
        }:
            raise ValueError("quantity determination item 字段非法")
        position = entry.get("item_position")
        if isinstance(position, bool) or not isinstance(position, int):
            raise ValueError("quantity item_position 必须是整数")
        if position in parsed:
            raise ValueError(f"quantity item_position 重复: {position}")
        if position not in expected_positions:
            raise ValueError(f"quantity item_position 非法: {position}")
        source = cell_text(entry.get("quantity_source"))
        if source not in {"user_explicit", "historical_median"}:
            raise ValueError(f"quantity_source 非法: item_position={position}")
        explanation = cell_text(entry.get("explanation"))
        if not explanation:
            raise ValueError(f"quantity explanation 不得为空: item_position={position}")
        raw_quantity = entry.get("quantity")
        quantity = None
        if source == "user_explicit":
            if isinstance(raw_quantity, bool) or not isinstance(raw_quantity, (int, float)):
                raise ValueError(f"user_explicit quantity 必须是正数: item_position={position}")
            quantity = numeric_or_none(raw_quantity)
            if quantity is None or quantity <= 0:
                raise ValueError(f"user_explicit quantity 必须是正数: item_position={position}")
        elif raw_quantity is not None:
            raise ValueError(f"historical_median quantity 必须为 null: item_position={position}")
        parsed[position] = {
            "quantity_source": source,
            "quantity": quantity,
            "explanation": explanation,
        }
    if set(parsed) != expected_positions:
        raise ValueError("quantity determination 必须完整覆盖全部最终清单")
    return parsed


def quantity_determination_fallback(plan_items: pd.DataFrame) -> dict[int, dict[str, Any]]:
    return {
        int(row["item_position"]): {
            "quantity_source": "historical_median",
            "quantity": None,
            "explanation": "未能从用户描述中获得可直接采用的当前清单工程量",
        }
        for _index, row in plan_items.iterrows()
    }


def calculate_quantities(
    plan_items: pd.DataFrame,
    determinations: dict[int, dict[str, Any]],
    quantity_statistics: dict[int, dict[str, Any]],
    sample_lookup: dict[str, dict[str, Any]],
    warnings: list[str] | None = None,
) -> dict[int, dict[str, Any]]:
    quantities: dict[int, dict[str, Any]] = {}
    for _index, row in plan_items.iterrows():
        position = int(row["item_position"])
        determination = determinations[position]
        source = cell_text(determination.get("quantity_source"))
        explanation = cell_text(determination.get("explanation"))
        stats = quantity_statistics.get(position, {})
        fallback_used = False
        fallback_reason = ""
        if source == "user_explicit":
            value = numeric_or_none(determination.get("quantity"))
            reason = explanation
            output_stats = {}
        else:
            value = numeric_or_none(stats.get("median"))
            output_stats = stats
            if value is None:
                append_warning(warnings, f"quantity_median_unavailable:{position}")
                stable_sample_id = cell_text(row.get("stable_sample_id"))
                best_sample = sample_lookup.get(stable_sample_id, {})
                best_quantity = numeric_or_none(best_sample.get("quantity"))
                target_unit = normalized_unit(cell_text(row.get("unit_normalized")) or row.get("unit"))
                best_unit = normalized_unit(cell_text(best_sample.get("unit_normalized")) or best_sample.get("unit"))
                if best_quantity is None or best_quantity <= 0 or not target_unit or best_unit != target_unit:
                    raise ValueError(f"无有效工程量中位数或最佳召回样本工程量: item_position={position}")
                value = best_quantity
                fallback_used = True
                fallback_reason = "best_sample_quantity"
                append_warning(warnings, f"quantity_best_sample_fallback:{position}")
                reason = f"{explanation}；全库同类样本无有效中位数，暂采用最佳召回样本工程量。"
            else:
                unit = cell_text(row.get("unit_normalized")) or cell_text(row.get("unit"))
                reason = (
                    f"{explanation}；采用全库召回的{int(stats.get('sample_count') or 0)}条同类历史样本"
                    f"工程量中位数{round(float(value), 4):g}{unit}暂估。"
                )
        rounded = None if value is None else round(float(value), 4)
        if rounded is None or rounded <= 0:
            raise ValueError(f"最终工程量必须大于 0: item_position={position}")
        quantities[position] = {
            "quantity": {"type": "exact", "value": rounded},
            "quantity_source": source,
            "quantity_explanation": explanation,
            "quantity_reason": reason,
            "quantity_sample_count": output_stats.get("sample_count"),
            "quantity_minimum": output_stats.get("minimum"),
            "quantity_median": output_stats.get("median"),
            "quantity_maximum": output_stats.get("maximum"),
            "quantity_fallback_used": fallback_used,
            "quantity_fallback_reason": fallback_reason,
        }
    return quantities


def build_scenario_from_plan_items(
    project_package_id: str,
    plan_items: pd.DataFrame,
    sample_lookup: dict[str, dict[str, Any]],
    quantities: dict[int, dict[str, Any]],
) -> EstimateScenario:
    items: list[ScenarioItem] = []
    for _index, row in plan_items.iterrows():
        position = int(row["item_position"])
        stable_sample_id = cell_text(row.get("stable_sample_id"))
        sample = sample_lookup.get(stable_sample_id)
        if sample is None:
            raise ValueError(f"最终清单无法回查证据: item_position={position}, stable_sample_id={stable_sample_id}")
        if cell_text(sample.get("project_package_id")) != project_package_id:
            raise ValueError(f"最终清单工程包映射不一致: item_position={position}")
        quantity_result = quantities[position]
        items.append(
            ScenarioItem(
                project_package_id=project_package_id,
                stable_sample_id=stable_sample_id,
                source_ref=cell_text(sample.get("source_ref")),
                display_id=cell_text(row.get("display_id")) or cell_text(sample.get("display_id")),
                practice_option_id=cell_text(row.get("selected_option_id")) or cell_text(sample.get("practice_option_id")),
                original_option_id=cell_text(row.get("original_option_id")) or cell_text(sample.get("practice_option_id")),
                original_family_id=cell_text(row.get("original_family_id")) or cell_text(sample.get("family_id")),
                representative_family_id=cell_text(row.get("representative_family_id")) or cell_text(sample.get("family_id")),
                selection_reason="",
                quantity=quantity_result["quantity"],
                quantity_reason=quantity_result["quantity_reason"],
                item_position=position,
                quantity_source=quantity_result["quantity_source"],
                quantity_explanation=quantity_result["quantity_explanation"],
                quantity_sample_count=quantity_result["quantity_sample_count"],
                quantity_minimum=quantity_result["quantity_minimum"],
                quantity_median=quantity_result["quantity_median"],
                quantity_maximum=quantity_result["quantity_maximum"],
                quantity_fallback_used=quantity_result["quantity_fallback_used"],
                quantity_fallback_reason=quantity_result["quantity_fallback_reason"],
            )
        )
    return EstimateScenario("S001", 1, "", "", items)


def generate_quantity_determination(
    raw_text: str,
    project_package_id: str,
    plan_items: pd.DataFrame,
    sample_lookup: dict[str, dict[str, Any]],
    displays_with_options: pd.DataFrame,
    candidate_families: pd.DataFrame,
    samples: pd.DataFrame,
    warnings: list[str] | None = None,
) -> tuple[EstimateScenario, str, dict[str, Any]]:
    if plan_items.empty:
        raise ValueError("最终清单为空，无法确定工程量")
    prompt = build_quantity_determination_prompt(raw_text, plan_items)
    max_tokens = 4096
    response = None
    error_message = ""
    try:
        response = request_llm_json_with_usage(
            prompt,
            max_tokens=max_tokens,
            system_prompt="你只输出一个 JSON object，不输出解释、Markdown 或思考过程。",
        )
        determinations = validate_quantity_determination_result(response.content, plan_items)
    except (LLMServiceError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        error_message = str(exc)
        determinations = quantity_determination_fallback(plan_items)
        append_warning(warnings, "quantity_determination_fallback_historical_median")
    display_map, option_map = display_option_maps(displays_with_options)
    quantity_statistics: dict[int, dict[str, Any]] = {}
    for _index, row in plan_items.iterrows():
        position = int(row["item_position"])
        if determinations[position]["quantity_source"] == "user_explicit":
            quantity_statistics[position] = {}
            continue
        display_id = cell_text(row.get("display_id"))
        option_id = cell_text(row.get("selected_option_id")) or cell_text(row.get("practice_option_id"))
        display = display_map.get(display_id)
        option = option_map.get((display_id, option_id))
        if display is None or option is None:
            raise ValueError(f"工程量统计 display/option 回查失败: item_position={position}")
        expanded = expand_samples_for_option(option, display, candidate_families, samples)
        target_unit = cell_text(row.get("unit_normalized")) or cell_text(row.get("unit"))
        quantity_statistics[position] = build_quantity_statistics(expanded, target_unit)
    quantities = calculate_quantities(
        plan_items, determinations, quantity_statistics, sample_lookup, warnings
    )
    scenario = build_scenario_from_plan_items(
        project_package_id, plan_items, sample_lookup, quantities
    )
    trace = trace_row(
        "quantity_determination",
        "确定最终连续区间内全部清单的工程量",
        not error_message,
        error=error_message,
        prompt=prompt,
        max_tokens=max_tokens,
        input_summary=json_text({
            "selected_project_package_id": project_package_id,
            "expected_item_positions": plan_items["item_position"].tolist(),
        }),
        usage=response.usage if response is not None else None,
        raw_response=getattr(response, "raw_content", "") if response is not None else "",
        scenario_count=1,
        scenario_item_count=len(scenario.items),
    )
    trace["fallback"] = bool(error_message)
    trace["quantity_items"] = json_text([
        {
            "item_position": position,
            "quantity_source": result["quantity_source"],
            "llm_quantity": determinations[position]["quantity"],
            "final_quantity": result["quantity"]["value"],
            "quantity_sample_count": result["quantity_sample_count"],
            "quantity_minimum": result["quantity_minimum"],
            "quantity_median": result["quantity_median"],
            "quantity_maximum": result["quantity_maximum"],
            "median_available": result["quantity_median"] is not None,
            "quantity_fallback_used": result["quantity_fallback_used"],
            "quantity_fallback_reason": result["quantity_fallback_reason"],
            "error_message": error_message,
        }
        for position, result in quantities.items()
    ])
    return scenario, prompt, trace


def build_stable_sample_lookup(
    samples: pd.DataFrame,
    evidence_items: pd.DataFrame,
    display_group_families: pd.DataFrame,
    displays_with_options: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    stable_ids = samples.get("stable_sample_id", pd.Series(dtype=object)).map(cell_text)
    if len(stable_ids) != len(samples) or stable_ids.eq("").any():
        raise ValueError("samples 中 stable_sample_id 不得为空")
    duplicate_ids = stable_ids[stable_ids.duplicated(keep=False)]
    if not duplicate_ids.empty:
        raise ValueError(f"stable_sample_id 必须全局唯一: {join_non_empty(duplicate_ids.tolist(), limit=10)}")

    family_to_display: dict[str, str] = {}
    for _index, row in display_group_families.iterrows():
        family_id = cell_text(row.get("family_id"))
        display_id = cell_text(row.get("display_id"))
        if not family_id or not display_id:
            raise ValueError("family/display 映射不得为空")
        if family_id in family_to_display and family_to_display[family_id] != display_id:
            raise ValueError(f"family_id 映射到多个 display: {family_id}")
        family_to_display[family_id] = display_id

    family_to_option: dict[str, str] = {}
    for _index, display in displays_with_options.iterrows():
        options = display.get("practice_options") if isinstance(display.get("practice_options"), list) else []
        for option in options:
            if not isinstance(option, dict):
                continue
            option_id = cell_text(option.get("practice_option_id"))
            for family_value in option.get("family_ids") or []:
                family_id = cell_text(family_value)
                if family_id in family_to_option and family_to_option[family_id] != option_id:
                    raise ValueError(f"family_id 映射到多个 practice option: {family_id}")
                family_to_option[family_id] = option_id

    lookup: dict[str, dict[str, Any]] = {}
    for _index, row in evidence_items.iterrows():
        stable_sample_id = cell_text(row.get("stable_sample_id"))
        if not stable_sample_id:
            raise ValueError("evidence item 缺少 stable_sample_id")
        if stable_sample_id in lookup:
            raise ValueError(f"evidence pool 中 stable_sample_id 重复: {stable_sample_id}")
        family_id = cell_text(row.get("family_id"))
        display_id = family_to_display.get(family_id, "")
        practice_option_id = family_to_option.get(family_id, "")
        if not display_id or not practice_option_id:
            raise ValueError(f"stable_sample_id 无法映射到 display/option: {stable_sample_id}")
        lookup[stable_sample_id] = {
            **row.to_dict(),
            "stable_sample_id": stable_sample_id,
            "source_ref": cell_text(row.get("source_ref")),
            "family_id": family_id,
            "display_id": display_id,
            "practice_option_id": practice_option_id,
            "project_package_id": cell_text(row.get("project_package_id")),
        }
    return lookup


def build_final_explanation_prompt(
    raw_text: str,
    scenario: EstimateScenario,
    estimate_scenarios: pd.DataFrame,
    matched_project_examples: list[dict[str, Any]],
) -> str:
    selected_project_name = next(
        (example for example in matched_project_examples if cell_text(example.get("project_package_id")) == scenario.items[0].project_package_id),
        {},
    )
    selected_project_name = cell_text(selected_project_name.get("project_name")) or cell_text(
        selected_project_name.get("project_name_text")
    )
    item_columns = [
        "清单名称", "项目特征", "单位",
        "工程量", "工程量来源", "工程量说明", "综合单价P10", "综合单价中位数", "综合单价P90",
        "合价P10", "合价中位数", "合价P90", "价格证据样本数",
    ]
    items = replace_nan_records(estimate_scenarios[item_columns])
    return f"""
清单、工艺、工程量、价格和金额均已确定。只补充方案名称、整体说明和逐项说明。
不得增加、删除、重排或修改任何清单、工艺、工程量或价格，不得提及输入中不存在的维修对象。

item_explanations 必须与 final_items 数量和原顺序完全一致，只输出：
{{
  "scenario_name": "方案名称",
  "scenario_summary": "方案整体说明",
  "item_explanations": [
    {{"item_explanation": "逐项说明"}}
  ]
}}

输入：
{json_text({"user_query": raw_text, "selected_project_name": selected_project_name, "final_items": items})}
""".strip()


def parse_final_explanation_result(result: dict[str, Any], scenario: EstimateScenario) -> EstimateScenario:
    if not isinstance(result, dict) or set(result) != {"scenario_name", "scenario_summary", "item_explanations"}:
        raise ValueError("final_explanation 顶层字段非法")
    scenario_name = cell_text(result.get("scenario_name"))
    scenario_summary = cell_text(result.get("scenario_summary"))
    if not scenario_name or not scenario_summary:
        raise ValueError("scenario_name 和 scenario_summary 不得为空")
    raw_items = result.get("item_explanations")
    if not isinstance(raw_items, list) or len(raw_items) != len(scenario.items):
        raise ValueError("item_explanations 数量必须等于最终 items 数量")
    explanations: list[str] = []
    for position, raw_item in enumerate(raw_items):
        if not isinstance(raw_item, dict) or set(raw_item) != {"item_explanation"}:
            raise ValueError("item_explanation 只允许 item_explanation")
        explanation = cell_text(raw_item.get("item_explanation"))
        if not explanation:
            raise ValueError(f"item_explanation 不得为空: position={position}")
        explanations.append(explanation)
    items = [
        ScenarioItem(
            project_package_id=item.project_package_id,
            stable_sample_id=item.stable_sample_id,
            source_ref=item.source_ref,
            display_id=item.display_id,
            practice_option_id=item.practice_option_id,
            original_option_id=item.original_option_id,
            original_family_id=item.original_family_id,
            representative_family_id=item.representative_family_id,
            selection_reason=explanations[position],
            quantity=item.quantity,
            quantity_reason=item.quantity_reason,
            item_position=item.item_position,
            quantity_source=item.quantity_source,
            quantity_explanation=item.quantity_explanation,
            quantity_sample_count=item.quantity_sample_count,
            quantity_minimum=item.quantity_minimum,
            quantity_median=item.quantity_median,
            quantity_maximum=item.quantity_maximum,
            quantity_fallback_used=item.quantity_fallback_used,
            quantity_fallback_reason=item.quantity_fallback_reason,
        )
        for position, item in enumerate(scenario.items)
    ]
    return EstimateScenario("S001", 1, scenario_name, scenario_summary, items)


def generate_final_explanation(
    raw_text: str,
    scenario: EstimateScenario,
    estimate_scenarios: pd.DataFrame,
    matched_project_examples: list[dict[str, Any]],
    warnings: list[str] | None = None,
) -> tuple[EstimateScenario, bool, str, str, dict[str, Any]]:
    prompt = build_final_explanation_prompt(raw_text, scenario, estimate_scenarios, matched_project_examples)
    max_tokens = 4096
    try:
        response = request_llm_json_with_usage(
            prompt, max_tokens=max_tokens,
            system_prompt="你只输出一个 JSON object，不输出解释、Markdown 或思考过程。",
        )
        explained = parse_final_explanation_result(response.content, scenario)
        trace = trace_row(
            "final_explanation", "补充唯一估价方案的整体和逐项说明", True,
            prompt=prompt, max_tokens=max_tokens,
            input_summary=json_text({
                "selected_project_package_id": scenario.items[0].project_package_id,
                "selected_stable_sample_ids": [item.stable_sample_id for item in scenario.items],
                "item_count": len(scenario.items),
            }),
            usage=response.usage, raw_response=getattr(response, "raw_content", ""),
            scenario_count=1, scenario_item_count=len(scenario.items),
        )
        return explained, True, "", prompt, trace
    except (LLMServiceError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        append_warning(warnings, "final_explanation_failed")
        trace = trace_row(
            "final_explanation", "补充唯一估价方案的整体和逐项说明", False,
            error=str(exc), prompt=prompt, max_tokens=max_tokens,
            input_summary=json_text({
                "selected_project_package_id": scenario.items[0].project_package_id,
                "selected_stable_sample_ids": [item.stable_sample_id for item in scenario.items],
                "item_count": len(scenario.items),
            }),
            scenario_count=1, scenario_item_count=len(scenario.items),
        )
        return scenario, False, str(exc), prompt, trace


def generate_optional_final_explanation(
    with_explanations: bool,
    raw_text: str,
    scenario: EstimateScenario,
    estimate_scenarios: pd.DataFrame,
    matched_project_examples: list[dict[str, Any]],
    warnings: list[str] | None = None,
) -> tuple[EstimateScenario, bool, str, str, dict[str, Any]]:
    if with_explanations:
        return generate_final_explanation(
            raw_text, scenario, estimate_scenarios, matched_project_examples, warnings=warnings
        )
    return scenario, True, "", "", trace_row(
        "final_explanation",
        "项目级和清单级解释已按运行配置跳过",
        True,
        input_summary=json_text({"with_explanations": False}),
        scenario_count=1,
        scenario_item_count=len(scenario.items),
    )


def build_option_evidence_expansion_prompt(
    target_option: dict[str, Any],
    candidate_families: list[dict[str, Any]],
) -> str:
    payload = {
        "target_option": target_option,
        "candidate_families": candidate_families,
    }

    return f"""
任务：判断哪些候选 family 可以加入目标 option 的价格证据。

只有在以下内容基本一致时才可加入：
- 维修对象
- 施工动作
- 主要工作范围
- 材料或关键规格
- 单位

判断原则：

1. 名称写法不同，不代表不能合并。
2. “含人工”“含安装”“含拆除及安装”“含运输”等附带说明不同，可以合并。
3. 维修对象不同，不得合并。
4. 施工动作或主要工作范围明显不同，不得合并。
5. 材料、型号、尺寸或关键规格存在冲突，不得合并。
6. 单位不兼容，不得合并。
7. 信息不足或无法确认时，不要加入。
8. 只能返回输入中存在的 family_id，不得重复。

只输出合法 JSON：

{{
  "accepted_family_ids": []
}}

输入：
{json.dumps(payload, ensure_ascii=False)}
""".strip()


def validate_option_evidence_expansion_result(
    result: Any,
    allowed_family_ids: list[str],
) -> list[str]:
    if not isinstance(result, dict) or set(result) != {"accepted_family_ids"}:
        raise ValueError("option evidence expansion 顶层字段非法")
    accepted = result.get("accepted_family_ids")
    if not isinstance(accepted, list):
        raise ValueError("option evidence expansion accepted_family_ids 必须是数组")
    allowed = set(allowed_family_ids)
    parsed: list[str] = []
    seen: set[str] = set()
    for value in accepted:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("option evidence expansion family_id 必须是非空字符串")
        family_id = value.strip()
        if family_id in seen:
            raise ValueError(f"option evidence expansion family_id 重复: {family_id}")
        if family_id not in allowed:
            raise ValueError(f"option evidence expansion family_id 非法: {family_id}")
        seen.add(family_id)
        parsed.append(family_id)
    return parsed


def option_evidence_expansion_candidates(
    item: ScenarioItem,
    option: dict[str, Any],
    candidate_families: pd.DataFrame,
) -> tuple[list[str], dict[str, Any], list[dict[str, Any]]]:
    original_family_ids = list(dict.fromkeys(
        cell_text(value) for value in option.get("family_ids", []) if cell_text(value)
    ))
    if not original_family_ids:
        raise ValueError(f"Option 缺少 family_ids: option_id={item.practice_option_id}")

    family_ids = candidate_families.get("family_id", pd.Series(dtype=object)).map(cell_text)
    representative_rows = candidate_families[family_ids.eq(item.representative_family_id)]
    if len(representative_rows) != 1:
        raise ValueError(
            f"代表 Family 无法唯一回查: family_id={item.representative_family_id}, "
            f"matched_rows={len(representative_rows)}"
        )
    representative = representative_rows.iloc[0]
    target_unit = cell_text(representative.get("unit_normalized"))
    target_option = {
        "cost_item_name": normalize_evidence_expansion_name(
            representative.get("representative_cost_item_name")
        ),
        "project_description": cell_text(
            representative.get("representative_project_description")
        ),
        "unit": target_unit,
    }

    candidates: list[dict[str, Any]] = []
    seen = set(original_family_ids)
    for _index, family in candidate_families.iterrows():
        family_id = cell_text(family.get("family_id"))
        if not family_id or family_id in seen:
            continue
        seen.add(family_id)
        if cell_text(family.get("unit_normalized")) != target_unit:
            continue
        candidates.append({
            "family_id": family_id,
            "cost_item_name": normalize_evidence_expansion_name(
                family.get("representative_cost_item_name")
            ),
            "project_description": cell_text(
                family.get("representative_project_description")
            ),
            "unit": cell_text(family.get("unit_normalized")),
        })
        if len(candidates) >= 20:
            break
    return original_family_ids, target_option, candidates


def expand_option_price_evidence_families(
    final_items: list[ScenarioItem],
    displays_with_options: pd.DataFrame,
    candidate_families: pd.DataFrame,
    samples: pd.DataFrame,
    warnings: list[str] | None = None,
) -> tuple[dict[int, list[str]], pd.DataFrame, list[dict[str, Any]]]:
    display_map, option_map = display_option_maps(displays_with_options)
    family_map = {
        cell_text(row.get("family_id")): row
        for _index, row in candidate_families.iterrows()
        if cell_text(row.get("family_id"))
    }
    expanded_by_position: dict[int, list[str]] = {}
    sheet_rows: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []

    for item in final_items:
        display = display_map.get(item.display_id)
        option = option_map.get((item.display_id, item.practice_option_id))
        representative = family_map.get(item.representative_family_id)
        if display is None or option is None or representative is None:
            raise ValueError(
                f"价格证据扩充 display/option/family 回查失败: item_position={item.item_position}"
            )
        original_family_ids, target_option, candidates = option_evidence_expansion_candidates(
            item, option, candidate_families
        )
        candidate_ids = [candidate["family_id"] for candidate in candidates]
        prompt = build_option_evidence_expansion_prompt(target_option, candidates) if candidates else ""
        response = None
        accepted_family_ids: list[str] = []
        error_message = ""
        max_tokens = 512 if candidates else 0
        try:
            if candidates:
                response = request_llm_json_with_usage(
                    prompt,
                    max_tokens=max_tokens,
                    system_prompt="你只输出一个 JSON object，不输出解释、Markdown 或思考过程。",
                )
                accepted_family_ids = validate_option_evidence_expansion_result(
                    response.content, candidate_ids
                )
        except (LLMServiceError, RuntimeError, ValueError, TypeError, KeyError) as exc:
            error_message = str(exc)
            accepted_family_ids = []
            append_warning(
                warnings,
                f"option_evidence_expansion_failed:item_position={item.item_position}",
            )

        expanded_family_ids = list(dict.fromkeys([
            *original_family_ids,
            *accepted_family_ids,
        ]))
        expanded_by_position[item.item_position] = expanded_family_ids
        original_option = {**option, "family_ids": original_family_ids}
        expanded_option = {**option, "family_ids": expanded_family_ids}
        original_evidence = expand_samples_for_option(
            original_option, display, candidate_families, samples
        )
        expanded_evidence = expand_samples_for_option(
            expanded_option, display, candidate_families, samples
        )
        sheet_rows.append({
            "final_item_position": item.item_position,
            "清单名称": cell_text(representative.get("representative_cost_item_name")),
            "practice_option_id": item.practice_option_id,
            "原family数": len(original_family_ids),
            "候选family数": len(candidates),
            "新增family数": len(accepted_family_ids),
            "扩展后family数": len(expanded_family_ids),
            "原价格证据样本数": len(original_evidence),
            "扩展后价格证据样本数": len(expanded_evidence),
            "新增family_ids": ",".join(accepted_family_ids),
        })
        raw_response = ""
        if response is not None:
            raw_response = cell_text(getattr(response, "raw_content", ""))
            if not raw_response:
                raw_response = json_text(response.content)
        trace = trace_row(
            "option_evidence_expansion",
            "判断可加入最终 option 价格证据的等价 family",
            not error_message,
            error=error_message,
            prompt=prompt,
            max_tokens=max_tokens,
            input_summary=(
                f"item_position={item.item_position}; "
                f"original_families={len(original_family_ids)}; "
                f"candidates={len(candidates)}; accepted={len(accepted_family_ids)}"
            ),
            usage=response.usage if response is not None else None,
            raw_response=raw_response,
            scenario_count=1,
            scenario_item_count=1,
        )
        trace["fallback"] = bool(error_message)
        traces.append(trace)

    return (
        expanded_by_position,
        pd.DataFrame(sheet_rows, columns=OPTION_EVIDENCE_EXPANSION_COLUMNS).fillna(""),
        traces,
    )


def expand_samples_for_option(
    option: dict[str, Any],
    display: pd.Series,
    candidate_families: pd.DataFrame,
    samples: pd.DataFrame,
) -> pd.DataFrame:
    option_id = cell_text(option.get("practice_option_id")) or cell_text(option.get("option_id"))
    family_ids = [cell_text(value) for value in option.get("family_ids", []) if cell_text(value)]
    if not family_ids:
        raise ValueError(f"Option 缺少 family_ids: option_id={option_id}")
    if "normalized_signature" not in samples.columns:
        raise ValueError(f"samples 缺少 normalized_signature: option_id={option_id}")

    candidate_family_ids = candidate_families.get("family_id", pd.Series(dtype=object)).map(cell_text)
    family_signatures: list[tuple[str, str]] = []
    expected_units_by_signature: dict[str, set[str]] = {}
    for family_id in family_ids:
        family_rows = candidate_families[candidate_family_ids.eq(family_id)]
        if len(family_rows) != 1:
            raise ValueError(
                f"Family 无法唯一映射 normalized_signature: option_id={option_id}, "
                f"family_id={family_id}, matched_rows={len(family_rows)}"
            )
        family = family_rows.iloc[0]
        signature = cell_text(family.get("normalized_signature"))
        if not signature:
            raise ValueError(
                f"Family normalized_signature 为空: option_id={option_id}, family_id={family_id}"
            )
        family_signatures.append((family_id, signature))
        expected_units_by_signature.setdefault(signature, set()).add(
            normalized_unit(cell_text(family.get("unit_normalized")) or family.get("unit"))
        )

    selected_signatures = list(dict.fromkeys(signature for _family_id, signature in family_signatures))
    sample_signatures = samples["normalized_signature"].map(cell_text)
    expanded = samples[sample_signatures.isin(selected_signatures)].copy()
    for family_id, signature in family_signatures:
        if not sample_signatures.eq(signature).any():
            raise ValueError(
                f"Family normalized_signature 在全量 samples 中无匹配: option_id={option_id}, "
                f"family_id={family_id}, normalized_signature={signature}"
            )

    display_unit = normalized_unit(display.get("unit"))
    for row_index, row in expanded.iterrows():
        signature = cell_text(row.get("normalized_signature"))
        sample_unit = normalized_unit(cell_text(row.get("unit_normalized")) or row.get("unit"))
        family_units = expected_units_by_signature.get(signature, set())
        if not sample_unit or "" in family_units or sample_unit not in family_units or sample_unit != display_unit:
            raise ValueError(
                f"全库价格证据单位不兼容: option_id={option_id}, "
                f"stable_sample_id={cell_text(row.get('stable_sample_id'))}, "
                f"normalized_signature={signature}, sample_unit={sample_unit}, "
                f"family_units={join_non_empty(sorted(family_units))}, display_unit={display_unit}, "
                f"row_index={row_index}"
            )

    source_refs: list[str] = []
    for _row_index, row in expanded.iterrows():
        source_refs.append(cell_text(row.get("source_ref")) or source_identity_for_row(row)[2])
    expanded["source_ref"] = source_refs

    stable_ids = expanded.get("stable_sample_id", pd.Series("", index=expanded.index)).map(cell_text)
    dedupe_keys = [
        f"stable:{stable_sample_id}" if stable_sample_id else f"source:{source_ref}"
        for stable_sample_id, source_ref in zip(stable_ids.tolist(), source_refs)
    ]
    expanded = expanded.loc[~pd.Series(dedupe_keys, index=expanded.index).duplicated(keep="first")].copy()

    signature_family = {}
    for family_id, signature in family_signatures:
        signature_family.setdefault(signature, family_id)
    expanded["family_id"] = expanded["normalized_signature"].map(
        lambda value: signature_family[cell_text(value)]
    )
    return expanded


def build_quantity_statistics(
    retrieved_samples: pd.DataFrame,
    target_unit: str,
) -> dict[str, Any]:
    normalized_target_unit = normalized_unit(target_unit)
    valid_quantities: list[float] = []
    seen_stable_ids: set[str] = set()
    for row_index, row in retrieved_samples.iterrows():
        stable_sample_id = cell_text(row.get("stable_sample_id"))
        if stable_sample_id:
            if stable_sample_id in seen_stable_ids:
                continue
            seen_stable_ids.add(stable_sample_id)
        sample_unit = normalized_unit(cell_text(row.get("unit_normalized")) or row.get("unit"))
        if not normalized_target_unit or sample_unit != normalized_target_unit:
            continue
        quantity = numeric_or_none(row.get("quantity"))
        if quantity is None or quantity <= 0:
            continue
        valid_quantities.append(float(quantity))
    if not valid_quantities:
        return {
            "sample_count": 0,
            "minimum": None,
            "median": None,
            "maximum": None,
            "fallback_used": False,
        }
    series = pd.Series(valid_quantities, dtype=float)
    return {
        "sample_count": int(len(series)),
        "minimum": float(series.min()),
        "median": float(series.median()),
        "maximum": float(series.max()),
        "fallback_used": False,
    }


def price_stats_for_option(
    option: dict[str, Any],
    display: pd.Series,
    candidate_families: pd.DataFrame,
    samples: pd.DataFrame,
) -> dict[str, Any]:
    expanded = expand_samples_for_option(option, display, candidate_families, samples)

    unit_price = p10_median_p90(expanded, "unit_price")
    labor_price = p10_median_p90(expanded, "labor_unit_price")
    machinery_price = p10_median_p90(expanded, "machinery_unit_price")
    return {
        "unit_price_p10": unit_price[0],
        "unit_price_median": unit_price[1],
        "unit_price_p90": unit_price[2],
        "labor_unit_price_p10": labor_price[0],
        "labor_unit_price_median": labor_price[1],
        "labor_unit_price_p90": labor_price[2],
        "machinery_unit_price_p10": machinery_price[0],
        "machinery_unit_price_median": machinery_price[1],
        "machinery_unit_price_p90": machinery_price[2],
        "source_refs": ordered_refs(expanded["source_ref"]),
        "evidence_count": int(len(expanded)),
        "expanded_evidence": expanded,
    }


def validate_price_stats(price_stats: dict[str, Any], stable_sample_id: str, practice_option_id: str) -> None:
    required = ["unit_price_p10", "unit_price_median", "unit_price_p90"]
    missing = [key for key in required if numeric_or_none(price_stats.get(key)) is None]
    if missing or int(numeric_or_none(price_stats.get("evidence_count")) or 0) <= 0:
        raise ValueError(
            f"价格回查失败: stable_sample_id={stable_sample_id}, "
            f"practice_option_id={practice_option_id}, missing={missing}"
        )


def quantity_display(quantity: dict[str, Any]) -> Any:
    return validate_quantity(quantity)["value"]


def quantity_values(quantity: dict[str, Any]) -> tuple[Any, Any, Any]:
    value = validate_quantity(quantity)["value"]
    return value, value, value


def quantity_amounts(quantity: dict[str, Any], price_stats: dict[str, Any]) -> tuple[Any, Any, Any]:
    value = validate_quantity(quantity)["value"]
    return (
        calc_amount(value, price_stats.get("unit_price_p10")),
        calc_amount(value, price_stats.get("unit_price_median")),
        calc_amount(value, price_stats.get("unit_price_p90")),
    )


def build_scenario_outputs(
    scenarios: list[EstimateScenario],
    displays_with_options: pd.DataFrame,
    candidate_families: pd.DataFrame,
    samples: pd.DataFrame,
    expanded_family_ids_by_position: dict[int, list[str]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    display_map, option_map = display_option_maps(displays_with_options)
    family_map = {cell_text(row.get("family_id")): row for _index, row in candidate_families.iterrows()}
    scenario_rows: list[dict[str, Any]] = []
    price_evidence_rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        for item in scenario.items:
            display_row = display_map.get(item.display_id)
            option = option_map.get((item.display_id, item.practice_option_id))
            if display_row is None or option is None:
                raise ValueError(f"最终清单 display/option 回查失败: {item.stable_sample_id}")
            representative = family_map.get(item.representative_family_id)
            if representative is None:
                raise ValueError(f"代表 Family 回查失败: {item.representative_family_id}")
            representative_unit = cell_text(representative.get("unit"))
            representative_unit_normalized = cell_text(representative.get("unit_normalized"))
            if not representative_unit and not representative_unit_normalized:
                continue
            family_ids = [cell_text(value) for value in option.get("family_ids", []) if cell_text(value)]
            if expanded_family_ids_by_position is not None:
                family_ids = list(expanded_family_ids_by_position.get(item.item_position, family_ids))
            price_option = {**option, "family_ids": family_ids}
            price_stats = price_stats_for_option(
                price_option, display_row, candidate_families, samples
            )
            validate_price_stats(price_stats, item.stable_sample_id, item.practice_option_id)
            amount_p10, amount_mid, amount_p90 = quantity_amounts(item.quantity, price_stats)
            quantity_value = quantity_display(item.quantity)
            expanded_evidence = price_stats["expanded_evidence"]
            for _evidence_index, evidence in expanded_evidence.iterrows():
                price_evidence_rows.append({
                    "final_item_position": item.item_position,
                    "清单名称": cell_text(representative.get("representative_cost_item_name")),
                    "display_id": item.display_id,
                    "practice_option_id": item.practice_option_id,
                    "family_id": cell_text(evidence.get("family_id")),
                    "normalized_signature": cell_text(evidence.get("normalized_signature")),
                    "stable_sample_id": cell_text(evidence.get("stable_sample_id")),
                    "project_key": cell_text(evidence.get("project_key")),
                    "source_ref": cell_text(evidence.get("source_ref")),
                    "工程名称": cell_text(evidence.get("工程名称")) or cell_text(evidence.get("来源工程名称")) or cell_text(evidence.get("project_name_text")),
                    "location": cell_text(evidence.get("location")),
                    "consultation_time": cell_text(evidence.get("consultation_time")),
                    "cost_item_name": cell_text(evidence.get("cost_item_name")),
                    "project_description": cell_text(evidence.get("project_description")),
                    "unit": cell_text(evidence.get("unit")),
                    "quantity": evidence.get("quantity", ""),
                    "unit_price": evidence.get("unit_price", ""),
                    "labor_unit_price": evidence.get("labor_unit_price", ""),
                    "machinery_unit_price": evidence.get("machinery_unit_price", ""),
                })
            scenario_rows.append(
                {
                    "清单名称": cell_text(representative.get("representative_cost_item_name")),
                    "项目特征": cell_text(representative.get("representative_project_description")),
                    "单位": cell_text(representative.get("unit_normalized")) or cell_text(representative.get("unit")),
                    "工程量": quantity_value,
                    "工程量来源": {
                        "user_explicit": "用户明确工程量",
                        "historical_median": "全库历史样本中位数",
                    }.get(item.quantity_source, item.quantity_source),
                    "工程量说明": item.quantity_reason,
                    "工程量样本数": item.quantity_sample_count,
                    "工程量最低值": item.quantity_minimum,
                    "工程量中位数": item.quantity_median,
                    "工程量最高值": item.quantity_maximum,
                    "综合单价": price_stats.get("unit_price_median"),
                    "暂估合价": amount_mid,
                    "合价P10": amount_p10,
                    "合价中位数": amount_mid,
                    "合价P90": amount_p90,
                    "综合单价P10": price_stats.get("unit_price_p10"),
                    "综合单价中位数": price_stats.get("unit_price_median"),
                    "综合单价P90": price_stats.get("unit_price_p90"),
                    "其中包含人工费单价P10": price_stats.get("labor_unit_price_p10"),
                    "其中包含人工费单价中位数": price_stats.get("labor_unit_price_median"),
                    "其中包含人工费单价P90": price_stats.get("labor_unit_price_p90"),
                    "其中包含机械费单价P10": price_stats.get("machinery_unit_price_p10"),
                    "其中包含机械费单价中位数": price_stats.get("machinery_unit_price_median"),
                    "其中包含机械费单价P90": price_stats.get("machinery_unit_price_p90"),
                    "价格证据样本数": price_stats.get("evidence_count"),
                    "来源样本": cell_text(price_stats.get("source_refs")),
                    "practice_option_id": item.practice_option_id,
                    "original_option_id": item.original_option_id,
                    "original_family_id": item.original_family_id,
                    "representative_family_id": item.representative_family_id,
                    "价格证据family": ",".join(family_ids),
                    "display_id": item.display_id,
                }
            )
    return (
        pd.DataFrame(scenario_rows, columns=ESTIMATE_SCENARIO_COLUMNS).fillna(""),
        pd.DataFrame(price_evidence_rows, columns=PRICE_EVIDENCE_ITEM_COLUMNS).fillna(""),
    )


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
    "方案说明",
    "主要施工内容",
    "待现场确认事项",
    "工程量来源",
    "工程量说明",
    "来源样本",
    "价格证据family",
}

INTEGER_COLUMNS = {
    "序号",
    "final_item_position",
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
    "合价P10",
    "合价中位数",
    "合价P90",
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
    description = short_description(row.get("项目特征"))
    if name and description:
        return f"{name}（{description}）"
    return name or description


def build_estimate_summary(
    scenarios: list[EstimateScenario],
    estimate_scenarios: pd.DataFrame,
) -> pd.DataFrame:
    if estimate_scenarios.empty:
        return pd.DataFrame(columns=ESTIMATE_SUMMARY_COLUMNS)
    scenario = scenarios[0] if scenarios else None
    row = {
        "方案名称": scenario.scenario_name if scenario is not None else "",
        "方案说明": scenario.scenario_summary if scenario is not None else "",
        "主要施工内容": join_non_empty([
            display_item_label(item) for _index, item in estimate_scenarios.iterrows()
        ]),
        "计价项目数": int(len(estimate_scenarios)),
        "合价P10": amount_sum(estimate_scenarios, "合价P10"),
        "合价中位数": amount_sum(estimate_scenarios, "合价中位数"),
        "合价P90": amount_sum(estimate_scenarios, "合价P90"),
        "待现场确认事项": "",
    }
    return pd.DataFrame([row], columns=ESTIMATE_SUMMARY_COLUMNS).fillna("")


def quantity_determination_status(quantity_trace: dict[str, Any]) -> str:
    if bool(quantity_trace.get("fallback")):
        return "fallback"
    if cell_text(quantity_trace.get("error_message")):
        return "failed"
    return "success"


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
    project_packages_before_constraint: int,
    project_packages_after_constraint: int,
    samples_before_constraint: int,
    samples_after_constraint: int,
    invalid_sample_consultation_time_count: int,
    invalid_project_package_consultation_time_count: int,
    retrieved_evidence_item_row_count: int,
    evidence_item_row_count: int,
    candidate_family_count: int,
    candidate_display_group_count: int,
    matched_project_example_count: int,
    matched_project_example_item_count: int,
    display_option_grouping_display_count: int,
    display_option_grouping_trace: dict[str, Any],
    display_option_grouping_fallback: bool,
    display_option_grouping_error: str,
    display_option_grouping_meta: dict[str, Any],
    selected_project_package_id: str,
    selected_item_count: int,
    selected_exact_quantity_count: int,
    selected_range_quantity_count: int,
    range_selection_trace: dict[str, Any],
    range_selection_error: str,
    quantity_trace: dict[str, Any],
    final_explanation_trace: dict[str, Any],
    final_explanation_error: str,
    output_path: Path | None,
    started_at: datetime,
    index_dir: Path,
    include_debug_text: bool,
    display_option_grouping_prompt: str,
    range_selection_prompt: str,
    quantity_prompt: str,
    final_explanation_prompt: str,
    with_explanations: bool,
    warnings: list[str] | None = None,
) -> pd.DataFrame:
    rows = [
        ("原始用户需求", rewrite.raw_query),
        ("project_package_query_text", rewrite.project_package_query_text),
        ("item_query_text", rewrite.item_query_text),
        ("query_location", rewrite.location),
        ("query_start_date", rewrite.start_date),
        ("query_end_date", rewrite.end_date),
        ("query_constraint_notes", "；".join(rewrite.notes)),
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
        ("project_packages_before_constraint", project_packages_before_constraint),
        ("project_packages_after_constraint", project_packages_after_constraint),
        ("samples_before_constraint", samples_before_constraint),
        ("samples_after_constraint", samples_after_constraint),
        ("invalid_sample_consultation_time_count", invalid_sample_consultation_time_count),
        ("invalid_project_package_consultation_time_count", invalid_project_package_consultation_time_count),
        ("LLM query rewrite 是否成功", "是" if rewrite.success else "否"),
        ("retrieved_evidence_item_row_count", retrieved_evidence_item_row_count),
        ("evidence_item_row_count", evidence_item_row_count),
        ("candidate_family_count", candidate_family_count),
        ("candidate_display_group_count", candidate_display_group_count),
        ("matched_project_example_count", matched_project_example_count),
        ("matched_project_example_item_count", matched_project_example_item_count),
        ("display_option_grouping_display_count", display_option_grouping_display_count),
        ("display_option_grouping_display_ids", json_text(display_option_grouping_meta.get("display_ids") or [])),
        ("display_option_grouping_llm_display_count", display_option_grouping_meta.get("llm_display_count", "")),
        ("display_option_grouping_programmatic_single_family_display_count", display_option_grouping_meta.get("programmatic_single_family_display_count", "")),
        ("display_option_grouping_practice_option_count", display_option_grouping_meta.get("practice_option_count", "")),
        ("display_option_grouping_families_grouped_count", display_option_grouping_meta.get("families_grouped_count", "")),
        ("display_option_grouping_option_count_by_display", json_text(display_option_grouping_meta.get("option_count_by_display") or {})),
        ("display_option_grouping_max_options_per_display", display_option_grouping_meta.get("max_options_per_display", "")),
        ("display_option_grouping_prompt_chars", display_option_grouping_trace.get("prompt_chars", "")),
        ("display_option_grouping_prompt_tokens", display_option_grouping_trace.get("prompt_tokens") or display_option_grouping_trace.get("estimated_tokens", "")),
        ("display_option_grouping_completion_tokens", display_option_grouping_trace.get("completion_tokens", "")),
        ("scenario_count", 1 if selected_item_count else 0),
        ("scenario_item_count", selected_item_count),
        ("scenario_exact_quantity_count", selected_exact_quantity_count),
        ("scenario_range_quantity_count", selected_range_quantity_count),
        ("selected_project_package_id", selected_project_package_id),
        ("range_selection_status", "fallback_full_project" if range_selection_error else "ok"),
        ("range_selection_prompt_chars", range_selection_trace.get("prompt_chars", "")),
        ("range_selection_prompt_tokens", range_selection_trace.get("prompt_tokens") or range_selection_trace.get("estimated_tokens", "")),
        ("range_selection_completion_tokens", range_selection_trace.get("completion_tokens", "")),
        (
            "quantity_determination_status",
            quantity_determination_status(quantity_trace),
        ),
        ("quantity_determination_prompt_chars", quantity_trace.get("prompt_chars", "")),
        ("quantity_determination_prompt_tokens", quantity_trace.get("prompt_tokens") or quantity_trace.get("estimated_tokens", "")),
        ("quantity_determination_completion_tokens", quantity_trace.get("completion_tokens", "")),
        ("with_explanations", with_explanations),
        ("final_explanation_status", "skipped" if not with_explanations else ("failed" if final_explanation_error else "success")),
        ("final_explanation_prompt_chars", final_explanation_trace.get("prompt_chars", "")),
        ("final_explanation_prompt_tokens", final_explanation_trace.get("prompt_tokens") or final_explanation_trace.get("estimated_tokens", "")),
        ("final_explanation_completion_tokens", final_explanation_trace.get("completion_tokens", "")),
        ("是否 display_option_grouping fallback", "是" if display_option_grouping_fallback else "否"),
        ("display_option_grouping LLM error", display_option_grouping_error),
        ("range_selection LLM error", range_selection_error),
        ("final_explanation LLM error", final_explanation_error),
        ("output_path", str(output_path or "")),
        ("运行时间", f"{(datetime.now() - started_at).total_seconds():.2f}s"),
        ("index_dir", str(index_dir)),
        ("主要文件路径", json_text((meta.get("files") or {}))),
        ("rewrite_notes", "；".join(rewrite.notes)),
        ("warnings", "；".join(warnings or [])),
    ]
    if include_debug_text:
        rows.append(("display_option_grouping_prompt_preview", display_option_grouping_prompt[:3000]))
        rows.append(("range_selection_prompt_preview", range_selection_prompt[:3000]))
        rows.append(("quantity_determination_prompt_preview", quantity_prompt[:3000]))
        rows.append(("final_explanation_prompt_preview", final_explanation_prompt[:3000]))
    return pd.DataFrame(rows, columns=["字段", "值"])


def write_query_result_workbook(output_path: Path, result: QueryResult, display: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        display_frame(result.estimate_summary, display).to_excel(writer, sheet_name="estimate_summary", index=False)
        display_frame(result.estimate_scenarios, display).to_excel(writer, sheet_name="estimate_scenarios", index=False)
        display_frame(result.option_evidence_expansion, display).to_excel(
            writer, sheet_name="option_evidence_expansion", index=False
        )
        display_frame(result.price_evidence_items, display).to_excel(
            writer, sheet_name="price_evidence_items", index=False
        )
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
        display_frame(result.display_option_grouping_trace, display).to_excel(
            writer,
            sheet_name="display_option_grouping_trace",
            index=False,
        )
        display_frame(result.option_selection_trace, display).to_excel(
            writer, sheet_name="option_selection_trace", index=False,
        )
        display_frame(result.matched_project_packages, display).to_excel(
            writer,
            sheet_name="matched_project_packages",
            index=False,
        )
        display_frame(result.matched_project_examples, display).to_excel(
            writer,
            sheet_name="matched_project_examples",
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
    package_weight_temperature: float = DEFAULT_PACKAGE_WEIGHT_TEMPERATURE,
    include_debug_text: bool = False,
    display: bool = False,
    with_explanations: bool = False,
) -> QueryResult:
    if package_weight_temperature <= 0:
        raise ValueError("package weight temperature 必须大于 0")
    started_at = datetime.now()
    warnings: list[str] = []
    samples, project_packages, project_package_embeddings, item_embeddings, meta = load_index(index_dir)
    rewrite, rewrite_trace = query_rewrite_for_embedding(raw_text)

    package_mask = build_constraint_mask(
        project_packages, rewrite.location, rewrite.start_date, rewrite.end_date
    )
    item_mask = build_constraint_mask(
        samples, rewrite.location, rewrite.start_date, rewrite.end_date
    )
    candidate_project_packages, candidate_project_package_embeddings = filter_rows_and_embeddings(
        project_packages, project_package_embeddings, package_mask, "工程包"
    )
    candidate_samples, candidate_item_embeddings = filter_rows_and_embeddings(
        samples, item_embeddings, item_mask, "清单样本"
    )
    ensure_project_package_candidates(candidate_project_packages, rewrite)

    model = load_embedding_model(str(meta.get("model") or "BAAI/bge-m3"))
    try:
        package_query_embedding = encode_query(model, rewrite.project_package_query_text)
        item_query_embedding = encode_query(model, rewrite.item_query_text)
    except Exception:
        release_embedding_model(model)
        del model
        gc.collect()
        raise

    if package_query_embedding.shape[0] != candidate_project_package_embeddings.shape[1]:
        raise ValueError("package query embedding 维度与索引 embedding 维度不一致")
    if item_query_embedding.shape[0] != candidate_item_embeddings.shape[1]:
        raise ValueError("item query embedding 维度与索引 embedding 维度不一致")

    package_query_similarities = candidate_project_package_embeddings @ package_query_embedding
    package_query_similarity_by_id = project_package_similarity_map(candidate_project_packages, package_query_similarities)
    matched_raw = score_project_packages(
        candidate_project_packages,
        candidate_project_package_embeddings,
        package_query_embedding,
        top_packages,
        max_packages_per_cache_subject=max_packages_per_cache_subject,
    )
    candidate_item_query_similarities = candidate_item_embeddings @ item_query_embedding
    direct_item_hits = score_direct_items(candidate_samples, candidate_item_query_similarities, top_items)
    item_query_similarities = np.zeros(len(samples), dtype=np.float32)
    constrained_sample_indices = pd.to_numeric(candidate_samples["sample_index"], errors="raise").astype(int).to_numpy()
    item_query_similarities[constrained_sample_indices] = candidate_item_query_similarities
    if direct_item_hits.empty:
        append_warning(warnings, "direct_item_hits_empty_after_constraints")
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
    release_embedding_model(model)
    del model
    gc.collect()

    selected_package, ranked_packages = select_representative_project_package(matched_raw)
    matched_project_packages = matched_project_packages_for_output(ranked_packages)
    selected_project_package_id = cell_text(selected_package.get("project_package_id"))
    selected_project_name = cell_text(selected_package.get("工程名称")) or cell_text(
        selected_package.get("project_name_text")
    )
    matched_project_examples = build_matched_project_examples(
        matched_project_packages, samples, limit=5
    )
    matched_project_examples_output = matched_project_examples_frame(matched_project_examples)
    selected_items = attach_family_and_display_ids_to_selected_items(
        expand_selected_project_items(samples, selected_package),
        evidence_items,
        display_group_families,
    )
    start, end, range_meta = select_contiguous_item_range(
        raw_text, selected_project_name, selected_items
    )
    if range_meta["fallback"]:
        append_warning(warnings, "range_selection_fallback_full_project")
    plan_items = selected_items.iloc[start : end + 1].copy()
    required_display_ids, candidate_display_groups, display_group_families = filter_required_display_groups(
        plan_items, candidate_display_groups, display_group_families
    )
    (
        displays_with_options,
        _display_option_grouping_success,
        display_option_grouping_fallback,
        display_option_grouping_error,
        display_option_grouping_prompt,
        display_option_grouping_trace,
        display_option_grouping_meta,
        display_option_grouping_trace_frame,
        display_option_grouping_llm_traces,
    ) = generate_display_option_grouping(
        candidate_display_groups,
        display_group_families,
        candidate_families,
        warnings=warnings,
    )
    displays_with_options = attach_option_support_counts(displays_with_options, evidence_items)
    display_option_grouping_trace_frame = build_display_option_grouping_trace_frame(
        displays_with_options, display_group_families, candidate_families
    )
    plan_items = attach_original_practice_options(plan_items, displays_with_options)
    required_family_ids = set(display_group_families["family_id"].map(cell_text).tolist())
    lookup_evidence_items = evidence_items[
        evidence_items["family_id"].map(cell_text).isin(required_family_ids)
    ].copy()
    sample_lookup = build_stable_sample_lookup(
        samples, lookup_evidence_items, display_group_families, displays_with_options
    )
    range_selection_trace = trace_row(
        "range_selection",
        "在确定性选中的完整历史工程内选择连续清单区间",
        not range_meta["fallback"],
        error=range_meta["error_message"],
        prompt=range_meta["prompt"],
        max_tokens=256,
        input_summary=json_text({
            "selected_project_package_id": selected_project_package_id,
            "project_item_count": len(selected_items),
            "start_item_position": start,
            "end_item_position": end,
            "selected_item_count": len(plan_items),
            "fallback": range_meta["fallback"],
        }),
        usage=range_meta["usage"],
        raw_response=range_meta["raw_response"],
        scenario_count=1,
        scenario_item_count=len(plan_items),
    )
    plan_items, option_selection_trace_frame, option_selection_llm_traces = select_final_options(
        raw_text, plan_items, sample_lookup, displays_with_options, candidate_families, warnings, evidence_items
    )
    display_option_grouping_trace_frame = apply_option_selection_to_grouping_trace(
        display_option_grouping_trace_frame, option_selection_trace_frame
    )
    scenario, quantity_prompt, quantity_trace = generate_quantity_determination(
        raw_text,
        selected_project_package_id,
        plan_items,
        sample_lookup,
        displays_with_options,
        candidate_families,
        candidate_samples,
        warnings,
    )
    scenarios = [scenario]
    (
        expanded_family_ids_by_position,
        option_evidence_expansion,
        option_evidence_expansion_traces,
    ) = expand_option_price_evidence_families(
        scenario.items,
        displays_with_options,
        candidate_families,
        candidate_samples,
        warnings,
    )
    estimate_scenarios, _price_evidence_items = build_scenario_outputs(
        scenarios,
        displays_with_options,
        candidate_families,
        candidate_samples,
        expanded_family_ids_by_position,
    )
    (
        scenario,
        final_explanation_success,
        final_explanation_error,
        final_explanation_prompt,
        final_explanation_trace,
    ) = generate_optional_final_explanation(
        with_explanations,
        raw_text,
        scenario,
        estimate_scenarios,
        matched_project_examples,
        warnings=warnings,
    )
    scenarios = [scenario]
    estimate_scenarios, price_evidence_items = build_scenario_outputs(
        scenarios,
        displays_with_options,
        candidate_families,
        candidate_samples,
        expanded_family_ids_by_position,
    )
    estimate_summary = build_estimate_summary(scenarios, estimate_scenarios)
    if warnings:
        append_trace_warnings(display_option_grouping_trace, warnings)
        append_trace_warnings(range_selection_trace, warnings)
        append_trace_warnings(quantity_trace, warnings)
        append_trace_warnings(final_explanation_trace, warnings)
    scenario_item_count = sum(len(scenario.items) for scenario in scenarios)
    scenario_exact_quantity_count = sum(1 for scenario in scenarios for item in scenario.items if cell_text(item.quantity.get("type")) == "exact")
    scenario_range_quantity_count = 0
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
        project_packages_before_constraint=len(project_packages),
        project_packages_after_constraint=len(candidate_project_packages),
        samples_before_constraint=len(samples),
        samples_after_constraint=len(candidate_samples),
        invalid_sample_consultation_time_count=int(parse_consultation_dates(samples["consultation_time"]).isna().sum()),
        invalid_project_package_consultation_time_count=int(parse_consultation_dates(project_packages["consultation_time"]).isna().sum()),
        retrieved_evidence_item_row_count=len(retrieved_evidence_items),
        evidence_item_row_count=len(evidence_items),
        candidate_family_count=len(candidate_families),
        candidate_display_group_count=len(candidate_display_groups),
        matched_project_example_count=len(matched_project_examples),
        matched_project_example_item_count=len(matched_project_examples_output),
        display_option_grouping_display_count=len(displays_with_options),
        display_option_grouping_trace=display_option_grouping_trace,
        display_option_grouping_fallback=display_option_grouping_fallback,
        display_option_grouping_error=display_option_grouping_error,
        display_option_grouping_meta=display_option_grouping_meta,
        selected_project_package_id=selected_project_package_id,
        selected_item_count=scenario_item_count,
        selected_exact_quantity_count=scenario_exact_quantity_count,
        selected_range_quantity_count=scenario_range_quantity_count,
        range_selection_trace=range_selection_trace,
        range_selection_error=range_meta["error_message"],
        quantity_trace=quantity_trace,
        final_explanation_trace=final_explanation_trace,
        final_explanation_error=final_explanation_error,
        output_path=output,
        started_at=started_at,
        index_dir=index_dir,
        include_debug_text=include_debug_text,
        display_option_grouping_prompt=display_option_grouping_prompt,
        range_selection_prompt=range_meta["prompt"],
        quantity_prompt=quantity_prompt,
        final_explanation_prompt=final_explanation_prompt,
        with_explanations=with_explanations,
        warnings=warnings,
    )
    llm_trace = pd.DataFrame(
        [
            rewrite_trace,
            *display_option_grouping_llm_traces,
            range_selection_trace,
            *option_selection_llm_traces,
            quantity_trace,
            *option_evidence_expansion_traces,
            final_explanation_trace,
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
        display_option_grouping_trace=display_option_grouping_trace_frame,
        option_selection_trace=option_selection_trace_frame,
        matched_project_examples=matched_project_examples_output,
        evidence_items=evidence_items,
        option_evidence_expansion=option_evidence_expansion,
        price_evidence_items=price_evidence_items,
        parse_info=parse_info,
        llm_trace=llm_trace,
        success=final_explanation_success,
        error_message=final_explanation_error,
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
    print(f"[DONE] matched project example items: {len(result.matched_project_examples)}")
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
            package_weight_temperature=args.package_weight_temperature,
            include_debug_text=args.include_debug_text,
            display=args.display,
            with_explanations=args.with_explanations,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    print_terminal_summary(result, output_path)
    if not getattr(result, "success", True):
        print(f"[ERROR] final_explanation 失败，已输出无说明估价: {getattr(result, 'error_message', '')}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
