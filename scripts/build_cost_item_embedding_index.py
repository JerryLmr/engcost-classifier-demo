#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CORE_REQUIRED_COLUMNS = [
    "工程名称",
    "project_name_text",
    "cost_item_name",
    "project_description",
]

OPTIONAL_COLUMNS = [
    "stable_sample_id",
    "batch_id",
    "source_row_id",
    "item_row_id",
    "consultation_time",
    "location",
    "cache_subject",
    "catalog_id",
    "一级分类",
    "二级分类",
    "维修状态",
    "标准对象",
    "是否复合工程",
    "复合目录",
    "seq",
    "unit",
    "unit_normalized",
    "quantity",
    "unit_price",
    "total_price",
    "labor_unit_price",
    "machinery_unit_price",
]

NUMERIC_COLUMNS = [
    "quantity",
    "unit_price",
    "total_price",
    "labor_unit_price",
    "machinery_unit_price",
]

NEW_OUTPUT_FILES = [
    "samples.parquet",
    "project_packages.parquet",
    "project_package_embeddings.npy",
    "item_embeddings.npy",
    "index_meta.json",
]

LEGACY_OUTPUT_FILES = [
    "project_groups.parquet",
    "project_name_embeddings.npy",
    "project_detail_embeddings.npy",
    "item_text_embeddings.npy",
    "project_group_embeddings.npy",
]

MANAGED_OUTPUT_FILES = [*NEW_OUTPUT_FILES, *LEGACY_OUTPUT_FILES]

PROJECT_PACKAGE_COLUMNS = [
    "project_package_id",
    "project_package_title",
    "project_key",
    "batch_id",
    "source_row_id",
    "工程名称",
    "project_name_text",
    "consultation_time",
    "location",
    "cache_subject",
    "item_count",
    "cost_item_names_summary",
    "catalog_id",
    "一级分类",
    "二级分类",
    "维修状态",
    "标准对象",
    "catalog_summary",
    "item_summary",
    "package_text",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建历史工程清单 package/item embedding 索引")
    parser.add_argument(
        "--samples",
        default="samples/cost_item_samples_all.xlsx",
        help="已审定清单总样本 Excel 路径，默认 samples/cost_item_samples_all.xlsx",
    )
    parser.add_argument("--output-dir", default="embeddings", help="索引输出目录，默认 embeddings")
    parser.add_argument("--model", default="BAAI/bge-m3", help="sentence-transformers 模型名")
    parser.add_argument("--batch-size", type=int, default=32, help="embedding 批大小")
    parser.add_argument("--overwrite", action="store_true", help="若索引输出目录已存在则覆盖")
    return parser.parse_args()


def validate_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"输出路径不是目录: {output_dir}")
    existing = [name for name in MANAGED_OUTPUT_FILES if (output_dir / name).exists()]
    if output_dir.exists() and not overwrite:
        existing.append(str(output_dir))
    if existing and not overwrite:
        raise ValueError(
            "输出已存在，请加 --overwrite 或更换输出目录: "
            f"{output_dir} ({', '.join(existing)})"
        )


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def normalize_source_row_id(value: Any) -> str:
    text = safe_text(value)
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def normalize_signature_text(value: Any) -> str:
    text = safe_text(value).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def first_non_empty(values: pd.Series) -> str:
    for value in values.tolist():
        text = safe_text(value)
        if text:
            return text
    return ""


def unique_join(values: list[str], separator: str = "；") -> str:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = safe_text(value)
        if text and text not in seen:
            output.append(text)
            seen.add(text)
    return separator.join(output)


def join_non_empty(parts: list[str], separator: str = " / ") -> str:
    return separator.join(part for part in parts if safe_text(part))


def build_project_key(batch_id: Any, source_row_id: Any) -> str:
    batch_text = safe_text(batch_id)
    source_text = normalize_source_row_id(source_row_id)
    if not batch_text or not source_text:
        raise ValueError("生成 project_key 需要 batch_id 和 source_row_id")
    return f"{batch_text}::{source_text}"


def ensure_project_key(samples: pd.DataFrame) -> pd.DataFrame:
    samples = samples.copy()
    if "project_key" in samples.columns:
        samples["project_key"] = samples["project_key"].map(safe_text)
        missing_mask = samples["project_key"].eq("")
    else:
        missing_mask = pd.Series(True, index=samples.index)
        samples["project_key"] = ""

    if missing_mask.any():
        if "batch_id" not in samples.columns or "source_row_id" not in samples.columns:
            raise ValueError("samples sheet 缺少 project_key，且无法从 batch_id/source_row_id 临时生成")
        samples.loc[missing_mask, "project_key"] = samples.loc[missing_mask].apply(
            lambda row: build_project_key(row.get("batch_id"), row.get("source_row_id")),
            axis=1,
        )
    return samples


def ensure_optional_columns(samples: pd.DataFrame) -> pd.DataFrame:
    samples = samples.copy()
    for column in OPTIONAL_COLUMNS:
        if column not in samples.columns:
            samples[column] = ""
    return samples


def normalize_numeric_columns(samples: pd.DataFrame) -> pd.DataFrame:
    samples = samples.copy()
    for column in NUMERIC_COLUMNS:
        if column in samples.columns:
            samples[column] = pd.to_numeric(samples[column], errors="coerce")
    return samples


def build_item_retrieval_text(row: pd.Series) -> str:
    lines: list[str] = []
    cost_item_name = safe_text(row.get("cost_item_name"))
    project_description = safe_text(row.get("project_description"))
    unit = safe_text(row.get("unit_normalized")) or safe_text(row.get("unit"))

    if cost_item_name:
        lines.append(f"清单项：{cost_item_name}")
    if project_description:
        lines.append(f"项目特征：{project_description}")
    if unit:
        lines.append(f"单位：{unit}")
    return "\n".join(lines)


def build_fine_signature(row: pd.Series) -> str:
    unit = safe_text(row.get("unit_normalized")) or safe_text(row.get("unit"))
    return " | ".join(
        [
            normalize_signature_text(row.get("cost_item_name")),
            normalize_signature_text(row.get("project_description")),
            normalize_signature_text(unit),
        ]
    )


def build_family_signature(row: pd.Series) -> str:
    unit = safe_text(row.get("unit_normalized")) or safe_text(row.get("unit"))
    return " | ".join(
        [
            normalize_signature_text(row.get("cost_item_name")),
            normalize_signature_text(unit),
        ]
    )


def load_samples(samples_path: Path) -> pd.DataFrame:
    if not samples_path.exists():
        raise ValueError(f"样本文件不存在: {samples_path}")
    if not samples_path.is_file():
        raise ValueError(f"样本路径不是文件: {samples_path}")

    try:
        samples = pd.read_excel(samples_path, sheet_name="samples", engine="openpyxl")
    except ValueError as exc:
        raise ValueError("样本 Excel 缺少 samples sheet") from exc

    missing = [column for column in CORE_REQUIRED_COLUMNS if column not in samples.columns]
    if missing:
        raise ValueError(f"samples sheet 缺少必要字段: {', '.join(missing)}")

    samples = normalize_numeric_columns(ensure_optional_columns(ensure_project_key(samples)))
    samples.insert(0, "sample_index", range(len(samples)))
    samples["project_package_id"] = samples["project_key"].map(safe_text)
    samples["item_retrieval_text"] = samples.apply(build_item_retrieval_text, axis=1)
    samples["fine_signature"] = samples.apply(build_fine_signature, axis=1)
    samples["family_signature"] = samples.apply(build_family_signature, axis=1)
    return samples


def catalog_summary_for_group(group: pd.DataFrame) -> str:
    rows: list[str] = []
    for _index, row in group.iterrows():
        catalog_label = join_non_empty(
            [
                safe_text(row.get("一级分类")),
                safe_text(row.get("二级分类")),
                safe_text(row.get("维修状态")),
                safe_text(row.get("标准对象")),
            ]
        )
        catalog_id = safe_text(row.get("catalog_id"))
        if catalog_id and catalog_label:
            rows.append(f"{catalog_id} | {catalog_label}")
        else:
            rows.append(catalog_id or catalog_label)
    return unique_join(rows)


def item_summary_for_group(group: pd.DataFrame) -> str:
    return cost_item_names_summary_for_group(group)


def cost_item_names_summary_for_group(group: pd.DataFrame) -> str:
    return unique_join([safe_text(value) for value in group.get("cost_item_name", pd.Series(dtype=object)).tolist()])


def unique_field_summary(group: pd.DataFrame, column: str) -> str:
    if column not in group.columns:
        return ""
    return unique_join([safe_text(value) for value in group[column].tolist()])


def package_text_for_group(first: pd.Series, cost_item_names_summary: str) -> str:
    lines = [
        f"工程名称：{safe_text(first.get('工程名称'))}",
        f"工程语义：{safe_text(first.get('project_name_text'))}",
        f"包含清单项：{cost_item_names_summary}",
    ]
    return "\n".join(line for line in lines if line.strip())


def build_project_package_title(project_name: str, project_name_text: str) -> str:
    if project_name and project_name_text:
        return f"{project_name} / {project_name_text}"
    return project_name or project_name_text


def build_project_packages(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for project_package_id, group in samples.groupby("project_package_id", sort=False, dropna=False):
        first = group.iloc[0]
        project_name = safe_text(first.get("工程名称"))
        project_name_text = safe_text(first.get("project_name_text"))
        catalog_summary = catalog_summary_for_group(group)
        cost_item_names_summary = cost_item_names_summary_for_group(group)
        item_summary = item_summary_for_group(group)
        rows.append(
            {
                "project_package_id": safe_text(project_package_id),
                "project_package_title": build_project_package_title(project_name, project_name_text),
                "project_key": safe_text(first.get("project_key")),
                "batch_id": safe_text(first.get("batch_id")),
                "source_row_id": normalize_source_row_id(first.get("source_row_id")),
                "工程名称": project_name,
                "project_name_text": project_name_text,
                "consultation_time": safe_text(first.get("consultation_time")),
                "location": safe_text(first.get("location")),
                "cache_subject": safe_text(first.get("cache_subject")),
                "item_count": int(len(group)),
                "cost_item_names_summary": cost_item_names_summary,
                "catalog_id": unique_field_summary(group, "catalog_id"),
                "一级分类": unique_field_summary(group, "一级分类"),
                "二级分类": unique_field_summary(group, "二级分类"),
                "维修状态": unique_field_summary(group, "维修状态"),
                "标准对象": unique_field_summary(group, "标准对象"),
                "catalog_summary": catalog_summary,
                "item_summary": item_summary,
                "package_text": package_text_for_group(first, cost_item_names_summary),
            }
        )
    return pd.DataFrame(rows, columns=PROJECT_PACKAGE_COLUMNS)


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
    return (array / norms).astype(np.float32, copy=False)


def encode_texts(model: Any, texts: list[str], batch_size: int) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return normalize_embeddings(embeddings)


def text_series(values: pd.Series) -> list[str]:
    return values.fillna("").astype(str).tolist()


def build_index_meta(
    samples_path: Path,
    model_name: str,
    sample_count: int,
    package_count: int,
    embedding_dim: int,
) -> dict[str, Any]:
    return {
        "model": model_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_samples": str(samples_path),
        "sample_count": sample_count,
        "package_count": package_count,
        "embedding_dim": embedding_dim,
        "files": {
            "samples": "samples.parquet",
            "project_packages": "project_packages.parquet",
            "project_package_embeddings": "project_package_embeddings.npy",
            "item_embeddings": "item_embeddings.npy",
        },
        "field_descriptions": {
            "sample_index": "samples.parquet 行号，与 item_embeddings.npy 行号一一对应。",
            "project_package_id": "当前阶段固定等于 project_key，用于历史工程包召回和展开。",
            "item_retrieval_text": "cost_item_name、project_description、unit_normalized(or unit) 拼接文本，用于 item embedding。",
            "fine_signature": "cost_item_name + project_description + unit，用于候选项精细聚合。",
            "family_signature": "cost_item_name + unit，用于相似工程包共现统计。",
            "cost_item_names_summary": "同一个 project_package 下 cost_item_name 去重列表。",
            "package_text": "工程名称、project_name_text、cost_item_names_summary 拼接文本，用于 project_package embedding。",
        },
    }


def write_index(
    samples: pd.DataFrame,
    project_packages: pd.DataFrame,
    project_package_embeddings: np.ndarray,
    item_embeddings: np.ndarray,
    output_dir: Path,
    meta: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in MANAGED_OUTPUT_FILES:
        path = output_dir / name
        if path.exists() and path.is_file():
            path.unlink()

    samples = normalize_numeric_columns(samples)
    samples.to_parquet(output_dir / "samples.parquet", index=False)
    project_packages.to_parquet(output_dir / "project_packages.parquet", index=False)
    np.save(output_dir / "project_package_embeddings.npy", project_package_embeddings)
    np.save(output_dir / "item_embeddings.npy", item_embeddings)
    (output_dir / "index_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_cost_item_embedding_index(
    samples_path: Path,
    output_dir: Path,
    model_name: str,
    batch_size: int,
) -> tuple[int, int, int]:
    samples = load_samples(samples_path)
    project_packages = build_project_packages(samples)
    model = load_embedding_model(model_name)

    project_package_embeddings = encode_texts(model, text_series(project_packages["package_text"]), batch_size)
    item_embeddings = encode_texts(model, text_series(samples["item_retrieval_text"]), batch_size)

    if project_package_embeddings.ndim != 2 or item_embeddings.ndim != 2:
        raise ValueError("embedding 必须是二维矩阵")
    if len(project_packages) != project_package_embeddings.shape[0]:
        raise ValueError("工程包数量与 project_package_embeddings 数量不一致")
    if len(samples) != item_embeddings.shape[0]:
        raise ValueError("样本数量与 item_embeddings 数量不一致")
    if project_package_embeddings.shape[1] != item_embeddings.shape[1]:
        raise ValueError("工程包 embedding 与清单项 embedding 维度不一致")

    embedding_dim = int(project_package_embeddings.shape[1])
    meta = build_index_meta(samples_path, model_name, len(samples), len(project_packages), embedding_dim)
    write_index(samples, project_packages, project_package_embeddings, item_embeddings, output_dir, meta)
    return len(samples), len(project_packages), embedding_dim


def main() -> int:
    args = parse_args()
    samples_path = Path(args.samples).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    try:
        validate_output_dir(output_dir, args.overwrite)
        sample_count, package_count, embedding_dim = build_cost_item_embedding_index(
            samples_path=samples_path,
            output_dir=output_dir,
            model_name=args.model,
            batch_size=args.batch_size,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    print(f"输入样本数: {sample_count}")
    print(f"工程包数: {package_count}")
    print(f"embedding 维度: {embedding_dim}")
    print(f"输出目录: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
