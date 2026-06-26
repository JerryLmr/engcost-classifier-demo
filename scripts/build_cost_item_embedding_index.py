#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "cost_item_name",
    "project_description",
]

MANAGED_OUTPUT_FILES = [
    "samples.parquet",
    "project_groups.parquet",
    "project_name_embeddings.npy",
    "project_detail_embeddings.npy",
    # 清理旧单路索引残留，查询阶段不再兼容读取。
    "project_group_embeddings.npy",
    "index_meta.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建已审定清单样本 embedding 索引")
    parser.add_argument("--samples", required=True, help="已审定清单样本 Excel 路径")
    parser.add_argument("--output-dir", required=True, help="索引输出目录")
    parser.add_argument("--model", default="BAAI/bge-m3", help="sentence-transformers 模型名")
    parser.add_argument("--batch-size", type=int, default=32, help="embedding 批大小")
    parser.add_argument("--overwrite", action="store_true", help="若索引输出文件已存在则覆盖")
    return parser.parse_args()


def validate_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError(f"输出路径不是目录: {output_dir}")
    existing = [name for name in MANAGED_OUTPUT_FILES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise ValueError(
            "输出已存在，请加 --overwrite 或更换输出目录: "
            f"{output_dir} ({', '.join(existing)})"
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

    missing = [column for column in REQUIRED_COLUMNS if column not in samples.columns]
    if missing:
        raise ValueError(f"samples sheet 缺少必要字段: {', '.join(missing)}")

    return samples


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def join_parts(parts: list[str], separator: str = " ") -> str:
    return separator.join(part for part in parts if part).strip()


PROJECT_GROUP_COLUMNS = [
    "source_row_id",
    "工程名称",
    "catalog_id",
    "一级分类",
    "二级分类",
    "维修状态",
    "标准对象",
    "consultation_time",
    "location",
    "project_name_text",
    "project_detail_text",
    "item_count",
]


def build_project_groups(samples: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for source_row_id, group in samples.groupby("source_row_id", sort=False, dropna=False):
        first = group.iloc[0]
        project_name_text = safe_text(first.get("工程名称"))
        item_summaries: list[str] = []
        seen: set[str] = set()
        for _index, item in group.iterrows():
            item_text = join_parts(
                [
                    safe_text(item.get("cost_item_name")),
                    safe_text(item.get("project_description")),
                ],
                " ",
            )
            if item_text and item_text not in seen:
                item_summaries.append(item_text)
                seen.add(item_text)
        project_detail_text = " ".join(item_summaries).strip()

        rows.append(
            {
                "source_row_id": source_row_id,
                "工程名称": project_name_text,
                "catalog_id": safe_text(first.get("catalog_id")),
                "一级分类": safe_text(first.get("一级分类")),
                "二级分类": safe_text(first.get("二级分类")),
                "维修状态": safe_text(first.get("维修状态")),
                "标准对象": safe_text(first.get("标准对象")),
                "consultation_time": safe_text(first.get("consultation_time")),
                "location": safe_text(first.get("location")),
                "project_name_text": project_name_text,
                "project_detail_text": project_detail_text,
                "item_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows, columns=PROJECT_GROUP_COLUMNS)


def load_embedding_model(model_name: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("缺少依赖 sentence-transformers，请先安装 requirements.txt") from exc

    try:
        return SentenceTransformer(model_name)
    except Exception as exc:
        raise RuntimeError(f"embedding 模型加载失败: {model_name}: {exc}") from exc


def text_series(values: pd.Series) -> list[str]:
    return values.fillna("").astype(str).tolist()


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return array / norms


def encode_texts(model: Any, texts: list[str], batch_size: int) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return normalize_embeddings(embeddings)


def build_index_meta(
    samples_path: Path,
    model_name: str,
    sample_count: int,
    embedding_dim: int,
) -> dict[str, Any]:
    return {
        "model": model_name,
        "sample_count": sample_count,
        "embedding_dim": embedding_dim,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_samples": str(samples_path),
        "files": {
            "samples": "samples.parquet",
            "project_groups": "project_groups.parquet",
            "project_name_embeddings": "project_name_embeddings.npy",
            "project_detail_embeddings": "project_detail_embeddings.npy",
        },
        "field_descriptions": {
            "project_name_text": "工程名称文本，用于 source_row_id 工程名称召回",
            "project_detail_text": "工程下清单项名称和项目特征文本，用于 source_row_id 工程明细召回",
            "unit_price": "综合单价，查询阶段用于参考区间计算",
            "labor_unit_price": "人工单价，查询阶段用于参考区间计算",
            "machinery_unit_price": "机械单价，查询阶段用于参考区间计算",
        },
    }


def write_index(
    samples: pd.DataFrame,
    project_groups: pd.DataFrame,
    project_name_embeddings: np.ndarray,
    project_detail_embeddings: np.ndarray,
    output_dir: Path,
    meta: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in MANAGED_OUTPUT_FILES:
        path = output_dir / name
        if path.exists() and path.is_file():
            path.unlink()
    samples.to_parquet(output_dir / "samples.parquet", index=False)
    project_groups.to_parquet(output_dir / "project_groups.parquet", index=False)
    np.save(output_dir / "project_name_embeddings.npy", project_name_embeddings)
    np.save(output_dir / "project_detail_embeddings.npy", project_detail_embeddings)
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
    project_groups = build_project_groups(samples)
    model = load_embedding_model(model_name)

    project_name_embeddings = encode_texts(model, text_series(project_groups["project_name_text"]), batch_size)
    project_detail_embeddings = encode_texts(model, text_series(project_groups["project_detail_text"]), batch_size)

    if len(project_groups) != project_name_embeddings.shape[0]:
        raise ValueError("工程分组数量与工程名称 embedding 数量不一致")
    if len(project_groups) != project_detail_embeddings.shape[0]:
        raise ValueError("工程分组数量与工程明细 embedding 数量不一致")
    if project_name_embeddings.shape[1] != project_detail_embeddings.shape[1]:
        raise ValueError("工程名称 embedding 与工程明细 embedding 维度不一致")

    embedding_dim = int(project_name_embeddings.shape[1]) if project_name_embeddings.ndim == 2 else 0
    meta = build_index_meta(samples_path, model_name, len(samples), embedding_dim)
    write_index(
        samples,
        project_groups,
        project_name_embeddings,
        project_detail_embeddings,
        output_dir,
        meta,
    )
    return len(samples), len(project_groups), embedding_dim


def main() -> int:
    args = parse_args()
    samples_path = Path(args.samples).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    try:
        validate_output_dir(output_dir, args.overwrite)
        input_count, group_count, embedding_dim = build_cost_item_embedding_index(
            samples_path=samples_path,
            output_dir=output_dir,
            model_name=args.model,
            batch_size=args.batch_size,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    print(f"输入样本数: {input_count}")
    print(f"工程组数: {group_count}")
    print(f"embedding 维度: {embedding_dim}")
    print(f"输出目录: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
