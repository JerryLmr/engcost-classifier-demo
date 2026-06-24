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
    "item_embeddings.npy",
    "project_embeddings.npy",
    "full_embeddings.npy",
    "index_meta.json",
    "item_similarity_embeddings.npy",
    "item_context_embeddings.npy",
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


def text_column(samples: pd.DataFrame, column: str) -> pd.Series:
    if column not in samples.columns:
        return pd.Series([""] * len(samples), index=samples.index, dtype=str)
    return samples[column].fillna("").astype(str).str.strip()


def join_parts(parts: list[str], separator: str = " ") -> str:
    return separator.join(part for part in parts if part).strip()


def add_embedding_text_columns(samples: pd.DataFrame) -> pd.DataFrame:
    indexed = samples.copy()
    cost_item_name = text_column(indexed, "cost_item_name")
    project_description = text_column(indexed, "project_description")
    project_name = text_column(indexed, "工程名称")
    consultation_project_name = text_column(indexed, "consultation_project_name")
    renovation_content = text_column(indexed, "renovation_content")
    units = text_column(indexed, "unit_normalized")

    item_texts: list[str] = []
    project_texts: list[str] = []
    full_texts: list[str] = []

    for index in indexed.index:
        item_text = join_parts(
            [
                safe_text(cost_item_name.loc[index]),
                safe_text(project_description.loc[index]),
            ],
            " ",
        )
        fallback_project_text = join_parts(
            [
                safe_text(consultation_project_name.loc[index]),
                safe_text(renovation_content.loc[index]),
            ],
            " ",
        )
        project_text = safe_text(project_name.loc[index]) or fallback_project_text
        unit = safe_text(units.loc[index])
        full_text = join_parts(
            [
                project_text,
                item_text,
                f"单位：{unit}" if unit else "",
            ],
            " ",
        )
        item_texts.append(item_text)
        project_texts.append(project_text)
        full_texts.append(full_text)

    indexed["item_text"] = item_texts
    indexed["project_text"] = project_texts
    indexed["full_text"] = full_texts
    return indexed


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
            "item_embeddings": "item_embeddings.npy",
            "project_embeddings": "project_embeddings.npy",
            "full_embeddings": "full_embeddings.npy",
        },
        "field_descriptions": {
            "item_text": "清单项和项目特征文本，用于 item_score 检索",
            "project_text": "工程/维修场景文本，用于 project_score 检索",
            "full_text": "工程场景、清单项和单位文本，用于 full_score 检索",
            "unit_price": "综合单价，查询阶段用于参考区间计算",
            "labor_unit_price": "人工单价，查询阶段用于参考区间计算",
            "machinery_unit_price": "机械单价，查询阶段用于参考区间计算",
        },
    }


def write_index(
    samples: pd.DataFrame,
    item_embeddings: np.ndarray,
    project_embeddings: np.ndarray,
    full_embeddings: np.ndarray,
    output_dir: Path,
    meta: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in MANAGED_OUTPUT_FILES:
        path = output_dir / name
        if path.exists() and path.is_file():
            path.unlink()
    samples.to_parquet(output_dir / "samples.parquet", index=False)
    np.save(output_dir / "item_embeddings.npy", item_embeddings)
    np.save(output_dir / "project_embeddings.npy", project_embeddings)
    np.save(output_dir / "full_embeddings.npy", full_embeddings)
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
    samples = add_embedding_text_columns(load_samples(samples_path))
    model = load_embedding_model(model_name)

    item_embeddings = encode_texts(model, text_series(samples["item_text"]), batch_size)
    project_embeddings = encode_texts(model, text_series(samples["project_text"]), batch_size)
    full_embeddings = encode_texts(model, text_series(samples["full_text"]), batch_size)

    if item_embeddings.shape != project_embeddings.shape or item_embeddings.shape != full_embeddings.shape:
        raise ValueError("三路 embedding 维度或样本数不一致")
    if len(samples) != item_embeddings.shape[0]:
        raise ValueError("样本数量与 embedding 数量不一致")

    embedding_dim = int(item_embeddings.shape[1]) if item_embeddings.ndim == 2 else 0
    meta = build_index_meta(samples_path, model_name, len(samples), embedding_dim)
    write_index(samples, item_embeddings, project_embeddings, full_embeddings, output_dir, meta)
    return len(samples), item_embeddings.shape[0], embedding_dim


def main() -> int:
    args = parse_args()
    samples_path = Path(args.samples).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    try:
        validate_output_dir(output_dir, args.overwrite)
        input_count, encoded_count, embedding_dim = build_cost_item_embedding_index(
            samples_path=samples_path,
            output_dir=output_dir,
            model_name=args.model,
            batch_size=args.batch_size,
        )
    except (RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    print(f"输入样本数: {input_count}")
    print(f"成功编码数: {encoded_count}")
    print(f"embedding 维度: {embedding_dim}")
    print(f"输出目录: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
