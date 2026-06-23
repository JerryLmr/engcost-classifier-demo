#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "item_similarity_text",
    "item_context_text",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建已审定清单样本 embedding 索引")
    parser.add_argument("--samples", required=True, help="已审定清单样本 Excel 路径")
    parser.add_argument("--output-dir", required=True, help="索引输出目录")
    parser.add_argument("--model", default="BAAI/bge-m3", help="sentence-transformers 模型名")
    parser.add_argument("--batch-size", type=int, default=32, help="embedding 批大小")
    return parser.parse_args()


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
            "item_similarity_embeddings": "item_similarity_embeddings.npy",
            "item_context_embeddings": "item_context_embeddings.npy",
        },
        "field_descriptions": {
            "item_similarity_text": "清单项自身语义文本，用于 item_score 检索",
            "item_context_text": "工程上下文文本，用于 context_score 检索",
            "unit_price": "综合单价，查询阶段用于参考区间计算",
            "labor_unit_price": "人工单价，查询阶段用于参考区间计算",
            "machinery_unit_price": "机械单价，查询阶段用于参考区间计算",
        },
    }


def write_index(
    samples: pd.DataFrame,
    item_embeddings: np.ndarray,
    context_embeddings: np.ndarray,
    output_dir: Path,
    meta: dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples.to_parquet(output_dir / "samples.parquet", index=False)
    np.save(output_dir / "item_similarity_embeddings.npy", item_embeddings)
    np.save(output_dir / "item_context_embeddings.npy", context_embeddings)
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
    model = load_embedding_model(model_name)

    item_embeddings = encode_texts(model, text_series(samples["item_similarity_text"]), batch_size)
    context_embeddings = encode_texts(model, text_series(samples["item_context_text"]), batch_size)

    if item_embeddings.shape != context_embeddings.shape:
        raise ValueError("两路 embedding 维度或样本数不一致")
    if len(samples) != item_embeddings.shape[0]:
        raise ValueError("样本数量与 embedding 数量不一致")

    embedding_dim = int(item_embeddings.shape[1]) if item_embeddings.ndim == 2 else 0
    meta = build_index_meta(samples_path, model_name, len(samples), embedding_dim)
    write_index(samples, item_embeddings, context_embeddings, output_dir, meta)
    return len(samples), item_embeddings.shape[0], embedding_dim


def main() -> int:
    args = parse_args()
    samples_path = Path(args.samples).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    try:
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
