from __future__ import annotations

import gc
from typing import Any

import numpy as np


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (array / norms).astype(np.float32, copy=False)


def load_embedding_model(model_name: str, device: str | None = None) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("缺少依赖 sentence-transformers，请先安装 requirements.txt") from exc

    try:
        if device is None:
            return SentenceTransformer(model_name)
        return SentenceTransformer(model_name, device=device)
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


def encode_texts(
    model: Any,
    texts: list[str],
    batch_size: int | None = None,
    *,
    show_progress_bar: bool | None = None,
    allow_empty: bool = True,
) -> np.ndarray:
    if not texts and allow_empty:
        return np.empty((0, 0), dtype=np.float32)
    encode_options: dict[str, Any] = {
        "show_progress_bar": batch_size is not None if show_progress_bar is None else show_progress_bar,
        "convert_to_numpy": True,
        "normalize_embeddings": False,
    }
    if batch_size is not None:
        encode_options["batch_size"] = batch_size
    embeddings = model.encode(texts, **encode_options)
    return normalize_embeddings(embeddings)
