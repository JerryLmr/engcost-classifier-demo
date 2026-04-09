import os
from pathlib import Path
from typing import Iterable, Optional


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:8b")
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "90"))

DEFAULT_FALLBACK_LEVEL1 = os.getenv("DEFAULT_FALLBACK_LEVEL1", "公共设施")
DEFAULT_FALLBACK_LEVEL2 = os.getenv("DEFAULT_FALLBACK_LEVEL2", "公共区域维修")

RULE_SOURCE = os.getenv("RULE_SOURCE", "json")
RULE_CONFIG_DIR = Path(
    os.getenv(
        "RULE_CONFIG_DIR",
        str(Path(__file__).resolve().parents[1] / "rules"),
    )
)
RULE_CONFIG_FALLBACK_DIR = Path(
    os.getenv(
        "RULE_CONFIG_FALLBACK_DIR",
        str(Path(__file__).resolve().parents[1] / "config"),
    )
)


def resolve_rule_file(
    filename: str,
    fallback_filenames: Optional[Iterable[str]] = None,
) -> Path:
    candidates = [filename, *(fallback_filenames or [])]
    for candidate in candidates:
        primary = RULE_CONFIG_DIR / candidate
        if primary.exists():
            return primary
    for candidate in candidates:
        fallback = RULE_CONFIG_FALLBACK_DIR / candidate
        if fallback.exists():
            return fallback
    return RULE_CONFIG_DIR / filename
