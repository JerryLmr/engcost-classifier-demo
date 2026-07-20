import re


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").strip())

