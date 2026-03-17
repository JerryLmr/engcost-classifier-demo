import json

import requests

from core.config import LLM_TIMEOUT_SECONDS, OLLAMA_BASE_URL, OLLAMA_MODEL
from data.categories import CATEGORY_LINES, CATEGORY_TREE


def request_llm_classification(text: str):
    prompt = f"""
你是物业维修工程分类助手。请根据工程名称，从给定分类体系中选择最合适的一级分类和二级分类。

分类体系：
{CATEGORY_LINES}

输出要求：
1. 只能从上述分类中选择。
2. 输出必须是 JSON。
3. JSON 字段固定为：level1, level2, reason。
4. 不要输出 markdown，不要输出解释。

输入工程名称：{text}
""".strip()

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0},
        "format": {
            "type": "object",
            "properties": {
                "level1": {"type": "string"},
                "level2": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["level1", "level2", "reason"],
        },
    }

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=LLM_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()
    return json.loads(data["response"])


def llm_classify(text: str):
    from services.classifier import fallback_classify

    try:
        content = request_llm_classification(text)
    except Exception as exc:
        return fallback_classify(text, f"LLM 不可用，返回默认分类：{exc}")

    level1 = content.get("level1", "")
    level2 = content.get("level2", "")
    if level1 not in CATEGORY_TREE:
        return fallback_classify(text, "LLM 返回了无效一级分类，已降级为默认分类")

    if level2 not in CATEGORY_TREE[level1]:
        level2 = CATEGORY_TREE[level1][0]

    return {
        "project_name": text,
        "level1": level1,
        "level2": level2,
        "method": "LLM 兜底",
        "reason": content.get("reason", "模型语义匹配"),
    }
