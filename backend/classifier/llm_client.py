import json
from typing import Any, Dict, Sequence

import requests

from classifier.catalog_loader import CatalogItem, get_catalog_by_id, load_catalog
from classifier.settings import LLM_TIMEOUT_SECONDS, OLLAMA_BASE_URL, OLLAMA_MODEL


def _catalog_prompt_lines(items: Sequence[CatalogItem]) -> str:
    return "\n".join(item.label for item in items)


def request_llm_classification(text: str, candidate_items: Sequence[CatalogItem] | None = None) -> Dict[str, Any]:
    items = list(candidate_items or load_catalog())
    prompt = f"""
你是物业工程目录分类助手。只能从给定 catalog id 中选择最合适的一个三级目录。

catalog:
{_catalog_prompt_lines(items)}

输出要求：
1. 只能返回 catalog 中已有 id，不允许创造目录。
2. 输出必须是 JSON。
3. JSON 字段固定为：id, reason, needs_review。
4. 如果输入是混合工程，选择最主要工程对象对应的 id，并在 reason 说明其他候选。
5. 不要输出 markdown，不要输出解释。

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
                "id": {"type": "string"},
                "reason": {"type": "string"},
                "needs_review": {"type": "boolean"},
            },
            "required": ["id", "reason", "needs_review"],
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


def llm_select_item(text: str, candidate_items: Sequence[CatalogItem] | None = None) -> tuple[CatalogItem, str, bool]:
    content = request_llm_classification(text, candidate_items)
    item_id = str(content.get("id", "")).strip()
    catalog_by_id = get_catalog_by_id()
    if item_id not in catalog_by_id:
        raise ValueError(f"LLM returned invalid catalog id: {item_id}")
    return (
        catalog_by_id[item_id],
        str(content.get("reason") or "模型语义匹配"),
        bool(content.get("needs_review", True)),
    )
