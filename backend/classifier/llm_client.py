import json
from typing import Any, Dict, Sequence

import requests

from classifier.catalog_loader import CatalogItem, get_catalog_by_id, load_catalog
from classifier.settings import LLM_TIMEOUT_SECONDS, OLLAMA_BASE_URL, OLLAMA_MODEL


def _catalog_prompt_lines(items: Sequence[CatalogItem]) -> str:
    lines = []
    for item in items:
        lines.append(item.label)
        lines.extend(f"  - {level3_item}" for level3_item in item.level3_items)
    return "\n".join(lines)


def request_llm_classification(text: str, candidate_items: Sequence[CatalogItem] | None = None) -> Dict[str, Any]:
    items = list(candidate_items or load_catalog())
    prompt = f"""
你是物业工程目录分类助手。只能从给定 catalog id 中选择最合适的一个三级目录。
输入可能是“项目名称 + 工程概况”的拼接文本，需要综合判断。

catalog:
{_catalog_prompt_lines(items)}

输出要求：
1. 只能返回 catalog 中已有 id，不允许创造目录。
2. 输出必须是 JSON。
3. JSON 字段固定为：id, level3_item, reason, needs_review。
4. 优先判断工程对象和设施对象，不要只根据“维修、改造、更换、整治”等通用动作词分类。
5. level3_item 必须从所选 id 下列出的原始细项中选择，不允许创造细项。
6. 如果没有完全对应细项，选择最接近的已列出细项，并设置 needs_review=true。
7. 如果输入是混合工程，选择最主要工程对象对应的 id，并在 reason 说明其他候选。
8. 不要输出 markdown，不要输出解释。

输入文本：{text}
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
                "level3_item": {"type": "string"},
                "reason": {"type": "string"},
                "needs_review": {"type": "boolean"},
            },
            "required": ["id", "level3_item", "reason", "needs_review"],
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


def llm_select_item(text: str, candidate_items: Sequence[CatalogItem] | None = None) -> tuple[CatalogItem, str, str, bool]:
    content = request_llm_classification(text, candidate_items)
    item_id = str(content.get("id", "")).strip()
    catalog_by_id = get_catalog_by_id()
    if item_id not in catalog_by_id:
        raise ValueError(f"LLM returned invalid catalog id: {item_id}")
    item = catalog_by_id[item_id]
    level3_item = str(content.get("level3_item") or "").strip()
    reason = str(content.get("reason") or "模型语义匹配")
    if level3_item and level3_item not in item.level3_items:
        reason = f"{reason}；LLM 返回的细项不在标准目录中，已忽略：{level3_item}"
        level3_item = ""
    return (
        item,
        level3_item,
        reason,
        bool(content.get("needs_review", True)),
    )
