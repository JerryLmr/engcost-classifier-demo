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


def _normalize_level3_format(value: str) -> str:
    compact = "".join(str(value).split())
    return compact.translate(
        str.maketrans(
            {
                "（": "(",
                "）": ")",
                "：": ":",
                "，": ",",
                "、": ",",
            }
        )
    )


def _match_catalog_level3_item(raw_level3_item: str, catalog_level3_items: Sequence[str]) -> str:
    if raw_level3_item in catalog_level3_items:
        return raw_level3_item

    normalized_item = _normalize_level3_format(raw_level3_item)
    normalized_matches = [
        catalog_item
        for catalog_item in catalog_level3_items
        if _normalize_level3_format(catalog_item) == normalized_item
    ]
    if len(normalized_matches) == 1:
        return normalized_matches[0]
    return ""


def request_llm_classification(text: str, candidate_items: Sequence[CatalogItem] | None = None) -> Dict[str, Any]:
    items = list(candidate_items or load_catalog())
    prompt = f"""
你是物业工程目录分类助手。只能从给定 catalog id 中选择最合适的一个三级目录。
输入可能是“项目名称 + 工程概况”的拼接文本，需要综合判断。

catalog:
{_catalog_prompt_lines(items)}

输出要求：
1. id 必须逐字复制 catalog 中已有的三位数字 id，只能返回纯 id。
   不能返回 "000"。
   不能返回 "069 空调 > 中央空调"。
   不能返回 "2023 绿化 > 花坛"。
   不能返回任何 catalog 外 id。
2. 输出必须是 JSON。
3. JSON 字段固定为：id, level3_item, reason, needs_review。
4. 优先判断工程对象和设施对象，不要只根据“维修、改造、更换、整治”等通用动作词分类。
5. level3_item 必须逐字复制所选 id 下列出的某一个原始细项。
   不能改字。
   不能增删词。
   不能合并多个细项。
   不能自行补充“及附件”“维修”“改造”等词。
   不能把两个细项合并成一个。
6. 如果无法确定具体细项，仍然要从所选 id 下选择最接近的一个原始细项，并设置 needs_review=true。
7. 如果文本同时出现“消防”和“管道、阀门、消防栓、消火栓、喷淋、报警、消防泵、风机、防排烟”之一，优先在“消防设备”目录中选择。
   不要归到给水或排水，除非文本明确是生活给水、污水、雨水、下水、排水。
8. 如果文本出现“电缆、强电、低压柜、高压柜、线路”，优先在强电目录中选择，不要归到排水或安防。
9. 如果文本出现“车辆识别、车牌识别、道闸、门禁、梯控、视频监控”，优先在安防系统中选择。
   车辆识别系统不要归到“车库、停车场地”这种土建附属设施，除非文本明确是车库地坪、车库墙面、车库结构维修。
10. 如果文本是“咨询、检测、评估、维保服务”而不是明确维修/更换/改造工程，允许选择最接近目录，但必须 needs_review=true，并在 reason 说明“服务类项目，目录可能不完全匹配”。
11. 如果是混合工程，选择主工程对象；如果主工程对象不清楚，选择最靠前、金额/数量/描述最多的工程对象，并 needs_review=true。
12. 不要输出 markdown，不要输出解释。

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
    matched_level3_item = _match_catalog_level3_item(level3_item, item.level3_items) if level3_item else ""
    if level3_item and not matched_level3_item:
        reason = f"{reason}；LLM 返回的细项不在标准目录中，已忽略：{level3_item}"
        level3_item = ""
    elif matched_level3_item:
        level3_item = matched_level3_item
    return (
        item,
        level3_item,
        reason,
        bool(content.get("needs_review", True)),
    )
