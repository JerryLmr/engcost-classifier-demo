import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Sequence

import requests

from classifier.catalog_loader import CatalogItem, get_catalog_by_id, load_catalog
from classifier.candidate_retriever import candidate_prompt_label
from classifier.settings import (
    LLM_PROVIDER,
    LLM_TIMEOUT_SECONDS,
    LMSTUDIO_API_KEY,
    LMSTUDIO_BASE_URL,
    LMSTUDIO_MAX_TOKENS,
    LMSTUDIO_MODEL,
    LMSTUDIO_RESPONSE_FORMAT,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)
from classifier.standard_catalog_loader import OUT_OF_SCOPE_ID, StandardCatalogItem


@dataclass(frozen=True)
class ItemSelection:
    catalog_id: str
    secondary_catalog_ids: tuple[str, ...]
    is_composite: bool
    needs_review: bool
    reason: str
    invalid_after_retry: bool = False


@dataclass(frozen=True)
class StatusSelection:
    repair_status: str
    needs_review: bool
    reason: str
    invalid_after_retry: bool = False


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


def build_classification_prompt(text: str, items: Sequence[CatalogItem]) -> str:
    return f"""
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


def _request_ollama_classification(prompt: str) -> Dict[str, Any]:
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


def _request_lmstudio_classification(prompt: str) -> Dict[str, Any]:
    payload = {
        "model": LMSTUDIO_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是物业工程目录分类助手。"
                    "不要输出思考过程，不要输出 <think>，不要输出 markdown。"
                    "最终答案只能是一个 JSON object。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": LMSTUDIO_MAX_TOKENS,
    }
    _apply_lmstudio_response_format(payload)
    response = requests.post(
        f"{LMSTUDIO_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {LMSTUDIO_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=LLM_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()
    return _extract_json_object(data["choices"][0]["message"]["content"])


def _apply_lmstudio_response_format(payload: Dict[str, Any]) -> None:
    response_format = LMSTUDIO_RESPONSE_FORMAT.strip().lower()
    if response_format == "text":
        payload["response_format"] = {"type": "text"}
    elif response_format == "json_schema":
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "classification_result",
                "schema": {
                    "type": "object",
                    "additionalProperties": True,
                },
            },
        }
    elif response_format == "none":
        pass
    else:
        raise ValueError(
            f"Unsupported LMSTUDIO_RESPONSE_FORMAT: {LMSTUDIO_RESPONSE_FORMAT}. "
            "Use one of: none, text, json_schema."
        )


def _strip_reasoning_and_fences(text: str) -> str:
    raw = str(text or "").strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE).strip()

    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    return raw


def _extract_json_object(text: str) -> Dict[str, Any]:
    raw = _strip_reasoning_and_fences(text)

    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("LLM response is not a JSON object")
        return data
    except json.JSONDecodeError:
        pass

    for start, char in enumerate(raw):
        if char != "{":
            continue

        depth = 0
        in_string = False
        escape = False

        for index in range(start, len(raw)):
            ch = raw[index]

            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = raw[start : index + 1]
                    try:
                        data = json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                    if not isinstance(data, dict):
                        raise ValueError("LLM response is not a JSON object")
                    return data

    raise ValueError("No valid JSON object found in LLM response")


def _request_ollama_json(prompt: str) -> Dict[str, Any]:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0},
        "format": "json",
    }
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json=payload,
        timeout=LLM_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()
    return _extract_json_object(data.get("response", ""))


def _request_lmstudio_json(prompt: str) -> Dict[str, Any]:
    payload = {
        "model": LMSTUDIO_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "你是物业工程分类助手。"
                    "不要输出思考过程，不要输出 <think>，不要输出 markdown。"
                    "最终答案只能是一个 JSON object。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": LMSTUDIO_MAX_TOKENS,
    }
    _apply_lmstudio_response_format(payload)
    response = requests.post(
        f"{LMSTUDIO_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {LMSTUDIO_API_KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=LLM_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    return _extract_json_object(content)


def request_llm_json(prompt: str) -> Dict[str, Any]:
    provider = LLM_PROVIDER.strip().lower()
    if provider == "ollama":
        return _request_ollama_json(prompt)
    if provider == "lmstudio":
        return _request_lmstudio_json(prompt)
    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


def request_llm_classification(text: str, candidate_items: Sequence[CatalogItem] | None = None) -> Dict[str, Any]:
    items = list(candidate_items or load_catalog())
    prompt = build_classification_prompt(text, items)
    provider = LLM_PROVIDER.strip().lower()
    if provider == "ollama":
        return _request_ollama_classification(prompt)
    if provider == "lmstudio":
        return _request_lmstudio_classification(prompt)
    raise ValueError(f"Unsupported LLM_PROVIDER: {LLM_PROVIDER}")


def build_item_selection_prompt(
    project_name: str,
    candidates: Sequence[StandardCatalogItem],
    context_hints: Sequence[str] | None = None,
) -> str:
    candidate_lines = "\n".join(candidate_prompt_label(item) for item in candidates)
    hint_lines = "\n".join(f"- {hint}" for hint in context_hints or [] if hint)
    hint_section = f"\n辅助提示（只作判断依据，不能替代候选目录）：\n{hint_lines}\n" if hint_lines else ""
    return f"""
你是物业工程标准目录分类助手。只能从给定 top 候选中选择最合适的 catalog_id。

工程名称：{project_name}
{hint_section}

候选目录：
{candidate_lines}

输出要求：
1. catalog_id 必须是候选中的 id，或者 OUT_OF_SCOPE。
2. secondary_catalog_ids 只能来自候选 id。
3. 如果候选目录都不适合，catalog_id 才返回 OUT_OF_SCOPE。
4. 如果是复合工程，主对象放 catalog_id，其它对象放 secondary_catalog_ids。
5. 如果主对象不明确，needs_review=true。
6. 不要创造候选外分类。
7. alias、动作词、复核提示只能辅助判断，不能绕过候选目录直接决定 catalog_id。
8. 不要因为出现咨询、设计、检测、维保、综合品质改造等提示词就直接 OUT；只有缺少明确维修对象时才 OUT_OF_SCOPE。
9. 不要输出思考过程。
10. 不要输出 <think>。
11. 不要输出 markdown。
12. reason 不超过 40 个汉字。
13. 只输出 JSON，不要自然语言段落。

JSON 字段固定为：catalog_id, secondary_catalog_ids, is_composite, needs_review, reason。
""".strip()


def build_status_selection_prompt(project_name: str, selected_item: StandardCatalogItem) -> str:
    basis_lines = "\n".join(
        f"{status}：{basis}" for status, basis in selected_item.status_basis.items()
    )
    return f"""
工程名称：{project_name}

已选标准目录：
id: {selected_item.id}
一级分类: {selected_item.category}
二级分类: {selected_item.item}

可选维修状态及依据：
{basis_lines}

请只从上述状态中选择一个。
如果工程名称无法判断具体状态，选择“不确定”。
不要输出思考过程。
不要输出 <think>。
不要输出 markdown。
reason 不超过 40 个汉字。
只输出 JSON。

JSON 字段固定为：repair_status, needs_review, reason。
""".strip()


def _validate_item_selection(content: Dict[str, Any], candidates: Sequence[StandardCatalogItem]) -> ItemSelection:
    candidate_ids = {item.id for item in candidates}
    catalog_id = str(content.get("catalog_id", "")).strip()
    if catalog_id not in candidate_ids and catalog_id != OUT_OF_SCOPE_ID:
        raise ValueError(f"LLM returned invalid catalog_id: {catalog_id}")

    raw_secondary_ids = content.get("secondary_catalog_ids")
    if not isinstance(raw_secondary_ids, list):
        raise ValueError("LLM returned invalid secondary_catalog_ids")
    secondary_ids: list[str] = []
    for value in raw_secondary_ids:
        item_id = str(value).strip()
        if item_id not in candidate_ids:
            raise ValueError(f"LLM returned invalid secondary catalog_id: {item_id}")
        if item_id != catalog_id and item_id not in secondary_ids:
            secondary_ids.append(item_id)

    is_composite = content.get("is_composite")
    needs_review = content.get("needs_review")
    reason = content.get("reason")
    if not isinstance(is_composite, bool):
        raise ValueError("LLM returned invalid is_composite")
    if not isinstance(needs_review, bool):
        raise ValueError("LLM returned invalid needs_review")
    if not isinstance(reason, str):
        raise ValueError("LLM returned invalid reason")

    return ItemSelection(
        catalog_id=catalog_id,
        secondary_catalog_ids=tuple(secondary_ids),
        is_composite=is_composite,
        needs_review=needs_review,
        reason=reason.strip() or "模型未提供目录选择依据",
    )


def _validate_status_selection(content: Dict[str, Any], selected_item: StandardCatalogItem) -> StatusSelection:
    repair_status = str(content.get("repair_status", "")).strip()
    allowed_statuses = set(selected_item.allowed_statuses)
    if repair_status not in allowed_statuses and repair_status != "不确定":
        raise ValueError(f"LLM returned invalid repair_status: {repair_status}")

    needs_review = content.get("needs_review")
    reason = content.get("reason")
    if not isinstance(needs_review, bool):
        raise ValueError("LLM returned invalid needs_review")
    if not isinstance(reason, str):
        raise ValueError("LLM returned invalid reason")
    if repair_status == "不确定":
        needs_review = True

    return StatusSelection(
        repair_status=repair_status,
        needs_review=needs_review,
        reason=reason.strip() or "模型未提供维修状态依据",
    )


def llm_select_catalog_item(
    project_name: str,
    candidates: Sequence[StandardCatalogItem],
    context_hints: Sequence[str] | None = None,
) -> ItemSelection:
    prompt = build_item_selection_prompt(project_name, candidates, context_hints)
    last_error: Exception | None = None
    for _attempt in range(2):
        try:
            return _validate_item_selection(request_llm_json(prompt), candidates)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    reason = "LLM item selection invalid after retry"
    if last_error:
        reason = f"{reason}: {last_error}"
    return ItemSelection(
        catalog_id=OUT_OF_SCOPE_ID,
        secondary_catalog_ids=(),
        is_composite=False,
        needs_review=True,
        reason=reason,
        invalid_after_retry=True,
    )


def llm_select_repair_status(project_name: str, selected_item: StandardCatalogItem) -> StatusSelection:
    prompt = build_status_selection_prompt(project_name, selected_item)
    last_error: Exception | None = None
    for _attempt in range(2):
        try:
            return _validate_status_selection(request_llm_json(prompt), selected_item)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    reason = "LLM status selection invalid after retry"
    if last_error:
        reason = f"{reason}: {last_error}"
    return StatusSelection(
        repair_status="不确定",
        needs_review=True,
        reason=reason,
        invalid_after_retry=True,
    )


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
