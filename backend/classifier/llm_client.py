import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Sequence

import requests

from classifier.settings import (
    LLM_TIMEOUT_SECONDS,
    LMSTUDIO_API_KEY,
    LMSTUDIO_BASE_URL,
    LMSTUDIO_MAX_TOKENS,
    LMSTUDIO_MODEL,
    LMSTUDIO_RESPONSE_FORMAT,
)
from classifier.standard_catalog_loader import OUT_OF_SCOPE_ID, StandardCatalogItem, get_standard_catalog_by_id


class LLMServiceError(RuntimeError):
    """Raised when the LM Studio service cannot be reached or returns a transport error."""


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


def check_lmstudio_service(timeout_seconds: float = 3.0) -> None:
    try:
        response = requests.get(f"{LMSTUDIO_BASE_URL}/models", timeout=timeout_seconds)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            "LM Studio 服务不可用，请先启动 LM Studio Server，并检查 "
            f"LMSTUDIO_BASE_URL={LMSTUDIO_BASE_URL}"
        ) from exc


def _request_lmstudio_json(
    prompt: str,
    max_tokens: int | None = None,
    timeout_seconds: int | None = None,
    system_prompt: str | None = None,
) -> Dict[str, Any]:
    payload = {
        "model": LMSTUDIO_MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
                or (
                    "你是物业工程分类助手。"
                    "不要输出思考过程，不要输出 <think>，不要输出 markdown。"
                    "最终答案只能是一个 JSON object。"
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": max_tokens or LMSTUDIO_MAX_TOKENS,
    }
    _apply_lmstudio_response_format(payload)
    try:
        response = requests.post(
            f"{LMSTUDIO_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {LMSTUDIO_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=timeout_seconds or LLM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except (
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
        requests.exceptions.RequestException,
    ) as exc:
        raise LLMServiceError(
            "LM Studio 服务请求失败，请检查 LM Studio Server 和 "
            f"LMSTUDIO_BASE_URL={LMSTUDIO_BASE_URL}: {exc}"
        ) from exc
    data = response.json()
    content = data["choices"][0]["message"]["content"]
    return _extract_json_object(content)


def request_llm_json(
    prompt: str,
    max_tokens: int | None = None,
    timeout_seconds: int | None = None,
    system_prompt: str | None = None,
) -> Dict[str, Any]:
    return _request_lmstudio_json(
        prompt,
        max_tokens=max_tokens,
        timeout_seconds=timeout_seconds,
        system_prompt=system_prompt,
    )


def compact_catalog_prompt_label(item: StandardCatalogItem) -> str:
    return (
        f"{item.id} | {item.standard_group} | {item.category} | {item.item} | "
        f"{'/'.join(item.allowed_statuses)}"
    )


def build_full_catalog_item_selection_prompt(
    project_name: str,
    catalog_items: Sequence[StandardCatalogItem],
    context_hints: Sequence[str] | None = None,
) -> str:
    catalog_lines = "\n".join(compact_catalog_prompt_label(item) for item in catalog_items)
    hint_lines = "\n".join(f"- {hint}" for hint in context_hints or [] if hint)
    hint_section = f"\n辅助提示（只作理解文本，不得替代标准目录判断）：\n{hint_lines}\n" if hint_lines else ""
    return f"""
你是物业工程标准目录分类助手。请读取完整 compact 标准目录，判断工程名称对应的标准目录。

工程名称：{project_name}
{hint_section}

完整 compact 标准目录：
catalog_id | 标准对象 | 一级分类 | 二级分类 | 可选状态
{catalog_lines}

输出要求：
1. 只能从完整标准目录中选择一个 catalog_id，或者 OUT_OF_SCOPE。
2. 如果工程名称能对应到标准目录中的共用部位、共用设施设备维修对象，不要返回 OUT_OF_SCOPE。
3. 如果有具体对象，优先选择具体二级目录。
4. 如果多个对象并列，主对象放 catalog_id，其它对象放 secondary_catalog_ids，并设置 is_composite=true 或 needs_review=true。
5. 如果只能判断一级系统，且标准目录里存在“未明确具体子项”，可以选择该项，并设置 needs_review=true。
6. 如果文本包含咨询、设计、检测、维保、审计等服务词，但同时有明确维修对象，应选择该对象并设置 needs_review=true，不要直接 OUT。
7. 只有完全没有共用部位/共用设施设备维修对象时，才返回 OUT_OF_SCOPE。
8. alias、动作词、复核提示只作为辅助理解，不能直接决定 catalog_id。
9. 不要输出思考过程。
10. 不要输出 <think>。
11. 不要输出 markdown。
12. reason 不超过 40 个汉字。
13. 只输出 JSON，不要自然语言段落。

JSON 字段固定为：
{{
  "catalog_id": "...",
  "secondary_catalog_ids": [],
  "is_composite": false,
  "needs_review": false,
  "reason": "..."
}}
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


def _validate_selection_fields(content: Dict[str, Any]) -> tuple[str, list[Any], bool, bool, str]:
    catalog_id = str(content.get("catalog_id", "")).strip()
    raw_secondary_ids = content.get("secondary_catalog_ids")
    if not isinstance(raw_secondary_ids, list):
        raise ValueError("LLM returned invalid secondary_catalog_ids")

    is_composite = content.get("is_composite")
    needs_review = content.get("needs_review")
    reason = content.get("reason")
    if not isinstance(is_composite, bool):
        raise ValueError("LLM returned invalid is_composite")
    if not isinstance(needs_review, bool):
        raise ValueError("LLM returned invalid needs_review")
    if not isinstance(reason, str):
        raise ValueError("LLM returned invalid reason")
    return catalog_id, raw_secondary_ids, is_composite, needs_review, reason


def _validate_full_catalog_item_selection(content: Dict[str, Any]) -> ItemSelection:
    catalog_by_id = get_standard_catalog_by_id()
    catalog_id, raw_secondary_ids, is_composite, needs_review, reason = _validate_selection_fields(content)
    if catalog_id not in catalog_by_id and catalog_id != OUT_OF_SCOPE_ID:
        raise ValueError(f"LLM returned invalid catalog_id: {catalog_id}")

    secondary_ids: list[str] = []
    dropped_secondary_ids: list[str] = []
    for value in raw_secondary_ids:
        item_id = str(value).strip()
        if item_id not in catalog_by_id:
            if item_id:
                dropped_secondary_ids.append(item_id)
            continue
        if item_id != catalog_id and item_id not in secondary_ids:
            secondary_ids.append(item_id)

    if secondary_ids:
        needs_review = True
    if dropped_secondary_ids:
        needs_review = True
        reason = "；".join(
            part
            for part in (
                reason.strip(),
                f"已丢弃标准外 secondary id: {','.join(dropped_secondary_ids)}",
            )
            if part
        )

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


def llm_select_catalog_item_from_full_catalog(
    project_name: str,
    catalog_items: Sequence[StandardCatalogItem],
    context_hints: Sequence[str] | None = None,
) -> ItemSelection:
    prompt = build_full_catalog_item_selection_prompt(project_name, catalog_items, context_hints)
    last_error: Exception | None = None
    for _attempt in range(2):
        try:
            return _validate_full_catalog_item_selection(request_llm_json(prompt))
        except LLMServiceError:
            raise
        except Exception as exc:  # noqa: BLE001
            last_error = exc
    reason = "LLM full catalog item selection invalid after retry"
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
        except LLMServiceError:
            raise
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
