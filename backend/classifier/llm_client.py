import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Sequence

import requests

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
9. 如果工程名称只写普通电梯维修、更新、改造、大修，未明确曳引机、钢丝绳、制动器、控制柜、层轿门、缓冲器、呼梯系统、液压梯、自动扶梯、自动人行道等具体部件或子系统，可以选择 CF-017-00，并设置 needs_review=true。
10. 不要把普通住宅垂直电梯、客梯、老旧电梯误选为 CF-017-13 自动扶梯及自动人行道。只有明确出现自动扶梯、自动人行道、扶手带、梯级、踏板、梯级链、滚轮等对象时，才选择 CF-017-13。
11. 如果文本中的“电梯”只是位置或系统修饰词，例如电梯厅、电梯间、电梯前室、电梯监控、轿厢监控、梯控、电梯门禁，不要因此选择电梯类目录，应根据墙面、吊顶、弱电、门禁、监控等实际对象判断。
12. 不要输出思考过程。
13. 不要输出 <think>。
14. 不要输出 markdown。
15. reason 不超过 40 个汉字。
16. 只输出 JSON，不要自然语言段落。

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
