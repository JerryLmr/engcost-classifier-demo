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
from classifier.semantic_prompt import SEMANTIC_PROJECT_TEXT_RULES
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
    project_name_text: str = ""
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
    item_group = getattr(item, "item_group", "") or "标准项"
    return (
        f"{item.id} | {item.standard_group} | {item.category} | {item.item} | "
        f"{item_group} | {'/'.join(item.allowed_statuses)}"
    )


def _is_unspecified_catalog_item(item: StandardCatalogItem) -> bool:
    item_group = getattr(item, "item_group", "") or ""
    return item.item == "未明确具体子项" or item_group == "内部扩展项" or "未明确" in item.item


def _catalog_prompt_sort_key(item: StandardCatalogItem) -> tuple[str, str, bool, str, str]:
    return (
        item.standard_group,
        item.category,
        _is_unspecified_catalog_item(item),
        item.item,
        item.id,
    )


def build_full_catalog_item_selection_prompt(
    consultation_project_name: str,
    classify_subject: str,
    catalog_items: Sequence[StandardCatalogItem],
    item_summary: Sequence[str] | str | None = None,
    context_hints: Sequence[str] | None = None,
) -> str:
    catalog_lines = "\n".join(
        compact_catalog_prompt_label(item)
        for item in sorted(catalog_items, key=_catalog_prompt_sort_key)
    )
    hint_lines = "\n".join(f"- {hint}" for hint in context_hints or [] if hint)
    hint_section = f"\n辅助提示（只作理解文本，不得替代标准目录判断）：\n{hint_lines}\n" if hint_lines else ""

    if isinstance(item_summary, str):
        item_summary_text = item_summary.strip()
    else:
        item_summary_text = "\n".join(
            f"{idx}. {name}"
            for idx, name in enumerate(item_summary or [], start=1)
            if str(name).strip()
        )

    item_summary_section = (
        f"\n清单摘要（只用于辅助判断本次分类对象，不参与缓存 key）：\n{item_summary_text}\n"
        if item_summary_text
        else ""
    )

    return f"""
你是物业工程标准目录分类助手。请读取完整 compact 标准目录，判断“本次分类对象”对应的标准目录。

报审项目名称：{consultation_project_name}
本次分类对象：{classify_subject}
{item_summary_section}
{hint_section}

完整 compact 标准目录：
catalog_id | 标准对象 | 一级分类 | 二级分类 | 目录属性 | 可选状态
{catalog_lines}

分类判断规则：
1. “本次分类对象”是本次要分类的工程对象，必须优先围绕它判断。
2. “报审项目名称”只作为背景信息，不是本次分类对象。
3. “清单摘要”只用于辅助判断本次分类对象实际涉及的维修对象，尤其用于“单项工程-安装”“单位工程”“安装工程”等笼统对象。
4. 如果“本次分类对象”很具体，例如“屋面渗漏维修”“外墙渗漏维修”“防火门维修”“塑胶跑道维修”，应优先依据本次分类对象判断。
5. 如果“本次分类对象”很笼统，例如“单项工程-安装”“单位工程”“安装工程”，应主要依据清单摘要判断。
6. 不要因为报审项目名称中包含多个部位，就把多个部位混在一起分类。
7. 你必须先阅读全部一级分类和二级分类，再从完整标准目录中选择最合适的一个 catalog_id。
8. 只能从完整标准目录中选择 catalog_id，不得创造目录外分类。
9. 如果本次分类对象能对应到标准目录中的共用部位、共用设施设备维修对象，不要返回 OUT_OF_SCOPE。
10. 必须遵守“最具体可匹配原则”：如果“本次分类对象”或“清单摘要”中出现了具体维修对象、设备、构件、材料、系统部件，且完整标准目录中存在对应具体二级目录，必须优先选择具体二级目录。
11. “未明确具体子项”或“内部扩展项”是兜底项，不是泛化首选项。只有当本次分类对象和清单摘要都无法对应任何具体二级目录时，才可以选择。
12. 只有在无法从“本次分类对象”和“清单摘要”判断任何具体二级对象时，才可以选择“未明确具体子项”或“内部扩展项”，并必须设置 needs_review=true。
13. 不得因为一个项目包含多个同一系统下的设备，就直接选择“未明确具体子项”。应该选择最能代表主维修对象的具体二级目录，并在确实存在多个并列对象时，用 secondary_catalog_ids 表示其它具体目录，同时设置 needs_review=true。
14. 如果清单摘要中多个具体对象都属于同一个一级系统，应优先从该一级系统下选择最具体、最核心的二级目录，而不是选择该一级系统的“未明确具体子项”。
15. 如果最终选择“未明确具体子项”或“内部扩展项”，reason 必须说明为什么没有任何具体二级目录适配，例如“未见更具体目录”，并设置 needs_review=true。
16. 如果文本包含咨询、设计、检测、维保、审计等服务词，但同时有明确维修对象，应选择该维修对象，并设置 needs_review=true，不要直接 OUT_OF_SCOPE。
17. 只有完全没有共用部位/共用设施设备维修对象时，才返回 OUT_OF_SCOPE。
18. alias、动作词、复核提示、辅助提示只作为辅助理解，不能直接决定 catalog_id。

复合项目规则：
1. 当前流程已经按 sub_project_id 拆分，本次通常只应输出一个主 catalog_id。
2. 如果“本次分类对象”本身仍然明确包含多个并列维修对象，主对象放 catalog_id，其它对象放 secondary_catalog_ids，并设置 is_composite=true 或 needs_review=true。
3. 如果只是报审项目名称中包含多个对象，但本次分类对象是单一对象，不要设置 is_composite=true。

输出要求：
1. 只能输出 JSON，不要自然语言段落。
2. 不要输出 markdown。
3. 不要输出思考过程。
4. 不要输出 <think>。
5. reason 不超过 40 个汉字。
6. catalog_id 必须是完整标准目录中的 catalog_id，或者 OUT_OF_SCOPE。
7. secondary_catalog_ids 必须只包含完整标准目录中的 catalog_id。
8. 如果 catalog_id=OUT_OF_SCOPE，secondary_catalog_ids 必须为空数组。

同时抽取 project_name_text。

project_name_text 是从“本次分类对象”中抽取出的用于相似项目检索的工程语义文本。
如果“本次分类对象”过于笼统，例如“单项工程-安装”“单位工程”“安装工程”，则可以结合清单摘要抽取工程语义文本。
project_name_text 不是分类理由，不是一级分类/二级分类/维修状态拼接，不要写成目录路径，不要补充输入中没有的信息。
project_name_text 应尽量去掉楼幢号、批次号、项目名前缀等位置或批次信息，保留真正的维修对象语义。
例如：
- “屋面维修工程15幢” -> “屋面维修工程”
- “16幢屋面渗漏维修” -> “屋面渗漏维修”
- “卓越雅苑一期第一批公区维修工程-防火门维修” -> “防火门维修”
- “单项工程-安装”，清单摘要包含监控摄像设备、录像设备、交换机、线缆 -> “监控系统安装”

project_name_text 抽取规则：
{SEMANTIC_PROJECT_TEXT_RULES}

JSON 字段固定为：
{{
  "catalog_id": "...",
  "secondary_catalog_ids": [],
  "is_composite": false,
  "needs_review": false,
  "reason": "...",
  "project_name_text": "..."
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
        project_name_text=str(content.get("project_name_text") or "").strip(),
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
    consultation_project_name: str,
    classify_subject: str,
    catalog_items: Sequence[StandardCatalogItem],
    item_summary: Sequence[str] | str | None = None,
    context_hints: Sequence[str] | None = None,
) -> ItemSelection:
    prompt = build_full_catalog_item_selection_prompt(
        consultation_project_name,
        classify_subject,
        catalog_items,
        item_summary,
        context_hints,
    )
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
        project_name_text="",
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
