from __future__ import annotations

import calendar
import sys
from datetime import date
from typing import Any, Callable

import pandas as pd

from estimator.paths import CLASSIFIER_BACKEND_DIR
from estimator.query_models import QueryRewrite
from estimator.retrieval.constraints import validate_query_constraints

if str(CLASSIFIER_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSIFIER_BACKEND_DIR))

from classifier.llm_client import LLMServiceError, request_llm_json  # noqa: E402


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _estimated_tokens(text: str) -> int:
    return max(1, int(len(text) / 2))


def _trace_row(
    stage: str,
    purpose: str,
    success: bool,
    error: str = "",
    prompt: str = "",
    max_tokens: int | str = "",
    input_summary: str = "",
    usage: dict[str, Any] | None = None,
    raw_response: str = "",
    scenario_count: int | str = "",
    scenario_item_count: int | str = "",
) -> dict[str, Any]:
    usage = usage or {}
    return {
        "stage": stage,
        "purpose": purpose,
        "prompt": prompt,
        "raw_response": raw_response,
        "parsed_status": "success" if success else "failed",
        "error_message": error,
        "scenario_count": scenario_count,
        "scenario_item_count": scenario_item_count,
        "prompt_chars": len(prompt),
        "estimated_tokens": _estimated_tokens(prompt) if prompt else "",
        "max_tokens": max_tokens,
        "input_summary": input_summary,
        "prompt_tokens": usage.get("prompt_tokens", ""),
        "completion_tokens": usage.get("completion_tokens", ""),
        "total_tokens": usage.get("total_tokens", ""),
    }


def shift_months(value: date, months: int) -> date:
    month_index = value.year * 12 + value.month - 1 + months
    year, zero_based_month = divmod(month_index, 12)
    month = zero_based_month + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def build_query_rewrite_prompt(query: str, current_date: date | None = None) -> str:
    current_date = current_date or date.today()
    current_date_text = current_date.isoformat()
    return f"""
你是维修工程需求解析和 embedding query rewrite 助手。请把用户原始需求解析为严格 JSON object。

只能输出 JSON object，不要 Markdown，不要解释，不要建议清单，不要计算价格。
当前日期：{current_date_text}

输出格式：
{{
  "project_package_query_text": "",
  "item_query_text": "",
  "location": "",
  "start_date": "",
  "end_date": ""
}}
不要增加其他字段。

当前 embedding 结构：
1. project_package_text 由“工程名称、project_name_text、cost_item_name 去重列表”组成。
   project_package_query_text 用于匹配相似历史工程包，应描述用户明确表达或直接相关的维修工程场景，保持短检索 query，不要预设建议清单、前置项、措施项或替代工艺。
2. item_retrieval_text 由“cost_item_name、project_description、unit_normalized”组成。
   item_query_text 用于匹配相似清单行，应贴近用户明确表达的维修对象、材料规格和做法，不要扩展未明确发生的清单项。
3. item_query_text 必须非空。如果用户问得很粗，也输出宽泛 item query，不要留空。
4. location 表示项目所属的标准地级行政区域。普通地级市必须输出“省级行政区 + 地级市”，例如“浙江省嘉兴市”“江苏省苏州市”“四川省成都市”；直辖市只输出“北京市”“上海市”“天津市”“重庆市”。
5. 用户未提出地域限制时 location 输出空字符串。不要输出简称、县、区或镇；县级行政区所属地级行政区明确时可输出标准地级区域，不确定时不要猜测。
6. start_date 和 end_date 只能是 YYYY-MM-DD 或空字符串。所有相对时间以当前日期 {current_date_text} 为基准转换为绝对日期。
7. “最近一年”“一年内”向前推 12 个月；“最近半年”向前推 6 个月；“最近三个月”向前推 3 个月，end_date 均为当前日期。
8. 整年使用当年 01-01 至 12-31；整月使用当月首日至末日；月份区间使用首月首日至末月末日；某日以后截至当前日期；“截至某日”只填写 end_date。
9. 用户未提出时间约束时 start_date 和 end_date 均输出空字符串，不输出相对时间自然语言。
10. 不扩展用户未明确提出的清单项，不输出数量分析、材料列表、不确定性、方案建议、价格或施工清单。

示例：
用户：屋面漏水，想做3mm SBS防水，面积大概500平
输出：{{"project_package_query_text":"屋面漏水维修工程 屋面防水维修 3mm SBS防水","item_query_text":"屋面卷材防水 3mm SBS防水卷材","location":"","start_date":"","end_date":""}}

用户：屋面漏水帮我估价
输出：{{"project_package_query_text":"屋面漏水维修工程 屋面防水维修","item_query_text":"屋面防水 防水层维修","location":"","start_date":"","end_date":""}}

用户：屋面漏水，参考嘉兴一年内的造价
输出：{{"project_package_query_text":"屋面漏水维修工程 屋面防水维修","item_query_text":"屋面防水 防水层维修","location":"浙江省嘉兴市","start_date":"{shift_months(current_date, -12).isoformat()}","end_date":"{current_date_text}"}}

用户：参考上海市2025年3月的消防报警主机更换造价
输出：{{"project_package_query_text":"消防报警主机更换工程","item_query_text":"消防报警主机更换","location":"上海市","start_date":"2025-03-01","end_date":"2025-03-31"}}

用户需求：{query}
""".strip()


def fallback_query_rewrite(query: str, note: str) -> QueryRewrite:
    return QueryRewrite(
        raw_query=query,
        project_package_query_text=query,
        item_query_text=query,
        location="",
        start_date="",
        end_date="",
        notes=[note],
        success=False,
    )


def query_rewrite_for_embedding(
    query: str,
    current_date: date | None = None,
    *,
    request_json: Callable[..., Any] | None = None,
    trace_factory: Callable[..., dict[str, Any]] | None = None,
) -> tuple[QueryRewrite, dict[str, Any]]:
    prompt = build_query_rewrite_prompt(query, current_date=current_date)
    max_tokens = 512
    request_json = request_json or request_llm_json
    trace_factory = trace_factory or _trace_row
    try:
        result = request_json(
            prompt,
            max_tokens=max_tokens,
            system_prompt="你只输出一个 JSON object，不输出解释、Markdown 或思考过程。",
        )
    except (LLMServiceError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        rewrite = fallback_query_rewrite(query, f"LLM query rewrite 失败，已回退为原始 query: {exc}")
        return rewrite, trace_factory(
            "query_rewrite_for_embedding",
            "生成 project_package_query_text 和 item_query_text",
            False,
            error=str(exc),
            prompt=prompt,
            max_tokens=max_tokens,
            input_summary=query,
        )

    notes: list[str] = []
    package_text = _cell_text(result.get("project_package_query_text")) if isinstance(result, dict) else ""
    item_text = _cell_text(result.get("item_query_text")) if isinstance(result, dict) else ""
    location, start_date, end_date, constraint_notes = validate_query_constraints(
        result.get("location") if isinstance(result, dict) else "",
        result.get("start_date") if isinstance(result, dict) else "",
        result.get("end_date") if isinstance(result, dict) else "",
    )
    notes.extend(constraint_notes)
    if not package_text:
        package_text = query
        notes.append("project_package_query_text 为空，已回退为原始 query")
    if not item_text:
        item_text = package_text or query
        notes.append("item_query_text 为空，已回退为 project_package_query_text 或原始 query")
    rewrite = QueryRewrite(
        raw_query=query,
        project_package_query_text=package_text,
        item_query_text=item_text,
        location=location,
        start_date=start_date,
        end_date=end_date,
        notes=notes,
        success=True,
    )
    return rewrite, trace_factory(
        "query_rewrite_for_embedding",
        "生成 project_package_query_text 和 item_query_text",
        True,
        prompt=prompt,
        max_tokens=max_tokens,
        input_summary=query,
    )
