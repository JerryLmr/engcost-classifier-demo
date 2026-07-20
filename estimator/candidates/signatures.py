from __future__ import annotations

import re
import unicodedata
import json
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping

import pandas as pd


def cell_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def truncate_text(value: Any, limit: int) -> str:
    text = cell_text(value)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip()


def normalized_unit(value: Any) -> str:
    text = cell_text(value).lower()
    text = text.replace("㎡", "m²").replace("平方米", "m²").replace("平方", "m²")
    text = re.sub(r"m\s*2|m\^2", "m²", text)
    text = text.replace("毫米", "mm")
    return text.strip()


def numeric_or_none(value: Any) -> float | None:
    if value is None or cell_text(value) == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(number):
        return None
    return number


def join_non_empty(values: list[Any], limit: int | None = None) -> str:
    texts: list[str] = []
    for value in values:
        text = cell_text(value)
        if text and text not in texts:
            texts.append(text)
        if limit is not None and len(texts) >= limit:
            break
    return "；".join(texts)


def append_warning(warnings: list[str] | None, code: str) -> None:
    if warnings is not None and code not in warnings:
        warnings.append(code)


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def estimated_tokens(text: str) -> int:
    return max(1, int(len(text) / 2))


def trace_row(
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
        "stage": stage, "purpose": purpose, "prompt": prompt,
        "raw_response": raw_response,
        "parsed_status": "success" if success else "failed",
        "error_message": error, "scenario_count": scenario_count,
        "scenario_item_count": scenario_item_count,
        "prompt_chars": len(prompt),
        "estimated_tokens": estimated_tokens(prompt) if prompt else "",
        "max_tokens": max_tokens, "input_summary": input_summary,
        "prompt_tokens": usage.get("prompt_tokens", ""),
        "completion_tokens": usage.get("completion_tokens", ""),
        "total_tokens": usage.get("total_tokens", ""),
    }


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _format_number(value: str) -> str:
    try:
        number = Decimal(value)
    except InvalidOperation:
        return value
    return format(number.normalize(), "f")


def _normalize_basic(value: Any) -> str:
    text = unicodedata.normalize("NFKC", _text(value)).lower()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("，", ",").replace("。", ".").replace("；", ";").replace("：", ":")
    text = text.replace("、", ",").replace("｜", "|")
    text = re.sub(r"[－—–−]", "-", text)
    text = re.sub(r"[～〜∼﹋]", "~", text)
    return text


def _normalize_thickness(text: str) -> str:
    number = r"(\d+(?:\.\d+)?)"
    text = re.sub(rf"厚\s*{number}\s*\(?\s*(?:mm|毫米)\s*\)?", lambda m: f"{_format_number(m.group(1))}mm厚", text)
    text = re.sub(rf"(?<![\d*x×]){number}\s*(?:mm|毫米)\s*厚", lambda m: f"{_format_number(m.group(1))}mm厚", text)
    text = re.sub(
        rf"(?<![\d*x×]){number}\s*(?:mm|毫米)(?=\s*(?:sbs|防水|涂料|卷材|弹性|自粘|聚氨酯|聚合物|$))",
        lambda m: f"{_format_number(m.group(1))}mm厚",
        text,
    )
    text = re.sub(rf"{number}\s*厚(?!度)", lambda m: f"{_format_number(m.group(1))}mm厚", text)
    return text


def _normalize_confirmed_process_terms(text: str) -> str:
    text = re.sub(r"(?:弹性体改性沥青|弹性改性沥青|sbs\s*改性沥青)", "sbs", text)
    text = re.sub(r"sbs\s*(?:沥青)?防水卷材", "sbs防水卷材", text)
    text = re.sub(r"sbs\s*防水沥青卷材", "sbs防水卷材", text)
    text = re.sub(r"(?:sbs){2,}", "sbs", text)
    text = text.replace("自粘性", "自粘").replace("单组份", "单组分")
    return text


def _remove_list_markers(text: str) -> str:
    # A dot followed by a digit is a decimal point, not a list marker.
    marker = r"(?<!\d)\d+\s*(?:[.,)]|\))\s*(?!\d)"
    text = re.sub(marker, "", text)
    # OCR commonly produces a leading list number before a decimal thickness: 1.3.0mm.
    text = re.sub(r"(?<!\d)\d+\.(?=\d+\.\d+\s*(?:mm|毫米|厚))", "", text)
    return text


def _remove_template_prefixes(text: str) -> str:
    return re.sub(
        r"(?:^|\n)(?:防水)?卷材品种[,、]规格(?:[,、]厚度)?[:：]?",
        "",
        text,
    )


def normalize_cost_item_name(value: Any) -> str:
    text = _normalize_basic(value)
    return re.sub(r"\s+", "", text)


def normalize_project_description(value: Any) -> str:
    text = _normalize_basic(value)
    text = _remove_template_prefixes(text)
    text = _remove_list_markers(text)
    text = _normalize_thickness(text)
    text = _normalize_confirmed_process_terms(text)
    text = re.sub(r"\s+", "", text)
    return text


def normalize_unit(value: Any) -> str:
    text = re.sub(r"\s+", "", _normalize_basic(value))
    if re.fullmatch(r"(?:㎡|m2|m\^2|m\^\{2\}|m²|平方米|平方|平)", text):
        return "m²"
    return text


def build_normalized_signature(row: Mapping[str, Any]) -> str:
    unit = _text(row.get("unit_normalized")) or _text(row.get("unit"))
    return " | ".join(
        [
            normalize_cost_item_name(row.get("cost_item_name")),
            normalize_project_description(row.get("project_description")),
            normalize_unit(unit),
        ]
    )
