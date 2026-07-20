from __future__ import annotations

import re
import unicodedata
from decimal import Decimal, InvalidOperation
from typing import Any, Mapping


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
