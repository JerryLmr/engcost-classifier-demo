import re
from typing import Any


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_unit(unit: Any) -> str:
    text = _cell_text(unit)
    if not text:
        return ""

    compact = re.sub(r"\s+", "", text).lower()
    compact = compact.replace("ｍ", "m")
    compact = compact.replace("$", "")
    compact = compact.replace("{", "").replace("}", "")
    compact = compact.replace("^", "")

    square_units = {
        "m2",
        "m²",
        "㎡",
        "平方米",
        "平米",
        "平方",
        "平",
        "m^{2}",
        "m^2",
    }
    cubic_units = {
        "m3",
        "m³",
        "立方米",
        "m^{3}",
        "m^3",
    }

    if compact in square_units or compact == "m2":
        return "m²"
    if compact in cubic_units or compact == "m3":
        return "m³"
    return re.sub(r"\s+", "", text)
