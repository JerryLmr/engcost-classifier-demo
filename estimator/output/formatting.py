from __future__ import annotations

from pathlib import Path
from typing import Any

from estimator.candidates.signatures import cell_text


TEXT_IDENTIFIER_COLUMNS = {
    "scenario_id", "display_id", "family_id", "project_package_id", "project_key",
    "item_key", "stable_sample_id", "source_ref", "catalog_id", "batch_id",
    "source_row_id", "item_row_id", "project_code",
}

TEXT_VALUE_COLUMNS = {
    "方案说明", "主要施工内容", "待现场确认事项", "工程量来源",
    "工程量说明", "来源样本", "价格证据family",
}

INTEGER_COLUMNS = {
    "序号", "final_item_position", "rank", "selection_rank", "support_rank",
    "family_count", "本次召回family数量", "默认family本次召回样本数",
    "默认family本次召回工程包数", "本次召回样本数", "本次召回工程包数",
    "retrieval_item_count", "retrieval_package_count", "本次召回清单行数",
    "本次召回支持度排序", "item_count", "page_no", "prompt_chars",
    "estimated_tokens", "max_tokens", "prompt_tokens", "completion_tokens", "total_tokens",
}

DECIMAL_VALUE_COLUMNS = {"quantity", "unit_price", "labor_unit_price", "machinery_unit_price"}

AMOUNT_VALUE_COLUMNS = {
    "total_price", "其中包含人工费P10", "其中包含人工费中位数", "其中包含人工费P90",
    "其中包含机械费P10", "其中包含机械费中位数", "其中包含机械费P90",
}

AMOUNT_WIDTH_COLUMNS = {
    "total_price", "合价P10", "合价中位数", "合价P90", *AMOUNT_VALUE_COLUMNS,
}


def is_text_identifier_column(column: Any) -> bool:
    name = cell_text(column)
    return name in TEXT_IDENTIFIER_COLUMNS or name.endswith("_id") or name.endswith("_ids")


def excel_number_format(column: Any) -> str | None:
    name = cell_text(column)
    lower_name = name.lower()
    if is_text_identifier_column(name):
        return "@"
    if name in TEXT_VALUE_COLUMNS:
        return None
    if name == "package_evidence_weight":
        return "0.000000"
    if "similarity" in lower_name or "相似度" in name or lower_name.endswith("_ratio") or "比例" in name:
        return "0.0000"
    if (
        name in INTEGER_COLUMNS
        or lower_name.endswith("_count")
        or lower_name.endswith("_rank")
        or "样本数" in name
        or "工程包数" in name
        or ("数量" in name and "工程量" not in name)
    ):
        return "0"
    if name in AMOUNT_VALUE_COLUMNS or "金额" in name or "合价" in name:
        return "#,##0.00"
    if name in DECIMAL_VALUE_COLUMNS or "工程量" in name or "单价" in name:
        return "0.00"
    return None


def excel_min_column_width(column: Any) -> float | None:
    name = cell_text(column)
    if name in AMOUNT_WIDTH_COLUMNS:
        return 15.0
    if name in DECIMAL_VALUE_COLUMNS or "工程量" in name or "单价" in name:
        return 12.0
    return None


def apply_workbook_style(path: Path) -> None:
    try:
        import openpyxl
        from openpyxl.styles import Alignment, Font
    except ImportError:
        return

    workbook = openpyxl.load_workbook(path)
    for worksheet in workbook.worksheets:
        worksheet.freeze_panes = None
        column_formats = {cell.column: excel_number_format(cell.value) for cell in worksheet[1]}
        for cell in worksheet[1]:
            min_width = excel_min_column_width(cell.value)
            if min_width is None:
                continue
            column_letter = cell.column_letter
            current_width = worksheet.column_dimensions[column_letter].width or 0
            worksheet.column_dimensions[column_letter].width = max(current_width, min_width)
        for cell in worksheet[1]:
            cell.font = Font(bold=True)
            cell.alignment = Alignment(wrap_text=False, vertical="top")
        for row in worksheet.iter_rows():
            for cell in row:
                cell.alignment = Alignment(wrap_text=False, vertical="top")
                number_format = column_formats.get(cell.column)
                if number_format:
                    cell.number_format = number_format
    workbook.save(path)
    workbook.close()
