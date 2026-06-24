#!/usr/bin/env python3
import argparse
import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import openpyxl


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from classifier.unit_normalizer import normalize_unit  # noqa: E402


PROJECT_HEADERS = [
    "file_name",
    "工程名称",
    "consultation_project_name",
    "renovation_content",
    "catalog_id",
    "一级分类",
    "二级分类",
    "维修状态",
    "标准对象",
    "是否复合工程",
    "复合目录",
    "是否紧急维修",
    "是否白蚁相关",
    "是否建议复核",
    "分类依据",
]

SAMPLE_HEADERS = [
    "source_row_id",
    "item_row_id",
    "file_name",
    "工程名称",
    "consultation_project_name",
    "renovation_content",
    "sub_project_id",
    "catalog_id",
    "一级分类",
    "二级分类",
    "维修状态",
    "标准对象",
    "是否复合工程",
    "复合目录",
    "是否紧急维修",
    "是否白蚁相关",
    "是否建议复核",
    "分类依据",
    "seq",
    "cost_item_name",
    "project_description",
    "unit",
    "unit_normalized",
    "quantity",
    "unit_price",
    "total_price",
    "labor_cost",
    "machinery_cost",
    "labor_unit_price",
    "machinery_unit_price",
    "item_similarity_text",
    "item_context_text",
    "source_json",
]

PARSE_ERROR_HEADERS = [
    "source_row_id",
    "error_type",
    "error_message",
    "raw_sub_item_project_rows",
    "raw_row_summary",
]

NUMERIC_FIELDS = [
    "quantity",
    "unit_price",
    "total_price",
    "labor_cost",
    "machinery_cost",
]

EXCEL_SUFFIXES = {".xlsx", ".xlsm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建已审定清单级样本 Excel")
    parser.add_argument("input_path", help="工程分类后的 Excel 文件")
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="输出清单样本 Excel 路径",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若输出文件已存在则覆盖",
    )
    return parser.parse_args()


def cell_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_header_map(worksheet) -> dict[str, int]:
    header_map: dict[str, int] = {}
    for column in range(1, worksheet.max_column + 1):
        header = cell_text(worksheet.cell(row=1, column=column).value)
        if header and header not in header_map:
            header_map[header] = column
    return header_map


def get_cell(worksheet, header_map: dict[str, int], row: int, header: str) -> Any:
    column = header_map.get(header)
    if not column:
        return None
    return worksheet.cell(row=row, column=column).value


def get_text(worksheet, header_map: dict[str, int], row: int, header: str) -> str:
    return cell_text(get_cell(worksheet, header_map, row, header))


def build_project_name(project_name: str, consultation_project_name: str, renovation_content: str) -> str:
    if project_name:
        return project_name
    return " ".join(part for part in [consultation_project_name, renovation_content] if part).strip()


def compact_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def raw_row_summary(row_values: dict[str, str]) -> str:
    summary = {
        "file_name": row_values.get("file_name", ""),
        "工程名称": row_values.get("工程名称", ""),
        "consultation_project_name": row_values.get("consultation_project_name", ""),
        "renovation_content": row_values.get("renovation_content", ""),
        "catalog_id": row_values.get("catalog_id", ""),
        "一级分类": row_values.get("一级分类", ""),
        "二级分类": row_values.get("二级分类", ""),
    }
    return compact_json(summary)


def make_error(
    source_row_id: int,
    error_type: str,
    error_message: str,
    raw_sub_item_project_rows: Any,
    row_summary: str,
) -> dict[str, Any]:
    return {
        "source_row_id": source_row_id,
        "error_type": error_type,
        "error_message": error_message,
        "raw_sub_item_project_rows": cell_text(raw_sub_item_project_rows),
        "raw_row_summary": row_summary,
    }


def parse_sub_item_rows(raw_value: Any) -> tuple[list[Any] | None, str | None]:
    if raw_value is None or cell_text(raw_value) == "":
        return None, "sub_item_project_rows 为空"
    if isinstance(raw_value, list):
        return raw_value, None
    if isinstance(raw_value, tuple):
        return list(raw_value), None
    if not isinstance(raw_value, str):
        return None, f"sub_item_project_rows 不是 JSON 字符串或数组: {type(raw_value).__name__}"

    raw_text = raw_value.strip()
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        return None, f"JSON 解析失败: {exc.msg} at pos {exc.pos}"

    if not isinstance(parsed, list):
        return None, f"sub_item_project_rows 解析后不是数组: {type(parsed).__name__}"
    return parsed, None


def parse_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    normalized = (
        text.replace(",", "")
        .replace("，", "")
        .replace("￥", "")
        .replace("¥", "")
        .replace("$", "")
        .replace("元", "")
        .strip()
    )
    normalized = re.sub(r"\s+", "", normalized)
    if not normalized:
        return None

    try:
        return float(Decimal(normalized))
    except (InvalidOperation, ValueError):
        return None


def divide_or_none(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def clean_text(value: Any) -> str:
    text = cell_text(value)
    if not text:
        return ""

    lines: list[str] = []
    for line in re.split(r"[\r\n]+", text):
        stripped = line.strip()
        stripped = re.sub(r"^\s*\d+\s*(?:、|\.(?!\d))\s*", "", stripped)
        if stripped:
            lines.append(stripped)

    cleaned = " ".join(lines)
    cleaned = re.sub(r"(?:(?<=\s)|^)\d+\s*(?:、|\.(?!\d))\s*", "", cleaned)
    cleaned = re.sub(r"(?i)(\d+(?:\.\d+)?\s*mm)(sbs)", r"\1 \2", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def add_numeric_parse_errors(
    errors: list[dict[str, Any]],
    source_row_id: int,
    seq_text: str,
    item: dict[str, Any],
    numeric_values: dict[str, float | None],
    raw_sub_item_rows: Any,
    row_summary: str,
) -> None:
    for field, parsed_value in numeric_values.items():
        raw_value = item.get(field)
        if parsed_value is None and raw_value is not None and cell_text(raw_value) != "":
            errors.append(
                make_error(
                    source_row_id,
                    "invalid_numeric_field",
                    f"数值字段无法解析: seq={seq_text}, field={field}, value={raw_value}",
                    raw_sub_item_rows,
                    row_summary,
                )
            )

    if numeric_values.get("unit_price") is None:
        errors.append(
            make_error(
                source_row_id,
                "missing_unit_price",
                f"清单行缺少或无法解析 unit_price: seq={seq_text}",
                raw_sub_item_rows,
                row_summary,
            )
        )


def join_text(parts: list[str], separator: str) -> str:
    return separator.join(part for part in parts if part)


def build_samples(input_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    workbook = openpyxl.load_workbook(input_path, read_only=True, data_only=True)
    worksheet = workbook.active
    header_map = load_header_map(worksheet)

    if "sub_item_project_rows" not in header_map:
        raise ValueError("输入 Excel 缺少 sub_item_project_rows 列")

    samples: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for source_row_id in range(2, worksheet.max_row + 1):
        raw_sub_item_rows = get_cell(worksheet, header_map, source_row_id, "sub_item_project_rows")
        row_values = {
            header: get_text(worksheet, header_map, source_row_id, header)
            for header in PROJECT_HEADERS
        }
        row_values["工程名称"] = build_project_name(
            row_values["工程名称"],
            row_values["consultation_project_name"],
            row_values["renovation_content"],
        )
        row_summary = raw_row_summary(row_values)

        item_rows, parse_error = parse_sub_item_rows(raw_sub_item_rows)
        if parse_error:
            errors.append(
                make_error(
                    source_row_id,
                    "invalid_sub_item_project_rows",
                    parse_error,
                    raw_sub_item_rows,
                    row_summary,
                )
            )
            continue

        if not item_rows:
            errors.append(
                make_error(
                    source_row_id,
                    "empty_sub_item_project_rows",
                    "sub_item_project_rows 数组为空",
                    raw_sub_item_rows,
                    row_summary,
                )
            )
            continue

        for index, item in enumerate(item_rows, start=1):
            if not isinstance(item, dict):
                errors.append(
                    make_error(
                        source_row_id,
                        "invalid_item_row",
                        f"清单行不是对象: index={index}, type={type(item).__name__}",
                        raw_sub_item_rows,
                        row_summary,
                    )
                )
                continue

            seq = item.get("seq")
            if seq is None or cell_text(seq) == "":
                seq = index
                errors.append(
                    make_error(
                        source_row_id,
                        "missing_seq",
                        f"清单行缺少 seq，已使用数组内序号: {index}",
                        raw_sub_item_rows,
                        row_summary,
                    )
                )
            seq_text = cell_text(seq)

            cost_item_name = cell_text(item.get("cost_item_name") or item.get("project_name"))
            if not cost_item_name:
                errors.append(
                    make_error(
                        source_row_id,
                        "missing_cost_item_name",
                        f"清单行缺少 cost_item_name/project_name: seq={seq_text}",
                        raw_sub_item_rows,
                        row_summary,
                    )
                )

            project_description = cell_text(item.get("project_description"))
            cleaned_description = clean_text(project_description)
            unit = cell_text(item.get("unit"))
            numeric_values = {field: parse_number(item.get(field)) for field in NUMERIC_FIELDS}
            add_numeric_parse_errors(
                errors,
                source_row_id,
                seq_text,
                item,
                numeric_values,
                raw_sub_item_rows,
                row_summary,
            )

            sample = {
                "source_row_id": source_row_id,
                "item_row_id": f"{source_row_id}-{seq_text}",
                **row_values,
                "sub_project_id": cell_text(item.get("sub_project_id")),
                "seq": seq,
                "cost_item_name": cost_item_name,
                "project_description": project_description,
                "unit": unit,
                "unit_normalized": normalize_unit(unit),
                "quantity": numeric_values["quantity"],
                "unit_price": numeric_values["unit_price"],
                "total_price": numeric_values["total_price"],
                "labor_cost": numeric_values["labor_cost"],
                "machinery_cost": numeric_values["machinery_cost"],
                "labor_unit_price": divide_or_none(numeric_values["labor_cost"], numeric_values["quantity"]),
                "machinery_unit_price": divide_or_none(
                    numeric_values["machinery_cost"],
                    numeric_values["quantity"],
                ),
                "item_similarity_text": join_text([cost_item_name, cleaned_description], "；"),
                "item_context_text": join_text(
                    [
                        row_values["工程名称"],
                        cell_text(item.get("sub_project_id")),
                        cost_item_name,
                        cleaned_description,
                    ],
                    " / ",
                ),
                "source_json": compact_json(item),
            }
            samples.append(sample)

    workbook.close()
    return samples, errors


def append_dict_rows(worksheet, headers: list[str], rows: list[dict[str, Any]]) -> None:
    worksheet.append(headers)
    for row in rows:
        worksheet.append([row.get(header) for header in headers])


def write_workbook(output_path: Path, samples: list[dict[str, Any]], errors: list[dict[str, Any]]) -> None:
    workbook = openpyxl.Workbook()
    samples_sheet = workbook.active
    samples_sheet.title = "samples"
    append_dict_rows(samples_sheet, SAMPLE_HEADERS, samples)

    errors_sheet = workbook.create_sheet("parse_errors")
    append_dict_rows(errors_sheet, PARSE_ERROR_HEADERS, errors)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)


def validate_paths(input_path: Path, output_path: Path, overwrite: bool) -> None:
    if not input_path.exists():
        raise ValueError(f"输入文件不存在: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"输入路径不是文件: {input_path}")
    if input_path.suffix.lower() not in EXCEL_SUFFIXES:
        raise ValueError(f"输入文件不是 Excel: {input_path}")
    if output_path.exists() and output_path.is_dir():
        raise ValueError(f"输出路径是目录，不是文件: {output_path}")
    if output_path.exists() and not overwrite:
        raise ValueError(f"输出已存在，请加 --overwrite 或更换输出路径: {output_path}")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    try:
        validate_paths(input_path, output_path, args.overwrite)
        samples, errors = build_samples(input_path)
        write_workbook(output_path, samples, errors)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 1

    print(f"[DONE] 输入文件: {input_path}")
    print(f"[DONE] 输出文件: {output_path}")
    print(f"[DONE] samples: {len(samples)}")
    print(f"[DONE] parse_errors: {len(errors)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
