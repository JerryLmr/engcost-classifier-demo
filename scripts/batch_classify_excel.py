#!/usr/bin/env python3
import argparse
import io
import sys
from pathlib import Path

import openpyxl


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


RESULT_HEADERS = [
    "工程名称",
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
    "file_name",
    "consultation_project_name",
    "renovation_content",
    "sub_item_project_rows",
]

EXCEL_SUFFIXES = {".xlsx", ".xlsm"}
OCR_HEADERS = [
    "file_name",
    "consultation_project_name",
    "renovation_content",
    "sub_item_project_rows",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量处理 Excel 工程标准目录分类文件")
    parser.add_argument("input_paths", nargs="+", help="待处理 Excel 文件；单个目录也可批量处理")
    parser.add_argument(
        "-o",
        "--output",
        help="输出文件或目录。多个输入或目录输入时按输出目录处理",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若输出文件已存在则覆盖",
    )
    parser.add_argument(
        "--include-classified",
        action="store_true",
        help="目录模式默认跳过已带 _分类结果 / _classified 后缀的文件；设置后不跳过",
    )
    return parser.parse_args()


def should_skip_file(path: Path, include_classified: bool) -> bool:
    if path.suffix.lower() not in EXCEL_SUFFIXES:
        return True
    if include_classified:
        return False
    stem = path.stem
    return stem.endswith("_分类结果") or stem.endswith("_classified")


def classified_output_name(input_file: Path) -> str:
    return f"{input_file.stem}_classified.xlsx"


def _bool_text(value: object) -> str:
    return "是" if bool(value) else "否"


def _cell_text(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _load_header_map(worksheet) -> dict[str, int]:
    header_map: dict[str, int] = {}
    for column in range(1, worksheet.max_column + 1):
        header = _cell_text(worksheet.cell(row=1, column=column).value)
        if header and header not in header_map:
            header_map[header] = column
    return header_map


def _get_cell(worksheet, header_map: dict[str, int], row: int, header: str) -> object:
    column = header_map.get(header)
    if not column:
        return None
    return worksheet.cell(row=row, column=column).value


def _get_text(worksheet, header_map: dict[str, int], row: int, header: str) -> str:
    return _cell_text(_get_cell(worksheet, header_map, row, header))


def _build_project_name(project_name: object, consultation_project_name: object, renovation_content: object) -> str:
    existing_project_name = _cell_text(project_name)
    if existing_project_name:
        return existing_project_name
    return " ".join(
        part for part in [_cell_text(consultation_project_name), _cell_text(renovation_content)] if part
    ).strip()


def _detect_project_name_column(header_map: dict[str, int]) -> int | None:
    if "工程名称" in header_map:
        return header_map["工程名称"]
    return None


def _validate_input_headers(header_map: dict[str, int], first_header: object) -> int | None:
    project_name_column = _detect_project_name_column(header_map)
    if project_name_column:
        return project_name_column

    missing_ocr_headers = [header for header in OCR_HEADERS if header not in header_map]
    if not missing_ocr_headers:
        return None

    if first_header is None or str(first_header).strip() == "":
        raise ValueError("第一列表头不能为空")
    raise ValueError(
        "输入 Excel 缺少 工程名称 列；若使用 OCR 原始表，必须包含字段: "
        + ", ".join(OCR_HEADERS)
        + f"。当前缺少: {', '.join(missing_ocr_headers)}"
    )


def _read_ocr_values(worksheet, header_map: dict[str, int], row: int) -> dict[str, object]:
    return {header: _get_cell(worksheet, header_map, row, header) for header in OCR_HEADERS}


def _write_result_row(
    worksheet,
    row: int,
    result: dict[str, object],
    ocr_values: dict[str, object],
) -> None:
    values = [
        result.get("project_name", ""),
        result.get("catalog_id", ""),
        result.get("category", ""),
        result.get("item", ""),
        result.get("repair_status", ""),
        result.get("standard_group", ""),
        _bool_text(result.get("is_composite")) if "is_composite" in result else "",
        " | ".join(result.get("secondary_catalog_labels") or []),
        _bool_text(result.get("is_emergency")) if "is_emergency" in result else "",
        _bool_text(result.get("termite_related")) if "termite_related" in result else "",
        _bool_text(result.get("needs_review")) if "needs_review" in result else "",
        result.get("reason", ""),
        ocr_values.get("file_name"),
        ocr_values.get("consultation_project_name"),
        ocr_values.get("renovation_content"),
        ocr_values.get("sub_item_project_rows"),
    ]
    for column, value in enumerate(values, start=1):
        worksheet.cell(row=row, column=column, value=value)


def classify_workbook(
    path: Path,
    output_path: Path,
    classify_project_func,
    classification_cache: dict[str, dict[str, object]],
) -> tuple[int, int, int, int]:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    source_sheet = workbook.active

    first_header = source_sheet.cell(row=1, column=1).value
    header_map = _load_header_map(source_sheet)
    project_name_column = _validate_input_headers(header_map, first_header)

    output_workbook = openpyxl.Workbook()
    output_sheet = output_workbook.active
    output_sheet.title = "分类结果"
    for column, header in enumerate(RESULT_HEADERS, start=1):
        output_sheet.cell(row=1, column=column, value=header)

    processed = 0
    classify_call_count = 0
    cache_hit_count = 0
    empty_project_name_rows = 0
    output_rows = 0
    output_row = 2
    for source_row in range(2, source_sheet.max_row + 1):
        ocr_values = _read_ocr_values(source_sheet, header_map, source_row)
        if project_name_column:
            project_name = source_sheet.cell(row=source_row, column=project_name_column).value
            project_text = _cell_text(project_name)
        else:
            project_text = _build_project_name(
                None,
                ocr_values.get("consultation_project_name"),
                ocr_values.get("renovation_content"),
            )
        if not project_text:
            empty_project_name_rows += 1
            print(f"[WARN] {path.name}:{source_row} 工程名称为空，保留原始行但跳过分类", flush=True)
            _write_result_row(output_sheet, output_row, {"project_name": ""}, ocr_values)
            output_row += 1
            output_rows += 1
            continue

        if project_text in classification_cache:
            result = classification_cache[project_text]
            cache_hit_count += 1
            print(f"[CACHE] {path.name}:{source_row} {project_text[:80]}", flush=True)
        else:
            print(f"[ROW ] {path.name}:{source_row} {project_text[:80]}", flush=True)

            result = classify_project_func(project_text)
            if result.get("pipeline_status") == "llm_service_error":
                raise RuntimeError(f"LLM 服务连接失败，已停止处理当前文件。失败行: {source_row}")
            classification_cache[project_text] = result
            classify_call_count += 1

            print(
                f"[DONE] {path.name}:{source_row} "
                f"{result.get('catalog_id')} "
                f"{result.get('category')} / {result.get('item')} "
                f"status={result.get('pipeline_status')}",
                flush=True,
            )

        _write_result_row(output_sheet, output_row, result, ocr_values)
        output_row += 1
        output_rows += 1
        processed += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = io.BytesIO()
    output_workbook.save(output)
    output_path.write_bytes(output.getvalue())
    workbook.close()
    return classify_call_count, cache_hit_count, output_rows, empty_project_name_rows


def _is_directory_output(raw_output: str, output_path: Path) -> bool:
    return (
        raw_output.endswith(("/", "\\"))
        or output_path.exists() and output_path.is_dir()
        or (not output_path.exists() and output_path.suffix == "")
    )


def resolve_single_file_output(input_path: Path, output_arg: str | None) -> Path:
    if not output_arg:
        return input_path.with_name(classified_output_name(input_path)).resolve()

    raw_output = output_arg.strip()
    output_path = Path(raw_output).expanduser()
    if _is_directory_output(raw_output, output_path):
        return (output_path / classified_output_name(input_path)).resolve()
    return output_path.resolve()


def _directory_jobs(input_dir: Path, output_arg: str | None, include_classified: bool) -> list[tuple[Path, Path]]:
    output_dir = Path(output_arg).expanduser().resolve() if output_arg else input_dir / "classified_results"
    excel_files = sorted(path for path in input_dir.iterdir() if not should_skip_file(path, include_classified))
    return [(path, output_dir / classified_output_name(path)) for path in excel_files]


def resolve_jobs(input_paths: list[str], output_arg: str | None, include_classified: bool) -> list[tuple[Path, Path]]:
    paths = [Path(raw).expanduser().resolve() for raw in input_paths]
    if len(paths) == 1 and paths[0].is_dir():
        return _directory_jobs(paths[0], output_arg, include_classified)

    if any(path.is_dir() for path in paths):
        raise ValueError("多个输入路径时不支持混用目录；请传多个 Excel 文件或单个目录")

    if len(paths) == 1:
        return [(paths[0], resolve_single_file_output(paths[0], output_arg))]

    if output_arg:
        output_dir = Path(output_arg).expanduser().resolve()
        if output_dir.suffix:
            raise ValueError("多个输入文件时 -o/--output 必须是目录")
    else:
        output_dir = paths[0].parent
    return [(path, output_dir / classified_output_name(path)) for path in paths]


def validate_jobs(jobs: list[tuple[Path, Path]], overwrite: bool) -> None:
    for input_path, output_path in jobs:
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
        output_path.parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
    from classifier.llm_client import check_lmstudio_service  # noqa: E402
    from services.standard_classifier import classify_project_standard  # noqa: E402

    args = parse_args()
    try:
        jobs = resolve_jobs(args.input_paths, args.output, args.include_classified)
        validate_jobs(jobs, args.overwrite)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 1

    if not jobs:
        print("[ERROR] 没有可处理的 Excel 文件")
        return 1

    try:
        check_lmstudio_service()
    except RuntimeError as exc:
        print(f"[ERROR] {exc}")
        return 1

    total_files = 0
    total_processed = 0
    total_classify_calls = 0
    total_cache_hits = 0
    total_output_rows = 0
    total_skipped = 0
    failed_files: list[tuple[str, str]] = []
    classification_cache: dict[str, dict[str, object]] = {}

    print(f"[INFO] 待处理文件数: {len(jobs)}")
    for path, output_path in jobs:
        print(f"[PLAN] {path} -> {output_path}")

    for path, output_path in jobs:
        print(f"[RUN ] {path.name}")
        try:
            (
                classify_call_count,
                cache_hit_count,
                output_rows,
                empty_project_name_rows,
            ) = classify_workbook(path, output_path, classify_project_standard, classification_cache)
            total_files += 1
            processed = classify_call_count + cache_hit_count
            total_processed += processed
            total_classify_calls += classify_call_count
            total_cache_hits += cache_hit_count
            total_output_rows += output_rows
            total_skipped += empty_project_name_rows
            print(
                f"[ OK ] {path.name} -> {output_path.name} "
                f"(分类 {processed} 行, 空工程名保留 {empty_project_name_rows} 行)"
            )
        except Exception as exc:  # noqa: BLE001
            failed_files.append((path.name, str(exc)))
            print(f"[FAIL] {path.name}: {exc}")

    print()
    print(f"[DONE] 成功文件: {total_files}, 失败文件: {len(failed_files)}")
    print(f"[DONE] 总分类行数: {total_processed}, 空工程名保留行数: {total_skipped}")
    print(f"[DONE] 实际分类调用次数: {total_classify_calls}")
    print(f"[DONE] 缓存命中次数: {total_cache_hits}")
    print(f"[DONE] 输出行数: {total_output_rows}")
    if failed_files:
        print("[DONE] 失败明细:")
        for name, error in failed_files:
            print(f"  - {name}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
