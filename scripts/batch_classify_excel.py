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
    "标准对象",
    "一级分类",
    "二级分类",
    "维修状态",
    "是否复合工程",
    "复合候选目录",
    "是否紧急维修",
    "是否白蚁相关",
    "是否建议复核",
    "候选目录",
    "分类依据",
    "原始文件",
    "原始行号",
]

EXCEL_SUFFIXES = {".xlsx", ".xlsm"}


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


def _write_result_row(worksheet, row: int, result: dict[str, object], source_file: Path, source_row: int) -> None:
    values = [
        result["project_name"],
        result["catalog_id"],
        result["standard_group"],
        result["category"],
        result["item"],
        result["repair_status"],
        _bool_text(result["is_composite"]),
        " | ".join(result["secondary_catalog_labels"]),
        _bool_text(result["is_emergency"]),
        _bool_text(result["termite_related"]),
        _bool_text(result["needs_review"]),
        " | ".join(result["candidate_labels"]),
        result["reason"],
        source_file.name,
        source_row,
    ]
    for column, value in enumerate(values, start=1):
        worksheet.cell(row=row, column=column, value=value)


def classify_workbook(path: Path, output_path: Path, classify_project_func) -> tuple[int, int]:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    source_sheet = workbook.active

    first_header = source_sheet.cell(row=1, column=1).value
    if first_header is None or str(first_header).strip() == "":
        raise ValueError("第一列表头不能为空")

    output_workbook = openpyxl.Workbook()
    output_sheet = output_workbook.active
    output_sheet.title = "分类结果"
    for column, header in enumerate(RESULT_HEADERS, start=1):
        output_sheet.cell(row=1, column=column, value=header)

    processed = 0
    skipped = 0
    output_row = 2
    consecutive_llm_service_errors = 0
    max_consecutive_llm_service_errors = 3
    for source_row in range(2, source_sheet.max_row + 1):
        project_name = source_sheet.cell(row=source_row, column=1).value
        if project_name is None or str(project_name).strip() == "":
            skipped += 1
            continue

        project_text = str(project_name).strip()
        print(f"[ROW ] {path.name}:{source_row} {project_text[:80]}", flush=True)

        result = classify_project_func(project_text)

        print(
            f"[DONE] {path.name}:{source_row} "
            f"{result.get('catalog_id')} "
            f"{result.get('category')} / {result.get('item')} "
            f"status={result.get('pipeline_status')}",
            flush=True,
        )

        _write_result_row(output_sheet, output_row, result, path, source_row)
        output_row += 1
        processed += 1

        if result.get("pipeline_status") == "llm_service_error":
            consecutive_llm_service_errors += 1
        else:
            consecutive_llm_service_errors = 0

        if consecutive_llm_service_errors >= max_consecutive_llm_service_errors:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output = io.BytesIO()
            output_workbook.save(output)
            output_path.write_bytes(output.getvalue())
            raise RuntimeError(
                f"连续 {max_consecutive_llm_service_errors} 行 LLM 服务连接失败，"
                f"已停止处理当前文件。最后失败行: {source_row}。"
                f"已保存部分结果: {output_path}"
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = io.BytesIO()
    output_workbook.save(output)
    output_path.write_bytes(output.getvalue())
    return processed, skipped


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

    total_files = 0
    total_processed = 0
    total_skipped = 0
    failed_files: list[tuple[str, str]] = []

    print(f"[INFO] 待处理文件数: {len(jobs)}")
    for path, output_path in jobs:
        print(f"[PLAN] {path} -> {output_path}")

    for path, output_path in jobs:
        print(f"[RUN ] {path.name}")
        try:
            processed, skipped = classify_workbook(path, output_path, classify_project_standard)
            total_files += 1
            total_processed += processed
            total_skipped += skipped
            print(
                f"[ OK ] {path.name} -> {output_path.name} "
                f"(处理 {processed} 行, 跳过 {skipped} 行空值)"
            )
        except Exception as exc:  # noqa: BLE001
            failed_files.append((path.name, str(exc)))
            print(f"[FAIL] {path.name}: {exc}")

    print()
    print(f"[DONE] 成功文件: {total_files}, 失败文件: {len(failed_files)}")
    print(f"[DONE] 总处理行数: {total_processed}, 总跳过空行: {total_skipped}")
    if failed_files:
        print("[DONE] 失败明细:")
        for name, error in failed_files:
            print(f"  - {name}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
