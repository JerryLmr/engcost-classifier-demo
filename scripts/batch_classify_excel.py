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

from services.classifier import classify_text  # noqa: E402


RESULT_HEADERS = [
    "一级分类",
    "二级分类",
    "分类方式",
    "分类依据",
    "是否复合工程",
    "是否建议复核",
    "结构类型",
    "复合原因",
    "候选分类",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量处理目录中的 Excel 工程分类文件")
    parser.add_argument("input_dir", help="待处理 Excel 所在目录")
    parser.add_argument(
        "-o",
        "--output-dir",
        help="结果输出目录，默认输出到 input_dir/classified_results",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若输出文件已存在则覆盖",
    )
    parser.add_argument(
        "--include-classified",
        action="store_true",
        help="默认会跳过已带 _分类结果 / _classified 后缀的文件；设置后不跳过",
    )
    return parser.parse_args()


def should_skip_file(path: Path, include_classified: bool) -> bool:
    if path.suffix.lower() not in {".xlsx", ".xlsm"}:
        return True
    if include_classified:
        return False
    stem = path.stem
    return stem.endswith("_分类结果") or stem.endswith("_classified")


def classify_workbook(path: Path, output_path: Path) -> tuple[int, int]:
    workbook = openpyxl.load_workbook(path)
    worksheet = workbook.active

    headers = [worksheet.cell(row=1, column=i).value for i in range(1, worksheet.max_column + 1)]
    if not headers or headers[0] is None:
        raise ValueError("第一列表头不能为空")

    result_start_col = worksheet.max_column + 1
    for offset, header in enumerate(RESULT_HEADERS):
        worksheet.cell(row=1, column=result_start_col + offset, value=header)

    processed = 0
    skipped = 0
    for row in range(2, worksheet.max_row + 1):
        project_name = worksheet.cell(row=row, column=1).value
        if project_name is None or str(project_name).strip() == "":
            skipped += 1
            continue

        result = classify_text(str(project_name))
        worksheet.cell(row=row, column=result_start_col, value=result["level1"])
        worksheet.cell(row=row, column=result_start_col + 1, value=result["level2"])
        worksheet.cell(row=row, column=result_start_col + 2, value=result["method"])
        worksheet.cell(row=row, column=result_start_col + 3, value=result["reason"])
        worksheet.cell(row=row, column=result_start_col + 4, value="是" if result["is_composite"] else "否")
        worksheet.cell(row=row, column=result_start_col + 5, value="是" if result["needs_review"] else "否")
        worksheet.cell(row=row, column=result_start_col + 6, value=result["structure_type"])
        worksheet.cell(row=row, column=result_start_col + 7, value=result["composite_reason"] or "")
        worksheet.cell(row=row, column=result_start_col + 8, value=" | ".join(result["secondary_candidates"]))
        processed += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = io.BytesIO()
    workbook.save(output)
    output_path.write_bytes(output.getvalue())
    return processed, skipped


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[ERROR] 输入目录不存在: {input_dir}")
        return 1

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else input_dir / "classified_results"
    )

    excel_files = sorted(
        path for path in input_dir.iterdir() if not should_skip_file(path, args.include_classified)
    )
    if not excel_files:
        print(f"[ERROR] 目录中没有可处理的 Excel 文件: {input_dir}")
        return 1

    total_files = 0
    total_processed = 0
    total_skipped = 0
    failed_files: list[tuple[str, str]] = []

    print(f"[INFO] 输入目录: {input_dir}")
    print(f"[INFO] 输出目录: {output_dir}")
    print(f"[INFO] 待处理文件数: {len(excel_files)}")

    for path in excel_files:
        output_path = output_dir / f"{path.stem}_分类结果.xlsx"
        if output_path.exists() and not args.overwrite:
            print(f"[SKIP] 输出已存在，跳过: {output_path.name}")
            continue

        print(f"[RUN ] {path.name}")
        try:
            processed, skipped = classify_workbook(path, output_path)
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
