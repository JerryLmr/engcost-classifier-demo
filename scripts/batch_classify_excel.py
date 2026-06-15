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
    "一级分类",
    "二级分类",
    "三级分类",
    "具体细项",
    "分类方式",
    "置信度",
    "匹配类型",
    "是否建议复核",
    "候选目录ID",
    "候选目录",
    "候选细项",
    "分类依据",
]

EXCEL_SUFFIXES = {".xlsx", ".xlsm"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量处理 Excel 工程三级目录分类文件")
    parser.add_argument("input_path", help="待处理 Excel 文件或目录")
    parser.add_argument(
        "-o",
        "--output",
        help="输出文件或目录。输入为目录时默认输出到 input/classified_results",
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
    parser.add_argument(
        "--mode",
        choices=("auto", "llm", "rule"),
        default="auto",
        help=(
            "分类模式：auto=规则优先并用 LLM 兜底；llm=直接用完整 catalog 调 LLM；"
            "rule=只跑规则，规则无结果时默认兜底。单文件示例："
            "python scripts/batch_classify_excel.py input.xlsx -o output.xlsx --overwrite --mode llm"
        ),
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
    return f"{input_file.stem}_分类结果.xlsx"


def classify_workbook(path: Path, output_path: Path, classify_text_func) -> tuple[int, int]:
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

        result = classify_text_func(str(project_name))
        worksheet.cell(row=row, column=result_start_col, value=result["level1"])
        worksheet.cell(row=row, column=result_start_col + 1, value=result["level2"])
        worksheet.cell(row=row, column=result_start_col + 2, value=result["level3_item"])
        worksheet.cell(row=row, column=result_start_col + 3, value=" | ".join(result["matched_level3_items"]))
        worksheet.cell(row=row, column=result_start_col + 4, value=result["method"])
        worksheet.cell(row=row, column=result_start_col + 5, value=result["confidence"])
        worksheet.cell(row=row, column=result_start_col + 6, value=result["match_type"])
        worksheet.cell(row=row, column=result_start_col + 7, value="是" if result["needs_review"] else "否")
        worksheet.cell(row=row, column=result_start_col + 8, value=" | ".join(result["candidate_ids"]))
        worksheet.cell(row=row, column=result_start_col + 9, value=" | ".join(result["candidate_labels"]))
        worksheet.cell(row=row, column=result_start_col + 10, value=" | ".join(result["candidate_level3_items"]))
        worksheet.cell(row=row, column=result_start_col + 11, value=result["reason"])
        processed += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = io.BytesIO()
    workbook.save(output)
    output_path.write_bytes(output.getvalue())
    return processed, skipped


def resolve_single_file_output(input_path: Path, output_arg: str | None) -> Path:
    if not output_arg:
        return input_path.with_name(classified_output_name(input_path))

    raw_output = output_arg.strip()
    output_path = Path(raw_output).expanduser()
    is_directory_output = (
        raw_output.endswith(("/", "\\"))
        or output_path.exists() and output_path.is_dir()
        or (not output_path.exists() and output_path.suffix == "")
    )
    if is_directory_output:
        return (output_path / classified_output_name(input_path)).resolve()

    return output_path.resolve()


def resolve_jobs(input_path: Path, output_arg: str | None, include_classified: bool) -> list[tuple[Path, Path]]:
    if input_path.is_file():
        if input_path.suffix.lower() not in EXCEL_SUFFIXES:
            raise ValueError(f"输入文件不是 Excel: {input_path}")
        output_path = resolve_single_file_output(input_path, output_arg)
        return [(input_path, output_path)]

    if not input_path.is_dir():
        raise ValueError(f"输入路径不存在: {input_path}")

    output_dir = Path(output_arg).expanduser().resolve() if output_arg else input_path / "classified_results"
    excel_files = sorted(
        path for path in input_path.iterdir() if not should_skip_file(path, include_classified)
    )
    return [(path, output_dir / classified_output_name(path)) for path in excel_files]


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

        output_parent = output_path.parent
        output_parent.mkdir(parents=True, exist_ok=True)
        if not output_parent.is_dir():
            raise ValueError(f"输出父路径不是目录: {output_parent}")

        probe_path = output_parent / f".batch_classify_excel_write_test_{id(output_path)}.tmp"
        try:
            probe_path.write_text("ok", encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"输出目录不可写: {output_parent}: {exc}") from exc
        finally:
            if probe_path.exists():
                probe_path.unlink()


def select_classifier(mode: str):
    from services.classifier import (  # noqa: E402
        classify_text,
        classify_text_llm_only,
        fallback_classify,
        rule_classify,
    )

    if mode == "auto":
        return classify_text
    if mode == "llm":
        return classify_text_llm_only
    if mode == "rule":
        return lambda text: rule_classify(text) or fallback_classify(
            text,
            "rule 模式未命中规则，返回默认分类",
        )
    raise ValueError(f"未知分类模式: {mode}")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path).expanduser().resolve()

    try:
        jobs = resolve_jobs(input_path, args.output, args.include_classified)
        validate_jobs(jobs, args.overwrite)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 1

    if not jobs:
        print(f"[ERROR] 没有可处理的 Excel 文件: {input_path}")
        return 1

    try:
        classify_text_func = select_classifier(args.mode)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 1

    total_files = 0
    total_processed = 0
    total_skipped = 0
    failed_files: list[tuple[str, str]] = []

    print(f"[INFO] 输入路径: {input_path}")
    print(f"[INFO] 待处理文件数: {len(jobs)}")
    print(f"[INFO] 分类模式: {args.mode}")
    for path, output_path in jobs:
        print(f"[PLAN] {path} -> {output_path}")

    for path, output_path in jobs:
        print(f"[RUN ] {path.name}")
        try:
            processed, skipped = classify_workbook(path, output_path, classify_text_func)
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
