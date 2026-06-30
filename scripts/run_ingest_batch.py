#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXCEL_SUFFIXES = {".xlsx", ".xlsm"}
BATCH_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")
OCR_EXPORT_PREFIX = "audit_ocr_export_"
OCR_EXPORT_SUFFIX = ".xlsx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按批次导入 OCR Excel 并生成单批次清单样本")
    parser.add_argument("--input", required=True, help="OCR xlsx 输入路径")
    parser.add_argument("--batch-id", help="批次 id；不传时从 audit_ocr_export_YYYYMMDD_NNN.xlsx 自动解析")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖该 batch 已有输出")
    return parser.parse_args()


def resolve_path(raw_path: str) -> Path:
    return Path(raw_path).expanduser().resolve()


def validate_batch_id(batch_id: str) -> str:
    normalized = batch_id.strip()
    if not normalized:
        raise ValueError("batch_id 不能为空")
    if not BATCH_ID_RE.fullmatch(normalized):
        raise ValueError("batch_id 只能包含字母、数字、下划线和连字符")
    return normalized


def infer_batch_id(input_path: Path) -> str:
    name = input_path.name
    if not name.startswith(OCR_EXPORT_PREFIX) or not name.endswith(OCR_EXPORT_SUFFIX):
        raise ValueError(
            "无法从文件名解析 batch_id。文件名应为 "
            "audit_ocr_export_YYYYMMDD_NNN.xlsx，或显式传入 --batch-id"
        )
    return validate_batch_id(name[len(OCR_EXPORT_PREFIX) : -len(OCR_EXPORT_SUFFIX)])


def batch_id_from_args(input_path: Path, explicit_batch_id: str | None) -> str:
    if explicit_batch_id is not None:
        return validate_batch_id(explicit_batch_id)
    return infer_batch_id(input_path)


def batch_outputs(batch_id: str) -> dict[str, Path]:
    return {
        "cleaned": ROOT / "cleaned_inputs" / batch_id / "ocr_required_cleaned.xlsx",
        "removed": ROOT / "removed_inputs" / batch_id / "ocr_required_removed.xlsx",
        "classified": ROOT / "classified_outputs" / batch_id / "classified_projects.xlsx",
        "samples": ROOT / "samples" / batch_id / "cost_item_samples.xlsx",
    }


def validate_input(input_path: Path) -> None:
    if not input_path.exists():
        raise ValueError(f"输入文件不存在: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"输入路径不是文件: {input_path}")
    if input_path.suffix.lower() not in EXCEL_SUFFIXES:
        raise ValueError(f"输入文件不是 Excel: {input_path}")


def validate_output_conflicts(outputs: dict[str, Path], overwrite: bool) -> None:
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not overwrite:
        formatted = "\n".join(f"  - {path}" for path in existing)
        raise ValueError(
            "批次输出已存在，不允许静默覆盖:\n"
            f"{formatted}\n"
            "请换一个唯一的 --batch-id，或使用 --overwrite 覆盖该批次结果"
        )


def command_steps(input_path: Path, outputs: dict[str, Path], overwrite: bool) -> list[list[str]]:
    filter_command = [
        sys.executable,
        str(ROOT / "scripts" / "filter_required_ocr_rows.py"),
        str(input_path),
        "--clean-output",
        str(outputs["cleaned"]),
        "--removed-output",
        str(outputs["removed"]),
    ]
    classify_command = [
        sys.executable,
        str(ROOT / "scripts" / "batch_classify_excel.py"),
        str(outputs["cleaned"]),
        "-o",
        str(outputs["classified"]),
    ]
    samples_command = [
        sys.executable,
        str(ROOT / "scripts" / "build_cost_item_samples.py"),
        str(outputs["classified"]),
        "-o",
        str(outputs["samples"]),
    ]
    commands = [filter_command, classify_command, samples_command]
    if overwrite:
        for command in commands:
            command.append("--overwrite")
    return commands


def run_commands(commands: list[list[str]]) -> int:
    for command in commands:
        print("[RUN ] " + " ".join(command), flush=True)
        completed = subprocess.run(command, cwd=ROOT, check=False)  # noqa: S603
        if completed.returncode != 0:
            return completed.returncode
    return 0


def main() -> int:
    args = parse_args()
    input_path = resolve_path(args.input)

    try:
        validate_input(input_path)
        batch_id = batch_id_from_args(input_path, args.batch_id)
        outputs = batch_outputs(batch_id)
        validate_output_conflicts(outputs, args.overwrite)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 1

    print(f"[INFO] batch_id: {batch_id}")
    for label, path in outputs.items():
        print(f"[PLAN] {label}: {path}")

    exit_code = run_commands(command_steps(input_path, outputs, args.overwrite))
    if exit_code:
        return exit_code

    print(f"[DONE] batch_id: {batch_id}")
    print(f"[DONE] samples: {outputs['samples']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
