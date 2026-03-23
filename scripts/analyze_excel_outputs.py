#!/usr/bin/env python3
import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List
import sys

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.analysis_service import load_records_from_path, summarize_records  # noqa: E402


SUMMARY_HEADERS = ["指标", "数值"]
CLASSIFIED_SUFFIXES = ("_分类结果", "_classified")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总分析目录中的分类结果 Excel")
    parser.add_argument("input_dir", help="分类结果 Excel 所在目录")
    parser.add_argument(
        "-o",
        "--output",
        help="汇总 Excel 输出路径，默认输出到 input_dir/分析汇总.xlsx",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="一级/二级分类统计输出前 N 项，默认 20",
    )
    return parser.parse_args()


def should_read_file(path: Path) -> bool:
    return (
        path.suffix.lower() in {".xlsx", ".xlsm"}
        and not path.name.startswith("~$")
        and path.stem.endswith(CLASSIFIED_SUFFIXES)
    )


def top_counter_rows(counter: Counter, top_n: int, label: str) -> List[List[object]]:
    rows = [[label, "数量"]]
    for name, count in counter.most_common(top_n):
        rows.append([name, count])
    return rows


def write_sheet(workbook: openpyxl.Workbook, title: str, rows: Iterable[Iterable[object]]) -> None:
    worksheet = workbook.create_sheet(title=title)
    for row in rows:
        worksheet.append(list(row))


def build_summary_rows(records: List[Dict[str, object]]) -> List[List[object]]:
    files = sorted({record["source_file"] for record in records})
    analysis = summarize_records(records)
    summary = analysis["summary"]
    structure = analysis["structure_counts"]
    rows = [SUMMARY_HEADERS]
    rows.extend(
        [
            ["文件数", len(files)],
            ["总记录数", len(records)],
            ["规则优先", summary["rule_method_count"]],
            ["LLM 辅助分类", summary["llm_method_count"]],
            ["体系外默认分类", summary["fallback_method_count"]],
            ["复合工程=是", summary["composite_count"]],
            ["建议复核=是", summary["review_count"]],
            ["single_project", structure["single_project"]],
            ["multi_system_same_domain", structure["multi_system_same_domain"]],
            ["composite_project", structure["composite_project"]],
        ]
    )
    return rows


def build_focus_rows(records: List[Dict[str, object]]) -> List[List[object]]:
    headers = [
        "来源文件",
        "行号",
        "工程名称",
        "一级分类",
        "二级分类",
        "分类方式",
        "是否复合工程",
        "是否建议复核",
        "结构类型",
        "分类依据",
    ]
    rows = [headers]
    focus_records = summarize_records(records)["focus_samples"]
    for record in focus_records:
        rows.append(
            [
                record["source_file"],
                record["row_num"],
                record["project_name"],
                record["level1"],
                record["level2"],
                record["method"],
                "是" if record["is_composite"] else "否",
                "是" if record["needs_review"] else "否",
                record["structure_type"],
                record["reason"],
            ]
        )
    return rows


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"[ERROR] 输入目录不存在: {input_dir}")
        return 1

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_dir / "分析汇总.xlsx"
    )

    excel_files = sorted(path for path in input_dir.iterdir() if should_read_file(path))
    if not excel_files:
        print(f"[ERROR] 目录中没有可分析的分类结果文件: {input_dir}")
        return 1

    records: List[Dict[str, object]] = []
    for path in excel_files:
        records.extend(load_records_from_path(path))

    if not records:
        print(f"[ERROR] 未读取到有效分类记录: {input_dir}")
        return 1

    level1_counter = Counter(record["level1"] for record in records)
    level2_counter = Counter(record["level2"] for record in records)
    analysis = summarize_records(records, top_n=args.top_n)

    workbook = openpyxl.Workbook()
    workbook.remove(workbook.active)
    write_sheet(workbook, "总览", build_summary_rows(records))
    write_sheet(workbook, "一级分类统计", top_counter_rows(level1_counter, args.top_n, "一级分类"))
    write_sheet(workbook, "二级分类统计", top_counter_rows(level2_counter, args.top_n, "二级分类"))
    write_sheet(workbook, "重点样本", build_focus_rows(records))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)

    summary = analysis["summary"]
    structure = analysis["structure_counts"]

    print(f"[DONE] 分析目录: {input_dir}")
    print(f"[DONE] 汇总文件: {output_path}")
    print(f"[DONE] 文件数: {len(excel_files)}")
    print(f"[DONE] 总记录数: {len(records)}")
    print(
        "[DONE] 分类方式: "
        f"规则优先={summary['rule_method_count']}, "
        f"LLM辅助分类={summary['llm_method_count']}, "
        f"体系外默认分类={summary['fallback_method_count']}"
    )
    print(
        "[DONE] 结构统计: "
        f"复合工程={summary['composite_count']}, "
        f"建议复核={summary['review_count']}, "
        f"single={structure['single_project']}, "
        f"multi_system={structure['multi_system_same_domain']}, "
        f"composite={structure['composite_project']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
