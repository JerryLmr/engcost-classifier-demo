#!/usr/bin/env python3
import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import openpyxl


SUMMARY_HEADERS = ["指标", "数值"]
CLASSIFIED_SUFFIXES = ("_分类结果", "_classified")
FOCUS_SAMPLE_LIMIT = 2000


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


def read_result_rows(path: Path) -> List[Dict[str, object]]:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    worksheet = workbook.active
    rows = worksheet.iter_rows(values_only=True)
    header = list(next(rows))
    index = {name: i for i, name in enumerate(header)}

    required_columns = [
        "一级分类",
        "二级分类",
        "分类方式",
        "分类依据",
        "是否复合工程",
        "是否建议复核",
        "结构类型",
    ]
    missing = [column for column in required_columns if column not in index]
    if missing:
        raise ValueError(f"{path.name} 缺少结果列: {', '.join(missing)}")

    result: List[Dict[str, object]] = []
    for row_num, row in enumerate(rows, start=2):
        if not any(row):
            continue
        project_name = row[0]
        if project_name is None or str(project_name).strip() == "":
            continue
        record = {
            "来源文件": path.name,
            "行号": row_num,
            "工程名称": str(project_name),
        }
        for column in required_columns:
            record[column] = row[index[column]]
        result.append(record)
    return result


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
    files = sorted({record["来源文件"] for record in records})
    method_counter = Counter(record["分类方式"] for record in records)
    composite_counter = Counter(record["是否复合工程"] for record in records)
    review_counter = Counter(record["是否建议复核"] for record in records)
    structure_counter = Counter(record["结构类型"] for record in records)

    rows = [SUMMARY_HEADERS]
    rows.extend(
        [
            ["文件数", len(files)],
            ["总记录数", len(records)],
            ["规则优先", method_counter.get("规则优先", 0)],
            ["LLM 兜底", method_counter.get("LLM 兜底", 0)],
            ["降级兜底", method_counter.get("降级兜底", 0)],
            ["复合工程=是", composite_counter.get("是", 0)],
            ["建议复核=是", review_counter.get("是", 0)],
            ["single_project", structure_counter.get("single_project", 0)],
            ["multi_system_same_domain", structure_counter.get("multi_system_same_domain", 0)],
            ["composite_project", structure_counter.get("composite_project", 0)],
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

    focus_records: List[Dict[str, object]] = []
    for record in records:
        if (
            record["分类方式"] != "规则优先"
            or record["是否复合工程"] == "是"
            or record["是否建议复核"] == "是"
        ):
            focus_records.append(record)

    focus_records.sort(
        key=lambda item: (
            item["来源文件"],
            item["分类方式"] != "规则优先",
            item["是否建议复核"] == "是",
            item["是否复合工程"] == "是",
            item["行号"],
        ),
        reverse=False,
    )

    for record in focus_records[:FOCUS_SAMPLE_LIMIT]:
        rows.append([record[column] for column in headers])
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
        records.extend(read_result_rows(path))

    if not records:
        print(f"[ERROR] 未读取到有效分类记录: {input_dir}")
        return 1

    level1_counter = Counter(record["一级分类"] for record in records)
    level2_counter = Counter(record["二级分类"] for record in records)

    workbook = openpyxl.Workbook()
    workbook.remove(workbook.active)
    write_sheet(workbook, "总览", build_summary_rows(records))
    write_sheet(workbook, "一级分类统计", top_counter_rows(level1_counter, args.top_n, "一级分类"))
    write_sheet(workbook, "二级分类统计", top_counter_rows(level2_counter, args.top_n, "二级分类"))
    write_sheet(workbook, "重点样本", build_focus_rows(records))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(output_path)

    method_counter = Counter(record["分类方式"] for record in records)
    composite_counter = Counter(record["是否复合工程"] for record in records)
    review_counter = Counter(record["是否建议复核"] for record in records)
    structure_counter = Counter(record["结构类型"] for record in records)

    print(f"[DONE] 分析目录: {input_dir}")
    print(f"[DONE] 汇总文件: {output_path}")
    print(f"[DONE] 文件数: {len(excel_files)}")
    print(f"[DONE] 总记录数: {len(records)}")
    print(
        "[DONE] 分类方式: "
        f"规则优先={method_counter.get('规则优先', 0)}, "
        f"LLM兜底={method_counter.get('LLM 兜底', 0)}, "
        f"降级兜底={method_counter.get('降级兜底', 0)}"
    )
    print(
        "[DONE] 结构统计: "
        f"复合工程={composite_counter.get('是', 0)}, "
        f"建议复核={review_counter.get('是', 0)}, "
        f"single={structure_counter.get('single_project', 0)}, "
        f"multi_system={structure_counter.get('multi_system_same_domain', 0)}, "
        f"composite={structure_counter.get('composite_project', 0)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
