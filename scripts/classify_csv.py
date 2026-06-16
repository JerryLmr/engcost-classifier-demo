#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


RESULT_HEADERS = [
    "case_id",
    "工程名称",
    "catalog_id",
    "一级分类",
    "二级分类",
    "维修状态",
    "标准对象",
    "是否复合工程",
    "复合候选目录",
    "是否紧急维修",
    "是否白蚁相关",
    "是否建议复核",
    "候选目录",
    "分类依据",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按标准目录批量分类 CSV 工程名称")
    parser.add_argument("--input", required=True, help="输入 CSV，需包含 工程名称，可选 case_id")
    parser.add_argument("--output", required=True, help="输出预测 CSV")
    return parser.parse_args()


def _bool_text(value: object) -> str:
    return "是" if bool(value) else "否"


def _prediction_row(source: dict[str, str], result: dict[str, object]) -> dict[str, str]:
    return {
        "case_id": source.get("case_id", ""),
        "工程名称": str(result.get("project_name") or source.get("工程名称") or ""),
        "catalog_id": str(result.get("catalog_id") or ""),
        "一级分类": str(result.get("category") or ""),
        "二级分类": str(result.get("item") or ""),
        "维修状态": str(result.get("repair_status") or ""),
        "标准对象": str(result.get("standard_group") or ""),
        "是否复合工程": _bool_text(result.get("is_composite")),
        "复合候选目录": " | ".join(result.get("secondary_catalog_labels") or []),
        "是否紧急维修": _bool_text(result.get("is_emergency")),
        "是否白蚁相关": _bool_text(result.get("termite_related")),
        "是否建议复核": _bool_text(result.get("needs_review")),
        "候选目录": " | ".join(result.get("candidate_labels") or []),
        "分类依据": str(result.get("reason") or ""),
    }


def main() -> int:
    from services.standard_classifier import classify_project_standard  # noqa: E402

    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    with input_path.open("r", encoding="utf-8-sig", newline="") as fp:
        rows = list(csv.DictReader(fp))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=RESULT_HEADERS)
        writer.writeheader()
        for row in rows:
            project_name = str(row.get("工程名称") or "").strip()
            if not project_name:
                continue
            result = classify_project_standard(project_name)
            writer.writerow(_prediction_row(row, result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
