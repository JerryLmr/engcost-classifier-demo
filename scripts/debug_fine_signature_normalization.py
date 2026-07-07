#!/usr/bin/env python3
from __future__ import annotations

import argparse
from typing import Any

from build_cost_item_embedding_index import build_fine_signature, normalize_fine_signature_text


DEFAULT_CASES = [
    "1.3.0mm厚弹性体改性沥青防水卷材",
    "1.3.0mm弹性体改性沥青防水卷材",
    "3.0mm厚弹性体改性沥青防水卷材",
    "厚 3.0 mm 弹性体改性沥青防水卷材",
    "3.0mm弹性改性沥青防水卷材",
    "3.0mm SBS防水卷材",
    "3.0mm SBS改性沥青防水卷材",
    "3.0mm SBS弹性体改性沥青防水卷材",
    "卷材品种、规格、厚度：3.0mm SBS防水卷材",
    "4.0mm SBS防水卷材",
    "3.0mm自粘SBS防水卷材",
    "3.0mm热熔SBS防水卷材",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="调试 fine_signature 归一化结果")
    parser.add_argument("--text", default="", help="待归一化的项目特征文本；不传则输出内置案例")
    parser.add_argument("--cost-item-name", default="屋面卷材防水", help="用于生成 fine_signature 的清单项名称")
    parser.add_argument("--unit", default="m²", help="用于生成 fine_signature 的单位")
    return parser.parse_args()


def fine_signature_for(cost_item_name: str, project_description: str, unit: str) -> str:
    row: dict[str, Any] = {
        "cost_item_name": cost_item_name,
        "project_description": project_description,
        "unit": unit,
        "unit_normalized": unit,
    }
    return build_fine_signature(row)  # type: ignore[arg-type]


def print_case(text: str, cost_item_name: str, unit: str) -> None:
    print(f"原文: {text}")
    print(f"归一化: {normalize_fine_signature_text(text)}")
    print(f"fine_signature: {fine_signature_for(cost_item_name, text, unit)}")


def main() -> None:
    args = parse_args()
    if args.text:
        print_case(args.text, args.cost_item_name, args.unit)
        return

    for index, text in enumerate(DEFAULT_CASES, start=1):
        if index > 1:
            print()
        print_case(text, args.cost_item_name, args.unit)


if __name__ == "__main__":
    main()
