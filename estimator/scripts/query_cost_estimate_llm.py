#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(_REPO_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(_REPO_BOOTSTRAP))

from estimator.output.frames import build_workbook_frames  # noqa: E402
from estimator.output.excel_writer import write_estimate_workbook  # noqa: E402
from estimator.paths import (  # noqa: E402
    CLASSIFIER_BACKEND_DIR,
    default_query_output_path,
    resolve_repo_path,
)
from estimator.query_models import QueryResult  # noqa: E402
from estimator.query_pipeline import run_estimate_query  # noqa: E402
from estimator.retrieval.weights import DEFAULT_PACKAGE_WEIGHT_TEMPERATURE  # noqa: E402

if str(CLASSIFIER_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(CLASSIFIER_BACKEND_DIR))
from classifier.llm_client import check_lmstudio_service  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="自然语言维修工程造价离线查询入口")
    parser.add_argument("--index-dir", default="embeddings", help="索引目录，默认 embeddings")
    parser.add_argument("--text", required=True, help="口语化维修需求")
    parser.add_argument("--top-packages", type=int, default=20, help="召回相似历史工程包数量")
    parser.add_argument("--top-items", type=int, default=300, help="召回直接相关历史清单行数量")
    parser.add_argument("--output", default=None, help="xlsx 输出路径，默认 query/YYYYMMDDHHMM.xlsx")
    parser.add_argument("--overwrite", action="store_true", help="若输出文件已存在则覆盖")
    parser.add_argument("--include-debug-text", action="store_true", help="在 parse_info 中保留 LLM 调试文本摘要")
    parser.add_argument("--display", action="store_true", help="输出时将部分数值格式化为易读文本")
    parser.add_argument(
        "--with-explanations", action="store_true",
        help="生成项目级和清单级 LLM 解释，默认关闭",
    )
    parser.add_argument(
        "--llm-check-timeout", type=float, default=3.0,
        help="启动前检查 LLM 服务可用性的超时时间，默认 3 秒",
    )
    parser.add_argument(
        "--max-packages-per-cache-subject", type=int, default=1,
        help="同一 cache_subject 最多保留的相似历史工程包数量，默认 1；设为 0 表示不限制",
    )
    parser.add_argument(
        "--package-weight-temperature", type=float,
        default=DEFAULT_PACKAGE_WEIGHT_TEMPERATURE,
        help=f"工程包证据权重 softmax temperature，必须大于 0，默认 {DEFAULT_PACKAGE_WEIGHT_TEMPERATURE}",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def validate_output_path(output_path: Path | None, overwrite: bool) -> None:
    if output_path is None:
        return
    if output_path.exists() and output_path.is_dir():
        raise ValueError(f"输出路径是目录，不是文件: {output_path}")
    if output_path.exists() and not overwrite:
        raise ValueError(f"输出已存在，请加 --overwrite 或更换输出路径: {output_path}")


def print_terminal_summary(result: QueryResult, output_path: Path | None) -> None:
    print(f"[DONE] package query: {result.rewrite.project_package_query_text}")
    print(f"[DONE] item query: {result.rewrite.item_query_text}")
    print(f"[DONE] matched project packages: {len(result.matched_project_packages)}")
    print(f"[DONE] candidate families: {len(result.candidate_families)}")
    print(f"[DONE] candidate display groups: {len(result.candidate_display_groups)}")
    print(f"[DONE] matched project example items: {len(result.matched_project_examples)}")
    print(f"[DONE] evidence items: {len(result.evidence_items)}")
    print(f"[DONE] estimate scenarios: {len(result.estimate_scenarios)}")
    if result.rewrite.notes:
        print(f"rewrite notes: {'；'.join(result.rewrite.notes)}")
    if output_path:
        print(f"输出文件: {output_path}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    index_dir = resolve_repo_path(args.index_dir)
    output_path = resolve_repo_path(args.output) if args.output is not None else default_query_output_path()

    try:
        validate_output_path(output_path, args.overwrite)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 1

    try:
        check_lmstudio_service(timeout_seconds=args.llm_check_timeout)
    except RuntimeError as exc:
        detail = str(exc)
        marker = "LMSTUDIO_BASE_URL="
        suffix = detail[detail.find(marker) :] if marker in detail else detail
        print(f"[ERROR] LLM 服务不可用，请先启动 LM Studio Server，并检查 {suffix}")
        return 1

    try:
        result = run_estimate_query(
            index_dir=index_dir,
            raw_text=args.text,
            top_packages=args.top_packages,
            top_items=args.top_items,
            reported_output_path=output_path,
            max_packages_per_cache_subject=args.max_packages_per_cache_subject,
            package_weight_temperature=args.package_weight_temperature,
            include_debug_text=args.include_debug_text,
            with_explanations=args.with_explanations,
        )
        frames = build_workbook_frames(result)
        write_estimate_workbook(output_path, frames)
    except (RuntimeError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    print_terminal_summary(result, output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
