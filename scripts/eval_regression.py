#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fp:
        return list(csv.DictReader(fp))


def yes(value: str | None) -> bool:
    return (value or "").strip() in {"是", "true", "True", "1", "yes", "Y"}


def acceptable_catalog_ids(row: dict[str, str]) -> set[str]:
    raw = row.get("acceptable_catalog_ids") or row.get("gold_primary_catalog_id") or ""
    return {value.strip() for value in raw.split(";") if value.strip()}


def evaluate(gold: list[dict[str, str]], pred: list[dict[str, str]]) -> tuple[dict[str, object], list[dict[str, object]]]:
    pred_by_case = {row.get("case_id", "").strip(): row for row in pred if row.get("case_id", "").strip()}
    pred_by_name = {row.get("工程名称", "").strip(): row for row in pred if row.get("工程名称", "").strip()}

    joined: list[dict[str, object]] = []
    mismatches: list[dict[str, object]] = []
    for gold_row in gold:
        case_id = gold_row.get("case_id", "").strip()
        name = gold_row.get("工程名称", "").strip()
        pred_row = pred_by_case.get(case_id) or pred_by_name.get(name)
        if not pred_row:
            mismatches.append({**gold_row, "error_type": "missing_prediction"})
            continue

        pred_catalog = (pred_row.get("catalog_id") or "").strip()
        accepted = acceptable_catalog_ids(gold_row)
        catalog_pass = pred_catalog in accepted
        row = {
            "case_id": case_id,
            "工程名称": name,
            "gold_primary_catalog_id": gold_row.get("gold_primary_catalog_id", ""),
            "acceptable_catalog_ids": gold_row.get("acceptable_catalog_ids", ""),
            "pred_catalog_id": pred_catalog,
            "catalog_pass": catalog_pass,
            "gold_维修状态": gold_row.get("gold_维修状态", ""),
            "pred_维修状态": pred_row.get("维修状态", ""),
            "gold_是否复合工程": gold_row.get("gold_是否复合工程", ""),
            "pred_是否复合工程": pred_row.get("是否复合工程", ""),
            "gold_是否建议复核": gold_row.get("gold_是否建议复核", ""),
            "pred_是否建议复核": pred_row.get("是否建议复核", ""),
            "候选目录": pred_row.get("候选目录", ""),
            "分类依据": pred_row.get("分类依据", ""),
        }
        joined.append(row)
        if not catalog_pass:
            mismatches.append({**row, "error_type": "catalog_mismatch"})

    catalog_pass_count = sum(1 for row in joined if row["catalog_pass"])
    metrics = {
        "total_predictions": len(pred),
        "gold_cases": len(gold),
        "joined_cases": len(joined),
        "out_of_scope_count": sum(1 for row in pred if (row.get("catalog_id") or "").strip() == "OUT_OF_SCOPE"),
        "review_required_count": sum(1 for row in pred if yes(row.get("是否建议复核"))),
        "no_candidate_count": sum(
            1
            for row in pred
            if not (row.get("候选目录") or "").strip()
            or "未召回候选目录" in (row.get("分类依据") or "")
        ),
        "composite_count": sum(1 for row in pred if yes(row.get("是否复合工程"))),
        "gold_catalog_pass_count": catalog_pass_count,
        "gold_catalog_pass_rate": round(catalog_pass_count / len(gold), 4) if gold else 0,
        "mismatch_count": len(mismatches),
    }
    return metrics, mismatches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估标准目录 OUT41 回归预测结果")
    parser.add_argument("--gold", required=True)
    parser.add_argument("--pred", required=True)
    parser.add_argument("--out", default="outputs/regression_metrics.json")
    parser.add_argument("--mismatch", default="outputs/regression_mismatch.csv")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics, mismatches = evaluate(read_csv(Path(args.gold)), read_csv(Path(args.pred)))

    out_path = Path(args.out)
    mismatch_path = Path(args.mismatch)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mismatch_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    if mismatches:
        fieldnames = sorted({key for row in mismatches for key in row})
        with mismatch_path.open("w", encoding="utf-8-sig", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(mismatches)
    else:
        mismatch_path.write_text("", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
