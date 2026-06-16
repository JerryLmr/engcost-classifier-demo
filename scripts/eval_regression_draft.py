#!/usr/bin/env python3
"""
Evaluate engcost classifier regression results against a gold file.

Expected gold columns:
case_id, 工程名称, gold_primary_catalog_id, acceptable_catalog_ids,
gold_维修状态, gold_是否复合工程, gold_是否建议复核

Expected prediction columns:
Either case_id or 工程名称, catalog_id, 维修状态, 是否复合工程, 是否建议复核, 候选目录, 分类依据
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def yes(v: str | None) -> bool:
    return (v or "").strip() in {"是", "true", "True", "1", "yes", "Y"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", default="regression_metrics.json")
    ap.add_argument("--mismatch", default="regression_mismatch.csv")
    args = ap.parse_args()

    gold = read_csv(Path(args.gold))
    pred = read_csv(Path(args.pred))

    pred_by_case = {r.get("case_id", "").strip(): r for r in pred if r.get("case_id", "").strip()}
    pred_by_name = {r.get("工程名称", "").strip(): r for r in pred if r.get("工程名称", "").strip()}

    joined = []
    mismatches = []
    for g in gold:
        key = g.get("case_id", "").strip()
        name = g.get("工程名称", "").strip()
        p = pred_by_case.get(key) or pred_by_name.get(name)
        if not p:
            mismatches.append({**g, "error_type": "missing_prediction"})
            continue

        pred_catalog = (p.get("catalog_id") or "").strip()
        acceptable = {
            x.strip()
            for x in (g.get("acceptable_catalog_ids") or g.get("gold_primary_catalog_id") or "").split(";")
            if x.strip()
        }
        catalog_pass = pred_catalog in acceptable

        row = {
            "case_id": key,
            "工程名称": name,
            "gold_primary_catalog_id": g.get("gold_primary_catalog_id", ""),
            "acceptable_catalog_ids": g.get("acceptable_catalog_ids", ""),
            "pred_catalog_id": pred_catalog,
            "catalog_pass": catalog_pass,
            "gold_维修状态": g.get("gold_维修状态", ""),
            "pred_维修状态": p.get("维修状态", ""),
            "gold_是否复合工程": g.get("gold_是否复合工程", ""),
            "pred_是否复合工程": p.get("是否复合工程", ""),
            "gold_是否建议复核": g.get("gold_是否建议复核", ""),
            "pred_是否建议复核": p.get("是否建议复核", ""),
            "候选目录": p.get("候选目录", ""),
            "分类依据": p.get("分类依据", ""),
        }
        joined.append(row)

        if not catalog_pass:
            row["error_type"] = "catalog_mismatch"
            mismatches.append(row)

    total = len(pred)
    out_of_scope_count = sum(1 for r in pred if (r.get("catalog_id") or "").strip() == "OUT_OF_SCOPE")
    review_required_count = sum(1 for r in pred if yes(r.get("是否建议复核")))
    no_candidate_count = sum(
        1
        for r in pred
        if not (r.get("候选目录") or "").strip()
        or "未召回候选目录" in (r.get("分类依据") or "")
    )
    composite_count = sum(1 for r in pred if yes(r.get("是否复合工程")))
    catalog_pass_count = sum(1 for r in joined if r["catalog_pass"])

    metrics = {
        "total_predictions": total,
        "gold_cases": len(gold),
        "joined_cases": len(joined),
        "out_of_scope_count": out_of_scope_count,
        "review_required_count": review_required_count,
        "no_candidate_count": no_candidate_count,
        "composite_count": composite_count,
        "gold_catalog_pass_count": catalog_pass_count,
        "gold_catalog_pass_rate": round(catalog_pass_count / len(gold), 4) if gold else 0,
        "mismatch_count": len(mismatches),
    }

    Path(args.out).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    if mismatches:
        fieldnames = sorted({k for row in mismatches for k in row.keys()})
        with Path(args.mismatch).open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(mismatches)
    else:
        Path(args.mismatch).write_text("", encoding="utf-8")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
