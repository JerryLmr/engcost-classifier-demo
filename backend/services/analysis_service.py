import io
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import openpyxl
from fastapi import HTTPException, UploadFile


REQUIRED_RESULT_COLUMNS = [
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

FOCUS_SAMPLE_LIMIT = 2000


def normalize_method(method: Any) -> str:
    text = str(method or "").strip()
    if text in {"LLM 兜底", "LLM 辅助分类"}:
        return "LLM兜底"
    if text in {"降级兜底", "体系外默认分类"}:
        return "默认兜底"
    return text


def _split_pipe_text(value: Any) -> List[str]:
    text = str(value or "").strip()
    if not text:
        return []
    return [part.strip() for part in text.split("|") if part.strip()]


def _read_result_rows_from_workbook(workbook: openpyxl.Workbook, source_name: str) -> List[Dict[str, Any]]:
    worksheet = workbook.active
    rows = worksheet.iter_rows(values_only=True)
    header = list(next(rows))
    index = {name: i for i, name in enumerate(header)}

    missing = [column for column in REQUIRED_RESULT_COLUMNS if column not in index]
    if missing:
        raise HTTPException(status_code=400, detail=f"结果文件缺少必要列: {', '.join(missing)}")

    records: List[Dict[str, Any]] = []
    for row_num, row in enumerate(rows, start=2):
        if not any(row):
            continue
        project_name = row[0]
        if project_name is None or str(project_name).strip() == "":
            continue
        records.append(
            {
                "source_file": source_name,
                "row_num": row_num,
                "project_name": str(project_name),
                "level1": row[index["一级分类"]],
                "level2": row[index["二级分类"]],
                "level3_item": row[index["三级分类"]],
                "matched_level3_items": _split_pipe_text(row[index["具体细项"]]),
                "method": normalize_method(row[index["分类方式"]]),
                "confidence": row[index["置信度"]],
                "match_type": row[index["匹配类型"]],
                "needs_review": row[index["是否建议复核"]] == "是",
                "candidate_ids": _split_pipe_text(row[index["候选目录ID"]]),
                "candidate_labels": _split_pipe_text(row[index["候选目录"]]),
                "candidate_level3_items": _split_pipe_text(row[index["候选细项"]]),
                "reason": row[index["分类依据"]],
            }
        )
    return records


def load_records_from_upload(file: UploadFile) -> List[Dict[str, Any]]:
    if not file.filename or not file.filename.lower().endswith((".xlsx", ".xlsm")):
        raise HTTPException(status_code=400, detail="请上传 .xlsx 或 .xlsm 文件")
    data = file.file.read()
    workbook = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    return _read_result_rows_from_workbook(workbook, file.filename)


def load_records_from_path(path: Path) -> List[Dict[str, Any]]:
    workbook = openpyxl.load_workbook(path, read_only=True, data_only=True)
    return _read_result_rows_from_workbook(workbook, path.name)


def summarize_records(records: List[Dict[str, Any]], top_n: int = 20) -> Dict[str, Any]:
    if not records:
        raise ValueError("没有可分析的分类记录")

    method_counter = Counter(record["method"] for record in records)
    level1_counter = Counter(record["level1"] for record in records)
    level2_counter = Counter(record["level2"] for record in records)
    match_type_counter = Counter(record["match_type"] for record in records)

    focus_samples = [
        record
        for record in records
        if record["method"] != "规则优先" or record["needs_review"]
    ]
    focus_samples.sort(
        key=lambda item: (
            item["source_file"],
            item["method"] == "规则优先",
            not item["needs_review"],
            item["row_num"],
        )
    )

    return {
        "summary": {
            "total_records": len(records),
            "rule_method_count": method_counter.get("规则优先", 0),
            "llm_method_count": method_counter.get("LLM兜底", 0),
            "fallback_method_count": method_counter.get("默认兜底", 0),
            "review_count": sum(1 for record in records if record["needs_review"]),
        },
        "match_type_counts": {
            "single": match_type_counter.get("single", 0),
            "cross_domain": match_type_counter.get("cross_domain", 0),
            "same_domain_multi_item": match_type_counter.get("same_domain_multi_item", 0),
            "low_confidence": match_type_counter.get("low_confidence", 0),
            "llm_fallback": match_type_counter.get("llm_fallback", 0),
            "fallback": match_type_counter.get("fallback", 0),
        },
        "level1_top": [{"name": name, "count": count} for name, count in level1_counter.most_common(top_n)],
        "level2_top": [{"name": name, "count": count} for name, count in level2_counter.most_common(top_n)],
        "focus_samples": focus_samples[:FOCUS_SAMPLE_LIMIT],
    }


def analyze_excel_file(file: UploadFile, top_n: int = 20) -> Dict[str, Any]:
    records = load_records_from_upload(file)
    return summarize_records(records, top_n=top_n)
