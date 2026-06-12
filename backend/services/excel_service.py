import io
from urllib.parse import quote

import openpyxl
from fastapi import HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from services.classifier import classify_text


RESULT_HEADERS = [
    "一级分类",
    "二级分类",
    "三级分类",
    "分类方式",
    "置信度",
    "匹配类型",
    "是否建议复核",
    "候选目录ID",
    "候选目录",
    "分类依据",
]


def build_classified_workbook_bytes(data: bytes) -> bytes:
    workbook = openpyxl.load_workbook(io.BytesIO(data))
    worksheet = workbook.active

    headers = [worksheet.cell(row=1, column=i).value for i in range(1, worksheet.max_column + 1)]
    if not headers or headers[0] is None:
        raise HTTPException(status_code=400, detail="第一列表头不能为空")

    result_start_col = worksheet.max_column + 1
    for offset, header in enumerate(RESULT_HEADERS):
        worksheet.cell(row=1, column=result_start_col + offset, value=header)

    for row in range(2, worksheet.max_row + 1):
        project_name = worksheet.cell(row=row, column=1).value
        if project_name is None or str(project_name).strip() == "":
            continue

        result = classify_text(str(project_name))
        worksheet.cell(row=row, column=result_start_col, value=result["level1"])
        worksheet.cell(row=row, column=result_start_col + 1, value=result["level2"])
        worksheet.cell(row=row, column=result_start_col + 2, value=result["level3"])
        worksheet.cell(row=row, column=result_start_col + 3, value=result["method"])
        worksheet.cell(row=row, column=result_start_col + 4, value=result["confidence"])
        worksheet.cell(row=row, column=result_start_col + 5, value=result["match_type"])
        worksheet.cell(row=row, column=result_start_col + 6, value="是" if result["needs_review"] else "否")
        worksheet.cell(row=row, column=result_start_col + 7, value=" | ".join(result["candidate_ids"]))
        worksheet.cell(row=row, column=result_start_col + 8, value=" | ".join(result["candidate_labels"]))
        worksheet.cell(row=row, column=result_start_col + 9, value=result["reason"])

    output = io.BytesIO()
    workbook.save(output)
    return output.getvalue()


def classify_excel_file(file: UploadFile):
    if not file.filename or not file.filename.lower().endswith((".xlsx", ".xlsm")):
        raise HTTPException(status_code=400, detail="请上传 .xlsx 或 .xlsm 文件")

    data = file.file.read()
    output = io.BytesIO(build_classified_workbook_bytes(data))

    base_name = file.filename.rsplit(".", 1)[0]
    safe_filename = f"{base_name}_classified.xlsx"
    encoded_filename = quote(f"{base_name}_分类结果.xlsx")
    headers = {
        "Content-Disposition": (
            f"attachment; filename={safe_filename}; "
            f"filename*=UTF-8''{encoded_filename}"
        )
    }

    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers=headers,
    )
