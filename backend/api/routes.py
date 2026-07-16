from fastapi import APIRouter, File, HTTPException, UploadFile

from classifier.settings import LMSTUDIO_MODEL
from models.schemas import ClassifyRequest
from services.analysis_service import analyze_excel_file
from services.excel_service import classify_excel_file
from services.standard_classifier import classify_project_standard


router = APIRouter(prefix="/api")


DEMO_RESULTS: dict[str, dict[str, object]] = {
    "某小区 3 号楼外墙渗水维修工程": {
        "project_name": "某小区 3 号楼外墙渗水维修工程",
        "level1": "防水工程",
        "level2": "外墙防水",
        "method": "规则优先",
        "reason": "一级命中：防水工程；边界判定：外墙或屋面场景中出现渗漏/防水词，优先归入防水工程；关键词：外墙、渗水",
        "is_composite": False,
        "needs_review": False,
        "composite_reason": None,
        "secondary_candidates": [],
        "structure_type": "single_project",
    },
    "地下车库道闸系统更新改造": {
        "project_name": "地下车库道闸系统更新改造",
        "level1": "停车交通",
        "level2": "道闸系统维修",
        "method": "规则优先",
        "reason": "一级命中：停车交通；关键词：道闸、系统",
        "is_composite": False,
        "needs_review": False,
        "composite_reason": None,
        "secondary_candidates": [],
        "structure_type": "single_project",
    },
    "小区生活水泵更换维修": {
        "project_name": "小区生活水泵更换维修",
        "level1": "给排水",
        "level2": "水泵维修更换",
        "method": "规则优先",
        "reason": "一级命中：给排水；关键词：水泵、泵、更换、维修",
        "is_composite": False,
        "needs_review": False,
        "composite_reason": None,
        "secondary_candidates": [],
        "structure_type": "single_project",
    },
    "屋面漏水及防水层翻修工程": {
        "project_name": "屋面漏水及防水层翻修工程",
        "level1": "防水工程",
        "level2": "屋面防水维修",
        "method": "规则优先",
        "reason": "一级命中：防水工程；边界判定：屋面或屋顶的防水/渗漏治理优先归入防水工程；关键词：屋面、防水、漏水",
        "is_composite": False,
        "needs_review": False,
        "composite_reason": None,
        "secondary_candidates": [],
        "structure_type": "single_project",
    },
}


def _presentation_result(result: dict[str, object]) -> dict[str, object]:
    catalog_id = str(result.get("catalog_id") or "")
    is_composite = bool(result.get("is_composite"))
    payload = dict(result)
    payload["level1"] = result.get("category") or ""
    payload["level2"] = result.get("item") or ""
    payload["level3_item"] = result.get("item") or ""
    payload["matched_level3_items"] = []
    payload["method"] = "体系外默认分类" if catalog_id == "OUT_OF_SCOPE" else "LLM主分类"
    payload["confidence"] = ""
    payload["match_type"] = "out_of_scope" if catalog_id == "OUT_OF_SCOPE" else "standard_catalog"
    payload["structure_type"] = "composite_project" if is_composite else "single_project"
    payload["composite_reason"] = "疑似复合工程" if is_composite else ""
    payload["secondary_candidates"] = result.get("secondary_catalog_labels") or []
    return payload


@router.get("/health")
def health_check():
    return {"status": "ok", "model": LMSTUDIO_MODEL}


@router.post("/classify")
def classify(req: ClassifyRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="工程名称不能为空")
    if text in DEMO_RESULTS:
        return dict(DEMO_RESULTS[text])
    return _presentation_result(classify_project_standard(text))


@router.post("/classify-excel")
def classify_excel(file: UploadFile = File(...)):
    return classify_excel_file(file)


@router.post("/analyze-excel")
def analyze_excel(file: UploadFile = File(...)):
    return analyze_excel_file(file)
