from fastapi import APIRouter, File, HTTPException, UploadFile

from classifier.settings import OLLAMA_MODEL
from models.schemas import ClassifyRequest
from services.analysis_service import analyze_excel_file
from services.excel_service import classify_excel_file
from services.standard_classifier import classify_project_standard


router = APIRouter(prefix="/api")


def _presentation_result(result: dict[str, object]) -> dict[str, object]:
    catalog_id = str(result.get("catalog_id") or "")
    is_composite = bool(result.get("is_composite"))
    payload = dict(result)
    payload["level1"] = result.get("category") or ""
    payload["level2"] = result.get("item") or ""
    payload["level3_item"] = result.get("item") or ""
    payload["matched_level3_items"] = []
    payload["method"] = "体系外默认分类" if catalog_id == "OUT_OF_SCOPE" else "LLM 辅助分类"
    payload["confidence"] = ""
    payload["match_type"] = "out_of_scope" if catalog_id == "OUT_OF_SCOPE" else "standard_catalog"
    payload["candidate_ids"] = []
    payload["candidate_level3_items"] = []
    payload["structure_type"] = "composite_project" if is_composite else "single_project"
    payload["composite_reason"] = "疑似复合工程" if is_composite else ""
    payload["secondary_candidates"] = result.get("secondary_catalog_labels") or []
    return payload


@router.get("/health")
def health_check():
    return {"status": "ok", "model": OLLAMA_MODEL}


@router.post("/classify")
def classify(req: ClassifyRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="工程名称不能为空")
    return _presentation_result(classify_project_standard(text))


@router.post("/classify-excel")
def classify_excel(file: UploadFile = File(...)):
    return classify_excel_file(file)


@router.post("/analyze-excel")
def analyze_excel(file: UploadFile = File(...)):
    return analyze_excel_file(file)
