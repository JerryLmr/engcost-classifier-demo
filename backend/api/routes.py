from fastapi import APIRouter, File, HTTPException, UploadFile

from classifier.settings import OLLAMA_MODEL
from models.schemas import ClassifyRequest
from services.analysis_service import analyze_excel_file
from services.excel_service import classify_excel_file
from services.standard_classifier import classify_project_standard


router = APIRouter(prefix="/api")


@router.get("/health")
def health_check():
    return {"status": "ok", "model": OLLAMA_MODEL}


@router.post("/classify")
def classify(req: ClassifyRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="工程名称不能为空")
    return classify_project_standard(text)


@router.post("/classify-excel")
def classify_excel(file: UploadFile = File(...)):
    return classify_excel_file(file)


@router.post("/analyze-excel")
def analyze_excel(file: UploadFile = File(...)):
    return analyze_excel_file(file)
