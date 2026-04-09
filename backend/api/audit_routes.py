from fastapi import APIRouter

from models.schemas import AuditRequest, AuditResponse
from services.audit_pipeline_service import run_audit_pipeline


router = APIRouter(prefix="/api")


@router.post("/audit", response_model=AuditResponse)
def audit(req: AuditRequest):
    payload = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    return run_audit_pipeline(payload)
