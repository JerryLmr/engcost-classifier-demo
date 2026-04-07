from fastapi import APIRouter

from models.schemas import AuditRequest, AuditResponse
from services.audit_service import audit_project
from services.mapping_service import map_project_name


router = APIRouter(prefix="/api")


@router.post("/audit", response_model=AuditResponse)
def audit(req: AuditRequest):
    payload = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    mapping_result = map_project_name(req.project_name)
    return audit_project(payload, mapping_result)
