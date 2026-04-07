from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.audit_routes import router as audit_router
from api.routes import router


app = FastAPI(title="物业工程分类 Demo", version="1.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)
app.include_router(audit_router)
