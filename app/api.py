import logging
from uuid import UUID

import psycopg
from fastapi import FastAPI, Header, HTTPException

from app.models import AskRequest, AskResponse, Scope
from app.rag import RagService

logging.basicConfig(level=logging.INFO)
app = FastAPI(title="Fireline RAG", version="0.1.0")
_service: RagService | None = None


def get_service() -> RagService:
    global _service
    if _service is None:
        _service = RagService()
    return _service


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


@app.post("/v1/ask", response_model=AskResponse)
def ask(req: AskRequest, x_tenant_id: UUID = Header(description="Tenant from your auth layer. Never from the model.")) -> AskResponse:
    scope = Scope(tenant_id=x_tenant_id, service=req.service, environment=req.environment, region=req.region)
    try:
        return get_service().ask(req.question, scope)
    except psycopg.Error:
        raise HTTPException(status_code=503, detail={"code": "RETRIEVAL_UNAVAILABLE"})
