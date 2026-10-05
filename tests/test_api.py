from uuid import uuid4

from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

import app.api as api
from app.models import TriageAnswer
from app.rag import RagService

DOC = Document(page_content="Restored the timeout.", metadata={"eid": "I-1", "kind": "incident", "ref": "INC1001", "source": "resolution", "similarity": 0.9})


def test_ask_requires_a_tenant_header():
    client = TestClient(api.app)
    assert client.post("/v1/ask", json={"question": "seen before?"}).status_code == 422


def test_ask_returns_grounded_answer():
    api._service = RagService(
        structured_llm=RunnableLambda(lambda _: TriageAnswer(insufficient_evidence=False, answer="Yes, INC1001.", citations=["I-1"], confidence=0.8)),
        retriever_factory=lambda scope: RunnableLambda(lambda q: [DOC]),
    )
    r = TestClient(api.app).post("/v1/ask", json={"question": "seen before?"}, headers={"X-Tenant-Id": str(uuid4())})
    assert r.status_code == 200 and r.json()["mode"] == "grounded"
    assert r.json()["prompt_version"] == "rag-answer-v1"
