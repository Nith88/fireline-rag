"""Builds (but does not send) the Gemini chain. Catches config bugs that stubbed-LLM tests
cannot, e.g. a structured-output schema the integration rejects."""
import pytest

from app.models import Evidence, TriageAnswer
from app.prompts import PROMPT, format_evidence


@pytest.fixture(autouse=True)
def _dummy_key(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")


def test_structured_chain_builds_and_prompt_formats():
    from app.rag import build_structured_llm

    llm = build_structured_llm()
    assert llm is not None
    ev = [Evidence(eid="I-1", kind="incident", ref="INC1", source="log", text="x", similarity=1)]
    msgs = PROMPT.format_messages(question="q?", evidence=format_evidence(ev))
    assert msgs
    assert {"insufficient_evidence", "answer", "citations", "confidence"} <= set(TriageAnswer.model_json_schema()["properties"])
