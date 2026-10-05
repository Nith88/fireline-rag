"""Builds (but does not send) the Anthropic request. Catches config bugs that stubbed-LLM tests
cannot, e.g. unsupported parameters or a structured-output mode the model rejects."""
import os
import warnings

import pytest

from app.models import Evidence
from app.prompts import PROMPT, format_evidence


@pytest.fixture(autouse=True)
def _dummy_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy")


def test_request_payload_is_valid_for_the_configured_model():
    from app.rag import build_structured_llm

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)  # e.g. "forced tool calling not supported"
        step = build_structured_llm().first
    ev = [Evidence(eid="I-1", kind="incident", ref="INC1", source="log", text="x", similarity=1)]
    msgs = PROMPT.format_messages(question="q?", evidence=format_evidence(ev))
    payload = step.bound._get_request_payload(msgs, **step.kwargs)  # raises on invalid parameters
    assert payload["output_config"]["format"]["type"] == "json_schema"
    assert "temperature" not in payload
