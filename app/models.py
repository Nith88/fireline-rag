from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


class Scope(BaseModel):
    """Trusted retrieval scope. Comes from auth + the current incident, never from the model."""

    tenant_id: UUID
    service: str | None = None
    environment: str | None = None
    region: str | None = None


class Evidence(BaseModel):
    eid: str  # "I-INC1001-resolution-0" (incident chunk) or "R-vpn-india-v3-0" (runbook chunk): what the model cites
    kind: Literal["incident", "runbook"]
    ref: str  # "INC1001" or "vpn-india v3"
    source: str  # title | description | timeline | log | resolution | runbook
    text: str
    similarity: float


class TriageAnswer(BaseModel):
    """Structured output contract for the LLM (Module 3, LU3.5)."""

    insufficient_evidence: bool = Field(
        description="True if the evidence does not answer the question."
    )
    answer: str | None = Field(
        description="Grounded answer, at most 120 words. Null when evidence is insufficient."
    )
    citations: list[str] = Field(
        description="Evidence ids (for example I-12, R-3) that support the answer."
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="How directly the evidence answers the question."
    )


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    service: str | None = None
    environment: str | None = None
    region: str | None = None


class AskResponse(BaseModel):
    mode: Literal["grounded", "insufficient_evidence", "fallback"]
    answer: str | None = None
    confidence: float | None = None
    citations: list[Evidence] = []  # evidence the answer cites
    evidence: list[Evidence] = []  # everything that was retrieved
    fallback_reason: str | None = None
    prompt_version: str
    latency_ms: int
