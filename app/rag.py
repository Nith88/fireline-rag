"""RAG pipeline: retrieve -> build context -> LLM -> validate -> policy -> answer or fallback.

The model suggests; the system controls . Anything that fails validation degrades to the
raw search results instead of an error, so engineers can still do the job during an outage.
"""
import logging
import time
from typing import Callable

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable

from app.config import get_settings
from app.embeddings import get_embeddings
from app.models import AskResponse, Evidence, Scope, TriageAnswer
from app.prompts import PROMPT, PROMPT_VERSION, format_evidence
from app.retriever import FirelineRetriever

log = logging.getLogger("fireline.rag")


def build_structured_llm() -> Runnable:
    s = get_settings()
    # No `temperature`: current Claude models reject non-default values. json_schema mode is used
    # because forced tool-calling is not supported for structured output on newer models.
    llm = ChatAnthropic(model=s.llm_model, max_tokens=800, timeout=s.llm_timeout_s, max_retries=0)
    return llm.with_structured_output(TriageAnswer, method="json_schema")


def to_evidence(doc: Document) -> Evidence:
    m = doc.metadata
    return Evidence(eid=m["eid"], kind=m["kind"], ref=m["ref"], source=m["source"], text=doc.page_content, similarity=round(m["similarity"], 4))


class RagService:
    def __init__(
        self,
        embeddings: Embeddings | None = None,
        structured_llm: Runnable | None = None,
        retriever_factory: Callable[[Scope], Runnable] | None = None,
    ):
        self._embeddings = embeddings
        self._llm = structured_llm
        self._retriever_factory = retriever_factory or self._default_retriever

    def _default_retriever(self, scope: Scope) -> Runnable:
        s = get_settings()
        return FirelineRetriever(
            embeddings=self._embeddings or get_embeddings(),
            scope=scope,
            k_incidents=s.k_incidents,
            k_runbooks=s.k_runbooks,
            min_similarity=s.min_similarity,
        )

    @property
    def chain(self) -> Runnable:
        if self._llm is None:  # built lazily so retrieval-only use needs no API key
            self._llm = build_structured_llm()
        return PROMPT | self._llm

    def ask(self, question: str, scope: Scope) -> AskResponse:
        start = time.perf_counter()

        def respond(mode, *, evidence, **kw) -> AskResponse:
            return AskResponse(mode=mode, evidence=evidence, prompt_version=PROMPT_VERSION, latency_ms=int((time.perf_counter() - start) * 1000), **kw)

        # 1. Retrieve (scoped by trusted metadata). A database failure propagates: nothing to fall back to.
        docs = self._retriever_factory(scope).invoke(question)
        evidence = [to_evidence(d) for d in docs]
        if not evidence:  # skip the LLM call entirely: nothing to ground an answer on
            return respond("insufficient_evidence", evidence=[])

        def fallback(reason: str) -> AskResponse:
            log.warning("rag_fallback reason=%s", reason)  # alert on the fallback rate (Module 4, LU4.9)
            return respond("fallback", evidence=evidence, fallback_reason=reason)

        # 2. Generate a structured answer from the selected evidence.
        try:
            out = self.chain.invoke({"question": question, "evidence": format_evidence(evidence)})
        except Exception as exc:  # timeout, rate limit, provider error, unparsable output
            return fallback(f"llm_error:{type(exc).__name__}")
        if not isinstance(out, TriageAnswer):
            return fallback("invalid_output")

        # 3. Policy: structurally valid is not the same as allowed.
        if out.insufficient_evidence:
            return respond("insufficient_evidence", evidence=evidence, confidence=out.confidence)
        by_id = {e.eid: e for e in evidence}
        if any(c not in by_id for c in out.citations):
            return fallback("unknown_citation")
        if not out.answer or not out.citations:
            return fallback("uncited_answer")

        return respond(
            "grounded",
            evidence=evidence,
            answer=out.answer,
            confidence=out.confidence,
            citations=[by_id[c] for c in dict.fromkeys(out.citations)],
        )
