"""RAG pipeline: retrieve -> build context -> LLM -> validate -> policy -> answer or fallback.

The model suggests; the system controls . Anything that fails validation degrades to the
raw search results instead of an error, so engineers can still do the job during an outage.
"""
import logging
import time
from typing import Callable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_settings
from app.embeddings import get_embeddings
from app.models import AskResponse, Evidence, Scope, TriageAnswer
from app.prompts import PROMPT, PROMPT_VERSION, format_evidence
from app.retriever import FirelineRetriever

log = logging.getLogger("fireline.rag")


def build_chat_model(max_tokens: int, timeout: float, max_retries: int, model: str | None = None) -> ChatGoogleGenerativeAI:
    s = get_settings()
    # max_tokens is generous because Gemini "thinking" tokens count against the output budget.
    return ChatGoogleGenerativeAI(
        model=model or s.llm_model, google_api_key=s.google_api_key, max_output_tokens=max_tokens, timeout=timeout, max_retries=max_retries
    )


def build_structured_llm() -> Runnable:
    s = get_settings()
    # Timeout is not retry: max_retries=0. Instead a 503 "high demand" (fast failure) switches to a
    # second model; if that also fails, RagService degrades to raw evidence.
    primary = build_chat_model(2000, s.llm_timeout_s, 0).with_structured_output(TriageAnswer, method="json_schema")
    if not s.llm_fallback_model:
        return primary
    backup = build_chat_model(2000, s.llm_timeout_s, 0, model=s.llm_fallback_model).with_structured_output(TriageAnswer, method="json_schema")
    return primary.with_fallbacks([backup])


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
            # Re-validate through this module's Scope: Streamlit reloads app.models on code changes, so a
            # cached RagService can receive a Scope from a newer copy of the class.
            scope=Scope.model_validate(scope.model_dump()),
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

        def fallback(reason: str, exc: Exception | None = None) -> AskResponse:
            log.warning("rag_fallback reason=%s", reason, exc_info=exc)  # alert on the fallback rate (Module 4, LU4.9)
            return respond("fallback", evidence=evidence, fallback_reason=reason)

        # 2. Generate a structured answer from the selected evidence.
        try:
            out = self.chain.invoke({"question": question, "evidence": format_evidence(evidence)})
        except Exception as exc:  # timeout, rate limit, provider error, unparsable output
            return fallback(f"llm_error:{type(exc).__name__}", exc)
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
