"""Retrieval + grounding evaluation (Module 3, LU3.9).

Retrieval checks are deterministic and need no LLM: expected evidence present, forbidden evidence
(wrong tenant, region, environment, retired runbook, private note) absent.
Generation checks (`--generate`) call the LLM. `--judge` adds claim-level support checking.
"""
import json
from pathlib import Path
from typing import Literal

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.config import get_settings
from app.embeddings import get_embeddings
from app.ids import tenant_uuid
from app.models import Scope
from app.prompts import format_evidence
from app.rag import RagService
from app.retriever import FirelineRetriever


class ClaimVerdict(BaseModel):
    claim: str
    verdict: Literal["supported", "unsupported"]
    reason: str


class JudgeResult(BaseModel):
    claims: list[ClaimVerdict]


JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You verify answers against evidence. Split the ANSWER into atomic factual claims. For each claim "
     "decide whether the EVIDENCE directly supports it. 'A happened before B' does NOT support 'A caused B'. "
     "A claim that is plausible but not stated in the evidence is unsupported."),
    ("human", "QUESTION\n{question}\n\nEVIDENCE\n{evidence}\n\nANSWER\n{answer}"),
])


def _scope(raw: dict) -> Scope:
    return Scope(tenant_id=tenant_uuid(raw["tenant"]), service=raw.get("service"), environment=raw.get("environment"), region=raw.get("region"))


def run(path: Path, generate: bool = False, judge: bool = False) -> bool:
    s = get_settings()
    cases = json.loads(path.read_text())
    service = RagService() if generate else None
    judge_llm = ChatAnthropic(model=s.llm_model, max_tokens=1200, timeout=30, max_retries=1).with_structured_output(JudgeResult, method="json_schema") if judge else None

    all_ok = True
    for case in cases:
        scope = _scope(case["scope"])
        retriever = FirelineRetriever(embeddings=get_embeddings(), scope=scope, k_incidents=s.k_incidents, k_runbooks=s.k_runbooks, min_similarity=s.min_similarity)
        docs = retriever.invoke(case["question"])
        inc_refs = {d.metadata["ref"] for d in docs if d.metadata["kind"] == "incident"}
        rb_refs = {d.metadata["ref"] for d in docs if d.metadata["kind"] == "runbook"}
        problems: list[str] = []

        missing = [r for r in case.get("expect_incidents", []) if r not in inc_refs] + [r for r in case.get("expect_runbooks", []) if r not in rb_refs]
        if missing:
            problems.append(f"missing evidence (recall): {missing}")
        leaked = [r for r in case.get("forbid_incidents", []) if r in inc_refs] + [r for r in case.get("forbid_runbooks", []) if r in rb_refs]
        if leaked:
            problems.append(f"SCOPE LEAK: {leaked}")
        text_leaks = [t for t in case.get("forbid_text", []) if any(t.lower() in d.page_content.lower() for d in docs)]
        if text_leaks:
            problems.append(f"TEXT LEAK: {text_leaks}")

        if generate and "generation" in case:
            res = service.ask(case["question"], scope)
            allowed = case["generation"].get("allowed_modes")
            if allowed and res.mode not in allowed:
                problems.append(f"mode {res.mode!r} not in {allowed} (fallback_reason={res.fallback_reason})")
            if judge_llm and res.answer:
                verdict = judge_llm.invoke(JUDGE_PROMPT.format_messages(question=case["question"], evidence=format_evidence(res.evidence), answer=res.answer))
                bad = [c for c in verdict.claims if c.verdict == "unsupported"]
                if bad:
                    problems.append("UNSUPPORTED CLAIMS: " + "; ".join(f"{c.claim} ({c.reason})" for c in bad))

        status = "PASS" if not problems else "FAIL"
        all_ok &= not problems
        print(f"[{status}] {case['id']}  incidents={sorted(inc_refs)} runbooks={sorted(rb_refs)}")
        for p in problems:
            print(f"        - {p}")
    print("\nALL PASSED" if all_ok else "\nFAILURES FOUND")
    return all_ok
