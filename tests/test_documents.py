import pytest
from langchain_core.runnables import RunnableLambda

from app import store
from app.analytics import Report, ReportSection
from app.documents import (DocumentError, DocumentReporter, DocumentRetriever, delete_document, document_service,
                           extract_pages, ingest_document, list_documents)
from app.embeddings import get_embeddings
from app.ids import tenant_uuid
from app.models import Scope, TriageAnswer
from tests.pdfs import make_pdf

pytestmark = pytest.mark.usefixtures("db")

POSTMORTEM = (
    "Incident INC-7001 postmortem. On 14 March the checkout service returned errors for 40 minutes. "
    "Root cause: a connection pool limit of 20 was exhausted after a deploy leaked database connections. "
    "Resolution: the deploy was rolled back and the pool limit was raised from 20 to 40. "
) * 2
PAGES = [POSTMORTEM, "Follow-up: add connection pool alerts. " * 8]


@pytest.fixture
def acme_pdf():
    t = tenant_uuid("acme")
    info = ingest_document(t, "postmortem-7001.pdf", make_pdf(PAGES))
    yield t, info
    delete_document(t, info.doc_id)


def report(text="The checkout service failed for 40 minutes.", refs=("INC-7001",)):
    return Report(title="Postmortem", summary=text, sections=[ReportSection(heading="Cause", body=text)], incident_refs=list(refs))


def test_extracts_text_per_page():
    pages = extract_pages(make_pdf(PAGES))
    assert [n for n, _ in pages] == [1, 2] and "INC-7001" in pages[0][1]


def test_rejects_non_pdf_scanned_and_corrupt_files():
    with pytest.raises(DocumentError, match="not a PDF"):
        extract_pages(b"hello")
    with pytest.raises(DocumentError, match="no extractable text"):
        extract_pages(make_pdf(["tiny"]))  # stands in for a scanned PDF: almost no text
    with pytest.raises(DocumentError, match="could not be read"):
        extract_pages(b"%PDF-1.4\ngarbage")


def test_ingest_lists_and_is_idempotent(acme_pdf):
    t, info = acme_pdf
    again = ingest_document(t, "renamed.pdf", make_pdf(PAGES))
    assert again.doc_id == info.doc_id  # same content = same document, replaced not duplicated
    docs = list_documents(t)
    assert [d.doc_id for d in docs] == [info.doc_id] and docs[0].filename == "renamed.pdf" and docs[0].pages == 2


def test_documents_are_tenant_isolated(acme_pdf):
    t, info = acme_pdf
    assert list_documents(tenant_uuid("globex")) == []
    delete_document(tenant_uuid("globex"), info.doc_id)  # another tenant cannot delete it
    assert [d.doc_id for d in list_documents(t)] == [info.doc_id]
    r = DocumentRetriever(embeddings=get_embeddings(), tenant_id=str(tenant_uuid("globex")), k=20, min_similarity=-1)
    assert r.invoke("connection pool") == []


def test_retriever_returns_scoped_cited_chunks(acme_pdf):
    t, _ = acme_pdf
    docs = DocumentRetriever(embeddings=get_embeddings(), tenant_id=str(t), k=20, min_similarity=-1).invoke("what caused the outage")
    assert docs and all(d.metadata["kind"] == "document" and d.metadata["ref"].startswith("postmortem") for d in docs)
    nothing = DocumentRetriever(embeddings=get_embeddings(), tenant_id=str(t), doc_ids=[], k=20, min_similarity=-1).invoke("x")
    assert nothing == []


def test_question_answering_reuses_the_rag_policy(acme_pdf):
    t, _ = acme_pdf
    svc = document_service()
    svc._llm = RunnableLambda(lambda _: TriageAnswer(insufficient_evidence=False, answer="Pool exhausted.", citations=["D-nope"], confidence=0.9))
    res = svc.ask("what happened?", Scope(tenant_id=t))
    assert res.evidence and res.mode == "fallback" and res.fallback_reason == "unknown_citation"  # invented citation rejected


def test_report_grounded_in_the_document(acme_pdf):
    t, info = acme_pdf
    seen = []
    res = DocumentReporter(RunnableLambda(lambda m: seen.append(m[-1].content) or report())).run("summarise", t, [info.doc_id])
    assert res.mode == "report" and res.documents == ["postmortem-7001.pdf"] and not res.truncated
    assert "connection pool limit" in seen[0] and "<document" in seen[0]


def test_report_rejects_identifiers_not_in_the_document(acme_pdf):
    t, info = acme_pdf
    bad = DocumentReporter(RunnableLambda(lambda _: report("See INC-9999 too.", refs=()))).run("summarise", t, [info.doc_id])
    assert bad.mode == "fallback" and bad.fallback_reason == "unknown_identifier"
    ok = DocumentReporter(RunnableLambda(lambda _: report("See INC 7001.", refs=()))).run("summarise", t, [info.doc_id])
    assert ok.mode == "report"  # spacing/punctuation differences do not trip the check


def test_report_cannot_read_another_tenants_document(acme_pdf):
    _, info = acme_pdf
    res = DocumentReporter(RunnableLambda(lambda _: report())).run("summarise", tenant_uuid("globex"), [info.doc_id])
    assert res.mode == "no_data"


def test_report_llm_failure_falls_back(acme_pdf):
    t, info = acme_pdf

    def boom(_):
        raise TimeoutError

    res = DocumentReporter(RunnableLambda(boom)).run("summarise", t, [info.doc_id])
    assert res.mode == "fallback" and res.fallback_reason == "llm_error:TimeoutError"


def test_deleting_removes_chunks(acme_pdf):
    t, info = acme_pdf
    delete_document(t, info.doc_id)
    assert list_documents(t) == []
    assert store.collection(store.DOCUMENTS).get(where={"doc_id": {"$eq": info.doc_id}})["ids"] == []
