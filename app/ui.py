"""Streamlit UI: streamlit run app/ui.py  (or: python -m app.cli ui)"""
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analytics import AnalyticsAgent, AnalyticsResult  # noqa: E402
from app.documents import (DocumentError, DocumentReporter, DocumentReportResult, delete_document, document_service,  # noqa: E402
                           ingest_document, list_documents)
from app.emailer import EmailError, email_configured, send_report  # noqa: E402
from app.ids import tenant_uuid  # noqa: E402
from app.models import AskResponse, Evidence, Scope  # noqa: E402
from app.rag import RagService  # noqa: E402
from app.store import STORE_ERRORS  # noqa: E402

MODE_BADGE = {
    "grounded": ("Grounded", "green"),
    "insufficient_evidence": ("Insufficient evidence", "orange"),
    "fallback": ("Fallback: raw evidence only", "red"),
}
REPORT_EXAMPLES = [
    "Give me an overview report of our incidents",
    "What are the recurring root causes and how were they fixed?",
    "Which incidents are still open and how severe are they?",
]
EXAMPLES = [
    "Have we seen VPN timeout after login before?",
    "How do I resolve a VPN authentication provider latency issue?",
    "What happened with the database connection failure?",
]

st.set_page_config(page_title="Fireline RAG", page_icon="🔥", layout="wide")

try:
    google_api_key = st.secrets.get("GOOGLE_API_KEY")
    chroma_secrets = {k: st.secrets.get(k) for k in ("CHROMA_API_KEY", "CHROMA_TENANT", "CHROMA_DATABASE", "SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASSWORD", "SMTP_FROM", "EMAIL_ALLOWLIST")}
except StreamlitSecretNotFoundError:
    google_api_key = None
    chroma_secrets = {}
if google_api_key and not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = str(google_api_key)
for _key, _value in chroma_secrets.items():
    if _value and not os.environ.get(_key):
        os.environ[_key] = str(_value)


@st.cache_resource
def get_service() -> RagService:
    return RagService()


@st.cache_resource
def get_analytics_agent() -> AnalyticsAgent:
    return AnalyticsAgent()


def render_evidence(items: list[Evidence], cited: set[str]) -> None:
    if not items:
        st.caption("Nothing retrieved for this scope.")
        return
    for e in items:
        mark = "✅ cited · " if e.eid in cited else ""
        with st.expander(f"{mark}{e.eid} · {e.ref} · {e.source} · sim {e.similarity:.3f}"):
            st.write(e.text)


def render_response(res: AskResponse) -> None:
    label, color = MODE_BADGE[res.mode]
    st.markdown(f":{color}-badge[{label}]")

    if res.mode == "grounded":
        st.markdown(res.answer)
    elif res.mode == "insufficient_evidence":
        st.warning("The retrieved evidence does not answer this question.")
    else:
        st.error(f"The model answer was not usable ({res.fallback_reason}). Showing retrieved evidence instead.")

    cols = st.columns(3)
    cols[0].metric("Confidence", f"{res.confidence:.0%}" if res.confidence is not None else "n/a")
    cols[1].metric("Latency", f"{res.latency_ms} ms")
    cols[2].metric("Prompt version", res.prompt_version)

    cited = {c.eid for c in res.citations}
    if res.citations:
        st.subheader("Citations")
        render_evidence(res.citations, cited)
    st.subheader(f"All retrieved evidence ({len(res.evidence)})")
    render_evidence(res.evidence, cited)


with st.sidebar:
    st.title("🔥 Fireline RAG")
    st.caption("Trusted scope. The model never chooses these.")
    tenant = st.text_input("Tenant (name or UUID)", value="acme")
    service = st.text_input("Service", value="vpn")
    environment = st.text_input("Environment", value="production")
    region = st.text_input("Region", value="india")
    st.caption("Leave a filter empty to search across all values.")
    st.divider()
    st.caption("Try an example:")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state["question"] = ex

def current_scope() -> Scope:
    return Scope(
        tenant_id=tenant_uuid(tenant.strip() or "acme"),
        service=service.strip() or None,
        environment=environment.strip() or None,
        region=region.strip() or None,
    )


def render_overview(o) -> None:
    cols = st.columns(3)
    cols[0].metric("Incidents in scope", o.total)
    cols[1].metric("Open", o.open)
    cols[2].metric("P1 / P2", f'{o.by_severity.get("P1", 0)} / {o.by_severity.get("P2", 0)}')
    for title, counts in (("By severity", o.by_severity), ("By service", o.by_service), ("By region", o.by_region), ("By status", o.by_status)):
        if counts:
            st.caption(title)
            st.bar_chart(pd.Series(counts), height=160)
    if o.open_incidents:
        st.caption("Open incidents")
        st.table(pd.DataFrame(o.open_incidents))


def render_report(res: AnalyticsResult) -> None:
    if res.mode == "report":
        st.markdown(":green-badge[Report]")
        st.subheader(res.report.title)
        st.markdown(res.report.summary)
        for sec in res.report.sections:
            st.markdown(f"**{sec.heading}**")
            st.markdown(sec.body)
    elif res.mode == "no_data":
        st.warning("No incidents found for this scope. Check the tenant and filters in the sidebar.")
    else:
        st.error(f"The model report was not usable ({res.fallback_reason}). Showing the computed overview instead.")
    st.caption(f"{res.latency_ms} ms · {res.prompt_version}")
    if res.steps:
        with st.expander(f"Agent steps ({len(res.steps)})"):
            for step in res.steps:
                st.code(step, language="text")
    st.subheader("Overview (computed, not model-written)")
    render_overview(res.overview)


def render_doc_report(res: DocumentReportResult) -> None:
    if res.mode == "report":
        st.markdown(":green-badge[Report]")
        st.subheader(res.report.title)
        st.markdown(res.report.summary)
        for sec in res.report.sections:
            st.markdown(f"**{sec.heading}**")
            st.markdown(sec.body)
    elif res.mode == "no_data":
        st.warning("Select at least one indexed document first.")
    else:
        st.error(f"The model report was not usable ({res.fallback_reason}). Try rephrasing or generate it again.")
    if res.truncated:
        st.warning("The documents were too long; the report is based on the first part only.")
    st.caption(f"{', '.join(res.documents)} · {res.latency_ms} ms · {res.prompt_version}")


def render_email_box(res: DocumentReportResult) -> None:
    st.subheader("Email this report")
    if not email_configured():
        st.caption("Email is not configured on this server (set SMTP_HOST, SMTP_FROM and EMAIL_ALLOWLIST).")
        return
    to = st.text_input("Send to", key="email_to", placeholder="name@your-company.com")
    st.caption("Reports can only be sent to addresses on the server's allowed recipients list.")
    if st.button("Send email", disabled=not to.strip()):
        try:
            with st.spinner("Sending..."):
                send_report(to, res.report, res.documents)
        except EmailError as exc:
            st.error(str(exc))
        else:
            st.success(f"Sent to {to.strip()}.")


def show_store_error() -> None:
    st.error("Retrieval is unavailable (database error). Is Chroma reachable and ingested?")


tab_ask, tab_reports, tab_docs = st.tabs(["Ask", "Reports", "Documents"])

with tab_ask:
    st.header("Ask about past incidents and runbooks")
    question = st.text_area("Question", key="question", height=90, placeholder="Have we seen VPN timeout after login before?")

    if st.button("Ask", type="primary", disabled=len(question.strip()) < 3):
        try:
            with st.spinner("Retrieving evidence and asking Gemini..."):
                res = get_service().ask(question.strip()[:1000], current_scope())
        except STORE_ERRORS:
            show_store_error()
        except Exception as exc:  # missing API keys etc.
            st.error(f"{type(exc).__name__}: {exc}")
        else:
            render_response(res)

with tab_reports:
    st.header("Incident analytics report")
    st.caption("An agent queries the incident records within the sidebar scope and writes a report. "
               "Clear the Service / Environment / Region filters to report across everything for the tenant.")
    for ex in REPORT_EXAMPLES:
        if st.button(ex, key=f"rex-{ex}"):
            st.session_state["report_request"] = ex
    request = st.text_area("What should the report cover?", key="report_request", height=90,
                           placeholder="Give me an overview report of our incidents")

    if st.button("Generate report", type="primary", disabled=len(request.strip()) < 3):
        try:
            with st.spinner("Gathering data and writing the report..."):
                report_res = get_analytics_agent().run(request.strip()[:1000], current_scope())
        except STORE_ERRORS:
            show_store_error()
        except Exception as exc:  # missing API keys etc.
            st.error(f"{type(exc).__name__}: {exc}")
        else:
            render_report(report_res)

with tab_docs:
    st.header("Incident report PDFs")
    st.caption("Upload text-based incident report PDFs (max 10 MB, 100 pages). They are indexed for the tenant "
               "in the sidebar and are only visible to that tenant. Scanned/image-only PDFs are not supported.")
    tenant_id = tenant_uuid(tenant.strip() or "acme")

    uploads = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    if uploads and st.button("Index uploaded PDFs", type="primary"):
        for f in uploads:
            try:
                with st.spinner(f"Indexing {f.name}..."):
                    info = ingest_document(tenant_id, f.name, f.getvalue())
            except DocumentError as exc:
                st.error(f"{f.name}: {exc}")
            except STORE_ERRORS:
                show_store_error()
            except Exception as exc:  # missing API keys, embedding quota etc.
                st.error(f"{f.name}: {type(exc).__name__}: {exc}")
            else:
                st.success(f"{info.filename}: {info.pages} pages, {info.chunks} chunks indexed.")

    try:
        docs = list_documents(tenant_id)
    except STORE_ERRORS:
        docs = []
        show_store_error()
    by_id = {d.doc_id: d for d in docs}
    selected = st.multiselect(
        "Documents to use", options=list(by_id), default=list(by_id), format_func=lambda i: f"{by_id[i].filename} ({by_id[i].pages} pages)",
        placeholder="No documents indexed yet" if not docs else "Choose documents",
    )
    if selected and st.button("Delete selected documents"):
        for i in selected:
            delete_document(tenant_id, i)
        st.rerun()

    st.subheader("Ask a question")
    doc_question = st.text_area("Question about the documents", key="doc_question", height=80,
                                placeholder="What was the root cause and how was it fixed?")
    if st.button("Ask documents", disabled=len(doc_question.strip()) < 3 or not selected):
        try:
            with st.spinner("Retrieving from the documents and asking Gemini..."):
                doc_res = document_service(selected).ask(doc_question.strip()[:1000], Scope(tenant_id=tenant_id))
        except STORE_ERRORS:
            show_store_error()
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")
        else:
            render_response(doc_res)

    st.subheader("Generate a report")
    doc_request = st.text_area("What should the report cover?", key="doc_request", height=80,
                               placeholder="Summarise the incident: timeline, root cause, impact and follow-up actions")
    if st.button("Generate document report", type="primary", disabled=len(doc_request.strip()) < 3 or not selected):
        try:
            with st.spinner("Reading the documents and writing the report..."):
                st.session_state["doc_report"] = DocumentReporter().run(doc_request.strip()[:1000], tenant_id, selected)
        except STORE_ERRORS:
            show_store_error()
        except Exception as exc:
            st.error(f"{type(exc).__name__}: {exc}")
    # Kept in session state so the email button (which reruns the script) does not lose the report.
    doc_report = st.session_state.get("doc_report")
    if doc_report is not None:
        render_doc_report(doc_report)
        if doc_report.mode == "report":
            render_email_box(doc_report)
