"""Streamlit UI: streamlit run app/ui.py  (or: python -m app.cli ui)"""
import os
import sys
from pathlib import Path

import psycopg
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.ids import tenant_uuid  # noqa: E402
from app.models import AskResponse, Evidence, Scope  # noqa: E402
from app.rag import RagService  # noqa: E402

MODE_BADGE = {
    "grounded": ("Grounded", "green"),
    "insufficient_evidence": ("Insufficient evidence", "orange"),
    "fallback": ("Fallback: raw evidence only", "red"),
}
EXAMPLES = [
    "Have we seen VPN timeout after login before?",
    "How do I resolve a VPN authentication provider latency issue?",
    "What happened with the database connection failure?",
]

st.set_page_config(page_title="Fireline RAG", page_icon="🔥", layout="wide")

try:
    google_api_key = st.secrets.get("GOOGLE_API_KEY")
    database_url = st.secrets.get("DATABASE_URL")
except StreamlitSecretNotFoundError:
    google_api_key = None
    database_url = None
if google_api_key and not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = str(google_api_key)
if database_url and not os.environ.get("DATABASE_URL"):
    os.environ["DATABASE_URL"] = str(database_url)


@st.cache_resource
def get_service() -> RagService:
    return RagService()


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

st.header("Ask about past incidents and runbooks")
question = st.text_area("Question", key="question", height=90, placeholder="Have we seen VPN timeout after login before?")

if st.button("Ask", type="primary", disabled=len(question.strip()) < 3):
    scope = Scope(
        tenant_id=tenant_uuid(tenant.strip() or "acme"),
        service=service.strip() or None,
        environment=environment.strip() or None,
        region=region.strip() or None,
    )
    try:
        with st.spinner("Retrieving evidence and asking Gemini..."):
            res = get_service().ask(question.strip()[:1000], scope)
    except psycopg.Error:
        st.error("Retrieval is unavailable (database error). Is Postgres up and migrated/ingested?")
    except Exception as exc:  # missing API keys etc.
        st.error(f"{type(exc).__name__}: {exc}")
    else:
        render_response(res)
