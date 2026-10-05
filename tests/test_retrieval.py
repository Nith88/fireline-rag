import pytest

from app.config import get_settings
from app.db import connect
from app.embeddings import get_embeddings
from app.ids import tenant_uuid
from app.models import Scope
from app.retriever import FirelineRetriever

pytestmark = pytest.mark.usefixtures("db")


def retrieve(tenant="acme", **scope):
    r = FirelineRetriever(embeddings=get_embeddings(), scope=Scope(tenant_id=tenant_uuid(tenant), **scope), k_incidents=50, k_runbooks=50, min_similarity=-1)
    docs = r.invoke("vpn timeout after login")
    return (
        {d.metadata["ref"] for d in docs if d.metadata["kind"] == "incident"},
        {d.metadata["ref"] for d in docs if d.metadata["kind"] == "runbook"},
        docs,
    )


def test_tenant_isolation():
    inc, rb, _ = retrieve("globex")
    assert inc == {"INC2001"}
    assert rb == set()  # acme runbooks are invisible to globex


def test_acme_never_sees_globex():
    inc, _, _ = retrieve("acme")
    assert "INC2001" not in inc


def test_metadata_filters_scope_incidents():
    inc, _, _ = retrieve(service="vpn", environment="production", region="india")
    assert inc == {"INC1001", "INC1002", "INC1005"}


def test_staging_is_separate_from_production():
    inc, _, _ = retrieve(service="vpn", environment="staging")
    assert inc == {"INC1006"}


def test_private_notes_are_never_retrieved():
    _, _, docs = retrieve()
    assert not any("Initech" in d.page_content for d in docs)
    with connect() as conn:  # ...but the private chunk does exist in storage
        n = conn.execute("SELECT count(*) AS n FROM incident_chunks WHERE visibility='private'").fetchone()["n"]
    assert n == 1


def test_retired_runbooks_are_excluded():
    _, rb, _ = retrieve(service="vpn", region="india")
    assert rb == {"vpn-india v3"}


def test_region_filter_keeps_global_runbooks():
    _, rb, _ = retrieve(service="payments", region="india")
    assert rb == {"payments-db v1"}  # runbook region is NULL = applies everywhere


def test_vectors_from_other_embedding_models_are_ignored():
    with connect() as conn:
        conn.execute("UPDATE incident_chunks SET embedding_model = 'other:model'")
    try:
        inc, _, _ = retrieve()
        assert inc == set()
    finally:
        with connect() as conn:
            conn.execute("UPDATE incident_chunks SET embedding_model = %s", (get_settings().embedding_model_id,))


def test_deleting_an_incident_cascades_to_its_chunks():
    with connect() as conn:
        conn.execute("DELETE FROM incidents WHERE external_key = 'INC1007'")
    inc, _, _ = retrieve(service="notifications")
    assert "INC1007" not in inc
    with connect() as conn:
        left = conn.execute("SELECT count(*) AS n FROM incident_chunks WHERE service='notifications'").fetchone()["n"]
    assert left == 0
