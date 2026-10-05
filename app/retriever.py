"""A LangChain retriever over the Fireline schema.

Why not langchain-postgres PGVector? It stores vectors in its own tables, so it cannot share the
tenant foreign keys, ON DELETE CASCADE or runbook status joins that the schema relies on. This
retriever keeps those guarantees and still plugs into LangChain (`retriever.invoke(question)`).
"""
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from app.config import get_settings
from app.db import connect, vec_literal
from app.models import Scope

INCIDENT_SQL = """
SELECT c.id, i.external_key AS ref, c.source, c.chunk_text,
       1 - (c.embedding <=> %(q)s::vector) AS similarity
FROM incident_chunks c
JOIN incidents i ON i.tenant_id = c.tenant_id AND i.id = c.incident_id
WHERE c.tenant_id = %(tenant)s                              -- never search another tenant
  AND c.embedding_model = %(model)s                         -- never mix embedding models
  AND c.visibility <> 'private'                             -- private notes are not evidence
  AND (%(service)s::text IS NULL     OR c.service     = %(service)s)
  AND (%(environment)s::text IS NULL OR c.environment = %(environment)s)
  AND (%(region)s::text IS NULL      OR c.region      = %(region)s)
ORDER BY c.embedding <=> %(q)s::vector
LIMIT %(k)s
"""

RUNBOOK_SQL = """
SELECT c.id, r.slug || ' v' || r.version AS ref, c.chunk_text,
       1 - (c.embedding <=> %(q)s::vector) AS similarity
FROM runbook_chunks c
JOIN runbooks r ON r.tenant_id = c.tenant_id AND r.id = c.runbook_id
WHERE c.tenant_id = %(tenant)s
  AND c.embedding_model = %(model)s
  AND r.status = 'active'                                   -- stale runbooks are not evidence
  AND (%(service)s::text IS NULL OR r.service = %(service)s)
  AND (%(region)s::text IS NULL  OR r.region IS NULL OR r.region = %(region)s)
ORDER BY c.embedding <=> %(q)s::vector
LIMIT %(k)s
"""


class FirelineRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    embeddings: Embeddings
    scope: Scope
    k_incidents: int = 5
    k_runbooks: int = 3
    min_similarity: float = 0.25

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        params = {
            "q": vec_literal(self.embeddings.embed_query(query)),
            "tenant": self.scope.tenant_id,
            "model": get_settings().embedding_model_id,
            "service": self.scope.service,
            "environment": self.scope.environment,
            "region": self.scope.region,
        }
        with connect() as conn:
            incidents = conn.execute(INCIDENT_SQL, {**params, "k": self.k_incidents}).fetchall()
            runbooks = conn.execute(RUNBOOK_SQL, {**params, "k": self.k_runbooks}).fetchall()

        docs = [
            Document(
                page_content=r["chunk_text"],
                metadata={"eid": f"I-{r['id']}", "kind": "incident", "ref": r["ref"], "source": r["source"], "similarity": float(r["similarity"])},
            )
            for r in incidents
        ] + [
            Document(
                page_content=r["chunk_text"],
                metadata={"eid": f"R-{r['id']}", "kind": "runbook", "ref": r["ref"], "source": "runbook", "similarity": float(r["similarity"])},
            )
            for r in runbooks
        ]
        docs = [d for d in docs if d.metadata["similarity"] >= self.min_similarity]
        return sorted(docs, key=lambda d: d.metadata["similarity"], reverse=True)
