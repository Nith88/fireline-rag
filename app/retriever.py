"""A LangChain retriever over the Fireline Chroma collections.

Isolation is enforced as metadata filters on every query: tenant, embedding model, visibility and
runbook status. Chroma has no joins, so each chunk carries the fields retrieval needs.
"""
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from app import store
from app.config import get_settings
from app.models import Scope


class FirelineRetriever(BaseRetriever):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    embeddings: Embeddings
    scope: Scope
    k_incidents: int = 5
    k_runbooks: int = 3
    min_similarity: float = 0.25

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        q = self.embeddings.embed_query(query)
        model = get_settings().embedding_model_id
        tenant = str(self.scope.tenant_id)

        incident_where = [
            store.eq("tenant_id", tenant),        # never search another tenant
            store.eq("embedding_model", model),   # never mix embedding models
            {"visibility": {"$ne": "private"}},   # private notes are not evidence
        ]
        runbook_where = [
            store.eq("tenant_id", tenant),
            store.eq("embedding_model", model),
            store.eq("status", "active"),         # stale runbooks are not evidence
        ]
        for field in ("service", "environment", "region"):
            if value := getattr(self.scope, field):
                incident_where.append(store.eq(field, value))
        if self.scope.service:
            runbook_where.append(store.eq("service", self.scope.service))
        if self.scope.region:  # runbooks with no region apply everywhere
            runbook_where.append({"region": {"$in": [self.scope.region, store.ANY_REGION]}})

        incidents = self._query(store.INCIDENTS, q, store.all_of(*incident_where), self.k_incidents)
        runbooks = self._query(store.RUNBOOKS, q, store.all_of(*runbook_where), self.k_runbooks)

        docs = [
            Document(
                page_content=text,
                metadata={"eid": f"I-{meta['ref']}-{meta['source']}-{meta['chunk_index']}", "kind": "incident", "ref": meta["ref"], "source": meta["source"], "similarity": sim},
            )
            for text, meta, sim in incidents
        ] + [
            Document(
                page_content=text,
                metadata={"eid": f"R-{meta['ref'].replace(' ', '-')}-{meta['chunk_index']}", "kind": "runbook", "ref": meta["ref"], "source": "runbook", "similarity": sim},
            )
            for text, meta, sim in runbooks
        ]
        docs = [d for d in docs if d.metadata["similarity"] >= self.min_similarity]
        return sorted(docs, key=lambda d: d.metadata["similarity"], reverse=True)

    @staticmethod
    def _query(kind: str, vector: list[float], where: dict, k: int) -> list[tuple[str, dict, float]]:
        res = store.collection(kind).query(
            query_embeddings=[vector], n_results=k, where=where, include=["documents", "metadatas", "distances"]
        )
        # Cosine distance -> similarity, same scale as the old `1 - (embedding <=> q)`.
        return list(zip(res["documents"][0], res["metadatas"][0], (1 - d for d in res["distances"][0])))
