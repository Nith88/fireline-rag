"""Chroma storage. Chroma Cloud when CHROMA_API_KEY is set, otherwise a local on-disk database.

Chroma has no joins, foreign keys or NULLs, so the rules the SQL schema enforced live here:
  * tenant / visibility / status / embedding-model checks are metadata filters on every query,
  * every chunk carries what retrieval needs (ref, service, region, runbook status), so there is no join,
  * a missing optional field (service, environment, region) is omitted, and NULL runbook region is stored as "*",
  * re-ingesting an incident or runbook deletes its old chunks first (what ON DELETE CASCADE used to give).
"""
from functools import lru_cache

import chromadb
import httpx
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.errors import ChromaError

from app.config import get_settings

INCIDENTS = "incident_chunks"
RUNBOOKS = "runbook_chunks"
ANY_REGION = "*"  # runbook region NULL = applies to every region

# What the API / UI map to "retrieval unavailable".
STORE_ERRORS = (ChromaError, httpx.HTTPError, ConnectionError)


@lru_cache
def get_client() -> ClientAPI:
    s = get_settings()
    if s.chroma_api_key:
        return chromadb.CloudClient(tenant=s.chroma_tenant, database=s.chroma_database, api_key=s.chroma_api_key)
    return chromadb.PersistentClient(path=s.chroma_path)


def collection(kind: str) -> Collection:
    """Cosine distance, so similarity = 1 - distance (same scale as the old pgvector query)."""
    s = get_settings()
    return get_client().get_or_create_collection(f"{s.chroma_prefix}{kind}", configuration={"hnsw": {"space": "cosine"}})


def reset() -> None:
    s = get_settings()
    client = get_client()
    for kind in (INCIDENTS, RUNBOOKS):
        try:
            client.delete_collection(f"{s.chroma_prefix}{kind}")
        except (ValueError, ChromaError):  # not created yet
            pass


def clean(meta: dict) -> dict:
    """Chroma metadata cannot hold None."""
    return {k: v for k, v in meta.items() if v is not None}


def all_of(*clauses: dict) -> dict:
    return clauses[0] if len(clauses) == 1 else {"$and": list(clauses)}


def eq(field: str, value) -> dict:
    return {field: {"$eq": value}}
