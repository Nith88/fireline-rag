"""Chunk -> embed -> store"""
import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app import store
from app.embeddings import get_embeddings
from app.ids import tenant_uuid, uid

BATCH = 100  # Chroma Cloud caps records per request

_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)

ChunkRow = tuple[str, int, str, str]  # (source, chunk_index, text, visibility)


def incident_chunks(inc: dict) -> list[ChunkRow]:
    rows: list[ChunkRow] = [("title", 0, inc["title"], "internal")]
    rows += [("description", i, t, "internal") for i, t in enumerate(_splitter.split_text(inc.get("description", "")))]
    rows += [
        ("timeline", i, e["text"], e.get("visibility", "internal"))
        for i, e in enumerate(inc.get("timeline", []))
    ]
    rows += [("log", i, t, "internal") for i, t in enumerate(_splitter.split_text("\n".join(inc.get("logs", []))))]
    rows += [("resolution", i, t, "internal") for i, t in enumerate(_splitter.split_text(inc.get("resolution", "")))]
    return rows


def _check_dim(vectors: list[list[float]]) -> None:
    want = get_settings().embedding_dim
    if vectors and len(vectors[0]) != want:
        raise ValueError(f"Embedding has {len(vectors[0])} dims but EMBEDDING_DIM={want}. Fix config and re-run migrate.")


def _replace_chunks(col, owner_field: str, owner_id: str, ids, documents, embeddings, metadatas) -> None:
    """Delete the owner's old chunks, then add the new ones (re-ingesting never duplicates)."""
    col.delete(where={owner_field: {"$eq": owner_id}})
    for i in range(0, len(ids), BATCH):
        col.add(ids=ids[i:i + BATCH], documents=documents[i:i + BATCH], embeddings=embeddings[i:i + BATCH], metadatas=metadatas[i:i + BATCH])


def ingest_incidents(path: Path) -> int:
    s, emb = get_settings(), get_embeddings()
    incidents = json.loads(path.read_text())
    col = store.collection(store.INCIDENTS)
    for inc in incidents:
        tenant = str(tenant_uuid(inc["tenant"]))
        inc_id = str(uid("incident", inc["tenant"], inc["key"]))
        chunks = incident_chunks(inc)
        # Prefix non-title chunks with the title: short chunks embed better with their context.
        texts = [t if src == "title" else f'{inc["title"]}\n{t}' for src, _, t, _ in chunks]
        vectors = emb.embed_documents(texts)
        _check_dim(vectors)

        _replace_chunks(
            col, "incident_id", inc_id,
            ids=[f"{inc_id}:{src}:{idx}" for src, idx, _, _ in chunks],
            documents=[text for _, _, text, _ in chunks],
            embeddings=vectors,
            metadatas=[
                store.clean({
                    "tenant_id": tenant, "incident_id": inc_id, "ref": inc["key"], "source": src, "chunk_index": idx,
                    "embedding_model": s.embedding_model_id, "service": inc.get("service"),
                    "environment": inc.get("environment"), "region": inc.get("region"), "visibility": vis,
                })
                for src, idx, _, vis in chunks
            ],
        )
    return len(incidents)


def ingest_runbooks(path: Path) -> int:
    s, emb = get_settings(), get_embeddings()
    runbooks = json.loads(path.read_text())
    col = store.collection(store.RUNBOOKS)
    for rb in runbooks:
        tenant = str(tenant_uuid(rb["tenant"]))
        rb_id = str(uid("runbook", rb["tenant"], rb["slug"], str(rb["version"])))
        pieces = _splitter.split_text(rb["content"])
        vectors = emb.embed_documents([f'{rb["title"]}\n{p}' for p in pieces])
        _check_dim(vectors)

        _replace_chunks(
            col, "runbook_id", rb_id,
            ids=[f"{rb_id}:{i}" for i in range(len(pieces))],
            documents=pieces,
            embeddings=vectors,
            metadatas=[
                store.clean({
                    "tenant_id": tenant, "runbook_id": rb_id, "ref": f'{rb["slug"]} v{rb["version"]}', "slug": rb["slug"],
                    "chunk_index": i, "embedding_model": s.embedding_model_id, "status": rb["status"],
                    "service": rb.get("service"), "region": rb.get("region") or store.ANY_REGION,
                })
                for i in range(len(pieces))
            ],
        )
    return len(runbooks)
