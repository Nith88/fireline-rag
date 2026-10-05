"""Chunk -> embed -> store"""
import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.db import connect, vec_literal
from app.embeddings import get_embeddings
from app.ids import tenant_uuid, uid

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


def ingest_incidents(path: Path) -> int:
    s, emb = get_settings(), get_embeddings()
    incidents = json.loads(path.read_text())
    with connect() as conn:
        for inc in incidents:
            tenant = tenant_uuid(inc["tenant"])
            inc_id = uid("incident", inc["tenant"], inc["key"])
            chunks = incident_chunks(inc)
            # Prefix non-title chunks with the title: short chunks embed better with their context.
            texts = [t if src == "title" else f'{inc["title"]}\n{t}' for src, _, t, _ in chunks]
            vectors = emb.embed_documents(texts)
            _check_dim(vectors)

            with conn.transaction():
                conn.execute(
                    """INSERT INTO incidents (id, tenant_id, external_key, title, status, severity, service, environment, region)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (id) DO UPDATE SET title=EXCLUDED.title, status=EXCLUDED.status,
                         severity=EXCLUDED.severity, service=EXCLUDED.service,
                         environment=EXCLUDED.environment, region=EXCLUDED.region""",
                    (inc_id, tenant, inc["key"], inc["title"], inc["status"], inc["severity"],
                     inc.get("service"), inc.get("environment"), inc.get("region")),
                )
                conn.execute("DELETE FROM incident_chunks WHERE incident_id = %s", (inc_id,))
                with conn.cursor() as cur:
                    cur.executemany(
                        """INSERT INTO incident_chunks
                             (tenant_id, incident_id, source, chunk_index, chunk_text, embedding,
                              embedding_model, service, environment, region, visibility)
                           VALUES (%s,%s,%s,%s,%s,%s::vector,%s,%s,%s,%s,%s)""",
                        [
                            (tenant, inc_id, src, idx, text, vec_literal(vec), s.embedding_model_id,
                             inc.get("service"), inc.get("environment"), inc.get("region"), vis)
                            for (src, idx, text, vis), vec in zip(chunks, vectors)
                        ],
                    )
    return len(incidents)


def ingest_runbooks(path: Path) -> int:
    s, emb = get_settings(), get_embeddings()
    runbooks = json.loads(path.read_text())
    with connect() as conn:
        # Retire first so a newly active version never collides with the old active one.
        for rb in sorted(runbooks, key=lambda r: r["status"] != "retired"):
            tenant = tenant_uuid(rb["tenant"])
            rb_id = uid("runbook", rb["tenant"], rb["slug"], str(rb["version"]))
            pieces = _splitter.split_text(rb["content"])
            vectors = emb.embed_documents([f'{rb["title"]}\n{p}' for p in pieces])
            _check_dim(vectors)

            with conn.transaction():
                conn.execute(
                    """INSERT INTO runbooks (id, tenant_id, slug, version, title, status, service, region)
                       VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                       ON CONFLICT (id) DO UPDATE SET title=EXCLUDED.title, status=EXCLUDED.status,
                         service=EXCLUDED.service, region=EXCLUDED.region""",
                    (rb_id, tenant, rb["slug"], rb["version"], rb["title"], rb["status"], rb.get("service"), rb.get("region")),
                )
                conn.execute("DELETE FROM runbook_chunks WHERE runbook_id = %s", (rb_id,))
                with conn.cursor() as cur:
                    cur.executemany(
                        """INSERT INTO runbook_chunks (tenant_id, runbook_id, chunk_index, chunk_text, embedding, embedding_model)
                           VALUES (%s,%s,%s,%s,%s::vector,%s)""",
                        [(tenant, rb_id, i, p, vec_literal(v), s.embedding_model_id) for i, (p, v) in enumerate(zip(pieces, vectors))],
                    )
    return len(runbooks)
