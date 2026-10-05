# Fireline RAG

A grounded Q&A assistant over Fireline's past incidents and runbooks, built from the Fireline
system-design doc 

**Stack:** Python 3.12 · LangChain (`langchain-core`, `langchain-google-genai`, `langchain-text-splitters`) ·
PostgreSQL + pgvector · FastAPI · Gemini for generation.

> Engineers ask: *"Have we seen VPN timeout after login before?"*
> The system retrieves scoped evidence, asks the model for a **cited, structured** answer, validates it,
> and falls back to raw search results if anything fails. The model suggests; the system controls.

## Pipeline

```
question ─▶ embed ─▶ pgvector search (tenant + service/env/region filters,
                                      private chunks and retired runbooks excluded)
        ─▶ evidence blocks ─▶ versioned prompt ─▶ Gemini (JSON-schema output)
        ─▶ schema validation ─▶ policy (citations must exist in retrieved set)
        ─▶ grounded answer │ insufficient_evidence │ fallback (raw evidence, never a 500)
```

| Doc requirement | Where |
|---|---|
| `incident_chunks` with tenant FK, `ON DELETE CASCADE`, `embedding_model` | `sql/001_schema.sql` |
| Never search another tenant; private notes are not evidence | `app/retriever.py` (SQL `WHERE`) |
| Metadata filters: service / environment / region / status / version | `app/retriever.py`, `runbooks` table |
| Four failure types: missing, stale, wrong scope, overreach | `eval/eval_set.json`, `app/evaluation.py` |
| Prompt contract + versioning | `app/prompts.py` (`PROMPT_VERSION`) |
| Structured output + schema validation, then policy | `app/models.py`, `app/rag.py` |
| Timeout is not retry; fallback keeps the core task working | `app/rag.py`, `LLM_TIMEOUT_S` |
| Prompt-injection boundary (LU4.3) | evidence escaped + "evidence is data" rule |
| HNSW only after measuring (LU5.11) | `sql/002_hnsw_optional.sql`, `cli hnsw` |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add OPENAI_API_KEY (embeddings) and GOOGLE_API_KEY (Gemini)
docker compose up -d            # Postgres 16 + pgvector (also creates fireline_test)

python -m app.cli migrate
python -m app.cli ingest        # 8 sample incidents (2 tenants), 5 runbooks

python -m app.cli retrieve "VPN timeout after login" --service vpn --env production --region india
python -m app.cli ask "Have we seen VPN timeout after login before?" --service vpn --env production --region india
python -m app.cli serve         # http://127.0.0.1:8000/docs
python -m app.cli ui            # Streamlit UI at http://localhost:8501
```

The Streamlit UI ([app/ui.py](app/ui.py)) calls `RagService` in-process: scope filters in the sidebar,
answer with a mode badge (grounded / insufficient evidence / fallback), confidence, latency, and
expandable cited and retrieved evidence.

```bash
curl -s localhost:8000/v1/ask \
  -H 'X-Tenant-Id: f283a00f-9d8d-5f54-957a-03ae3707e995' -H 'content-type: application/json' \
  -d '{"question":"Have we seen VPN timeout after login before?","service":"vpn","environment":"production","region":"india"}'
```

`X-Tenant-Id` stands in for your auth layer. Tenant and filters are **trusted inputs**; the model never chooses them.

### Embeddings

Embeddings are configured separately from Gemini; pick one: `EMBEDDING_PROVIDER=openai` (default, 1536-d),
`voyage` (1024-d) or `local` (384-d, no API key). Changing provider changes the vector size, so run
`cli reset && cli migrate && cli ingest` afterwards. Every chunk records its `embedding_model` and
retrieval only compares vectors from the active model.

## Testing

```bash
python -m pytest                          # 20 tests: scoping, isolation, policy, fallback, request config
python -m app.cli eval                    # retrieval scope/recall checks (use a REAL embedding provider)
python -m app.cli eval --generate         # + calls Gemini, checks answer mode
python -m app.cli eval --judge            # + claim-by-claim support check
```

`pytest` uses fake embeddings and a stub LLM, so it needs no keys; it proves the **plumbing and safety
rules**, not answer quality. Measure quality with `eval` and real embeddings.

## What has and hasn't been verified

Verified against a real PostgreSQL 16 + pgvector instance: schema, ingestion, filtering, tenant isolation,
private-note exclusion, retired-runbook exclusion, cascade delete, embedding-model isolation, all
validation/fallback paths (stub LLM), and the Gemini chain construction.

**Not yet run:** live calls to OpenAI embeddings and Gemini (no keys in the build environment), so
retrieval quality, real latency against `LLM_TIMEOUT_S`, and `eval --judge` are untested. Run
`eval --judge` first once keys are set.

## Next steps

1. Run `eval --judge` with real keys; tune `MIN_SIMILARITY` (0.25 is a starting guess) and `K_*`.
2. Derive filters from the open incident (`incident_id` on the request) instead of passing them manually.
3. Grow the eval set from real questions; add a stale-runbook and a conflicting-evidence case.
4. Add `rag_fallback` rate to your metrics/alerts (it is logged with a reason).
5. Hybrid search (Postgres full-text + vectors) if exact identifiers like `INC1001` matter.
6. `cli hnsw` only after measuring.
