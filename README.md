# Fireline RAG

A grounded Q&A assistant over Fireline's past incidents and runbooks, built from the Fireline
system-design doc.

**🔴 Live demo:** https://fireline-rag.streamlit.app/

**Stack:** Python 3.12 · LangChain (`langchain-core`, `langchain-google-genai`, `langchain-text-splitters`) ·
Chroma Cloud (vector store) · FastAPI · Gemini for generation.

> Engineers ask: *"Have we seen VPN timeout after login before?"*
> The system retrieves scoped evidence, asks the model for a **cited, structured** answer, validates it,
> and falls back to raw search results if anything fails. The model suggests; the system controls.

## Pipeline

```
question ─▶ embed ─▶ Chroma search (tenant + service/env/region filters,
                                      private chunks and retired runbooks excluded)
        ─▶ evidence blocks ─▶ versioned prompt ─▶ Gemini (JSON-schema output)
        ─▶ schema validation ─▶ policy (citations must exist in retrieved set)
        ─▶ grounded answer │ insufficient_evidence │ fallback (raw evidence, never a 500)
```

| Doc requirement | Where |
|---|---|
| `incident_chunks` carrying tenant, `embedding_model`, visibility; re-ingest replaces old chunks | `app/store.py`, `app/ingest.py` |
| Never search another tenant; private notes are not evidence | `app/retriever.py` (Chroma `where` filters) |
| Metadata filters: service / environment / region / status / version | `app/retriever.py`, runbook chunk metadata |
| Four failure types: missing, stale, wrong scope, overreach | `eval/eval_set.json`, `app/evaluation.py` |
| Prompt contract + versioning | `app/prompts.py` (`PROMPT_VERSION`) |
| Structured output + schema validation, then policy | `app/models.py`, `app/rag.py` |
| Timeout is not retry; fallback keeps the core task working | `app/rag.py`, `LLM_TIMEOUT_S` |
| Prompt-injection boundary (LU4.3) | evidence escaped + "evidence is data" rule |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add CHROMA_* (Chroma Cloud), embeddings key and GOOGLE_API_KEY (Gemini)
python -m app.cli migrate
python -m app.cli ingest        # 8 sample incidents (2 tenants), 5 runbooks

python -m app.cli retrieve "VPN timeout after login" --service vpn --env production --region india
python -m app.cli ask "Have we seen VPN timeout after login before?" --service vpn --env production --region india
python -m app.cli serve         # http://127.0.0.1:8000/docs
python -m app.cli ui            # Streamlit UI at http://localhost:8501
```

The Streamlit UI ([app/ui.py](app/ui.py)) calls `RagService` in-process: scope filters in the sidebar,
answer with a mode badge (grounded / insufficient evidence / fallback), confidence, latency, and
expandable cited and retrieved evidence. A hosted copy is running at
[fireline-rag.streamlit.app](https://fireline-rag.streamlit.app/).

### Analytics reports

The **Reports** tab in the UI ([app/ui.py](app/ui.py)) runs an analytics agent ([app/analytics.py](app/analytics.py)) that
writes a report from a plain-language request ("What are the recurring root causes?"). It works in two steps:
a tool-calling loop gathers data (`incident_counts`, `list_incidents`, `search_incident_details`), then a
structured-output call writes the report from that data only.

- Tools are bound to the sidebar scope in code, so the model cannot switch tenant or widen the scope.
- The overview (counts, open incidents, charts) is computed without the model and is always shown.
- A report that mentions an incident id that is not in the tenant's data is rejected and the overview is shown instead.
- Reports use the `incident_records` collection (one record per incident with severity, status, service, region).
  After upgrading, run `python -m app.cli ingest` once to create it. It is idempotent.
- The data has no timestamps, so there are no trend, duration or MTTR metrics yet.
- Each report makes 2-4 Gemini calls. The free tier allows about 5 per minute per model, so back-to-back reports
  can fall back to the overview with `llm_error:GoogleRateLimitError`.

### Uploaded incident report PDFs

The **Documents** tab lets a tenant upload text-based incident report PDFs ([app/documents.py](app/documents.py)),
ask questions about them and generate reports, then email a report ([app/emailer.py](app/emailer.py)).

- PDFs are extracted with `pypdf`, chunked, embedded and stored in the `document_chunks` collection, tagged with the
  tenant. Every read and delete filters on the tenant, so one tenant cannot see or remove another's files.
- Limits: 10 MB, 100 pages, 400 chunks. Scanned/image-only and password-protected PDFs are rejected with a message.
  Re-uploading identical content replaces the document instead of duplicating it.
- Questions reuse the same pipeline as **Ask** (cited answers, invalid citations rejected, raw-evidence fallback).
- Reports are one structured Gemini call over the document text. A report that mentions an identifier
  (`INC-1234`, ...) that is not in the source text is rejected.
- Uploaded text is untrusted: it is escaped and quoted as data, and the model is told never to follow instructions in it.

**Email** uses SMTP and can only send to addresses on `EMAIL_ALLOWLIST` (exact addresses and/or `@domain` entries),
because a public app with an open "send to anyone" button would be a spam relay. The subject and body are a fixed
template around the report. Set `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASSWORD`, `SMTP_FROM` and
`EMAIL_ALLOWLIST` in `.env` or in the Streamlit **Secrets** (Gmail needs an App Password). Without them the email box
explains that email is not configured. Report PDFs are not attached; the report is sent as plain text.

### Chroma Cloud

Create a database at [trychroma.com](https://www.trychroma.com/), then put its credentials in `.env`:

```
CHROMA_API_KEY=...
CHROMA_TENANT=...        # tenant id from the Chroma Cloud dashboard
CHROMA_DATABASE=...      # database name
```

Run `python -m app.cli migrate` then `python -m app.cli ingest` once to create the collections and load the
sample data. With `CHROMA_API_KEY` unset, the same code uses a local on-disk Chroma database (`CHROMA_PATH`,
default `.chroma/`), which is what you want for offline work.

For the hosted Streamlit app, add the same values under **App settings → Secrets**:

```toml
GOOGLE_API_KEY = "your-gemini-api-key"
CHROMA_API_KEY = "..."
CHROMA_TENANT = "..."
CHROMA_DATABASE = "..."
```

Keep `EMBEDDING_PROVIDER` and `EMBEDDING_DIM` consistent between ingestion and the hosted app. Existing
non-empty environment variables take precedence over Streamlit Secrets.

```bash
curl -s localhost:8000/v1/ask \
  -H 'X-Tenant-Id: f283a00f-9d8d-5f54-957a-03ae3707e995' -H 'content-type: application/json' \
  -d '{"question":"Have we seen VPN timeout after login before?","service":"vpn","environment":"production","region":"india"}'
```

`X-Tenant-Id` stands in for your auth layer. Tenant and filters are **trusted inputs**; the model never chooses them.

### Embeddings

Embeddings are configured separately from the Gemini generation model (primary `gemini-3.6-flash`, with
`gemini-3.5-flash-lite` as a fallback on errors such as 503). Pick one provider: `EMBEDDING_PROVIDER=openai`
(default, 1536-d), `gemini` (`gemini-embedding-001`, only needs `GOOGLE_API_KEY`), `voyage` (1024-d),
`local` (384-d, no API key) or `fake` (offline tests only). Changing provider changes the vector size, so run
`cli reset && cli migrate && cli ingest` afterwards. Every chunk records its `embedding_model` and
retrieval only compares vectors from the active model.

## Testing

```bash
python -m pytest                          # 20 tests: scoping, isolation, policy, fallback, request config
python -m app.cli eval                    # retrieval scope/recall checks (use a REAL embedding provider)
python -m app.cli eval --generate         # + calls Gemini, checks answer mode
python -m app.cli eval --judge            # + claim-by-claim support check
```

`pytest` uses a throwaway local Chroma database (never Chroma Cloud), fake embeddings and a stub LLM, so it needs no keys; it proves the **plumbing and safety
rules**, not answer quality. Measure quality with `eval` and real embeddings.

## What has and hasn't been verified

Verified against a local Chroma database (not yet against Chroma Cloud): ingestion, filtering, tenant isolation,
private-note exclusion, retired-runbook exclusion, re-ingest and delete, embedding-model isolation, all
validation/fallback paths (stub LLM), and the Gemini chain construction.

**Not yet run:** live calls to OpenAI embeddings and Gemini (no keys in the build environment), so
retrieval quality, real latency against `LLM_TIMEOUT_S`, and `eval --judge` are untested. Run
`eval --judge` first once keys are set.

## Version history

The four most recent commits, newest first. Versions are numbered oldest to newest from v0.1; they are labels in this
table only (no git tags).

| Version | Commit | Change |
|---|---|---|---|
| **v0.4** | `329db28` | **PDF incident reports.** Tenants upload text PDFs, ask cited questions, generate reports and email them over SMTP to allow-listed recipients. Adds `app/documents.py`, `app/emailer.py`, a Documents tab, a `document_chunks` collection, and `SMTP_*` / `EMAIL_ALLOWLIST` settings. The analytics gathering step now falls back to the backup model on a 503. |
| **v0.3** | `c8d455b` | **Analytics agent.** A Reports tab where an agent gathers counts, listings and semantic-search results with tenant-bound tools, then writes a report. Reports that cite an unknown incident id are rejected. Adds `app/analytics.py` and an `incident_records` collection (re-run `cli ingest`). |
| **v0.2** | `b4384ff` | **Hosted-app fixes.** Gemini API key passed explicitly to the chat model; scope passed to the retriever as a dict so Streamlit module reloads cannot break the `Scope` type check; the LLM fallback now logs its traceback. |
| **v0.1** | `180da4e` | **`Scope` re-validation.** First attempt at the Streamlit reload `Scope` error, by re-validating through `Scope.model_validate`. Superseded by v0.2. |

## Next steps

1. Run `eval --judge` with real keys; tune `MIN_SIMILARITY` (0.25 is a starting guess) and `K_*`.
2. Derive filters from the open incident (`incident_id` on the request) instead of passing them manually.
3. Grow the eval set from real questions; add a stale-runbook and a conflicting-evidence case.
4. Add `rag_fallback` rate to your metrics/alerts (it is logged with a reason).
5. Hybrid search (Chroma supports sparse/full-text search alongside vectors) if exact identifiers like `INC1001` matter.
