-- Fireline RAG schema (Module 2, LU2.11), PostgreSQL + pgvector.
-- __EMBEDDING_DIM__ is substituted by `python -m app.cli migrate`.

CREATE EXTENSION IF NOT EXISTS vector;

-- Minimal slice of the Module 2 incidents table: only what retrieval needs.
CREATE TABLE IF NOT EXISTS incidents (
  id           UUID PRIMARY KEY,
  tenant_id    UUID NOT NULL,
  external_key TEXT NOT NULL,                       -- INC1001
  title        TEXT NOT NULL,
  status       TEXT NOT NULL CHECK (status   IN ('OPEN', 'RESOLVED', 'CLOSED')),
  severity     TEXT NOT NULL CHECK (severity IN ('P1', 'P2', 'P3', 'P4')),
  service      TEXT,
  environment  TEXT,
  region       TEXT,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (tenant_id, id),                           -- target of the composite FK below
  UNIQUE (tenant_id, external_key)
);

CREATE TABLE IF NOT EXISTS incident_chunks (
  id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  tenant_id       UUID NOT NULL,
  incident_id     UUID NOT NULL,
  source          TEXT NOT NULL CHECK (source IN ('title', 'description', 'timeline', 'log', 'resolution')),
  chunk_index     INT  NOT NULL,
  chunk_text      TEXT NOT NULL,
  embedding       vector(__EMBEDDING_DIM__) NOT NULL,
  embedding_model TEXT NOT NULL,                    -- never compare vectors from different models
  service         TEXT,                             -- metadata filters used at query time
  environment     TEXT,
  region          TEXT,
  visibility      TEXT NOT NULL DEFAULT 'internal' CHECK (visibility IN ('internal', 'private')),
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (incident_id, source, chunk_index),
  FOREIGN KEY (tenant_id, incident_id) REFERENCES incidents (tenant_id, id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS incident_chunks_filter
  ON incident_chunks (tenant_id, service, environment);

-- Runbooks are versioned. Exactly one version per (tenant, slug) may be 'active'.
-- Status lives here, not on the chunks, so retiring a runbook takes effect instantly.
CREATE TABLE IF NOT EXISTS runbooks (
  id         UUID PRIMARY KEY,
  tenant_id  UUID NOT NULL,
  slug       TEXT NOT NULL,                         -- vpn-india
  version    INT  NOT NULL,
  title      TEXT NOT NULL,
  status     TEXT NOT NULL CHECK (status IN ('active', 'retired')),
  service    TEXT,
  region     TEXT,                                  -- NULL = applies to every region
  UNIQUE (tenant_id, id),
  UNIQUE (tenant_id, slug, version)
);

CREATE UNIQUE INDEX IF NOT EXISTS one_active_runbook_version
  ON runbooks (tenant_id, slug) WHERE status = 'active';

CREATE TABLE IF NOT EXISTS runbook_chunks (
  id              BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  tenant_id       UUID NOT NULL,
  runbook_id      UUID NOT NULL,
  chunk_index     INT  NOT NULL,
  chunk_text      TEXT NOT NULL,
  embedding       vector(__EMBEDDING_DIM__) NOT NULL,
  embedding_model TEXT NOT NULL,
  UNIQUE (runbook_id, chunk_index),
  FOREIGN KEY (tenant_id, runbook_id) REFERENCES runbooks (tenant_id, id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS runbook_chunks_tenant ON runbook_chunks (tenant_id);
