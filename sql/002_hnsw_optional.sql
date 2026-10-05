-- OPTIONAL. The design doc (LU5.11) lists "HNSW before measuring volume" as over-built.
-- Start with exact search. Add these only after `python -m app.cli eval` shows latency is a
-- problem AND recall stays acceptable with your real tenant/service filters.
CREATE INDEX IF NOT EXISTS incident_chunks_embedding
  ON incident_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS runbook_chunks_embedding
  ON runbook_chunks USING hnsw (embedding vector_cosine_ops);
