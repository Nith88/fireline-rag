"""Tests use a throwaway local Chroma database (never Chroma Cloud) and fake embeddings, so they need no
keys or services. Fake embeddings make *similarity* meaningless; the tests therefore check scoping,
isolation and policy, which are deterministic. Retrieval *quality* is measured with
`python -m app.cli eval` using a real embedding provider."""
import os
import tempfile

os.environ["CHROMA_API_KEY"] = ""  # even if .env has one: tests must never touch Chroma Cloud
os.environ["CHROMA_PATH"] = tempfile.mkdtemp(prefix="fireline-chroma-")
os.environ["CHROMA_PREFIX"] = "test_"
os.environ["EMBEDDING_PROVIDER"] = "fake"
os.environ["MIN_SIMILARITY"] = "-1"   # keep every row so filters are what is under test
os.environ["K_INCIDENTS"] = "50"
os.environ["K_RUNBOOKS"] = "50"

from pathlib import Path  # noqa: E402

import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def db():
    from app import cli
    from app.config import get_settings
    from app.ingest import ingest_incidents, ingest_runbooks

    s = get_settings()
    assert not s.chroma_api_key and s.chroma_prefix == "test_", "refusing to reset a non-test store"
    cli.cmd_reset(None)
    cli.cmd_migrate(None)
    ingest_incidents(ROOT / "data" / "incidents.json")
    ingest_runbooks(ROOT / "data" / "runbooks.json")
