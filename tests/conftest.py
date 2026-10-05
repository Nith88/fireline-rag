"""Tests run against a separate database and use fake embeddings, so they need no API keys.
Fake embeddings make *similarity* meaningless; the tests therefore check scoping, isolation and
policy, which are deterministic. Retrieval *quality* is measured with `python -m app.cli eval`
using a real embedding provider."""
import os

os.environ["DATABASE_URL"] = os.environ.get("TEST_DATABASE_URL", "postgresql://fireline:fireline@localhost:5432/fireline_test")
os.environ["EMBEDDING_PROVIDER"] = "fake"
os.environ["MIN_SIMILARITY"] = "-1"   # keep every row so filters are what is under test
os.environ["K_INCIDENTS"] = "50"
os.environ["K_RUNBOOKS"] = "50"

from pathlib import Path  # noqa: E402

import psycopg  # noqa: E402
import pytest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def db():
    from app import cli
    from app.config import get_settings
    from app.ingest import ingest_incidents, ingest_runbooks

    try:
        cli.cmd_reset(None)
    except psycopg.OperationalError:
        pytest.skip("test database not reachable (start it with docker compose up -d)")
    assert "test" in get_settings().database_url, "refusing to reset a non-test database"
    cli.cmd_migrate(None)
    ingest_incidents(ROOT / "data" / "incidents.json")
    ingest_runbooks(ROOT / "data" / "runbooks.json")
