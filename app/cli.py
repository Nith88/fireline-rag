"""python -m app.cli <command>"""
import argparse
import json
import sys
from pathlib import Path

from app.config import get_settings
from app.ids import tenant_uuid

ROOT = Path(__file__).resolve().parent.parent


def cmd_migrate(_):
    """Create the collections (idempotent). Chroma needs no schema; the vector size is set by the first insert."""
    from app import store

    for kind in store.ALL_COLLECTIONS:
        store.collection(kind)
    print(f"collections ready (embedding dim {get_settings().embedding_dim})")


def cmd_reset(_):
    from app import store

    store.reset()
    print("collections dropped")


def cmd_ingest(args):
    from app.ingest import ingest_incidents, ingest_runbooks

    n_inc = ingest_incidents(Path(args.incidents))
    n_rb = ingest_runbooks(Path(args.runbooks))
    print(f"ingested {n_inc} incidents, {n_rb} runbooks")


def _scope(args):
    from app.models import Scope

    return Scope(tenant_id=tenant_uuid(args.tenant), service=args.service, environment=args.env, region=args.region)


def cmd_retrieve(args):
    from app.embeddings import get_embeddings
    from app.retriever import FirelineRetriever

    s = get_settings()
    docs = FirelineRetriever(embeddings=get_embeddings(), scope=_scope(args), k_incidents=s.k_incidents, k_runbooks=s.k_runbooks, min_similarity=s.min_similarity).invoke(args.question)
    for d in docs:
        m = d.metadata
        print(f'{m["similarity"]:.3f}  {m["eid"]:<8} {m["ref"]:<14} {m["source"]:<12} {d.page_content[:90]!r}')


def cmd_ask(args):
    from app.rag import RagService

    res = RagService().ask(args.question, _scope(args))
    print(json.dumps(res.model_dump(mode="json"), indent=2))


def cmd_serve(args):
    import uvicorn

    uvicorn.run("app.api:app", host=args.host, port=args.port, reload=False)


def cmd_ui(args):
    import subprocess

    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "ui.py"), "--server.port", str(args.port)]))


def cmd_eval(args):
    from app.evaluation import run

    ok = run(Path(args.file), generate=args.generate or args.judge, judge=args.judge)
    sys.exit(0 if ok else 1)


def main():
    p = argparse.ArgumentParser(prog="fireline-rag")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("migrate").set_defaults(fn=cmd_migrate)
    sub.add_parser("reset").set_defaults(fn=cmd_reset)

    ing = sub.add_parser("ingest")
    ing.add_argument("--incidents", default=str(ROOT / "data" / "incidents.json"))
    ing.add_argument("--runbooks", default=str(ROOT / "data" / "runbooks.json"))
    ing.set_defaults(fn=cmd_ingest)

    for name, fn in (("retrieve", cmd_retrieve), ("ask", cmd_ask)):
        q = sub.add_parser(name)
        q.add_argument("question")
        q.add_argument("--tenant", default="acme", help="tenant name from the sample data, or a UUID")
        q.add_argument("--service")
        q.add_argument("--env")
        q.add_argument("--region")
        q.set_defaults(fn=fn)

    sv = sub.add_parser("serve")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.set_defaults(fn=cmd_serve)

    ui = sub.add_parser("ui", help="Streamlit demo UI")
    ui.add_argument("--port", type=int, default=8501)
    ui.set_defaults(fn=cmd_ui)

    ev = sub.add_parser("eval")
    ev.add_argument("--file", default=str(ROOT / "eval" / "eval_set.json"))
    ev.add_argument("--generate", action="store_true", help="also call the LLM (needs GOOGLE_API_KEY)")
    ev.add_argument("--judge", action="store_true", help="claim-level support check (implies --generate)")
    ev.set_defaults(fn=cmd_eval)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
