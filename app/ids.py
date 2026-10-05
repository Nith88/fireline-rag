"""Deterministic ids, so re-ingesting the same data updates rows instead of duplicating them."""
import uuid

_NS = uuid.UUID("0d1f0c1e-5a7e-4b1e-9a55-00000000f1fe")


def uid(*parts: str) -> uuid.UUID:
    return uuid.uuid5(_NS, "/".join(parts))


def tenant_uuid(name_or_uuid: str) -> uuid.UUID:
    """Accept a real UUID, or a dev tenant name such as 'acme'."""
    try:
        return uuid.UUID(name_or_uuid)
    except ValueError:
        return uid("tenant", name_or_uuid)
