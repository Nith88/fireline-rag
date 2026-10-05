import psycopg
from psycopg.rows import dict_row

from app.config import get_settings


def connect(**kwargs) -> psycopg.Connection:
    return psycopg.connect(get_settings().database_url, row_factory=dict_row, **kwargs)


def vec_literal(vector: list[float]) -> str:
    """pgvector text format. Sent as a parameter and cast with ::vector in SQL."""
    return "[" + ",".join(f"{x:.8g}" for x in vector) + "]"
