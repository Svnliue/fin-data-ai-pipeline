import atexit
import os
from contextlib import contextmanager
from dotenv import load_dotenv
from psycopg_pool import ConnectionPool

load_dotenv()

_pool: ConnectionPool | None = None


def _build_conninfo() -> str:
    return (
        f"host={os.getenv('POSTGRES_HOST', 'localhost')} "
        f"port={os.getenv('POSTGRES_PORT', '5433')} "
        f"dbname={os.getenv('POSTGRES_DB')} "
        f"user={os.getenv('POSTGRES_USER')} "
        f"password={os.getenv('POSTGRES_PASSWORD')}"
    )


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(conninfo=_build_conninfo(), min_size=1, max_size=5, open=True)
        atexit.register(_pool.close)
    return _pool


@contextmanager
def get_db_connection():
    pool = get_pool()
    with pool.connection() as conn:
        yield conn
