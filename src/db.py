import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    """Creates and returns a PostgreSQL database connection using .env variables."""
    conn_str = (
        f"host=localhost "
        f"port={os.getenv('POSTGRES_PORT', '5432')} "
        f"dbname={os.getenv('POSTGRES_DB')} "
        f"user={os.getenv('POSTGRES_USER')} "
        f"password={os.getenv('POSTGRES_PASSWORD')}"
    )
    return psycopg.connect(conn_str)