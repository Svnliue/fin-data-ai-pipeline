import os
from dotenv import load_dotenv

load_dotenv()

DEFAULT_SYMBOLS = ["AAPL", "NVDA", "MSFT", "GLD", "XLE"]
BENCHMARK_SYMBOL = "QQQ"
PRICE_PERIOD = "1y"

PORTFOLIO_CAPITAL = float(os.getenv("PORTFOLIO_CAPITAL", "10000.0"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.25"))
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "3"))
TRAILING_ATR_MULT = float(os.getenv("TRAILING_ATR_MULT", "2.5"))
COMMISSION_RATE = float(os.getenv("COMMISSION_RATE", "0.001"))
SCHEDULE_TIME = os.getenv("SCHEDULE_TIME", "23:30")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FINBERT_MODEL = "ProsusAI/finbert"


def get_active_symbols() -> list[str]:
    try:
        from db import get_db_connection
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT symbol FROM watchlist WHERE is_active = TRUE ORDER BY symbol;")
                rows = [r[0] for r in cur.fetchall()]
                return rows if rows else DEFAULT_SYMBOLS
    except Exception:
        return DEFAULT_SYMBOLS


SYMBOLS = get_active_symbols()
