import os
from dotenv import load_dotenv

load_dotenv()

DEFAULT_ASSET_MAP = {
    "AAPL": "STOCK",
    "NVDA": "STOCK",
    "MSFT": "STOCK",
    "GLD": "COMMODITY",
    "XLE": "SECTOR_ETF",
    "IREX": "LEVERAGED",
}
DEFAULT_SYMBOLS = list(DEFAULT_ASSET_MAP.keys())
BENCHMARK_SYMBOL = "QQQ"
PRICE_PERIOD = os.getenv("PRICE_PERIOD", "5y")

PORTFOLIO_CAPITAL = float(os.getenv("PORTFOLIO_CAPITAL", "10000.0"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.25"))
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "3"))
COMMISSION_RATE = float(os.getenv("COMMISSION_RATE", "0.001"))
SCHEDULE_TIME = os.getenv("SCHEDULE_TIME", "00:15")

_ETF_PROFILE = {
    "require_macro_bull": False,
    "require_stock_sma200": True,
    "sl_atr": 2.0,
    "tp1_atr": 1.5,
    "tp2_atr": 3.5,
    "max_hold_days": None,
}

ASSET_PROFILES = {
    "STOCK": {
        "require_macro_bull": True,
        "require_stock_sma200": True,
        "sl_atr": 2.0,
        "tp1_atr": 1.5,
        "tp2_atr": 3.5,
        "max_hold_days": None,
    },
    "COMMODITY": dict(_ETF_PROFILE),
    "SECTOR_ETF": dict(_ETF_PROFILE),
    "ETF": dict(_ETF_PROFILE),
    "LEVERAGED": {
        "require_macro_bull": False,
        "require_stock_sma200": False,
        "sl_atr": 1.2,
        "tp1_atr": 1.2,
        "tp2_atr": 2.5,
        "max_hold_days": 10,
    },
}

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
FINBERT_MODEL = "ProsusAI/finbert"


def get_active_asset_map() -> dict[str, str]:
    try:
        from db import get_db_connection
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT symbol, asset_type FROM watchlist WHERE is_active = TRUE ORDER BY symbol;")
                rows = dict(cur.fetchall())
                return rows if rows else dict(DEFAULT_ASSET_MAP)
    except Exception:
        return dict(DEFAULT_ASSET_MAP)


def get_held_symbols() -> list[str]:
    try:
        from db import get_db_connection
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT symbol FROM portfolio_positions ORDER BY symbol;")
                return [r[0] for r in cur.fetchall()]
    except Exception:
        return []


def get_active_symbols() -> list[str]:
    return sorted(set(get_active_asset_map().keys()) | set(get_held_symbols()))
