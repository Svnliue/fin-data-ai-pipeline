import os
from dotenv import load_dotenv
from logger import get_logger

load_dotenv()
log = get_logger("config")

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
PRICE_PERIOD = "5y"

PORTFOLIO_CAPITAL = float(os.getenv("PORTFOLIO_CAPITAL", "10000.0"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", "0.01"))
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.25"))
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "3"))
MAX_SAME_GROUP_POSITIONS = int(os.getenv("MAX_SAME_GROUP_POSITIONS", "2"))
MAX_DRAWDOWN_PCT = float(os.getenv("MAX_DRAWDOWN_PCT", "0.12"))
MAX_DAILY_LOSS_PCT = float(os.getenv("MAX_DAILY_LOSS_PCT", "0.03"))
COMMISSION_RATE = float(os.getenv("COMMISSION_RATE", "0.001"))
SCHEDULE_TIME = os.getenv("SCHEDULE_TIME", "16:15")

ASSET_PROFILES = {
    "STOCK": {
        "require_macro_bull": True,
        "require_stock_sma200": True,
        "sl_atr": 2.0,
        "tp1_atr": 1.5,
        "tp2_atr": 3.5,
        "max_hold_days": None,
    },
    "COMMODITY": {
        "require_macro_bull": False,
        "require_stock_sma200": True,
        "sl_atr": 2.0,
        "tp1_atr": 1.5,
        "tp2_atr": 3.5,
        "max_hold_days": None,
    },
    "SECTOR_ETF": {
        "require_macro_bull": False,
        "require_stock_sma200": True,
        "sl_atr": 2.0,
        "tp1_atr": 1.5,
        "tp2_atr": 3.5,
        "max_hold_days": None,
    },
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


def get_all_asset_map() -> dict[str, str]:
    try:
        from db import get_db_connection
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT symbol, asset_type FROM watchlist ORDER BY symbol;")
                rows = dict(cur.fetchall())
                return rows if rows else DEFAULT_ASSET_MAP
    except Exception as exc:
        log.error(f"Watchlist tam tablosu okunamadi: {exc}")
        return DEFAULT_ASSET_MAP


def get_active_asset_map() -> dict[str, str]:
    try:
        from db import get_db_connection
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT symbol, asset_type FROM watchlist WHERE is_active = TRUE ORDER BY symbol;")
                rows = dict(cur.fetchall())
                return rows if rows else DEFAULT_ASSET_MAP
    except Exception as exc:
        log.error(f"Aktif watchlist okunamadi, varsayilan listeye dusuldu: {exc}")
        return DEFAULT_ASSET_MAP


def get_active_symbols() -> list[str]:
    return list(get_active_asset_map().keys())


SYMBOLS = get_active_symbols()
