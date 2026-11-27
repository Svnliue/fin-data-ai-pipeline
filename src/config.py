import os
from dotenv import load_dotenv

load_dotenv()

SYMBOLS = ["AAPL", "NVDA", "MSFT", "SAGT", "IREX"]
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
