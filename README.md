# End-to-End Financial Data, Hybrid RAG & Quantitative Signal Pipeline

An end-to-end Data Engineering, Retrieval-Augmented Generation (RAG), and Quantitative Research pipeline that ingests 1-year historical stock market prices and financial news, computes vectorized technical and volatility indicators (`SMA-20`, `SMA-50`, `RSI-14`, `ATR-14`, `Volume Ratio`) using **Polars**, runs financial sentiment classification via **ProsusAI/finbert**, and validates trading rules through a commission-adjusted **Walk-Forward Backtest Engine** backed by **PostgreSQL + pgvector**.

---

## Architecture Overview

```text
+-------------------+       +------------------------+       +-------------------------------+
|   Yahoo Finance   | ----> |  Polars ETL Pipeline   | ----> |   PostgreSQL 16 + pgvector    |
| (1y Prices & News)|       |  (src/fetch_prices.py) |       |  - market_prices (Time-Series)|
+-------------------+       +------------------------+       |  - market_news (vector(384) +   |
          |                                                  |    HNSW Cosine Index)         |
          |                 +------------------------+       +-------------------------------+
          +---------------> | AI Embedding Pipeline  | ----------------------|
                            | (all-MiniLM-L6-v2,     |                       |
                            |  src/fetch_news.py)    |                       v
                            +------------------------+       +-------------------------------+
                                                             |  RAG, Signal & Risk Engines   |
                                                             |  - src/rag_query.py (RAG)     |
                                                             |  - src/signal_engine.py       |
                                                             |    (Tech + FinBERT + ATR Risk)|
                                                             |  - src/backtest.py (1y Sim)   |
                                                             +-------------------------------+
```

---

## Key Engineering Highlights

* **High-Performance ETL & Vectorized Indicators (Polars):** Ingests 1-year OHLCV market history and computes rolling indicators (`SMA-20`, `SMA-50`, `RSI-14`, `ATR-14`, and 20-day `Volume Ratio`) across multiple symbols using window expressions.
* **Idempotent Data Ingestion:** Enforces PostgreSQL `ON CONFLICT ... DO UPDATE` (UPSERT) constraints on `(symbol, price_date)` and `UNIQUE(url)` to guarantee zero duplicate records across repeated pipeline runs.
* **Dual Local AI Models (Zero API Cost):**
  * **Semantic Embeddings (`all-MiniLM-L6-v2`):** Generates normalized 384-dimensional news vectors indexed with HNSW cosine similarity in `pgvector`.
  * **Financial Sentiment (`ProsusAI/finbert`):** Computes domain-specific positive/negative/neutral sentiment probabilities on financial news titles and summaries.
* **ATR Dynamic Risk Management & Position Sizing:** Calculates volatility-adjusted Stop-Loss (`2 x ATR-14`), Take-Profit (`3 x ATR-14`, `1:1.5` Risk/Reward), and fixed-fractional position sizing (`1%` max portfolio risk per trade) alongside volume confirmation filters.
* **Commission-Adjusted Walk-Forward Backtester (`src/backtest.py`):** Simulates trend, volume, and ATR exit rules across 1-year historical OHLCV data including `0.1%` commission/slippage per leg, reporting trade count, win rate, and net portfolio return.
* **Containerized Infrastructure:** Reproducible PostgreSQL 16 + `pgvector` environment provisioned via Docker Compose and automated `init.sql` schema migrations.

---

## Tech Stack

* **Language:** Python 3.12+
* **Data Processing & Quant Analytics:** Polars, PyArrow
* **Database & Vector Store:** PostgreSQL 16, `pgvector` (HNSW Index), `psycopg` (v3)
* **AI / NLP Models:** PyTorch, HuggingFace `transformers` (`ProsusAI/finbert`), `sentence-transformers` (`all-MiniLM-L6-v2`)
* **Infrastructure & Source:** Docker, Docker Compose, `yfinance`

---

## Quickstart

### 1. Start PostgreSQL + `pgvector` Container
```bash
cp .env.example .env
docker compose up -d
```

### 2. Install Python Dependencies
```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt
```

### 3. Run ETL & Vector Pipelines
```bash
python src/fetch_prices.py
python src/fetch_news.py
```

### 4. Run Hybrid RAG, Signal & Backtest Engines
```bash
python src/rag_query.py
python src/signal_engine.py
python src/backtest.py
```

---

> **Disclaimer:** This project is built strictly for educational, software architecture, and data engineering research purposes. Generated technical, sentiment, risk, and backtest outputs do not constitute financial or investment advice.
