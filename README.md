# End-to-End Financial Data, Hybrid RAG & Autonomous Quantitative Pipeline

An end-to-end Data Engineering, Retrieval-Augmented Generation (RAG), and Quantitative Research pipeline that ingests 1-year historical stock market prices and financial news, computes vectorized technical and volatility indicators (`SMA-20`, `SMA-50`, `SMA-200`, `RSI-14`, `ATR-14`, `Volume Ratio`) using **Polars**, runs domain-specific financial sentiment classification via **ProsusAI/finbert**, enforces a **QQQ SMA-200 Macro Market Regime Lock**, and validates rules through a commission-adjusted **Walk-Forward Backtest Engine** backed by **PostgreSQL + pgvector**.

---

## Architecture Overview

```text
                        +---------------------------------------+
                        |   Autonomous Pipeline Orchestrator    |
                        |        (src/run_pipeline.py)          |
                        +---------------------------------------+
                                            |
        +-----------------------------------+-----------------------------------+
        v                                   v                                   v
+--------------------+            +--------------------+            +------------------------+
| 1. Polars ETL      |            | 2. AI Vector ETL   |            | 3. Quant & AI Engines  |
| (fetch_prices.py)  |            | (fetch_news.py)    |            | - rag_query.py         |
| 1y OHLCV + QQQ     |            | all-MiniLM-L6-v2   |            | - signal_engine.py     |
+--------------------+            +--------------------+            |   (QQQ SMA-200 Regime +|
        |                                   |                       |    FinBERT + ATR Risk) |
        +-----------------+-----------------+                       | - backtest.py (1y Sim) |
                          v                                         +------------------------+
          +-------------------------------+                                     ^
          |   PostgreSQL 16 + pgvector    | ------------------------------------+
          | - market_prices (Time-Series) |
          | - market_news (vector(384) +  |
          |   HNSW Cosine Index)          |
          +-------------------------------+
```

---

## Key Engineering Highlights

* **Single-Command Autonomous Orchestration (`src/run_pipeline.py`):** Executes the full 4-stage pipeline (Market ETL -> Vector Embeddings -> Macro/Signal/Risk Engine -> Walk-Forward Backtest) with step-level latency telemetry and fail-fast exit handling.
* **Macro Market Regime Filter (`QQQ` vs `SMA-200`):** Evaluates benchmark index (`QQQ`) trend before authorizing long signals; automatically locks new `BUY` entries (`BEKLE / REGIME LOCK`) during bear regimes (`QQQ < SMA-200`) to preserve capital.
* **High-Performance ETL & Vectorized Indicators (Polars):** Ingests 1-year OHLCV market history across equities, leveraged ETFs, and benchmark indices, computing rolling indicators (`SMA-20`, `SMA-50`, `RSI-14`, `ATR-14`, and 20-day `Volume Ratio`).
* **Idempotent Data Ingestion:** Enforces PostgreSQL `ON CONFLICT ... DO UPDATE` (UPSERT) constraints on `(symbol, price_date)` and `UNIQUE(url)` to guarantee zero duplicate records across repeated runs.
* **Dual Local AI Models (Zero API Cost):**
  * **Semantic Embeddings (`all-MiniLM-L6-v2`):** Generates normalized 384-dimensional news vectors indexed with HNSW cosine similarity in `pgvector`.
  * **Financial Sentiment (`ProsusAI/finbert`):** Computes domain-specific positive/negative/neutral sentiment probabilities on financial news titles and summaries.
* **ATR Dynamic Risk Management & Walk-Forward Backtest:** Calculates volatility-adjusted Stop-Loss (`2 x ATR-14`), Take-Profit (`3 x ATR-14`, `1:1.5` Risk/Reward), fixed-fractional position sizing (`1%` max risk per trade), and commission-adjusted (`0.1%`) 1-year historical simulation (`src/backtest.py`).
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

### 3. Run the Full Autonomous Pipeline (Single Command)
```bash
python src/run_pipeline.py
```

### 4. Query the Hybrid Financial RAG Engine (Interactive Semantic Search)
```bash
python src/rag_query.py
```

---

> **Disclaimer:** This project is built strictly for educational, software architecture, and data engineering research purposes. Generated technical, macro regime, sentiment, risk, and backtest outputs do not constitute financial or investment advice.
