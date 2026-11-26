# End-to-End Financial Data, Hybrid RAG & Autonomous Quantitative Pipeline

An end-to-end Data Engineering, Retrieval-Augmented Generation (RAG), and Quantitative Research pipeline that ingests 1-year historical stock market prices and financial news, computes vectorized technical, volatility, and relative strength indicators (`SMA-20`, `SMA-50`, `SMA-200`, `RSI-14`, `ATR-14`, `Volume Ratio`, `20d RS vs. QQQ`) using **Polars**, runs domain-specific financial sentiment classification via **ProsusAI/finbert**, enforces a **Two-Tier Institutional Risk & Regime Architecture**, and validates rules through a commission-adjusted **Walk-Forward Backtest Engine** backed by **PostgreSQL + pgvector**.

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
+--------------------+            +--------------------+            |   (Tier-1 SMA-200 Gates|
        |                                   |                       |   +Tier-2 Multi-Factor)|
        +-----------------+-----------------+                       | - backtest.py (MaxDD & |
                          v                                         |   Profit Factor Scorecard|
          +-------------------------------+                         +------------------------+
          |   PostgreSQL 16 + pgvector    | <-----------------------------------+
          | - market_prices (Time-Series) |
          | - market_news (vector(384) +  |
          |   HNSW Cosine Index)          |
          +-------------------------------+
```

---

## Key Engineering Highlights

* **Two-Tier Institutional Decision Architecture (`src/signal_engine.py`):**
  * **Tier-1 Hard Safety Gates (Veto Layer):** Blocks new `BUY` signals automatically unless both the macro benchmark (`QQQ >= SMA-200`) and the individual equity (`Close >= SMA-200`) trade in long-term bull regimes.
  * **Tier-2 Multi-Factor Scoring:** Combines trend alignment (`SMA-20 > SMA-50`), 20-day Relative Strength outperformance vs. `QQQ`, volume confirmation (`>= 1.2x` 20d avg), RSI balance, and **ProsusAI/finbert** financial news sentiment into a composite confidence score.
* **Institutional Backtest Scorecard (`src/backtest.py`):** Evaluates 1-year walk-forward performance with `0.1%` commission/slippage, reporting trade count, blocked regime entries, Win Rate, **Max Drawdown (`MaxDD %`)**, and **Profit Factor (`Gross Profit / Gross Loss`)**.
* **ATR Dynamic Risk Management & Position Sizing:** Calculates volatility-adjusted Stop-Loss (`2 x ATR-14`), Take-Profit (`3 x ATR-14`, `1:1.5` Risk/Reward), and fixed-fractional position sizing (`1%` max portfolio risk per trade).
* **Single-Command Autonomous Orchestration (`src/run_pipeline.py`):** Executes the full 4-stage pipeline (Market ETL -> Vector Embeddings -> Two-Tier Signal/Risk Engine -> Walk-Forward Backtest) with step-level latency telemetry and fail-fast exit handling.
* **Idempotent Data Ingestion & Local AI Models:** Enforces PostgreSQL `ON CONFLICT ... DO UPDATE` (UPSERT) constraints and runs `all-MiniLM-L6-v2` (384-dim `pgvector` HNSW index) alongside `ProsusAI/finbert` locally with zero external API costs.

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
