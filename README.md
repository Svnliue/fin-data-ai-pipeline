# End-to-End Financial Data, Hybrid RAG & AI Signal Pipeline

An end-to-end Data Engineering, Retrieval-Augmented Generation (RAG), and Quantitative Signal pipeline that ingests 1-year historical stock market prices and financial news, computes vectorized technical indicators (`SMA-20`, `SMA-50`, `RSI-14`) using **Polars**, runs financial sentiment classification via **ProsusAI/finbert**, and stores both time-series data and 384-dimensional vector embeddings inside **PostgreSQL + pgvector**.

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
                                                             |   Hybrid RAG & Signal Engines |
                                                             |  - src/rag_query.py (RAG)     |
                                                             |  - src/signal_engine.py       |
                                                             |    (Polars Tech + FinBERT)    |
                                                             +-------------------------------+
```

---

## Key Engineering Highlights

* **High-Performance ETL & Technical Indicators (Polars):** Ingests 1-year OHLCV history and computes vectorized rolling indicators (`SMA-20`, `SMA-50`, `RSI-14`) across multiple tickers using window expressions.
* **Idempotent Data Ingestion:** Implements PostgreSQL `ON CONFLICT ... DO UPDATE` (UPSERT) constraints on `(symbol, price_date)` and `UNIQUE(url)` to guarantee zero duplicate records across repeated runs.
* **Dual Local AI Models (Zero API Cost):**
  * **Semantic Embeddings (`all-MiniLM-L6-v2`):** Generates normalized 384-dimensional news vectors stored with an HNSW cosine index in `pgvector`.
  * **Financial Sentiment (`ProsusAI/finbert`):** Evaluates domain-specific positive/negative/neutral sentiment probabilities on financial news headlines and summaries.
* **Hybrid Decision & RAG Engines:** Combines SQL window analytics and semantic search (`src/rag_query.py`) alongside a rule-based Technical + Sentiment scoring engine (`src/signal_engine.py`).
* **Containerized Infrastructure:** Reproducible PostgreSQL 16 + `pgvector` environment provisioned via Docker Compose and automated `init.sql` schema migrations.

---

## Tech Stack

* **Language:** Python 3.12+
* **Data Processing & Indicators:** Polars, PyArrow
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

### 4. Run Hybrid RAG & AI Signal Engines
```bash
python src/rag_query.py
python src/signal_engine.py
```

---

> **Disclaimer:** This project is built strictly for educational, software architecture, and data engineering research purposes. Generated technical and sentiment signals do not constitute financial or investment advice.
