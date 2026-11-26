# End-to-End Financial Data & Hybrid RAG Pipeline

An end-to-end Data Engineering and AI Retrieval-Augmented Generation (RAG) pipeline that ingests historical stock market prices and financial news, transforms them using **Polars** and **HuggingFace Sentence-Transformers**, and stores both relational time-series data and 384-dimensional vector embeddings inside a single **PostgreSQL + pgvector** database.

---

## Architecture Overview

```text
+-------------------+       +------------------------+       +-------------------------------+
|   Yahoo Finance   | ----> |  Polars ETL Pipeline   | ----> |   PostgreSQL 16 + pgvector    |
|  (Prices & News)  |       |  (src/fetch_prices.py) |       |  - market_prices (Time-Series)|
+-------------------+       +------------------------+       |  - market_news (vector(384) +   |
          |                                                  |    HNSW Cosine Index)         |
          |                 +------------------------+       +-------------------------------+
          +---------------> | AI Embedding Pipeline  | ----------------------|
                            | (all-MiniLM-L6-v2,     |                       |
                            |  src/fetch_news.py)    |                       v
                            +------------------------+       +-------------------------------+
                                                             |   Hybrid RAG Query Engine     |
                                                             |       (src/rag_query.py)      |
                                                             |  SQL Analytics + Cosine Sim   |
                                                             +-------------------------------+
```

---

## Key Engineering Highlights

* **High-Performance ETL with Polars:** Uses vectorized transformations and strict schema casting for market OHLCV data.
* **Idempotent Data Ingestion:** Implements PostgreSQL `ON CONFLICT ... DO UPDATE` (UPSERT) constraints on `(symbol, price_date)` and `UNIQUE(url)` to guarantee zero duplicate records across repeated runs.
* **Local AI Vectorization (Zero API Cost):** Generates normalized 384-dimensional embeddings in batch mode via `sentence-transformers/all-MiniLM-L6-v2` on PyTorch.
* **Hybrid Retrieval (`pgvector` + SQL Window Functions):** Combines deterministic financial price metrics (`FIRST_VALUE`, period min/max, % return) with HNSW-indexed cosine similarity search (`1 - (embedding <=> query_vector)`).
* **Containerized Infrastructure:** Reproducible PostgreSQL + `pgvector` environment provisioned via Docker Compose and automated `init.sql` schema migrations.

---

## Tech Stack

* **Language:** Python 3.12+
* **Data Processing:** Polars, PyArrow
* **Database & Vector Store:** PostgreSQL 16, `pgvector` (HNSW Index), `psycopg` (v3)
* **AI / Embeddings:** PyTorch, HuggingFace `sentence-transformers` (`all-MiniLM-L6-v2`)
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

### 4. Query the Hybrid RAG Engine
```bash
python src/rag_query.py
```
