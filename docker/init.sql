CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS market_prices (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    price_date DATE NOT NULL,
    open_price NUMERIC(12, 4) NOT NULL,
    high_price NUMERIC(12, 4) NOT NULL,
    low_price NUMERIC(12, 4) NOT NULL,
    close_price NUMERIC(12, 4) NOT NULL,
    volume BIGINT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, price_date)
);

CREATE INDEX IF NOT EXISTS idx_market_prices_symbol_date
ON market_prices(symbol, price_date DESC);

CREATE TABLE IF NOT EXISTS market_news (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    url TEXT UNIQUE NOT NULL,
    published_at TIMESTAMPTZ,
    embedding vector(384),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_market_news_embedding_hnsw
ON market_news USING hnsw (embedding vector_cosine_ops);

CREATE TABLE IF NOT EXISTS signal_history (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    signal_date DATE NOT NULL DEFAULT CURRENT_DATE,
    decision VARCHAR(255) NOT NULL,
    confidence_score NUMERIC(6, 2),
    close_price NUMERIC(12, 4),
    stop_loss NUMERIC(12, 4),
    take_profit NUMERIC(12, 4),
    finbert_score NUMERIC(6, 3),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(symbol, signal_date)
);

CREATE TABLE IF NOT EXISTS watchlist (
    symbol VARCHAR(20) PRIMARY KEY,
    asset_type VARCHAR(20) DEFAULT 'STOCK',
    is_active BOOLEAN DEFAULT TRUE,
    added_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO watchlist (symbol, asset_type, is_active) VALUES
    ('AAPL', 'STOCK', TRUE),
    ('NVDA', 'STOCK', TRUE),
    ('MSFT', 'STOCK', TRUE),
    ('GLD', 'ETF', TRUE),
    ('XLE', 'ETF', TRUE)
ON CONFLICT (symbol) DO NOTHING;
