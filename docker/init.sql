CREATE EXTENSION IF NOT EXISTS vector;

-- 1. Zaman Serisi Fiyat Tablosu
CREATE TABLE IF NOT EXISTS market_prices (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    price_date DATE NOT NULL,
    open_price NUMERIC(12, 4),
    high_price NUMERIC(12, 4),
    low_price NUMERIC(12, 4),
    close_price NUMERIC(12, 4) NOT NULL,
    volume BIGINT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uq_symbol_date UNIQUE (symbol, price_date)
);

CREATE INDEX IF NOT EXISTS idx_market_prices_symbol_date 
ON market_prices(symbol, price_date DESC);

-- 2. Haberler ve AI Vektör Tablosu
CREATE TABLE IF NOT EXISTS market_news (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20),
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    source VARCHAR(100),
    url TEXT UNIQUE,
    published_at TIMESTAMP WITH TIME ZONE,
    embedding VECTOR(384),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_market_news_embedding 
ON market_news USING hnsw (embedding vector_cosine_ops);