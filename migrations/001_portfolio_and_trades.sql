UPDATE watchlist SET asset_type = 'COMMODITY' WHERE symbol = 'GLD' AND asset_type = 'ETF';
UPDATE watchlist SET asset_type = 'SECTOR_ETF' WHERE symbol = 'XLE' AND asset_type = 'ETF';

ALTER TABLE portfolio_positions ADD COLUMN IF NOT EXISTS initial_atr NUMERIC(12, 4);
ALTER TABLE portfolio_positions ADD COLUMN IF NOT EXISTS stop_loss NUMERIC(12, 4);
ALTER TABLE portfolio_positions ADD COLUMN IF NOT EXISTS tp1 NUMERIC(12, 4);
ALTER TABLE portfolio_positions ADD COLUMN IF NOT EXISTS tp2 NUMERIC(12, 4);
ALTER TABLE portfolio_positions ADD COLUMN IF NOT EXISTS scaled_out BOOLEAN DEFAULT FALSE;

CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    shares NUMERIC(14, 4) NOT NULL,
    entry_price NUMERIC(14, 4) NOT NULL,
    exit_price NUMERIC(14, 4) NOT NULL,
    realized_pnl NUMERIC(14, 4) NOT NULL,
    exit_reason VARCHAR(255),
    opened_at DATE,
    closed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
