ALTER TABLE market_news ADD COLUMN IF NOT EXISTS finbert_score NUMERIC(6, 3);
ALTER TABLE signal_history ADD COLUMN IF NOT EXISTS outcome_5d_pct NUMERIC(8, 2);
ALTER TABLE signal_history ADD COLUMN IF NOT EXISTS outcome_10d_pct NUMERIC(8, 2);
