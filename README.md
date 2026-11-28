# Fin-Data AI Pipeline (Alpha Engine v4.9.9)

Cok varlikli (STOCK, COMMODITY, SECTOR_ETF, LEVERAGED) kantitatif karar destek, portfoy yonetim, finansal NLP duygu analizi ve rejim dogrulama boru hatti.

## Ozellikler
- 5 yillik OHLCV fiyat ETL ve haber deduplication (Polars + yfinance)
- PostgreSQL + pgvector ve numarali SQL migration sistemi (src/migrate.py)
- FinBERT duygu onbellegi (market_news.finbert_score)
- Varlik tipine duyarli risk yonetimi, Trailing Stop ve trades defteri
- Next-open + slippage rejim dogrulama backtesti (Win Rate, Profit Factor, Sharpe, Expectancy, MaxDD)
- 5 ve 10 gunluk Forward-Test sonuc takibi ve Telegram bildirimi
