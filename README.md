# Fin-Data AI Pipeline (Alpha Engine v4.9.17)

Çok varlıklı (**STOCK**, **COMMODITY**, **SECTOR_ETF**, **LEVERAGED**) kantitatif karar destek, portföy yönetim, finansal NLP duygu analizi ve rejim doğrulama boru hattı.

## Mimari Akış

    +-------------------+     +-----------------------+     +-------------------------+
    |  Yahoo Finance    |     |  Polars ETL & Kalite  |     |  PostgreSQL + pgvector  |
    |  (5Y OHLCV + News)| --> |  Kontrol (Dedupe)     | --> |  (market_prices, news,  |
    +-------------------+     +-----------------------+     |   watchlist, portfolio, |
                                                            |   trades, signals)      |
                                                            +------------+------------+
                                                                         |
    +-------------------+     +-----------------------+                  v
    |  Telegram Bot     | <-- |  Alpha Engine v4.9.17  | <-- +-------------------------+
    |  & CLI Raporu     |     |  (Risk, Trailing SL,  |     |  ProsusAI/finbert       |
    +-------------------+     |   Forward-Test, OOS)  |     |  (DB Skor Önbelleği)    |
                              +-----------------------+     +-------------------------+

## Temel Yetenekler ve Mimari Katmanlar

### 1. Veri Katmanı ve Kalite Koruması (`src/fetch_prices.py`, `src/fetch_news.py`)
* `yfinance` ve `Polars` ile **5 yıllık (`5y`)** günlük OHLCV fiyat verileri çekilir; ağ hatalarında üstel geri çekilme (exponential backoff) uygulanır.
* NYSE seansı açıkken oluşan yarım günlük barlar filtrelenir; tüm sembollerin başarısız olması (`total_records == 0`) durumunda boru hattı hata fırlatarak durur ve 5 günden eski bayat fiyat barları loglanır.
* Haberler vektörleştirilmeden (`sentence-transformers/all-MiniLM-L6-v2`) önce mevcut URL kontrolüyle filtrelenir; yinelenen haberler modeli gereksiz yere çalıştırmaz.

### 2. Veritabanı ve Numaralı Migration Sistemi (`docker/init.sql`, `migrations/`, `src/migrate.py`)
* PostgreSQL 16 + `pgvector` HNSW indeksi üzerinde çalışır; port yerel arayüzle (`127.0.0.1:5433:5432`) sınırlandırılmıştır ve hassas dosyalar `.dockerignore` ile imaj dışında tutulur.
* `migrations/001_portfolio_and_trades.sql` ve `migrations/002_news_score_and_outcomes.sql` dosyaları `schema_migrations` tablosu üzerinden izlenerek güvenli ve tekrarlanabilir şekilde uygulanır.

### 3. FinBERT Duygu Önbelleği (`ProsusAI/finbert`)
* Son 7 günün haberleri analiz edilirken yalnızca `market_news.finbert_score IS NULL` olan yeni haberler FinBERT modelinden geçirilir ve net skor veritabanına kaydedilir; önceden skorlanmış haberler önbellekten okunur.

### 4. Varlık Sınıfına Duyarlı Karar ve Risk Motoru (`src/signal_engine.py`, `src/indicators.py`)
* **STOCK:** Makro rejim filtresi (`QQQ >= SMA-200`), `Fiyat >= SMA-200`, 3 günlük bilanço koruması, haber ve hacim/fiyat (`price_diff`) şok kilitleri ve dinamik ATR risk planı (`SL: 2.5x ATR`, `TP-1: 2.0x ATR`, `TP-2: 4.5x ATR`).
* **COMMODITY / SECTOR_ETF:** Makro hisse ayı rejiminden bağımsız değerlendirilir (`GLD`, `XLE` vb.; `SL: 2.5x ATR`, `TP-1: 2.0x ATR`, `TP-2: 4.5x ATR`).
* **LEVERAGED:** Sıkı stop (`1.5x ATR`), hedefler (`1.5x / 2.5x ATR`) ve 15 takvim günü maksimum elde tutma koruması (`IREX` vb.).

### 5. Portföy Yönetimi, Trailing Stop ve İşlem Defteri (`src/portfolio_cli.py`)
* Açık pozisyonlarda yükselen **Trailing Stop** uygulanır (stop seviyesi asla aşağı çekilmez).
* `TP-1` hedefine ulaşıldığında `%50` kısmi kâr alımı önerilir ve stop maliyete (breakeven) çekilir.
* Kapanan işlemler `trades` tablosuna işlenir; gerçekleşmiş PnL (`Realized PnL`), serbest nakit ve maksimum açık pozisyon limiti (`MAX_OPEN_POSITIONS`) canlı takip edilir.

### 6. İki Dönemli Rejim Doğrulama Backtesti (`src/backtest.py`)
* 200-bar teknik gösterge ısınma (warm-up) süresinin ardından sinyalin üretildiği günün ertesi iş günü açılışından (**next-open**) slippage ve komisyon dahil işlem açar.
* **Win Rate, Profit Factor, Sharpe Ratio, Expectancy ($)** ve **Max Drawdown** metriklerini ortak kasalı portföy ve iki ayrı piyasa döneminde raporlar.

### 7. Forward-Test Sonuç Takibi, Telegram Uyarısı ve NYSE Zamanlayıcı (`src/notifier.py`, `src/scheduler.py`)
* `signal_history` tablosuna gerçek fiyat barı tarihi (`bar_date`) yazılır ve geçmiş sinyallerin 5 ve 10 işlem günü sonraki gerçek getirileri (`outcome_5d_pct`, `outcome_10d_pct`) otomatik güncellenir.
* `.env` içerisinde `TELEGRAM_BOT_TOKEN` ve `TELEGRAM_CHAT_ID` tanımlıysa aksiyon gerektiren sinyaller ve zamanlayıcı boru hattı hataları Telegram üzerinden iletilir.
* `scheduler.py` hafta sonlarını ve ABD NYSE resmi tatillerini atlayarak `SCHEDULE_TIME` (`America/New_York`) saatinde tam boru hattını çalıştırır.

## Proje Dizin Yapısı

    fin-data-ai-pipeline/
    ├── docker/
    │   └── init.sql                 # İlk kurulum veritabanı şeması ve pgvector uzantısı
    ├── migrations/
    │   ├── 001_portfolio_and_trades.sql
    │   └── 002_news_score_and_outcomes.sql
    ├── scripts/
    │   ├── ablation.py              # Kural/filtre etki analizi betiği
    │   └── walk_forward_and_exits.py# 70/30 OOS ve çıkış parametresi tarama betiği
    ├── src/
    │   ├── config.py                # Varlık profilleri, risk limitleri ve dinamik watchlist
    │   ├── db.py                    # PostgreSQL bağlantı yöneticisi
    │   ├── migrate.py               # Numaralı SQL migration uygulayıcı
    │   ├── indicators.py            # Ortak teknik indikatörler (SMA, RSI, ATR, RS vs QQQ)
    │   ├── fetch_prices.py          # 5Y OHLCV fiyat ETL ve bayat veri kontrolü
    │   ├── fetch_news.py            # Haber çekme, URL deduplication ve vektörleştirme
    │   ├── signal_engine.py         # FinBERT skor önbelleği, karar motoru ve forward-test
    │   ├── backtest.py              # Next-open + slippage iki dönemli rejim doğrulama
    │   ├── portfolio_cli.py         # Portföy alım/satım, trailing stop ve trades defteri
    │   ├── watchlist_cli.py         # İzleme listesi ve varlık tipi yönetimi
    │   ├── notifier.py              # Telegram bildirim modülü
    │   ├── pipeline_runner.py       # Uçtan uca ana boru hattı orkestratörü
    │   └── scheduler.py             # NYSE tatil duyarlı otonom zamanlayıcı
    ├── tests/
    │   └── test_engine.py           # Birim ve entegrasyon test paketi (22 test)
    ├── .dockerignore
    ├── .env.example
    ├── docker-compose.yml
    ├── requirements.txt
    └── README.md

## Kurulum ve Hızlı Başlangıç

    # 1. PostgreSQL + pgvector konteynerini başlat
    docker compose up -d

    # 2. Ortam değişkenlerini oluştur
    cp .env.example .env

    # 3. Python bağımlılıklarını kur ve şema migration dosyalarını uygula
    pip install -r requirements.txt
    python src/migrate.py

    # 4. Uçtan uca veri çekme, FinBERT skorlama ve sinyal üretimini çalıştır
    python src/pipeline_runner.py

    # 5. İki dönemli rejim doğrulama backtest raporunu al
    python src/backtest.py

## Portföy ve İzleme Listesi Komutları

    # Portföy durumunu görüntüle
    python src/portfolio_cli.py list

    # Yeni pozisyon ekle (opsiyonel giriş tarihi ile)
    python src/portfolio_cli.py buy NVDA 5 150.00 2026-11-28

    # Pozisyon sat / kapat (kısmi veya tam)
    python src/portfolio_cli.py sell NVDA 2 168.50

    # İzleme listesini yönet
    python src/watchlist_cli.py list
    python src/watchlist_cli.py add AMD
    python src/watchlist_cli.py add IREX LEVERAGED

## Doğrulama ve Testler

    pytest -v
