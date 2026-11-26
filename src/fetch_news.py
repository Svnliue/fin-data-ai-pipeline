import time
from datetime import datetime, timezone
from sentence_transformers import SentenceTransformer
import yfinance as yf
from db import get_db_connection

SYMBOLS = ["AAPL", "NVDA", "MSFT", "SAGT", "IREX"]
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def parse_yahoo_news(symbol: str, raw_news: list[dict]) -> list[dict]:
    """Yahoo Finance'in farklı JSON versiyonlarını standart şemaya oturtur."""
    parsed_items = []

    for item in raw_news:
        # Güncel yfinance sürümlerinde veriler 'content' sözlüğü içinde gelir
        data = item.get("content", item)

        title = data.get("title")
        summary = data.get("summary") or data.get("description") or title

        # URL çıkarımı
        url = None
        if isinstance(data.get("canonicalUrl"), dict):
            url = data["canonicalUrl"].get("url")
        elif isinstance(data.get("clickThroughUrl"), dict):
            url = data["clickThroughUrl"].get("url")
        else:
            url = data.get("link") or data.get("url")

        # Kaynak (Publisher) çıkarımı
        provider = data.get("provider")
        if isinstance(provider, dict):
            source = provider.get("displayName", "Yahoo Finance")
        else:
            source = data.get("publisher") or "Yahoo Finance"

        # Yayınlanma tarihi çıkarımı (ISO string veya Unix timestamp)
        pub_raw = data.get("pubDate") or data.get("providerPublishTime")
        published_at = None
        if isinstance(pub_raw, str):
            try:
                published_at = datetime.fromisoformat(pub_raw.replace("Z", "+00:00"))
            except ValueError:
                published_at = datetime.now(timezone.utc)
        elif isinstance(pub_raw, (int, float)):
            published_at = datetime.fromtimestamp(pub_raw, tz=timezone.utc)
        else:
            published_at = datetime.now(timezone.utc)

        # Başlık veya URL yoksa bozuk kaydı ele
        if not title or not url:
            continue

        parsed_items.append({
            "symbol": symbol,
            "title": title.strip(),
            "content": summary.strip(),
            "source": str(source)[:100],
            "url": url.strip(),
            "published_at": published_at,
        })

    return parsed_items


def fetch_and_embed_news(model: SentenceTransformer, symbol: str) -> list[tuple]:
    """Bir hissenin haberlerini çeker ve 384 boyutlu AI vektörlerini üretir."""
    ticker = yf.Ticker(symbol)
    raw_news = ticker.news or []
    cleaned_news = parse_yahoo_news(symbol, raw_news)

    if not cleaned_news:
        return []

    # Başlık ve içeriği birleştirerek yapay zekaya zengin bağlam veriyoruz
    texts_to_embed = [
        f"{item['symbol']} | {item['title']}: {item['content']}"
        for item in cleaned_news
    ]

    # Toplu (batch) vektör üretimi — tek tek üretmekten kat kat hızlıdır
    vectors = model.encode(texts_to_embed, normalize_embeddings=True)

    rows = []
    for item, vec in zip(cleaned_news, vectors):
        # pgvector formatı için listeyi '[0.12, -0.04, ...]' stringine çeviriyoruz
        vec_str = "[" + ",".join(f"{x:.6f}" for x in vec.tolist()) + "]"
        rows.append((
            item["symbol"],
            item["title"],
            item["content"],
            item["source"],
            item["url"],
            item["published_at"],
            vec_str,
        ))

    return rows


def load_news_to_postgres(rows: list[tuple]) -> int:
    """Haberleri ve vektörleri idempotent şekilde PostgreSQL'e yazar."""
    if not rows:
        return 0

    upsert_query = """
        INSERT INTO market_news (symbol, title, content, source, url, published_at, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
        ON CONFLICT (url)
        DO UPDATE SET
            title = EXCLUDED.title,
            content = EXCLUDED.content,
            embedding = EXCLUDED.embedding;
    """

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(upsert_query, rows)
        conn.commit()

    return len(rows)


def run_news_pipeline():
    print(f"--- AI Haber & Vektör Boru Hattı Başlatıldı ({EMBEDDING_MODEL_NAME}) ---")
    # Modeli döngü dışında SADECE 1 KEZ yüklüyoruz
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    total_processed = 0
    for symbol in SYMBOLS:
        try:
            rows = fetch_and_embed_news(model, symbol)
            count = load_news_to_postgres(rows)
            total_processed += count
            print(f"[BAŞARILI] {symbol}: {count} haber vektörleştirildi ve DB'ye yazıldı.")
        except Exception as exc:
            print(f"[HATA] {symbol} haberleri işlenemedi: {exc}")

        time.sleep(1.5)

    print(f"--- Tamamlandı: Toplam {total_processed} haber vektörü işlendi ---")


if __name__ == "__main__":
    run_news_pipeline()