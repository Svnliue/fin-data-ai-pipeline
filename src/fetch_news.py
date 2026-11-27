from datetime import datetime, timezone
import yfinance as yf
from sentence_transformers import SentenceTransformer
from db import get_db_connection
from config import get_active_symbols, EMBEDDING_MODEL


def fetch_and_embed_news(model: SentenceTransformer, symbol: str) -> int:
    ticker = yf.Ticker(symbol)
    raw_news = ticker.news or []
    if not raw_news:
        return 0

    records = []
    for item in raw_news:
        content_obj = item.get("content", item)
        title = content_obj.get("title") or ""
        summary = content_obj.get("summary") or content_obj.get("description") or title

        canonical = content_obj.get("canonicalUrl") or {}
        url = canonical.get("url") if isinstance(canonical, dict) else (content_obj.get("link") or "")

        pub_date_raw = content_obj.get("pubDate")
        published_at = None
        if isinstance(pub_date_raw, str):
            try:
                published_at = datetime.fromisoformat(pub_date_raw.replace("Z", "+00:00"))
            except ValueError:
                published_at = datetime.now(timezone.utc)

        if not title or not url:
            continue

        text_to_embed = f"{title}. {summary}"
        vec = model.encode(text_to_embed, normalize_embeddings=True).tolist()
        vec_str = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
        records.append((symbol, title, summary, url, published_at, vec_str))

    if not records:
        return 0

    query = """
        INSERT INTO market_news (symbol, title, content, url, published_at, embedding)
        VALUES (%s, %s, %s, %s, %s, %s::vector)
        ON CONFLICT (url) DO NOTHING;
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(query, records)
        conn.commit()

    return len(records)


def main():
    print(f"--- AI Haber & Vektör Boru Hattı Başlatıldı ({EMBEDDING_MODEL}) ---")
    model = SentenceTransformer(EMBEDDING_MODEL)
    total = 0

    for sym in get_active_symbols():
        count = fetch_and_embed_news(model, sym)
        total += count
        print(f"[BAŞARILI] {sym}: {count} haber vektörleştirildi ve DB'ye yazıldı.")

    print(f"--- Tamamlandı: Toplam {total} haber vektörü işlendi ---")


if __name__ == "__main__":
    main()
