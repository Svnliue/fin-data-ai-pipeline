from datetime import datetime, timezone
import yfinance as yf
from sentence_transformers import SentenceTransformer
from db import get_db_connection
from fetch_prices import get_target_symbols
from config import EMBEDDING_MODEL, BENCHMARK_SYMBOL
from logger import get_logger

log = get_logger("fetch_news")


def _get_existing_urls(urls: list[str]) -> set[str]:
    if not urls:
        return set()
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT url FROM market_news WHERE url = ANY(%s);", (urls,))
            return {row[0] for row in cur.fetchall()}


def fetch_news_for_symbol(symbol: str, model: SentenceTransformer) -> int:
    ticker = yf.Ticker(symbol)
    raw_news = ticker.news or []
    if not raw_news:
        return 0

    candidates = []
    for item in raw_news:
        content = item.get("content") if isinstance(item.get("content"), dict) else item
        title = (content.get("title") or "").strip()
        summary = (content.get("summary") or content.get("description") or "").strip()
        canonical = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
        url = (canonical.get("url") if isinstance(canonical, dict) else content.get("link")) or ""
        pub_raw = content.get("pubDate") or content.get("providerPublishTime")

        if not title or not url:
            continue

        pub_dt = None
        if isinstance(pub_raw, (int, float)):
            pub_dt = datetime.fromtimestamp(pub_raw, tz=timezone.utc)
        elif isinstance(pub_raw, str):
            try:
                pub_dt = datetime.fromisoformat(pub_raw.replace("Z", "+00:00"))
            except ValueError:
                pub_dt = None

        candidates.append({"title": title, "summary": summary, "url": url, "pub_dt": pub_dt})

    existing_urls = _get_existing_urls([c["url"] for c in candidates])
    new_items = [c for c in candidates if c["url"] not in existing_urls]
    if not new_items:
        return 0

    texts = [f"{c['title']}. {c['summary']}" for c in new_items]
    embeddings = model.encode(texts, normalize_embeddings=True)

    rows = [
        (symbol, c["title"], c["summary"], c["url"], c["pub_dt"], emb.tolist())
        for c, emb in zip(new_items, embeddings)
    ]
    query = """
        INSERT INTO market_news (symbol, title, summary, url, published_at, embedding)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (url) DO NOTHING;
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(query, rows)
        conn.commit()
    return len(rows)


def main() -> int:
    log.info("Haber & Vektor ETL Boru Hatti Baslatildi (Pre-Embed Dedupe + Acik Pozisyonlar)")
    model = SentenceTransformer(EMBEDDING_MODEL)
    symbols = [s for s in get_target_symbols() if s != BENCHMARK_SYMBOL]
    total_inserted = 0
    for sym in symbols:
        try:
            count = fetch_news_for_symbol(sym, model)
            total_inserted += count
            print(f"[HABER] {sym}: {count} yeni haber vektorlestirildi ve kaydedildi.")
        except Exception as exc:
            log.error(f"{sym} haber cekme hatasi: {exc}")
    return total_inserted


if __name__ == "__main__":
    main()
