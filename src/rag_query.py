import re
from sentence_transformers import SentenceTransformer
from db import get_db_connection

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
KNOWN_SYMBOLS = {"AAPL", "NVDA", "MSFT", "SAGT", "IREX"}


def extract_symbol_from_query(query: str) -> str | None:
    """Kullanıcı sorusundan hisse sembolünü veya şirket adını tespit eder."""
    upper_q = query.upper()

    # Doğrudan sembol geçiyorsa yakala
    tokens = set(re.findall(r"\b[A-Z]{2,5}\b", upper_q))
    matched = tokens.intersection(KNOWN_SYMBOLS)
    if matched:
        return sorted(matched)[0]

    # Şirket ismi geçiyorsa sembole eşle
    aliases = {
        "APPLE": "AAPL",
        "IPHONE": "AAPL",
        "NVIDIA": "NVDA",
        "MICROSOFT": "MSFT",
    }
    for name, sym in aliases.items():
        if name in upper_q:
            return sym

    return None


def get_price_summary(symbol: str) -> dict | None:
    """PostgreSQL'den ilgili hissenin yapısal fiyat özetini çeker."""
    sql = """
        WITH ordered_prices AS (
            SELECT
                price_date,
                close_price,
                FIRST_VALUE(close_price) OVER (ORDER BY price_date ASC) AS first_close,
                FIRST_VALUE(close_price) OVER (ORDER BY price_date DESC) AS last_close
            FROM market_prices
            WHERE symbol = %s
        )
        SELECT
            MIN(price_date) AS start_date,
            MAX(price_date) AS end_date,
            COUNT(*) AS trading_days,
            ROUND(MIN(close_price), 2) AS min_price,
            ROUND(MAX(close_price), 2) AS max_price,
            ROUND(MAX(first_close), 2) AS first_close,
            ROUND(MAX(last_close), 2) AS last_close,
            ROUND(((MAX(last_close) - MAX(first_close)) / NULLIF(MAX(first_close), 0)) * 100, 2) AS pct_change
        FROM ordered_prices;
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (symbol,))
            row = cur.fetchone()

    if not row or row[2] == 0:
        return None

    return {
        "symbol": symbol,
        "start_date": str(row[0]),
        "end_date": str(row[1]),
        "trading_days": row[2],
        "min_price": float(row[3]),
        "max_price": float(row[4]),
        "first_close": float(row[5]),
        "last_close": float(row[6]),
        "pct_change": float(row[7]),
    }


def search_semantic_news(
    model: SentenceTransformer,
    query: str,
    symbol: str | None = None,
    top_k: int = 3,
) -> list[dict]:
    """Kullanıcı sorusunu vektöre çevirip pgvector üzerinde kosinüs benzerliği arar."""
    query_vec = model.encode([query], normalize_embeddings=True)[0]
    vec_str = "[" + ",".join(f"{x:.6f}" for x in query_vec.tolist()) + "]"

    if symbol:
        sql = """
            SELECT
                symbol,
                title,
                content,
                source,
                published_at,
                ROUND((1 - (embedding <=> %s::vector))::numeric, 4) AS similarity
            FROM market_news
            WHERE symbol = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        params = (vec_str, symbol, vec_str, top_k)
    else:
        sql = """
            SELECT
                symbol,
                title,
                content,
                source,
                published_at,
                ROUND((1 - (embedding <=> %s::vector))::numeric, 4) AS similarity
            FROM market_news
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        params = (vec_str, vec_str, top_k)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    results = []
    for r in rows:
        results.append({
            "symbol": r[0],
            "title": r[1],
            "content": r[2],
            "source": r[3],
            "published_at": str(r[4])[:10] if r[4] else "N/A",
            "similarity": float(r[5]),
        })
    return results


def ask_financial_rag(query: str, model: SentenceTransformer):
    """SQL fiyat verileri ile Vektör haber aramasını birleştiren Hybrid RAG motoru."""
    print("\n" + "=" * 72)
    print(f"SORU: {query}")
    print("=" * 72)

    symbol = extract_symbol_from_query(query)
    price_info = get_price_summary(symbol) if symbol else None
    news_hits = search_semantic_news(model, query, symbol=symbol, top_k=3)

    # 1. Yapısal Fiyat Analizi Çıktısı
    if price_info:
        direction = "YÜKSELİŞ" if price_info["pct_change"] >= 0 else "DÜŞÜŞ"
        print(f"\n[SQL FİYAT ANALİZİ - {price_info['symbol']}]")
        print(
            f"• Dönem ({price_info['start_date']} -> {price_info['end_date']}, "
            f"{price_info['trading_days']} işlem günü): "
            f"${price_info['first_close']} -> ${price_info['last_close']} "
            f"(%{price_info['pct_change']} {direction})"
        )
        print(
            f"• Dönem Aralığı: En Düşük ${price_info['min_price']} | "
            f"En Yüksek ${price_info['max_price']}"
        )
    else:
        print("\n[SQL FİYAT ANALİZİ]: Soruda spesifik bir hisse tespit edilmedi, genel haber araması yapıldı.")

    # 2. Anlamsal Vektör Arama Sonuçları
    print("\n[PGVECTOR ANLAMSAL HABER EŞLEŞMELERİ (Top-3)]")
    if not news_hits:
        print("• İlgili haber bulunamadı.")
    else:
        for idx, item in enumerate(news_hits, start=1):
            print(
                f"{idx}. [{item['symbol']}] (Benzerlik: {item['similarity']:.4f}) "
                f"{item['title']} — ({item['source']}, {item['published_at']})"
            )
            print(f"   Özet: {item['content'][:160]}...")

    print("=" * 72 + "\n")


if __name__ == "__main__":
    print(f"--- Hybrid Financial RAG Motoru Yükleniyor ({EMBEDDING_MODEL_NAME}) ---")
    embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    sample_questions = [
        "Apple AI selloff and App Store safety news performance",
        "NVIDIA data center chips and artificial intelligence growth",
        "Michael Burry stock market short position",
    ]

    for q in sample_questions:
        ask_financial_rag(q, embed_model)