from sentence_transformers import SentenceTransformer
from db import get_db_connection
from config import EMBEDDING_MODEL


def get_price_summary(symbol: str) -> dict | None:
    query = """
        WITH stats AS (
            SELECT
                FIRST_VALUE(close_price) OVER (ORDER BY price_date ASC) AS start_close,
                FIRST_VALUE(close_price) OVER (ORDER BY price_date DESC) AS end_close,
                MIN(low_price) OVER () AS min_low,
                MAX(high_price) OVER () AS max_high,
                COUNT(*) OVER () AS trading_days
            FROM market_prices
            WHERE symbol = %s
        )
        SELECT start_close, end_close, min_low, max_high, trading_days
        FROM stats
        LIMIT 1;
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (symbol,))
            row = cur.fetchone()

    if not row:
        return None

    start_c, end_c, min_l, max_h, days = row
    pct_change = ((float(end_c) - float(start_c)) / float(start_c)) * 100.0
    return {
        "symbol": symbol,
        "trading_days": days,
        "start_close": round(float(start_c), 2),
        "end_close": round(float(end_c), 2),
        "min_low": round(float(min_l), 2),
        "max_high": round(float(max_h), 2),
        "pct_change": round(pct_change, 2),
    }


def semantic_news_search(model: SentenceTransformer, question: str, symbol: str | None = None, top_k: int = 3) -> list[dict]:
    vec = model.encode(question, normalize_embeddings=True).tolist()
    vec_str = "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

    if symbol:
        sql = """
            SELECT symbol, title, content, published_at,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM market_news
            WHERE symbol = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        params = (vec_str, symbol.upper(), vec_str, top_k)
    else:
        sql = """
            SELECT symbol, title, content, published_at,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM market_news
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        params = (vec_str, vec_str, top_k)

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()

    return [
        {
            "symbol": r[0],
            "title": r[1],
            "content": r[2],
            "published_at": str(r[3]) if r[3] else "N/A",
            "similarity": round(float(r[4]), 4),
        }
        for r in rows
    ]


def main():
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("\n--- Hibrit Finansal RAG Sorgu Motoru Hazır (Çıkmak için 'q') ---")

    while True:
        q = input("\nSorunuz: ").strip()
        if q.lower() in {"q", "quit", "exit"}:
            break
        if not q:
            continue

        sym_input = input("Hisse filtresi (Örn: NVDA, AAPL veya tümü için Enter): ").strip().upper()
        sym = sym_input if sym_input else None

        if sym:
            summary = get_price_summary(sym)
            if summary:
                print(
                    f"\n[FİYAT ÖZETİ - {sym}] {summary['trading_days']} gün | "
                    f"Başlangıç: ${summary['start_close']} -> Son: ${summary['end_close']} "
                    f"({summary['pct_change']:+.2f}%) | Dip: ${summary['min_low']} | Zirve: ${summary['max_high']}"
                )

        hits = semantic_news_search(model, q, symbol=sym, top_k=3)
        print("\n[EN İLGİLİ HABERLER]")
        for i, h in enumerate(hits, 1):
            print(f"{i}. [{h['symbol']}] (Benzerlik: {h['similarity']}) {h['title']}")


if __name__ == "__main__":
    main()
