import os
import polars as pl
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from db import get_db_connection

SYMBOLS = ["AAPL", "NVDA", "MSFT", "SAGT", "IREX"]
FINBERT_MODEL = "ProsusAI/finbert"


def load_prices_df() -> pl.DataFrame:
    query = """
        SELECT symbol, price_date, close_price::float AS close
        FROM market_prices
        ORDER BY symbol, price_date ASC;
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()

    return pl.DataFrame(rows, schema=["symbol", "price_date", "close"], orient="row")


def compute_technical_indicators(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort(["symbol", "price_date"])

    df = df.with_columns([
        pl.col("close").rolling_mean(window_size=20).over("symbol").alias("sma_20"),
        pl.col("close").rolling_mean(window_size=50).over("symbol").alias("sma_50"),
        (pl.col("close") - pl.col("close").shift(1).over("symbol")).alias("delta"),
    ])

    df = df.with_columns([
        pl.when(pl.col("delta") > 0).then(pl.col("delta")).otherwise(0.0).alias("gain"),
        pl.when(pl.col("delta") < 0).then(-pl.col("delta")).otherwise(0.0).alias("loss"),
    ])

    df = df.with_columns([
        pl.col("gain").rolling_mean(window_size=14).over("symbol").alias("avg_gain"),
        pl.col("loss").rolling_mean(window_size=14).over("symbol").alias("avg_loss"),
    ])

    df = df.with_columns(
        (100.0 - (100.0 / (1.0 + (pl.col("avg_gain") / (pl.col("avg_loss") + 1e-9))))).alias("rsi_14")
    )

    return df.group_by("symbol").last().sort("symbol")


def compute_finbert_sentiment() -> dict[str, tuple[float, int]]:
    print(f"--- FinBERT Finansal Duygu Modeli Yükleniyor ({FINBERT_MODEL}) ---")
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
    model.eval()

    query = """
        SELECT symbol, title, content
        FROM market_news
        ORDER BY published_at DESC NULLS LAST;
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()

    scores_by_symbol: dict[str, list[float]] = {s: [] for s in SYMBOLS}

    for symbol, title, content in rows:
        if symbol not in scores_by_symbol:
            continue
        text = f"{title}. {content}"[:512]
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).squeeze().tolist()
        net_score = probs[0] - probs[1]
        scores_by_symbol[symbol].append(net_score)

    summary = {}
    for sym, vals in scores_by_symbol.items():
        if vals:
            summary[sym] = (round(sum(vals) / len(vals), 3), len(vals))
        else:
            summary[sym] = (0.0, 0)
    return summary


def generate_signal(close: float, sma20: float | None, sma50: float | None, rsi: float | None, sentiment: float) -> tuple[str, float, list[str]]:
    score = 0.0
    reasons = []

    if sma20 is not None and sma50 is not None:
        if close > sma20 > sma50:
            score += 1.5
            reasons.append("Yükseliş trendi (Fiyat > SMA20 > SMA50)")
        elif close < sma20 < sma50:
            score -= 1.5
            reasons.append("Düşüş trendi (Fiyat < SMA20 < SMA50)")
        else:
            reasons.append("Karışık/Yatay hareketli ortalama")

    if rsi is not None:
        if rsi < 35:
            score += 1.0
            reasons.append(f"RSI aşırı satım bölgesinde ({rsi:.1f})")
        elif rsi > 70:
            score -= 1.0
            reasons.append(f"RSI aşırı alım bölgesinde ({rsi:.1f})")
        else:
            reasons.append(f"RSI dengeli ({rsi:.1f})")

    if sentiment > 0.15:
        score += 1.0
        reasons.append(f"Haber duygusu pozitif ({sentiment:+.2f})")
    elif sentiment < -0.15:
        score -= 1.0
        reasons.append(f"Haber duygusu negatif ({sentiment:+.2f})")
    else:
        reasons.append(f"Haber duygusu nötr ({sentiment:+.2f})")

    if score >= 2.0:
        decision = "GÜÇLÜ AL"
    elif score >= 0.75:
        decision = "AL"
    elif score <= -2.0:
        decision = "GÜÇLÜ SAT"
    elif score <= -0.75:
        decision = "SAT"
    else:
        decision = "BEKLE / TUT"

    return decision, round(score, 2), reasons


def main():
    prices_df = load_prices_df()
    tech_df = compute_technical_indicators(prices_df)
    sentiment_map = compute_finbert_sentiment()

    print("\n" + "=" * 76)
    print("HİBRİT TEKNİK + FINBERT SİNYAL MOTORU RAPORU")
    print("=" * 76)

    for row in tech_df.iter_rows(named=True):
        sym = row["symbol"]
        close = row["close"]
        sma20 = row["sma_20"]
        sma50 = row["sma_50"]
        rsi = row["rsi_14"]
        sent_score, news_cnt = sentiment_map.get(sym, (0.0, 0))

        decision, total_score, reasons = generate_signal(close, sma20, sma50, rsi, sent_score)

        sma20_str = f"${sma20:.2f}" if sma20 is not None else "N/A"
        sma50_str = f"${sma50:.2f}" if sma50 is not None else "N/A"
        rsi_str = f"{rsi:.1f}" if rsi is not None else "N/A"

        print(f"\n[{sym}] KARAR: {decision} (Skor: {total_score:+.2f})")
        print(f"  - Fiyat: ${close:.2f} | SMA-20: {sma20_str} | SMA-50: {sma50_str} | RSI-14: {rsi_str}")
        print(f"  - FinBERT Haber Skoru ({news_cnt} haber): {sent_score:+.3f}")
        print(f"  - Gerekçeler: {' | '.join(reasons)}")

    print("\n" + "=" * 76)


if __name__ == "__main__":
    main()
