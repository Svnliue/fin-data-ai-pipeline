import polars as pl
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from db import get_db_connection
from logger import get_logger
from config import (
    SYMBOLS,
    BENCHMARK_SYMBOL,
    FINBERT_MODEL,
    PORTFOLIO_CAPITAL,
    RISK_PER_TRADE,
    MAX_POSITION_PCT,
)

log = get_logger("signal_engine")


def load_prices_df() -> pl.DataFrame:
    query = """
        SELECT symbol, price_date,
               high_price::float AS high,
               low_price::float AS low,
               close_price::float AS close,
               volume::float AS volume
        FROM market_prices
        ORDER BY symbol, price_date ASC;
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()

    return pl.DataFrame(rows, schema=["symbol", "price_date", "high", "low", "close", "volume"], orient="row")


def get_benchmark_metrics(df: pl.DataFrame) -> tuple[bool, float, float, float]:
    qqq = df.filter(pl.col("symbol") == BENCHMARK_SYMBOL).sort("price_date")
    if qqq.height < 200:
        return True, 0.0, 0.0, 0.0

    qqq = qqq.with_columns([
        pl.col("close").rolling_mean(window_size=200).alias("sma_200"),
        ((pl.col("close") / pl.col("close").shift(20)) - 1.0).alias("ret_20d"),
    ])
    last_row = qqq.tail(1).to_dicts()[0]
    close = last_row["close"]
    sma200 = last_row["sma_200"]
    ret_20d = last_row["ret_20d"] or 0.0
    return (close >= sma200), round(close, 2), round(sma200, 2), ret_20d


def compute_technical_and_risk(df: pl.DataFrame, qqq_ret_20d: float) -> pl.DataFrame:
    df = df.filter(pl.col("symbol").is_in(SYMBOLS)).sort(["symbol", "price_date"])

    df = df.with_columns([
        pl.col("close").rolling_mean(window_size=20).over("symbol").alias("sma_20"),
        pl.col("close").rolling_mean(window_size=50).over("symbol").alias("sma_50"),
        pl.col("close").rolling_mean(window_size=200).over("symbol").alias("sma_200"),
        pl.col("volume").rolling_mean(window_size=20).over("symbol").alias("vol_sma_20"),
        ((pl.col("close") / pl.col("close").shift(20).over("symbol")) - 1.0).alias("ret_20d"),
        (pl.col("close") - pl.col("close").shift(1).over("symbol")).alias("delta"),
        pl.max_horizontal([
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("close").shift(1).over("symbol")).abs(),
            (pl.col("low") - pl.col("close").shift(1).over("symbol")).abs(),
        ]).alias("tr"),
    ])

    df = df.with_columns([
        pl.when(pl.col("delta") > 0).then(pl.col("delta")).otherwise(0.0).alias("gain"),
        pl.when(pl.col("delta") < 0).then(-pl.col("delta")).otherwise(0.0).alias("loss"),
        pl.col("tr").rolling_mean(window_size=14).over("symbol").alias("atr_14"),
        (pl.col("volume") / (pl.col("vol_sma_20") + 1e-9)).alias("vol_ratio"),
        ((pl.col("ret_20d") - qqq_ret_20d) * 100.0).alias("rs_vs_qqq_pct"),
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
    log.info(f"FinBERT Finansal Duygu Modeli Yükleniyor ({FINBERT_MODEL})")
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

    return {
        sym: (round(sum(vals) / len(vals), 3), len(vals)) if vals else (0.0, 0)
        for sym, vals in scores_by_symbol.items()
    }


def generate_signal(row: dict, sentiment: float, is_macro_bull: bool) -> tuple[str, float, list[str], dict]:
    close = row["close"]
    sma20 = row["sma_20"]
    sma50 = row["sma_50"]
    sma200 = row["sma_200"]
    rsi = row["rsi_14"]
    atr = row["atr_14"] or (close * 0.02)
    vol_ratio = row["vol_ratio"] or 1.0
    rs_pct = row["rs_vs_qqq_pct"] or 0.0

    is_stock_bull = (close >= sma200) if sma200 is not None else True

    score = 0.0
    reasons = []

    if sma20 is not None and sma50 is not None:
        if close > sma20 > sma50:
            score += 1.5
            reasons.append("Kısa/Orta yükseliş trendi (Fiyat > SMA20 > SMA50)")
        elif close < sma20 < sma50:
            score -= 1.5
            reasons.append("Kısa/Orta düşüş trendi (Fiyat < SMA20 < SMA50)")
        else:
            reasons.append("Yatay/Karışık ortalama")

    if rs_pct >= 2.0:
        score += 0.5
        reasons.append(f"Endeks lideri (RS vs QQQ: {rs_pct:+.1f}%)")
    elif rs_pct <= -2.0:
        score -= 0.5
        reasons.append(f"Endeksten zayıf (RS vs QQQ: {rs_pct:+.1f}%)")
    else:
        reasons.append(f"Endekse paralel (RS: {rs_pct:+.1f}%)")

    if vol_ratio >= 1.2:
        score += 0.5 if score > 0 else -0.5
        reasons.append(f"Güçlü hacim ({vol_ratio:.2f}x)")
    else:
        reasons.append(f"Normal hacim ({vol_ratio:.2f}x)")

    if rsi is not None:
        if rsi < 35:
            score += 1.0
            reasons.append(f"RSI aşırı satım ({rsi:.1f})")
        elif rsi > 70:
            score -= 1.0
            reasons.append(f"RSI aşırı alım ({rsi:.1f})")
        else:
            reasons.append(f"RSI dengeli ({rsi:.1f})")

    if sentiment > 0.15:
        score += 1.0
        reasons.append(f"FinBERT pozitif ({sentiment:+.2f})")
    elif sentiment < -0.15:
        score -= 1.0
        reasons.append(f"FinBERT negatif ({sentiment:+.2f})")
    else:
        reasons.append(f"FinBERT nötr ({sentiment:+.2f})")

    hard_lock_reasons = []
    if not is_macro_bull:
        hard_lock_reasons.append("QQQ < SMA200 Makro Ayı Rejimi")
    if not is_stock_bull:
        hard_lock_reasons.append(f"Hisse < SMA200 (${sma200:.2f})")

    if score >= 2.25:
        decision = "GÜÇLÜ AL" if not hard_lock_reasons else f"BEKLE (KİLİTLİ: {' + '.join(hard_lock_reasons)})"
    elif score >= 0.75:
        decision = "AL" if not hard_lock_reasons else f"BEKLE (KİLİTLİ: {' + '.join(hard_lock_reasons)})"
    elif score <= -2.25:
        decision = "GÜÇLÜ SAT"
    elif score <= -0.75:
        decision = "SAT"
    else:
        decision = "BEKLE / TUT"

    stop_loss = max(0.01, close - (2.0 * atr))
    partial_tp = close + (1.5 * atr)
    take_profit = close + (3.5 * atr)
    risk_per_share = max(0.01, close - stop_loss)

    max_risk_usd = PORTFOLIO_CAPITAL * RISK_PER_TRADE
    risk_based_shares = int(max_risk_usd // risk_per_share)
    cap_based_shares = int((PORTFOLIO_CAPITAL * MAX_POSITION_PCT) // max(0.01, close))
    suggested_shares = min(risk_based_shares, cap_based_shares)

    risk_plan = {
        "sma_200": round(sma200, 2) if sma200 else 0.0,
        "atr": round(atr, 2),
        "stop_loss": round(stop_loss, 2),
        "partial_tp": round(partial_tp, 2),
        "take_profit": round(take_profit, 2),
        "shares": suggested_shares,
        "position_usd": round(suggested_shares * close, 2),
    }

    return decision, round(score, 2), reasons, risk_plan


def persist_signal(symbol: str, decision: str, score: float, close: float, risk: dict, sentiment: float) -> None:
    query = """
        INSERT INTO signal_history (symbol, signal_date, decision, confidence_score, close_price, stop_loss, take_profit, finbert_score)
        VALUES (%s, CURRENT_DATE, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (symbol, signal_date)
        DO UPDATE SET
            decision = EXCLUDED.decision,
            confidence_score = EXCLUDED.confidence_score,
            close_price = EXCLUDED.close_price,
            stop_loss = EXCLUDED.stop_loss,
            take_profit = EXCLUDED.take_profit,
            finbert_score = EXCLUDED.finbert_score;
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (symbol, decision, score, close, risk["stop_loss"], risk["take_profit"], sentiment))
        conn.commit()


def main():
    prices_df = load_prices_df()
    is_macro_bull, qqq_close, qqq_sma200, qqq_ret_20d = get_benchmark_metrics(prices_df)
    tech_df = compute_technical_and_risk(prices_df, qqq_ret_20d)
    sentiment_map = compute_finbert_sentiment()

    regime_label = "BOĞA REJİMİ (Makro Kilit Açık)" if is_macro_bull else "AYI REJİMİ (Makro Kilit Aktif)"
    log.info(f"Makro Rejim [{BENCHMARK_SYMBOL}]: ${qqq_close:.2f} vs SMA-200 ${qqq_sma200:.2f} -> {regime_label}")

    print("\n" + "=" * 88)
    print("ALPHA ENGINE v4.4 SİNYAL, GÖRECELİ GÜÇ (RS) VE PARÇALI KÂR (SCALE-OUT) RİSK RAPORU")
    print(f"KATMAN-1 MAKRO KİLİT [{BENCHMARK_SYMBOL}]: ${qqq_close:.2f} vs SMA-200 ${qqq_sma200:.2f} -> {regime_label}")
    print("=" * 88)

    for row in tech_df.iter_rows(named=True):
        sym = row["symbol"]
        sent_score, _ = sentiment_map.get(sym, (0.0, 0))
        decision, total_score, reasons, risk = generate_signal(row, sent_score, is_macro_bull)
        persist_signal(sym, decision, total_score, row["close"], risk, sent_score)
        log.info(f"[{sym}] Karar={decision} | Skor={total_score:+.2f} | SL=${risk['stop_loss']:.2f} | TP1=${risk['partial_tp']:.2f}")

        print(f"\n[{sym}] KARAR: {decision} (Güven Skoru: {total_score:+.2f})")
        print(f"  - Fiyat: ${row['close']:.2f} | SMA-50: ${row['sma_50']:.2f} | SMA-200: ${risk['sma_200']:.2f} | RSI-14: {row['rsi_14']:.1f}")
        print(f"  - ATR-14: ${risk['atr']:.2f} | SL: ${risk['stop_loss']:.2f} | TP-1 (%50+Breakeven): ${risk['partial_tp']:.2f} | TP-2: ${risk['take_profit']:.2f}")
        print(f"  - Volatilite Bazlı Lot ($10k kasa, %1 risk): {risk['shares']} adet (~${risk['position_usd']:.2f})")
        print(f"  - Katman-2 Puan Detayı: {' | '.join(reasons)}")

    print("\n[DB] Tüm sinyaller PostgreSQL signal_history tablosuna kaydedildi.")
    print("=" * 88)


if __name__ == "__main__":
    main()
