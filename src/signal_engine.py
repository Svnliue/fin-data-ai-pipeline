from datetime import datetime, date, timezone
import polars as pl
import torch
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from db import get_db_connection
from logger import get_logger
from config import (
    get_active_asset_map,
    ASSET_PROFILES,
    BENCHMARK_SYMBOL,
    FINBERT_MODEL,
    PORTFOLIO_CAPITAL,
    RISK_PER_TRADE,
    MAX_POSITION_PCT,
)

log = get_logger("signal_engine")

EARNINGS_BLACKOUT_DAYS = 3
SHOCK_NEWS_THRESHOLD = -0.75
SHOCK_VOL_RATIO = 2.5


def load_prices_df() -> pl.DataFrame:
    query = """
        SELECT symbol, price_date,
               open_price::float AS high_open,
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

    return pl.DataFrame(rows, schema=["symbol", "price_date", "open", "high", "low", "close", "volume"], orient="row")


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


def compute_technical_and_risk(df: pl.DataFrame, symbols: list[str], qqq_ret_20d: float) -> pl.DataFrame:
    df = df.filter(pl.col("symbol").is_in(symbols)).sort(["symbol", "price_date"])

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


def compute_finbert_sentiment(symbols: list[str]) -> dict[str, tuple[float, float, int]]:
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

    scores_by_symbol: dict[str, list[float]] = {s: [] for s in symbols}

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
        sym: (round(sum(vals) / len(vals), 3), round(min(vals), 3), len(vals)) if vals else (0.0, 0.0, 0)
        for sym, vals in scores_by_symbol.items()
    }


def get_days_to_earnings(symbol: str, asset_type: str) -> int | None:
    if asset_type != "STOCK":
        return None
    try:
        cal = yf.Ticker(symbol).calendar
        if not cal or "Earnings Date" not in cal:
            return None
        edates = cal["Earnings Date"]
        if not edates:
            return None
        next_edate = edates[0]
        if isinstance(next_edate, datetime):
            next_edate = next_edate.date()
        if isinstance(next_edate, date):
            today = datetime.now(timezone.utc).date()
            diff = (next_edate - today).days
            return diff if diff >= 0 else None
    except Exception:
        return None
    return None


def generate_signal(
    row: dict,
    asset_type: str,
    sentiment: float,
    is_macro_bull: bool,
    min_news_score: float = 0.0,
    days_to_earnings: int | None = None,
) -> tuple[str, float, list[str], dict]:
    profile = ASSET_PROFILES.get(asset_type, ASSET_PROFILES["STOCK"])
    close = row["close"]
    sma20 = row["sma_20"]
    sma50 = row["sma_50"]
    sma200 = row["sma_200"]
    rsi = row["rsi_14"]
    atr = row["atr_14"] or (close * 0.02)
    vol_ratio = row["vol_ratio"] or 1.0
    rs_pct = row["rs_vs_qqq_pct"] or 0.0
    delta = row.get("delta") or 0.0

    is_stock_bull = (close >= sma200) if sma200 is not None else True

    score = 0.0
    reasons = [f"Profil: {asset_type}"]

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

    news_shock = min_news_score <= SHOCK_NEWS_THRESHOLD
    vol_price_shock = (vol_ratio >= SHOCK_VOL_RATIO) and (delta <= -2.5 * atr)
    shock_active = news_shock or vol_price_shock

    hard_lock_reasons = []
    if profile["require_macro_bull"] and not is_macro_bull:
        hard_lock_reasons.append("QQQ < SMA200 Makro Ayı Rejimi")
    if profile["require_stock_sma200"] and not is_stock_bull:
        hard_lock_reasons.append(f"Varlık < SMA200 (${sma200:.2f})")
    if days_to_earnings is not None and days_to_earnings <= EARNINGS_BLACKOUT_DAYS:
        hard_lock_reasons.append(f"Bilanço Koruması ({days_to_earnings} gün kaldı)")
    if news_shock:
        hard_lock_reasons.append(f"Haber Şoku Kilidi (Min FinBERT: {min_news_score:+.2f})")
    if vol_price_shock:
        hard_lock_reasons.append(f"Hacim/Fiyat Şoku ({vol_ratio:.1f}x hacim)")

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

    effective_sl_mult = 0.75 if shock_active else profile["sl_atr"]
    stop_loss = max(0.01, close - (effective_sl_mult * atr))
    partial_tp = close + (profile["tp1_atr"] * atr)
    take_profit = close + (profile["tp2_atr"] * atr)
    risk_per_share = max(0.01, close - stop_loss)

    max_risk_usd = PORTFOLIO_CAPITAL * RISK_PER_TRADE
    risk_based_shares = int(max_risk_usd // risk_per_share)
    cap_based_shares = int((PORTFOLIO_CAPITAL * MAX_POSITION_PCT) // max(0.01, close))
    suggested_shares = min(risk_based_shares, cap_based_shares)

    risk_plan = {
        "asset_type": asset_type,
        "max_hold": profile["max_hold_days"],
        "shock_active": shock_active,
        "days_to_earnings": days_to_earnings,
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
            cur.execute(query, (symbol, decision[:250], score, close, risk["stop_loss"], risk["take_profit"], sentiment))
        conn.commit()


def main():
    asset_map = get_active_asset_map()
    symbols = list(asset_map.keys())
    prices_df = load_prices_df()
    is_macro_bull, qqq_close, qqq_sma200, qqq_ret_20d = get_benchmark_metrics(prices_df)
    tech_df = compute_technical_and_risk(prices_df, symbols, qqq_ret_20d)
    sentiment_map = compute_finbert_sentiment(symbols)

    regime_label = "BOĞA REJİMİ (Hisse Kilidi Açık)" if is_macro_bull else "AYI REJİMİ (STOCK Kilidi Aktif | COMMODITY/ETF Bağımsız)"
    log.info(f"Makro Rejim [{BENCHMARK_SYMBOL}]: ${qqq_close:.2f} vs SMA-200 ${qqq_sma200:.2f} -> {regime_label}")

    print("\n" + "=" * 100)
    print("ALPHA ENGINE v4.8 SİNYAL RAPORU [VARLIK SINIFI + BİLANÇO KORUMASI + HABER/HACİM ŞOKU KİLİDİ]")
    print(f"MAKRO REJİM [{BENCHMARK_SYMBOL}]: ${qqq_close:.2f} vs SMA-200 ${qqq_sma200:.2f} -> {regime_label}")
    print("=" * 100)

    for row in tech_df.iter_rows(named=True):
        sym = row["symbol"]
        atype = asset_map.get(sym, "STOCK")
        sent_score, min_sent, _ = sentiment_map.get(sym, (0.0, 0.0, 0))
        earnings_days = get_days_to_earnings(sym, atype)
        decision, total_score, reasons, risk = generate_signal(
            row, atype, sent_score, is_macro_bull, min_news_score=min_sent, days_to_earnings=earnings_days
        )
        persist_signal(sym, decision, total_score, row["close"], risk, sent_score)

        hold_note = f" | Max Süre: {risk['max_hold']} gün" if risk["max_hold"] else ""
        shock_note = " [ŞOK: SIKI STOP 0.75x ATR]" if risk["shock_active"] else ""
        earn_note = f" | Bilanço: {earnings_days}g" if earnings_days is not None else ""

        print(f"\n[{sym} | {atype}] KARAR: {decision} (Güven Skoru: {total_score:+.2f}){earn_note}")
        print(f"  - Fiyat: ${row['close']:.2f} | SMA-50: ${row['sma_50']:.2f} | SMA-200: ${risk['sma_200']:.2f} | RSI-14: {row['rsi_14']:.1f}")
        print(f"  - ATR-14: ${risk['atr']:.2f} | SL: ${risk['stop_loss']:.2f}{shock_note} | TP-1 (%50+Breakeven): ${risk['partial_tp']:.2f} | TP-2: ${risk['take_profit']:.2f}{hold_note}")
        print(f"  - Volatilite Bazlı Lot ($10k kasa, %1 risk): {risk['shares']} adet (~${risk['position_usd']:.2f})")
        print(f"  - Puan Detayı: {' | '.join(reasons)}")

    print("\n[DB] Tüm sinyaller PostgreSQL signal_history tablosuna kaydedildi.")
    print("=" * 100)


if __name__ == "__main__":
    main()
