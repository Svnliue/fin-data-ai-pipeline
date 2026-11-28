from datetime import datetime, date, timezone
import polars as pl
import torch
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from db import get_db_connection
from logger import get_logger
from indicators import add_technical_indicators
from config import (
    MAX_SAME_GROUP_POSITIONS,
    MAX_DRAWDOWN_PCT,
    MAX_DAILY_LOSS_PCT,
    get_active_asset_map,
    get_all_asset_map,
    ASSET_PROFILES,
    BENCHMARK_SYMBOL,
    FINBERT_MODEL,
    PORTFOLIO_CAPITAL,
    RISK_PER_TRADE,
    MAX_POSITION_PCT,
    MAX_OPEN_POSITIONS,
)

log = get_logger("signal_engine")

EARNINGS_BLACKOUT_DAYS = 3
SHOCK_NEWS_THRESHOLD = -0.75
SHOCK_VOL_RATIO = 2.5
NEWS_LOOKBACK_DAYS = 7

_FINBERT_CACHE: tuple | None = None


def _get_finbert_pipeline():
    global _FINBERT_CACHE
    if _FINBERT_CACHE is None:
        log.info(f"FinBERT Finansal Duygu Modeli RAM'e Yükleniyor ({FINBERT_MODEL})")
        tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
        model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
        model.eval()
        _FINBERT_CACHE = (tokenizer, model)
    return _FINBERT_CACHE


def load_prices_df() -> pl.DataFrame:
    query = """
        SELECT symbol, price_date,
               open_price::float AS open,
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


def load_portfolio_positions() -> dict[str, dict]:
    query = """
        SELECT symbol, shares::float, entry_price::float,
               initial_atr::float, stop_loss::float, tp1::float, tp2::float,
               COALESCE(scaled_out, FALSE), entry_date
        FROM portfolio_positions;
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
    return {
        sym: {
            "shares": shares,
            "entry_price": entry_px,
            "initial_atr": i_atr,
            "stop_loss": sl,
            "tp1": tp1,
            "tp2": tp2,
            "scaled_out": scaled,
            "entry_date": edate,
        }
        for sym, shares, entry_px, i_atr, sl, tp1, tp2, scaled, edate in rows
    }


def load_realized_pnl() -> float:
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COALESCE(SUM(realized_pnl), 0)::float FROM trades;")
                row = cur.fetchone()
                return float(row[0]) if row else 0.0
    except Exception as exc:
        log.error(f"Realized PnL okunamadı: {exc}")
        return 0.0


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
    sub = df.filter(pl.col("symbol").is_in(symbols))
    qqq_ret_df = df.filter(pl.col("symbol") == BENCHMARK_SYMBOL).sort("price_date").with_columns(
        ((pl.col("close") / pl.col("close").shift(20)) - 1.0).alias("qqq_ret_20d")
    ).select(["price_date", "qqq_ret_20d"])

    if qqq_ret_df.is_empty():
        sub = add_technical_indicators(sub, None).with_columns(
            ((pl.col("ret_20d") - qqq_ret_20d) * 100.0).alias("rs_vs_qqq_pct")
        )
    else:
        sub = add_technical_indicators(sub, qqq_ret_df)

    return sub.group_by("symbol").last().sort("symbol")


def compute_finbert_sentiment(symbols: list[str], lookback_days: int = NEWS_LOOKBACK_DAYS) -> dict[str, tuple[float, float, int]]:
    """
    Sadece DB'de finbert_score IS NULL olan yeni haberleri skorlar ve DB'ye yazar;
    daha once skorlanmis haberleri dogrudan DB onbelleginden okur.
    """
    query = """
        SELECT id, symbol, title, content, finbert_score::float
        FROM market_news
        WHERE COALESCE(published_at, created_at) >= NOW() - (%s * INTERVAL '1 day')
        ORDER BY COALESCE(published_at, created_at) DESC;
    """
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (lookback_days,))
            rows = cur.fetchall()

    scores_by_symbol: dict[str, list[float]] = {s: [] for s in symbols}
    to_update: list[tuple[float, int]] = []

    unscored_rows = [r for r in rows if r[1] in scores_by_symbol and r[4] is None]
    tokenizer, model = _get_finbert_pipeline() if unscored_rows else (None, None)

    for news_id, symbol, title, content, cached_score in rows:
        if symbol not in scores_by_symbol:
            continue
        if cached_score is not None:
            scores_by_symbol[symbol].append(float(cached_score))
        else:
            text = f"{title}. {content}"[:512]
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).squeeze().tolist()
            net_score = round(float(probs[0] - probs[1]), 3)
            scores_by_symbol[symbol].append(net_score)
            to_update.append((net_score, news_id))

    if to_update:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.executemany("UPDATE market_news SET finbert_score = %s WHERE id = %s;", to_update)
            conn.commit()
        log.info(f"FinBERT önbellek güncellendi: {len(to_update)} yeni haber skorlandı.")

    return {
        sym: (round(sum(vals) / len(vals), 3), round(min(vals), 3), len(vals)) if vals else (0.0, 0.0, 0)
        for sym, vals in scores_by_symbol.items()
    }


def evaluate_signal_outcomes() -> int:
    """
    signal_history tablosundaki gecmis sinyallerin 5 ve 10 islem gunu sonraki
    gercek getiri yuzdelerini (forward test) otomatik doldurur.
    """
    updated = 0
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, symbol, signal_date, close_price::float
                    FROM signal_history
                    WHERE (outcome_5d_pct IS NULL OR outcome_10d_pct IS NULL)
                      AND close_price IS NOT NULL
                      AND close_price > 0;
                    """
                )
                pending = cur.fetchall()

                for sig_id, sym, sig_date, entry_close in pending:
                    cur.execute(
                        """
                        SELECT close_price::float
                        FROM market_prices
                        WHERE symbol = %s AND price_date > %s
                        ORDER BY price_date ASC
                        LIMIT 10;
                        """,
                        (sym, sig_date),
                    )
                    future_bars = [r[0] for r in cur.fetchall()]
                    out_5 = round(((future_bars[4] - entry_close) / entry_close) * 100.0, 2) if len(future_bars) >= 5 else None
                    out_10 = round(((future_bars[9] - entry_close) / entry_close) * 100.0, 2) if len(future_bars) >= 10 else None
                    if out_5 is not None or out_10 is not None:
                        cur.execute(
                            """
                            UPDATE signal_history
                            SET outcome_5d_pct = COALESCE(%s, outcome_5d_pct),
                                outcome_10d_pct = COALESCE(%s, outcome_10d_pct)
                            WHERE id = %s;
                            """,
                            (out_5, out_10, sig_id),
                        )
                        updated += 1
            conn.commit()
    except Exception as exc:
        log.error(f"Forward-test sonuç takibi güncellenemedi: {exc}")
    return updated


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


def _score_factors(row: dict, asset_type: str, sentiment: float) -> tuple[float, list[str], float, float]:
    close = row["close"]
    sma20 = row.get("sma_20")
    sma50 = row.get("sma_50")
    rsi = row.get("rsi_14")
    atr = row.get("atr_14") or (close * 0.02)
    vol_ratio = row.get("vol_ratio") or 1.0
    rs_pct = row.get("rs_vs_qqq_pct") or 0.0

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
    else:
        reasons.append("Yetersiz ortalama geçmişi (Warm-up)")

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

    return score, reasons, atr, vol_ratio


def generate_signal(
    row: dict,
    asset_type: str,
    sentiment: float,
    is_macro_bull: bool,
    min_news_score: float = 0.0,
    days_to_earnings: int | None = None,
    available_cash: float = PORTFOLIO_CAPITAL,
    open_positions_count: int = 0,
    effective_capital: float = PORTFOLIO_CAPITAL,
) -> tuple[str, float, list[str], dict]:
    profile = ASSET_PROFILES.get(asset_type, ASSET_PROFILES["STOCK"])
    close = row["close"]
    sma200 = row.get("sma_200")
    delta = row.get("delta") or 0.0
    is_stock_bull = (close >= sma200) if sma200 is not None else True

    score, reasons, atr, vol_ratio = _score_factors(row, asset_type, sentiment)

    news_shock = min_news_score <= SHOCK_NEWS_THRESHOLD
    vol_price_shock = (vol_ratio >= SHOCK_VOL_RATIO) and (delta <= -2.5 * atr)
    shock_active = news_shock or vol_price_shock

    hard_lock_reasons = []
    if open_positions_count >= MAX_OPEN_POSITIONS:
        hard_lock_reasons.append(f"Max Açık Pozisyon Dolu ({open_positions_count}/{MAX_OPEN_POSITIONS})")
    if profile["require_macro_bull"] and not is_macro_bull:
        hard_lock_reasons.append("QQQ < SMA200 Makro Ayı Rejimi")
    if profile["require_stock_sma200"] and not is_stock_bull:
        hard_lock_reasons.append(f"Varlık < SMA200 (${sma200:.2f})")
    if days_to_earnings is not None and days_to_earnings <= EARNINGS_BLACKOUT_DAYS:
        hard_lock_reasons.append(f"Bilanço Koruması ({days_to_earnings} gün kaldı)")
    if news_shock:
        hard_lock_reasons.append(f"Haber Şoku Kilidi (7g Min FinBERT: {min_news_score:+.2f})")
    if vol_price_shock:
        hard_lock_reasons.append(f"Hacim/Fiyat Şoku ({vol_ratio:.1f}x hacim)")

    effective_sl_mult = 0.75 if shock_active else profile["sl_atr"]
    stop_loss = max(0.01, close - (effective_sl_mult * atr))
    partial_tp = close + (profile["tp1_atr"] * atr)
    take_profit = close + (profile["tp2_atr"] * atr)
    risk_per_share = max(0.01, close - stop_loss)

    max_risk_usd = effective_capital * RISK_PER_TRADE
    risk_based_shares = int(max_risk_usd // risk_per_share)
    cap_based_shares = int((effective_capital * MAX_POSITION_PCT) // max(0.01, close))
    cash_based_shares = int(max(0.0, available_cash) // max(0.01, close))
    suggested_shares = min(risk_based_shares, cap_based_shares, cash_based_shares)

    if suggested_shares <= 0 and score >= 0.75:
        hard_lock_reasons.append(f"Yetersiz Serbest Nakit (${available_cash:.2f})")

    if score >= 2.25:
        decision = "GÜÇLÜ AL" if not hard_lock_reasons else f"BEKLE (KİLİTLİ: {' + '.join(hard_lock_reasons)})"
    elif score >= 0.75:
        decision = "AL" if not hard_lock_reasons else f"BEKLE (KİLİTLİ: {' + '.join(hard_lock_reasons)})"
    else:
        decision = "GİRİŞ UYGUN DEĞİL (ZAYIF / NÖTR TREND)"

    risk_plan = {
        "asset_type": asset_type,
        "bar_date": str(row.get("price_date") or date.today()),
        "max_hold": profile["max_hold_days"],
        "shock_active": shock_active,
        "days_to_earnings": days_to_earnings,
        "sma_50": round(row["sma_50"], 2) if row.get("sma_50") is not None else None,
        "sma_200": round(sma200, 2) if sma200 is not None else None,
        "rsi_14": round(row["rsi_14"], 1) if row.get("rsi_14") is not None else None,
        "atr": round(atr, 2),
        "stop_loss": round(stop_loss, 2),
        "partial_tp": round(partial_tp, 2),
        "take_profit": round(take_profit, 2),
        "shares": suggested_shares,
        "position_usd": round(suggested_shares * close, 2),
        "available_cash": round(available_cash, 2),
    }

    return decision, round(score, 2), reasons, risk_plan


def evaluate_held_position(
    row: dict,
    position: dict,
    asset_type: str,
    sentiment: float,
    min_news_score: float = 0.0,
    days_to_earnings: int | None = None,
    as_of_date: date | None = None,
) -> tuple[str, float, list[str], dict]:
    profile = ASSET_PROFILES.get(asset_type, ASSET_PROFILES["STOCK"])
    close = row["close"]
    sma20 = row.get("sma_20")
    sma200 = row.get("sma_200")
    delta = row.get("delta") or 0.0

    entry_price = position["entry_price"]
    shares = position["shares"]
    entry_date = position.get("entry_date")
    scaled_out = bool(position.get("scaled_out", False))
    ref_date = as_of_date or datetime.now(timezone.utc).date()

    days_held = 0
    if isinstance(entry_date, datetime):
        entry_date = entry_date.date()
    if isinstance(entry_date, date):
        days_held = max(0, (ref_date - entry_date).days)

    pnl_usd = (close - entry_price) * shares
    pnl_pct = ((close - entry_price) / entry_price) * 100.0

    score, reasons, current_atr, vol_ratio = _score_factors(row, asset_type, sentiment)
    base_atr = position.get("initial_atr") or current_atr

    news_shock = min_news_score <= SHOCK_NEWS_THRESHOLD
    vol_price_shock = (vol_ratio >= SHOCK_VOL_RATIO) and (delta <= -2.5 * current_atr)
    shock_active = news_shock or vol_price_shock

    tp1_target = position.get("tp1") or (entry_price + (profile["tp1_atr"] * base_atr))
    tp2_target = position.get("tp2") or (entry_price + (profile["tp2_atr"] * base_atr))

    prev_sl = position.get("stop_loss") or max(0.01, entry_price - (profile["sl_atr"] * base_atr))
    effective_sl_mult = 0.75 if shock_active else profile["sl_atr"]
    candidate_sl = max(0.01, close - (effective_sl_mult * current_atr))
    trailing_sl = round(max(prev_sl, candidate_sl), 2)

    max_hold = profile["max_hold_days"]
    expired_leveraged = (max_hold is not None) and (days_held >= max_hold)

    if close <= trailing_sl:
        decision = f"POZİSYONU KAPAT (TRAILING STOP KIRILDI: ${trailing_sl:.2f})"
    elif expired_leveraged:
        decision = f"POZİSYONU KAPAT (LEVERAGED MAX SÜRE DOLDU: {days_held}/{max_hold} GÜN — VOLATILITY DECAY)"
    elif score <= -1.5 or (sma20 is not None and close < sma20 and score < 0):
        decision = "POZİSYONU KAPAT (SAT — TREND / MOMENTUM BOZULDU)"
    elif close >= tp2_target:
        decision = "KÂR AL VE KAPAT (TP-2 ANA HEDEF GELDİ)"
    elif close >= tp1_target and not scaled_out:
        trailing_sl = round(max(trailing_sl, entry_price * 1.003), 2)
        decision = "KISMİ KÂR AL (%50 SAT + STOP MALİYETE ÇEKİLDİ)"
    else:
        notes = []
        if scaled_out:
            notes.append("TP-1 Kısmi Kâr Alındı (Breakeven Koruma)")
        if max_hold is not None:
            notes.append(f"Kaldıraç Süresi: {days_held}/{max_hold}g")
        if shock_active:
            notes.append("Şok Aktif: Trailing Stop Sıkılaştırıldı")
        if days_to_earnings is not None and days_to_earnings <= EARNINGS_BLACKOUT_DAYS:
            notes.append(f"Bilançoya {days_to_earnings}g kaldı — risk azaltılabilir")
        suffix = f" ({' | '.join(notes)})" if notes else ""
        decision = f"TUT (POZİSYONU KORU){suffix}"

    risk_plan = {
        "asset_type": asset_type,
        "bar_date": str(row.get("price_date") or date.today()),
        "shares": shares,
        "entry_price": round(entry_price, 2),
        "days_held": days_held,
        "max_hold": max_hold,
        "scaled_out": scaled_out,
        "pnl_usd": round(pnl_usd, 2),
        "pnl_pct": round(pnl_pct, 2),
        "shock_active": shock_active,
        "days_to_earnings": days_to_earnings,
        "sma_50": round(row["sma_50"], 2) if row.get("sma_50") is not None else None,
        "sma_200": round(sma200, 2) if sma200 is not None else None,
        "rsi_14": round(row["rsi_14"], 1) if row.get("rsi_14") is not None else None,
        "atr": round(current_atr, 2),
        "stop_loss": trailing_sl,
        "partial_tp": round(tp1_target, 2),
        "take_profit": round(tp2_target, 2),
        "position_usd": round(shares * close, 2),
    }

    return decision, round(score, 2), reasons, risk_plan


def load_daily_realized_pnl() -> float:
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT COALESCE(SUM(realized_pnl), 0) FROM trades WHERE closed_at >= NOW() - INTERVAL '24 hours';")
                val = cur.fetchone()
                return float(val[0]) if val else 0.0
    except Exception as exc:
        log.error(f"Gunluk realized PnL okunamadi: {exc}")
        return 0.0


def evaluate_circuit_breaker(effective_capital: float, daily_pnl: float, base_capital: float = PORTFOLIO_CAPITAL) -> tuple[bool, str]:
    dd_pct = (base_capital - effective_capital) / base_capital if base_capital > 0 else 0.0
    if dd_pct >= MAX_DRAWDOWN_PCT:
        return True, f"Max Drawdown Devre Kesici (-{dd_pct*100:.1f}% >= -{MAX_DRAWDOWN_PCT*100:.0f}%)"
    daily_loss_pct = (-daily_pnl) / base_capital if base_capital > 0 else 0.0
    if daily_loss_pct >= MAX_DAILY_LOSS_PCT:
        return True, f"Gunluk Zarar Devre Kesici (-{daily_loss_pct*100:.1f}% >= -{MAX_DAILY_LOSS_PCT*100:.0f}%)"
    return False, ""


def _update_position_trailing_state(symbol: str, stop_loss: float, tp1: float, tp2: float) -> None:
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE portfolio_positions
                    SET stop_loss = GREATEST(COALESCE(stop_loss, 0), %s),
                        tp1 = COALESCE(tp1, %s),
                        tp2 = COALESCE(tp2, %s),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE symbol = %s;
                    """,
                    (stop_loss, tp1, tp2, symbol),
                )
            conn.commit()
    except Exception as exc:
        log.error(f"{symbol} trailing stop state DB'ye yazılamadı: {exc}")


def persist_signal(symbol: str, decision: str, score: float, close: float, risk: dict, sentiment: float, bar_date=None) -> None:
    sig_dt = bar_date or date.today()
    query = """
        INSERT INTO signal_history (symbol, signal_date, decision, confidence_score, close_price, stop_loss, take_profit, finbert_score)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
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
            cur.execute(query, (symbol, sig_dt, decision[:250], score, close, risk["stop_loss"], risk["take_profit"], sentiment))
        conn.commit()


def run_analysis(persist: bool = True) -> dict:
    active_map = get_active_asset_map()
    all_asset_map = get_all_asset_map()
    positions = load_portfolio_positions()
    all_symbols = sorted(set(active_map.keys()) | set(positions.keys()))

    realized_pnl = load_realized_pnl()
    effective_capital = round(PORTFOLIO_CAPITAL + realized_pnl, 2)
    invested_cost = round(sum(p["shares"] * p["entry_price"] for p in positions.values()), 2)
    available_cash = round(max(0.0, effective_capital - invested_cost), 2)

    prices_df = load_prices_df()
    is_macro_bull, qqq_close, qqq_sma200, qqq_ret_20d = get_benchmark_metrics(prices_df)
    tech_df = compute_technical_and_risk(prices_df, all_symbols, qqq_ret_20d)
    sentiment_map = compute_finbert_sentiment(all_symbols)
    rows_by_sym = {r["symbol"]: r for r in tech_df.iter_rows(named=True)}

    held_results = []
    warnings = []
    for sym in sorted(positions.keys()):
        if sym not in rows_by_sym:
            msg = f"Acik pozisyon {sym} icin fiyat verisi bulunamadi, degerlendirme atlandi."
            log.warning(msg)
            warnings.append(msg)
            continue
        row = rows_by_sym[sym]
        atype = all_asset_map.get(sym, "STOCK")
        sent_score, min_sent, _ = sentiment_map.get(sym, (0.0, 0.0, 0))
        earnings_days = get_days_to_earnings(sym, atype)
        decision, total_score, reasons, risk = evaluate_held_position(
            row, positions[sym], atype, sent_score, min_news_score=min_sent, days_to_earnings=earnings_days
        )
        if persist:
            _update_position_trailing_state(sym, risk["stop_loss"], risk["partial_tp"], risk["take_profit"])
            persist_signal(sym, f"[PORTFÖY] {decision}", total_score, row["close"], risk, sent_score, bar_date=row.get("price_date"))
        held_results.append({
            "symbol": sym,
            "asset_type": atype,
            "decision": decision,
            "score": total_score,
            "close": round(row["close"], 2),
            "sentiment": sent_score,
            "reasons": reasons,
            "risk": risk,
        })

    radar_results = []
    cb_active, cb_reason = evaluate_circuit_breaker(effective_capital, load_daily_realized_pnl())
    open_count = len(positions)
    group_counts: dict[str, int] = {}
    for p_sym in positions:
        g = all_asset_map.get(p_sym, 'STOCK')
        group_counts[g] = group_counts.get(g, 0) + 1
    for sym in [s for s in sorted(active_map.keys()) if s not in positions]:
        if sym not in rows_by_sym:
            msg = f"Izleme listesi sembolu {sym} icin fiyat verisi bulunamadi, analiz atlandi."
            log.warning(msg)
            warnings.append(msg)
            continue
        row = rows_by_sym[sym]
        atype = active_map.get(sym, "STOCK")
        sent_score, min_sent, _ = sentiment_map.get(sym, (0.0, 0.0, 0))
        earnings_days = get_days_to_earnings(sym, atype)
        decision, total_score, reasons, risk = generate_signal(
            row,
            atype,
            sent_score,
            is_macro_bull,
            min_news_score=min_sent,
            days_to_earnings=earnings_days,
            available_cash=available_cash,
            open_positions_count=len(positions),
            effective_capital=effective_capital,
        )
        if decision in ('AL', 'GÜÇLÜ AL'):
            if cb_active:
                decision = f'BEKLE (KİLİTLİ: {cb_reason})'
            elif group_counts.get(atype, 0) >= MAX_SAME_GROUP_POSITIONS:
                decision = f'BEKLE (KİLİTLİ: {atype} Grup Limiti Dolu {group_counts[atype]}/{MAX_SAME_GROUP_POSITIONS})'
            elif open_count >= MAX_OPEN_POSITIONS:
                decision = f'BEKLE (KİLİTLİ: Maksimum pozisyon {open_count}/{MAX_OPEN_POSITIONS})'
            else:
                available_cash = max(0.0, round(available_cash - risk['position_usd'], 2))
                open_count += 1
                group_counts[atype] = group_counts.get(atype, 0) + 1
            if decision.startswith('BEKLE'):
                risk = {**risk, 'shares': 0, 'position_usd': 0.0}
        if persist:
            persist_signal(sym, decision, total_score, row["close"], risk, sent_score, bar_date=row.get("price_date"))
        radar_results.append({
            "symbol": sym,
            "asset_type": atype,
            "decision": decision,
            "score": total_score,
            "close": round(row["close"], 2),
            "sentiment": sent_score,
            "reasons": reasons,
            "risk": risk,
        })

    outcomes_updated = evaluate_signal_outcomes() if persist else 0

    return {
        "macro": {
            "benchmark": BENCHMARK_SYMBOL,
            "is_bull": is_macro_bull,
            "close": qqq_close,
            "sma_200": qqq_sma200,
        },
        "capital": {
            "base_capital": PORTFOLIO_CAPITAL,
            "realized_pnl": realized_pnl,
            "effective_capital": effective_capital,
            "invested_cost": invested_cost,
            "available_cash": available_cash,
            "open_positions": len(positions),
            "max_positions": MAX_OPEN_POSITIONS,
        },
        "outcomes_updated": outcomes_updated,
        "portfolio": held_results,
        "radar": radar_results,
    }


def _fmt_opt(val: float | None, prefix: str = "$") -> str:
    return f"{prefix}{val:.2f}" if val is not None else "N/A"


def main():
    report = run_analysis(persist=True)
    macro = report["macro"]
    cap = report["capital"]
    regime_label = "BOĞA REJİMİ (Hisse Kilidi Açık)" if macro["is_bull"] else "AYI REJİMİ (STOCK Kilidi Aktif | COMMODITY/ETF Bağımsız)"

    print("\n" + "=" * 100)
    print("ALPHA ENGINE v4.9.14 RAPORU [1. AÇIK POZİSYONLARIM  |  2. İZLEME LİSTESİ FIRSAT RADARI]")
    print(f"MAKRO REJİM [{macro['benchmark']}]: ${macro['close']:.2f} vs SMA-200 ${macro['sma_200']:.2f} -> {regime_label}")
    print(
        f"SERMAYE DURUMU: Efektif ${cap['effective_capital']:,.2f} (Realized PnL: ${cap['realized_pnl']:+,.2f}) | "
        f"Bağlı: ${cap['invested_cost']:,.2f} | Serbest Nakit: ${cap['available_cash']:,.2f} | "
        f"Pozisyon: {cap['open_positions']}/{cap['max_positions']}"
    )
    print("=" * 100)

    print("\n>>> BÖLÜM 1: AÇIK POZİSYONLARIM (PORTFÖYDEKİ VARLIKLAR)")
    if not report["portfolio"]:
        print("  Portföyde şu an açık pozisyon yok. (Eklemek için: python src/portfolio_cli.py buy <SEMBOL> <ADET> <MALİYET>)")
    else:
        for item in report["portfolio"]:
            risk = item["risk"]
            shock_note = " [ŞOK: SIKI TRAILING STOP]" if risk["shock_active"] else ""
            print(f"\n[{item['symbol']} | {item['asset_type']}] POZİSYON KARARI: {item['decision']} (Skor: {item['score']:+.2f})")
            print(f"  - Adet: {risk['shares']:.2f} | Maliyet: ${risk['entry_price']:.2f} -> Güncel: ${item['close']:.2f} | Kâr/Zarar: ${risk['pnl_usd']:+.2f} ({risk['pnl_pct']:+.2f}%)")
            print(f"  - Sabit Trailing Stop: ${risk['stop_loss']:.2f}{shock_note} | Sabit TP-1: ${risk['partial_tp']:.2f} | Sabit TP-2: ${risk['take_profit']:.2f}")
            print(f"  - Puan Detayı: {' | '.join(item['reasons'])}")

    print("\n>>> BÖLÜM 2: İZLEME LİSTESİ (YENİ GİRİŞ FIRSAT RADARI)")
    for item in report["radar"]:
        risk = item["risk"]
        hold_note = f" | Max Süre: {risk['max_hold']} gün" if risk["max_hold"] else ""
        earn_note = f" | Bilanço: {risk['days_to_earnings']}g" if risk["days_to_earnings"] is not None else ""
        print(f"\n[{item['symbol']} | {item['asset_type']}] RADAR KARARI: {item['decision']} (Güven Skoru: {item['score']:+.2f}){earn_note}")
        print(f"  - Fiyat: ${item['close']:.2f} | SMA-50: {_fmt_opt(risk['sma_50'])} | SMA-200: {_fmt_opt(risk['sma_200'])} | RSI-14: {_fmt_opt(risk['rsi_14'], '')}")
        if item["decision"].startswith("AL") or item["decision"].startswith("GÜÇLÜ AL"):
            print(f"  - Giriş Planı -> SL: ${risk['stop_loss']:.2f} | TP-1: ${risk['partial_tp']:.2f} | TP-2: ${risk['take_profit']:.2f}{hold_note}")
            print(f"  - Önerilen Lot (Serbest Nakit ${cap['available_cash']:,.2f}): {risk['shares']} adet (~${risk['position_usd']:.2f})")
        print(f"  - Puan Detayı: {' | '.join(item['reasons'])}")

    print(f"\n[DB] Tüm sinyaller bar tarihiyle signal_history tablosuna kaydedildi (Forward-test güncellenen: {report['outcomes_updated']}).")
    print("=" * 100)


if __name__ == "__main__":
    main()
