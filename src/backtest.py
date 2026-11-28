import math
import polars as pl
from indicators import add_technical_indicators
from signal_engine import load_prices_df
from config import (
    get_active_asset_map,
    ASSET_PROFILES,
    BENCHMARK_SYMBOL,
    PORTFOLIO_CAPITAL,
    RISK_PER_TRADE,
    MAX_POSITION_PCT,
    COMMISSION_RATE,
    SLIPPAGE_RATE,
)

WARMUP_BARS = 200


def _simulate_slice(rows: list[dict], asset_type: str) -> dict:
    profile = ASSET_PROFILES.get(asset_type, ASSET_PROFILES["STOCK"])
    cash = PORTFOLIO_CAPITAL
    shares = 0.0
    entry_px = 0.0
    position_cost = 0.0
    partial_realized = 0.0
    stop_loss = 0.0
    tp1 = 0.0
    tp2 = 0.0
    scaled_out = False
    bars_held = 0
    pending_entry = False

    closed_trade_pnls = []
    equity_curve = []

    for i, r in enumerate(rows):
        open_px = r["open"]
        high_px = r["high"]
        low_px = r["low"]
        close = r["close"]
        atr = r["atr_14"] or (close * 0.02)
        sma20 = r["sma_20"]
        sma50 = r["sma_50"]
        sma200 = r["sma_200"]
        rsi = r["rsi_14"]
        vol_ratio = r["vol_ratio"] or 1.0
        rs_pct = r["rs_vs_qqq_pct"] or 0.0
        is_macro_bull = bool(r["qqq_bull"]) if r["qqq_bull"] is not None else True

        # Bir onceki bar kapanisinda uretilen sinyale ertesi gun acilista (next_open + slippage) giris
        if pending_entry and shares == 0:
            exec_entry = open_px * (1.0 + SLIPPAGE_RATE)
            risk_per_share = max(0.01, profile["sl_atr"] * atr)
            qty = min(
                int((cash * RISK_PER_TRADE) // risk_per_share),
                int((cash * MAX_POSITION_PCT) // max(0.01, exec_entry)),
            )
            total_cost = qty * exec_entry * (1.0 + COMMISSION_RATE)
            if qty > 0 and total_cost <= cash:
                cash -= total_cost
                shares = float(qty)
                entry_px = exec_entry
                position_cost = total_cost
                partial_realized = 0.0
                stop_loss = exec_entry - (profile["sl_atr"] * atr)
                tp1 = exec_entry + (profile["tp1_atr"] * atr)
                tp2 = exec_entry + (profile["tp2_atr"] * atr)
                scaled_out = False
                bars_held = 0
            pending_entry = False

        # Teknik skor (canli _score_factors ile birebir hizali)
        score = 0.0
        if sma20 and sma50:
            if close > sma20 > sma50:
                score += 1.5
            elif close < sma20 < sma50:
                score -= 1.5
        if rs_pct >= 2.0:
            score += 0.5
        elif rs_pct <= -2.0:
            score -= 0.5
        if vol_ratio >= 1.2:
            score += 0.5 if score > 0 else -0.5
        if rsi is not None:
            if rsi < 35:
                score += 1.0
            elif rsi > 70:
                score -= 1.0

        if shares > 0:
            bars_held += 1
            # Gercek Trailing Stop: sadece yukari gider
            stop_loss = max(stop_loss, close - (profile["sl_atr"] * atr))
            max_hold = profile["max_hold_days"]

            # Gun ici stop (gap dahil) ve trend bozulmasi cikisi (canli kuralla ayni)
            hit_stop = low_px <= stop_loss
            hit_tp2 = high_px >= tp2
            trend_broken = (sma20 is not None and close < sma20 and score < 0)
            time_expired = (max_hold is not None and bars_held >= max_hold)

            if hit_stop or hit_tp2 or trend_broken or time_expired:
                if hit_stop:
                    raw_exit = min(open_px, stop_loss) * (1.0 - SLIPPAGE_RATE)
                elif hit_tp2:
                    raw_exit = max(open_px, tp2) * (1.0 - SLIPPAGE_RATE)
                else:
                    raw_exit = close * (1.0 - SLIPPAGE_RATE)

                proceeds = shares * raw_exit * (1.0 - COMMISSION_RATE)
                total_trade_pnl = (partial_realized + proceeds) - position_cost
                closed_trade_pnls.append(total_trade_pnl)
                cash += proceeds
                shares = 0.0
                bars_held = 0
                scaled_out = False
            elif high_px >= tp1 and not scaled_out and shares >= 2:
                sell_qty = math.floor(shares / 2.0)
                raw_tp1 = max(open_px, tp1) * (1.0 - SLIPPAGE_RATE)
                proceeds = sell_qty * raw_tp1 * (1.0 - COMMISSION_RATE)
                partial_realized += proceeds
                cash += proceeds
                shares -= sell_qty
                scaled_out = True
                stop_loss = max(stop_loss, entry_px * 1.002)
        else:
            macro_ok = is_macro_bull if profile["require_macro_bull"] else True
            sma200_ok = (close >= sma200) if (profile["require_stock_sma200"] and sma200 is not None) else True
            if score >= 0.75 and macro_ok and sma200_ok and i < len(rows) - 1:
                pending_entry = True

        equity_curve.append(cash + (shares * close))

    final_eq = equity_curve[-1] if equity_curve else PORTFOLIO_CAPITAL
    ret_pct = ((final_eq - PORTFOLIO_CAPITAL) / PORTFOLIO_CAPITAL) * 100.0
    bh_ret = (((rows[-1]["close"] - rows[0]["close"]) / rows[0]["close"]) * 100.0) if len(rows) > 1 else 0.0

    wins = [p for p in closed_trade_pnls if p > 0]
    losses = [abs(p) for p in closed_trade_pnls if p < 0]
    win_rate = (len(wins) / len(closed_trade_pnls) * 100.0) if closed_trade_pnls else 0.0
    profit_factor = (sum(wins) / sum(losses)) if losses else (99.0 if wins else 0.0)
    expectancy = (sum(closed_trade_pnls) / len(closed_trade_pnls)) if closed_trade_pnls else 0.0

    # Gunluk getiri uzerinden yillik Sharpe
    daily_rets = []
    for idx in range(1, len(equity_curve)):
        prev_e = equity_curve[idx - 1]
        if prev_e > 0:
            daily_rets.append((equity_curve[idx] - prev_e) / prev_e)
    if len(daily_rets) > 1:
        mean_r = sum(daily_rets) / len(daily_rets)
        std_r = math.sqrt(sum((x - mean_r) ** 2 for x in daily_rets) / (len(daily_rets) - 1))
        sharpe = (mean_r / std_r) * math.sqrt(252) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    peak = PORTFOLIO_CAPITAL
    max_dd = 0.0
    for eq in equity_curve:
        peak = max(peak, eq)
        dd = ((eq - peak) / peak) * 100.0
        max_dd = min(max_dd, dd)

    return {
        "return_pct": round(ret_pct, 2),
        "buy_hold_pct": round(bh_ret, 2),
        "alpha_vs_bh_pct": round(ret_pct - bh_ret, 2),
        "trades": len(closed_trade_pnls),
        "win_rate_pct": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "expectancy_usd": round(expectancy, 2),
        "sharpe": round(sharpe, 2),
        "max_dd_pct": round(max_dd, 2),
    }


def run_backtest() -> list[dict]:
    df = load_prices_df()
    asset_map = get_active_asset_map()

    qqq = df.filter(pl.col("symbol") == BENCHMARK_SYMBOL).sort("price_date").with_columns([
        pl.col("close").rolling_mean(window_size=200).alias("qqq_sma_200"),
        ((pl.col("close") / pl.col("close").shift(20)) - 1.0).alias("qqq_ret_20d"),
    ]).select([
        "price_date",
        (pl.col("close") >= pl.col("qqq_sma_200")).alias("qqq_bull"),
        "qqq_ret_20d",
    ])

    enriched = add_technical_indicators(df, qqq.select(["price_date", "qqq_ret_20d"])).join(
        qqq.select(["price_date", "qqq_bull"]), on="price_date", how="left"
    )

    results = []
    for sym, atype in sorted(asset_map.items()):
        all_rows = enriched.filter(pl.col("symbol") == sym).sort("price_date").to_dicts()
        if len(all_rows) <= WARMUP_BARS + 30:
            continue
        valid_rows = all_rows[WARMUP_BARS:]
        split_idx = int(len(valid_rows) * 0.7)
        regime_a = _simulate_slice(valid_rows[:split_idx], atype)
        regime_b = _simulate_slice(valid_rows[split_idx:], atype)
        results.append({
            "symbol": sym,
            "asset_type": atype,
            "bars_tested": len(valid_rows),
            "period_1": regime_a,
            "period_2": regime_b,
        })
    return results




def run_portfolio_simulation(symbol_dfs: dict[str, pl.DataFrame], asset_map: dict[str, str], qqq_df: pl.DataFrame) -> dict:
    """Ortak kasali coklu-sembol portfoy simulasyonu (MAX_OPEN_POSITIONS ve nakit paylasimi ile)."""
    prepared = {}
    qqq_prep = _prepare_symbol_df(qqq_df, qqq_df)
    qqq_bull_map = dict(zip(qqq_prep["price_date"].to_list(), (qqq_prep["close_price"] >= qqq_prep["sma_200"]).to_list()))

    all_dates = set()
    for sym, df in symbol_dfs.items():
        if sym == BENCHMARK_SYMBOL or df.height < 220:
            continue
        pdf = _prepare_symbol_df(df, qqq_df)
        rows = pdf.to_dicts()
        prepared[sym] = {r["price_date"]: r for r in rows}
        all_dates.update(r["price_date"] for r in rows[200:])

    sorted_dates = sorted(all_dates)
    cash = PORTFOLIO_CAPITAL
    positions = {}
    pending_entries = {}
    closed_pnls = []
    equity_curve = [PORTFOLIO_CAPITAL]

    for dt in sorted_dates:
        macro_bull = bool(qqq_bull_map.get(dt, False))

        # 1. Bekleyen ertesi gun acilis (next-open) girislerini uygula
        for sym, entry_meta in list(pending_entries.items()):
            row = prepared.get(sym, {}).get(dt)
            if row and sym not in positions and len(positions) < MAX_OPEN_POSITIONS:
                exec_px = float(row["open_price"]) * (1.0 + SLIPPAGE_PCT)
                atr = entry_meta["atr"]
                profile = entry_meta["profile"]
                sl = exec_px - (profile["sl_atr"] * atr)
                tp1 = exec_px + (profile["tp1_atr"] * atr)
                tp2 = exec_px + (profile["tp2_atr"] * atr)
                risk_per_share = max(exec_px - sl, 0.01)
                shares = min((cash * RISK_PER_TRADE) / risk_per_share, (cash * MAX_POSITION_PCT) / exec_px)
                cost = shares * exec_px * (1.0 + COMMISSION_RATE)
                if shares > 0 and cost <= cash:
                    cash -= cost
                    positions[sym] = {
                        "shares": shares,
                        "entry_px": exec_px,
                        "entry_date": dt,
                        "sl": sl,
                        "tp1": tp1,
                        "tp2": tp2,
                        "scaled_out": False,
                        "profile": profile,
                    }
            pending_entries.pop(sym, None)

        # 2. Acik pozisyonlari yonet
        for sym in list(positions.keys()):
            row = prepared.get(sym, {}).get(dt)
            if not row:
                continue
            pos = positions[sym]
            open_px = float(row["open_price"])
            high_px = float(row["high_price"])
            low_px = float(row["low_price"])
            close_px = float(row["close_price"])
            sma_20 = row["sma_20"]
            score, _ = _compute_tech_score(row)
            days_held = (dt - pos["entry_date"]).days
            max_hold = pos["profile"]["max_hold_days"]

            exit_px = None
            if open_px <= pos["sl"]:
                exit_px = open_px * (1.0 - SLIPPAGE_PCT)
            elif low_px <= pos["sl"]:
                exit_px = pos["sl"] * (1.0 - SLIPPAGE_PCT)
            elif high_px >= pos["tp2"]:
                exit_px = pos["tp2"] * (1.0 - SLIPPAGE_PCT)
            elif max_hold is not None and days_held >= max_hold:
                exit_px = close_px * (1.0 - SLIPPAGE_PCT)
            elif sma_20 is not None and close_px < float(sma_20) and score < 0:
                exit_px = close_px * (1.0 - SLIPPAGE_PCT)

            if exit_px is not None:
                proceeds = pos["shares"] * exit_px * (1.0 - COMMISSION_RATE)
                pnl = proceeds - (pos["shares"] * pos["entry_px"])
                cash += proceeds
                closed_pnls.append(pnl)
                del positions[sym]

        # 3. Yeni giris sinyallerini topla
        if len(positions) + len(pending_entries) < MAX_OPEN_POSITIONS:
            for sym, date_map in prepared.items():
                if sym in positions or sym in pending_entries:
                    continue
                row = date_map.get(dt)
                if not row or row["sma_200"] is None or row["atr_14"] is None:
                    continue
                atype = asset_map.get(sym, "STOCK")
                profile = ASSET_PROFILES.get(atype, ASSET_PROFILES["STOCK"])
                if profile["require_macro_bull"] and not macro_bull:
                    continue
                if profile["require_stock_sma200"] and float(row["close_price"]) < float(row["sma_200"]):
                    continue
                score, _ = _compute_tech_score(row)
                if score >= 2.0:
                    pending_entries[sym] = {"atr": float(row["atr_14"]), "profile": profile}
                    if len(positions) + len(pending_entries) >= MAX_OPEN_POSITIONS:
                        break

        mtm_val = cash + sum(
            pos["shares"] * float(prepared[sym][dt]["close_price"])
            for sym, pos in positions.items()
            if dt in prepared.get(sym, {})
        )
        equity_curve.append(mtm_val)

    total_ret = ((equity_curve[-1] - PORTFOLIO_CAPITAL) / PORTFOLIO_CAPITAL) * 100.0
    wins = [p for p in closed_pnls if p > 0]
    losses = [abs(p) for p in closed_pnls if p <= 0]
    pf = (sum(wins) / sum(losses)) if losses else (99.0 if wins else 0.0)
    win_rate = (len(wins) / len(closed_pnls) * 100.0) if closed_pnls else 0.0
    return {
        "final_equity": round(equity_curve[-1], 2),
        "total_return_pct": round(total_ret, 2),
        "total_trades": len(closed_pnls),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(pf, 2),
    }


def main():
    results = run_backtest()
    print("\n" + "=" * 112)
    print("İKİ DÖNEMLİ REJİM DOĞRULAMA RAPORU (NEXT-OPEN + SLIPPAGE + CANLI ÇIKIŞ KURALLARI | 200-BAR WARM-UP)")
    print("=" * 112)
    for r in results:
        p1 = r["period_1"]
        p2 = r["period_2"]
        print(
            f"[{r['symbol']:<6} | {r['asset_type']:<10}] Bar: {r['bars_tested']} | "
            f"Dönem-1: {p1['return_pct']:+6.2f}% (PF: {p1['profit_factor']:.2f}) | "
            f"Dönem-2: {p2['return_pct']:+6.2f}% (B&H: {p2['buy_hold_pct']:+6.2f}%) | "
            f"İşlem: {p2['trades']} | Win: %{p2['win_rate_pct']:.1f} | PF: {p2['profit_factor']:.2f} | "
            f"Sharpe: {p2['sharpe']:.2f} | Exp: ${p2['expectancy_usd']:+.2f} | MaxDD: {p2['max_dd_pct']:.2f}%"
        )
    print("=" * 112)
    pf_sim = run_portfolio_simulation(symbol_dfs, asset_map, qqq_df)
    print(f"ORTAK KASALI PORTFÖY SİMÜLASYONU (Max {MAX_OPEN_POSITIONS} Pozisyon): "
          f"Bitiş: ${pf_sim['final_equity']:,.2f} | Getiri: {pf_sim['total_return_pct']:+.2f}% | "
          f"İşlem: {pf_sim['total_trades']} | Win: %{pf_sim['win_rate']:.1f} | PF: {pf_sim['profit_factor']:.2f}")
    print("=" * 112)


if __name__ == "__main__":
    main()
