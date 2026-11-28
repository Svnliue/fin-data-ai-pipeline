import math
import polars as pl
from indicators import add_technical_indicators, compute_tech_score
from signal_engine import load_prices_df
from config import (
    get_active_asset_map,
    ASSET_PROFILES,
    BENCHMARK_SYMBOL,
    PORTFOLIO_CAPITAL,
    RISK_PER_TRADE,
    MAX_POSITION_PCT,
    MAX_OPEN_POSITIONS,
    MAX_SAME_GROUP_POSITIONS,
    COMMISSION_RATE,
    SLIPPAGE_RATE,
)


def _simulate_slice(rows: list[dict], asset_type: str) -> dict:
    """Tek varlik icin iki donemli rejim dogrulama simulasyonu (next-open + slippage + trailing SL + TP1/TP2)."""
    profile = ASSET_PROFILES.get(asset_type, ASSET_PROFILES["STOCK"])
    cash = PORTFOLIO_CAPITAL
    pos = None
    pending_entry = None
    closed_pnls = []
    equity_curve = [PORTFOLIO_CAPITAL]

    for r in rows:
        dt = r["price_date"]
        open_px = float(r["open"])
        high_px = float(r["high"])
        low_px = float(r["low"])
        close_px = float(r["close"])
        sma_20 = r["sma_20"]
        sma_200 = r["sma_200"]
        atr = r["atr_14"]
        macro_bull = bool(r.get("macro_bull", False))

        if sma_200 is None or atr is None or sma_20 is None:
            continue

        # 1. Bekleyen ertesi gun acilis girisini uygula
        if pending_entry is not None and pos is None:
            exec_px = open_px * (1.0 + SLIPPAGE_RATE)
            e_atr = pending_entry["atr"]
            sl = exec_px - (profile["sl_atr"] * e_atr)
            tp1 = exec_px + (profile["tp1_atr"] * e_atr)
            tp2 = exec_px + (profile["tp2_atr"] * e_atr)
            risk_per_share = max(exec_px - sl, 0.01)
            shares = min((cash * RISK_PER_TRADE) / risk_per_share, (cash * MAX_POSITION_PCT) / exec_px)
            cost = shares * exec_px * (1.0 + COMMISSION_RATE)
            if shares > 0 and cost <= cash:
                cash -= cost
                pos = {
                    "shares": shares,
                    "entry_px": exec_px,
                    "entry_date": dt,
                    "initial_atr": e_atr,
                    "sl": sl,
                    "tp1": tp1,
                    "tp2": tp2,
                    "scaled_out": False,
                    "realized_partial": 0.0,
                }
            pending_entry = None

        score, _ = compute_tech_score(r)

        # 2. Acik pozisyonu yonet
        if pos is not None:
            days_held = (dt - pos["entry_date"]).days
            max_hold = profile["max_hold_days"]

            # Trailing stop guncelle (asla dusmez)
            trail_cand = close_px - (profile["sl_atr"] * pos["initial_atr"])
            if trail_cand > pos["sl"]:
                pos["sl"] = trail_cand

            exit_px = None
            if open_px <= pos["sl"]:
                exit_px = open_px * (1.0 - SLIPPAGE_RATE)
            elif low_px <= pos["sl"]:
                exit_px = pos["sl"] * (1.0 - SLIPPAGE_RATE)
            elif high_px >= pos["tp2"]:
                exit_px = pos["tp2"] * (1.0 - SLIPPAGE_RATE)
            elif not pos["scaled_out"] and high_px >= pos["tp1"]:
                sell_shares = pos["shares"] * 0.5
                exec_tp1 = pos["tp1"] * (1.0 - SLIPPAGE_RATE)
                proceeds = sell_shares * exec_tp1 * (1.0 - COMMISSION_RATE)
                cash += proceeds
                pos["realized_partial"] += proceeds - (sell_shares * pos["entry_px"])
                pos["shares"] -= sell_shares
                pos["scaled_out"] = True
                pos["sl"] = max(pos["sl"], pos["entry_px"])
            elif max_hold is not None and days_held >= max_hold:
                exit_px = close_px * (1.0 - SLIPPAGE_RATE)
            elif close_px < float(sma_20) and score < 0:
                exit_px = close_px * (1.0 - SLIPPAGE_RATE)

            if exit_px is not None:
                proceeds = pos["shares"] * exit_px * (1.0 - COMMISSION_RATE)
                total_trade_pnl = pos["realized_partial"] + (proceeds - (pos["shares"] * pos["entry_px"]))
                cash += proceeds
                closed_pnls.append(total_trade_pnl)
                pos = None

        # 3. Yeni giris sinyali kontrolu
        if pos is None and pending_entry is None:
            if (not profile["require_macro_bull"] or macro_bull) and (
                not profile["require_stock_sma200"] or close_px >= float(sma_200)
            ):
                if score >= 2.0:
                    pending_entry = {"atr": float(atr)}

        mtm = cash + (pos["shares"] * close_px if pos is not None else 0.0)
        equity_curve.append(mtm)

    if pos is not None:
        last_close = float(rows[-1]["close"]) * (1.0 - SLIPPAGE_RATE)
        proceeds = pos["shares"] * last_close * (1.0 - COMMISSION_RATE)
        total_trade_pnl = pos["realized_partial"] + (proceeds - (pos["shares"] * pos["entry_px"]))
        cash += proceeds
        closed_pnls.append(total_trade_pnl)
        equity_curve[-1] = cash

    total_ret = ((cash - PORTFOLIO_CAPITAL) / PORTFOLIO_CAPITAL) * 100.0
    bh_ret = ((float(rows[-1]["close"]) - float(rows[0]["close"])) / float(rows[0]["close"])) * 100.0
    wins = [p for p in closed_pnls if p > 0]
    losses = [abs(p) for p in closed_pnls if p <= 0]
    pf = (sum(wins) / sum(losses)) if losses else (99.0 if wins else 0.0)
    win_rate = (len(wins) / len(closed_pnls) * 100.0) if closed_pnls else 0.0
    expectancy = (sum(closed_pnls) / len(closed_pnls)) if closed_pnls else 0.0

    rets = [
        (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
        for i in range(1, len(equity_curve))
        if equity_curve[i - 1] > 0
    ]
    if len(rets) > 1:
        mean_r = sum(rets) / len(rets)
        std_r = math.sqrt(sum((x - mean_r) ** 2 for x in rets) / (len(rets) - 1))
        sharpe = (mean_r / std_r) * math.sqrt(252) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = ((eq - peak) / peak) * 100.0 if peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd

    return {
        "return_pct": round(total_ret, 2),
        "bh_pct": round(bh_ret, 2),
        "trades": len(closed_pnls),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(pf, 2),
        "sharpe": round(sharpe, 2),
        "expectancy_usd": round(expectancy, 2),
        "max_dd_pct": round(max_dd, 2),
    }


def _prepare_all_symbols(raw_df: pl.DataFrame, asset_map: dict[str, str]) -> tuple[dict[str, list[dict]], dict]:
    qqq_df = raw_df.filter(pl.col("symbol") == BENCHMARK_SYMBOL).sort("price_date")
    qqq_ind = add_technical_indicators(qqq_df).with_columns(
        (pl.col("close") >= pl.col("sma_200")).fill_null(False).alias("macro_bull")
    )
    qqq_ret_df = qqq_ind.select(["price_date", pl.col("ret_20d").alias("qqq_ret_20d")])
    qqq_bull_map = dict(zip(qqq_ind["price_date"].to_list(), qqq_ind["macro_bull"].to_list()))

    prepared = {}
    for sym in asset_map.keys():
        sub = raw_df.filter(pl.col("symbol") == sym).sort("price_date")
        if sub.height < 220:
            continue
        ind_df = add_technical_indicators(sub, qqq_ret_df)
        rows = ind_df.to_dicts()
        for r in rows:
            r["macro_bull"] = bool(qqq_bull_map.get(r["price_date"], False))
        prepared[sym] = rows[200:]
    return prepared, qqq_bull_map


def run_portfolio_simulation(prepared: dict[str, list[dict]], asset_map: dict[str, str]) -> dict:
    """Ortak kasali coklu-sembol portfoy simulasyonu (Trailing Stop + TP1 kismi satis + Grup Limiti)."""
    date_lookup = {sym: {r["price_date"]: r for r in rows} for sym, rows in prepared.items()}
    all_dates = sorted({r["price_date"] for rows in prepared.values() for r in rows})

    cash = PORTFOLIO_CAPITAL
    positions = {}
    pending_entries = {}
    closed_pnls = []
    equity_curve = [PORTFOLIO_CAPITAL]

    for dt in all_dates:
        # 1. Ertesi gun acilis (next-open) girislerini uygula
        for sym, meta in list(pending_entries.items()):
            row = date_lookup.get(sym, {}).get(dt)
            atype = asset_map.get(sym, "STOCK")
            same_group = sum(1 for p_sym in positions if asset_map.get(p_sym, "STOCK") == atype)
            if (
                row
                and sym not in positions
                and len(positions) < MAX_OPEN_POSITIONS
                and same_group < MAX_SAME_GROUP_POSITIONS
            ):
                exec_px = float(row["open"]) * (1.0 + SLIPPAGE_RATE)
                profile = meta["profile"]
                atr = meta["atr"]
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
                        "initial_atr": atr,
                        "sl": sl,
                        "tp1": tp1,
                        "tp2": tp2,
                        "scaled_out": False,
                        "realized_partial": 0.0,
                        "profile": profile,
                    }
            pending_entries.pop(sym, None)

        # 2. Acik pozisyonlari yonet (Trailing SL + TP1 + TP2 + Trend Bozulmasi)
        for sym in list(positions.keys()):
            row = date_lookup.get(sym, {}).get(dt)
            if not row:
                continue
            pos = positions[sym]
            profile = pos["profile"]
            open_px = float(row["open"])
            high_px = float(row["high"])
            low_px = float(row["low"])
            close_px = float(row["close"])
            sma_20 = row["sma_20"]
            score, _ = compute_tech_score(row)
            days_held = (dt - pos["entry_date"]).days
            max_hold = profile["max_hold_days"]

            trail_cand = close_px - (profile["sl_atr"] * pos["initial_atr"])
            if trail_cand > pos["sl"]:
                pos["sl"] = trail_cand

            exit_px = None
            if open_px <= pos["sl"]:
                exit_px = open_px * (1.0 - SLIPPAGE_RATE)
            elif low_px <= pos["sl"]:
                exit_px = pos["sl"] * (1.0 - SLIPPAGE_RATE)
            elif high_px >= pos["tp2"]:
                exit_px = pos["tp2"] * (1.0 - SLIPPAGE_RATE)
            elif not pos["scaled_out"] and high_px >= pos["tp1"]:
                sell_shares = pos["shares"] * 0.5
                exec_tp1 = pos["tp1"] * (1.0 - SLIPPAGE_RATE)
                proceeds = sell_shares * exec_tp1 * (1.0 - COMMISSION_RATE)
                cash += proceeds
                pos["realized_partial"] += proceeds - (sell_shares * pos["entry_px"])
                pos["shares"] -= sell_shares
                pos["scaled_out"] = True
                pos["sl"] = max(pos["sl"], pos["entry_px"])
            elif max_hold is not None and days_held >= max_hold:
                exit_px = close_px * (1.0 - SLIPPAGE_RATE)
            elif sma_20 is not None and close_px < float(sma_20) and score < 0:
                exit_px = close_px * (1.0 - SLIPPAGE_RATE)

            if exit_px is not None:
                proceeds = pos["shares"] * exit_px * (1.0 - COMMISSION_RATE)
                total_pnl = pos["realized_partial"] + (proceeds - (pos["shares"] * pos["entry_px"]))
                cash += proceeds
                closed_pnls.append(total_pnl)
                del positions[sym]

        # 3. Yeni giris sinyallerini topla
        if len(positions) + len(pending_entries) < MAX_OPEN_POSITIONS:
            for sym, dmap in date_lookup.items():
                if sym in positions or sym in pending_entries:
                    continue
                row = dmap.get(dt)
                if not row or row["sma_200"] is None or row["atr_14"] is None:
                    continue
                atype = asset_map.get(sym, "STOCK")
                profile = ASSET_PROFILES.get(atype, ASSET_PROFILES["STOCK"])
                macro_bull = bool(row.get("macro_bull", False))
                if profile["require_macro_bull"] and not macro_bull:
                    continue
                if profile["require_stock_sma200"] and float(row["close"]) < float(row["sma_200"]):
                    continue
                score, _ = compute_tech_score(row)
                if score >= 2.0:
                    pending_entries[sym] = {"atr": float(row["atr_14"]), "profile": profile}
                    if len(positions) + len(pending_entries) >= MAX_OPEN_POSITIONS:
                        break

        mtm = cash + sum(
            pos["shares"] * float(date_lookup[sym][dt]["close"])
            for sym, pos in positions.items()
            if dt in date_lookup.get(sym, {})
        )
        equity_curve.append(mtm)

    final_eq = equity_curve[-1]
    total_ret = ((final_eq - PORTFOLIO_CAPITAL) / PORTFOLIO_CAPITAL) * 100.0
    wins = [p for p in closed_pnls if p > 0]
    losses = [abs(p) for p in closed_pnls if p <= 0]
    pf = (sum(wins) / sum(losses)) if losses else (99.0 if wins else 0.0)
    win_rate = (len(wins) / len(closed_pnls) * 100.0) if closed_pnls else 0.0

    peak = equity_curve[0]
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = ((eq - peak) / peak) * 100.0 if peak > 0 else 0.0
        if dd < max_dd:
            max_dd = dd

    return {
        "final_equity": round(final_eq, 2),
        "total_return_pct": round(total_ret, 2),
        "total_trades": len(closed_pnls),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(pf, 2),
        "max_dd_pct": round(max_dd, 2),
    }


def run_backtest() -> tuple[list[dict], dict]:
    asset_map = get_active_asset_map()
    symbols = list(asset_map.keys()) + [BENCHMARK_SYMBOL]
    raw_df = load_prices_df(symbols)
    prepared, _ = _prepare_all_symbols(raw_df, asset_map)

    results = []
    for sym, valid_rows in prepared.items():
        atype = asset_map.get(sym, "STOCK")
        split_idx = int(len(valid_rows) * 0.70)
        if split_idx < 30 or (len(valid_rows) - split_idx) < 20:
            continue
        regime_a = _simulate_slice(valid_rows[:split_idx], atype)
        regime_b = _simulate_slice(valid_rows[split_idx:], atype)
        results.append({
            "symbol": sym,
            "asset_type": atype,
            "bars": len(valid_rows),
            "regime_a": regime_a,
            "regime_b": regime_b,
        })

    portfolio_summary = run_portfolio_simulation(prepared, asset_map)
    return results, portfolio_summary


def main():
    results, pf_sim = run_backtest()
    print("\n" + "=" * 112)
    print("İKİ DÖNEMLİ REJİM DOĞRULAMA RAPORU (NEXT-OPEN + SLIPPAGE + CANLI ÇIKIŞ KURALLARI | 200-BAR WARM-UP)")
    print("=" * 112)
    for r in results:
        p1 = r["regime_a"]
        p2 = r["regime_b"]
        print(
            f"[{r['symbol']:<6} | {r['asset_type']:<10}] Bar: {r['bars']:<4} | "
            f"Dönem-1: {p1['return_pct']:+6.2f}% (PF: {p1['profit_factor']:.2f}) | "
            f"Dönem-2: {p2['return_pct']:+6.2f}% (B&H: {p2['bh_pct']:+6.2f}%) | "
            f"İşlem: {p2['trades']:<2} | Win: %{p2['win_rate']:4.1f} | PF: {p2['profit_factor']:.2f} | "
            f"Sharpe: {p2['sharpe']:5.2f} | Exp: ${p2['expectancy_usd']:+6.2f} | MaxDD: {p2['max_dd_pct']:6.2f}%"
        )
    print("=" * 112)
    print(
        f"ORTAK KASALI PORTFÖY SİMÜLASYONU (Max {MAX_OPEN_POSITIONS} Pozisyon | Trailing SL + TP1): "
        f"Bitiş: ${pf_sim['final_equity']:,.2f} | Getiri: {pf_sim['total_return_pct']:+.2f}% | "
        f"İşlem: {pf_sim['total_trades']} | Win: %{pf_sim['win_rate']:.1f} | "
        f"PF: {pf_sim['profit_factor']:.2f} | MaxDD: {pf_sim['max_dd_pct']:.2f}%"
    )
    print("=" * 112)


if __name__ == "__main__":
    main()
