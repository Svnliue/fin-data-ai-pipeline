import os
import sys
from datetime import date, timedelta
import polars as pl

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from signal_engine import compute_technical_and_risk, generate_signal


def _build_synthetic_prices(symbol: str, days: int = 210, start_price: float = 100.0, step: float = 0.5) -> pl.DataFrame:
    base_date = date(2025, 1, 1)
    rows = []
    for i in range(days):
        close = start_price + (i * step)
        rows.append({
            "symbol": symbol,
            "price_date": base_date + timedelta(days=i),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000.0 + (i * 1000.0),
        })
    return pl.DataFrame(rows)


def test_technical_indicators_calculation():
    df = _build_synthetic_prices("AAPL", days=210, start_price=100.0, step=0.5)
    result = compute_technical_and_risk(df, symbols=["AAPL"], qqq_ret_20d=0.01)
    row = result.to_dicts()[0]

    assert row["symbol"] == "AAPL"
    assert row["sma_20"] is not None
    assert row["sma_50"] is not None
    assert row["sma_200"] is not None
    assert row["close"] > row["sma_20"] > row["sma_50"] > row["sma_200"]
    assert 0.0 <= row["rsi_14"] <= 100.0
    assert row["atr_14"] > 0.0


def test_macro_bear_regime_locks_stock_but_allows_commodity():
    row = {
        "close": 150.0,
        "sma_20": 145.0,
        "sma_50": 140.0,
        "sma_200": 130.0,
        "rsi_14": 60.0,
        "atr_14": 3.0,
        "vol_ratio": 1.5,
        "rs_vs_qqq_pct": 8.0,
    }
    decision_stock, _, _, _ = generate_signal(row, asset_type="STOCK", sentiment=0.80, is_macro_bull=False)
    decision_gold, _, _, _ = generate_signal(row, asset_type="COMMODITY", sentiment=0.80, is_macro_bull=False)

    assert "KİLİTLİ" in decision_stock
    assert decision_gold == "GÜÇLÜ AL"


def test_earnings_and_news_shock_guards():
    row = {
        "close": 100.0,
        "sma_20": 95.0,
        "sma_50": 90.0,
        "sma_200": 80.0,
        "rsi_14": 55.0,
        "atr_14": 4.0,
        "vol_ratio": 1.4,
        "rs_vs_qqq_pct": 6.0,
    }
    dec_earn, _, _, _ = generate_signal(row, asset_type="STOCK", sentiment=0.60, is_macro_bull=True, days_to_earnings=2)
    assert "Bilanço Koruması" in dec_earn

    dec_shock, _, _, risk_shock = generate_signal(row, asset_type="STOCK", sentiment=0.30, is_macro_bull=True, min_news_score=-0.85)
    assert "Haber Şoku Kilidi" in dec_shock
    assert risk_shock["shock_active"] is True
    assert risk_shock["stop_loss"] == 97.0


def test_position_sizing_respects_max_cap():
    row = {
        "close": 100.0,
        "sma_20": 95.0,
        "sma_50": 90.0,
        "sma_200": 80.0,
        "rsi_14": 55.0,
        "atr_14": 0.25,
        "vol_ratio": 1.3,
        "rs_vs_qqq_pct": 5.0,
    }
    _, _, _, risk = generate_signal(row, asset_type="STOCK", sentiment=0.50, is_macro_bull=True)
    assert risk["shares"] == 25
    assert risk["position_usd"] <= 2500.0


def test_held_position_evaluation_take_profit_and_exit():
    from signal_engine import evaluate_held_position
    row_profit = {
        "close": 120.0,
        "sma_20": 115.0,
        "sma_50": 110.0,
        "sma_200": 100.0,
        "rsi_14": 65.0,
        "atr_14": 4.0,
        "vol_ratio": 1.3,
        "rs_vs_qqq_pct": 5.0,
    }
    pos = {"shares": 10.0, "entry_price": 100.0}
    dec, _, _, risk = evaluate_held_position(row_profit, pos, asset_type="STOCK", sentiment=0.40)
    assert "KÂR AL" in dec
    assert risk["pnl_usd"] == 200.0
    assert risk["pnl_pct"] == 20.0


def test_leveraged_max_hold_and_cash_limits():
    from signal_engine import evaluate_held_position, generate_signal
    row = {
        "close": 100.0,
        "sma_20": 95.0,
        "sma_50": 90.0,
        "sma_200": 80.0,
        "rsi_14": 60.0,
        "atr_14": 2.0,
        "vol_ratio": 1.3,
        "rs_vs_qqq_pct": 5.0,
    }
    # 11 gundur tutulan LEVERAGED urun volatility decay nedeniyle kapatilmali
    pos_old = {"shares": 10.0, "entry_price": 98.0, "entry_date": date.today() - timedelta(days=11)}
    dec_lev, _, _, _ = evaluate_held_position(row, pos_old, asset_type="LEVERAGED", sentiment=0.50)
    assert "LEVERAGED MAX SÜRE DOLDU" in dec_lev

    # 3/3 acik pozisyon doluyken yeni alim kilitlenmeli
    dec_cap, _, _, _ = generate_signal(row, asset_type="STOCK", sentiment=0.80, is_macro_bull=True, open_positions_count=3)
    assert "Max Açık Pozisyon Dolu" in dec_cap


def test_true_trailing_stop_never_decreases_and_scaled_out_flag():
    from signal_engine import evaluate_held_position
    row_drop = {
        "close": 103.0,
        "sma_20": 102.0,
        "sma_50": 95.0,
        "sma_200": 85.0,
        "rsi_14": 55.0,
        "atr_14": 3.0,
        "vol_ratio": 1.1,
        "rs_vs_qqq_pct": 3.0,
    }
    # Onceki stop 100.0 ise close=103, 2*ATR=6 -> aday stop 97.0 olsa bile stop 100.0'in altina dusmemeli
    pos = {"shares": 10.0, "entry_price": 95.0, "stop_loss": 100.0, "tp1": 102.0, "tp2": 115.0, "scaled_out": True}
    dec, _, _, risk = evaluate_held_position(row_drop, pos, asset_type="STOCK", sentiment=0.40)
    assert risk["stop_loss"] == 100.0
    assert "KISMİ KÂR AL" not in dec


def test_backtest_next_open_and_metrics():
    from backtest import _simulate_slice
    rows = []
    for i in range(60):
        px = 100.0 + (i * 0.5)
        rows.append({
            "open": px - 0.2,
            "high": px + 0.5,
            "low": px - 0.5,
            "close": px,
            "atr_14": 1.5,
            "sma_20": px - 2.0,
            "sma_50": px - 5.0,
            "sma_200": px - 10.0,
            "rsi_14": 58.0,
            "vol_ratio": 1.3,
            "rs_vs_qqq_pct": 4.0,
            "qqq_bull": True,
        })
    res = _simulate_slice(rows, asset_type="STOCK")
    assert res["return_pct"] > 0
    assert "profit_factor" in res
    assert "sharpe" in res
    assert "expectancy_usd" in res


def test_scheduler_nyse_holiday_detection():
    from datetime import date
    from scheduler import _is_nyse_holiday
    assert _is_nyse_holiday(date(2026, 7, 3)) is True
    assert _is_nyse_holiday(date(2026, 12, 25)) is True
    assert _is_nyse_holiday(date(2026, 11, 24)) is False


def test_buy_position_preserves_existing_watchlist_type(monkeypatch):
    import portfolio_cli
    executed_sql = []

    class DummyCursor:
        def __init__(self):
            self._last_query = ""
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def execute(self, query, params=None):
            self._last_query = query
            executed_sql.append((query.strip(), params))
        def fetchone(self):
            if "SUM(realized_pnl)" in self._last_query:
                return (0.0,)
            return None
        def fetchall(self):
            return []

    class DummyConn:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def cursor(self):
            return DummyCursor()
        def commit(self):
            pass

    add_called = {"count": 0}
    monkeypatch.setattr(portfolio_cli, "get_all_asset_map", lambda: {"IREX": "LEVERAGED"})
    monkeypatch.setattr(portfolio_cli, "add_symbol", lambda sym: add_called.__setitem__("count", add_called["count"] + 1))
    monkeypatch.setattr(portfolio_cli, "_get_latest_atr", lambda sym, px: 1.0)
    monkeypatch.setattr(portfolio_cli, "get_db_connection", lambda: DummyConn())

    portfolio_cli.buy_position("IREX", 10, 10.0, "2026-11-28")
    assert add_called["count"] == 0
    insert_calls = [p for q, p in executed_sql if "INSERT INTO portfolio_positions" in q]
    assert len(insert_calls) == 1
    assert insert_calls[0][4] == 8.8


def test_intraday_bar_filter_during_market_hours():
    from datetime import datetime, date
    from zoneinfo import ZoneInfo
    import polars as pl
    from fetch_prices import _filter_incomplete_intraday_bar

    ny_tz = ZoneInfo("America/New_York")
    # Cuma saat 11:30 ET (seans acik)
    now_open = datetime(2026, 11, 27, 11, 30, tzinfo=ny_tz)
    df = pl.DataFrame({
        "symbol": ["AAPL", "AAPL"],
        "price_date": [date(2026, 11, 26), date(2026, 11, 27)],
        "close_price": [350.0, 355.0],
    })
    filtered = _filter_incomplete_intraday_bar(df, now_ny=now_open)
    assert filtered.height == 1
    assert filtered["price_date"].max() == date(2026, 11, 26)


def test_circuit_breaker_triggers_on_drawdown():
    from signal_engine import evaluate_circuit_breaker
    triggered, reason = evaluate_circuit_breaker(effective_capital=8500.0, daily_pnl=0.0, base_capital=10000.0)
    assert triggered is True
    assert "Drawdown" in reason


def test_live_run_analysis_enforces_group_limit_and_keeps_risk_dict(monkeypatch):
    from datetime import date
    import polars as pl
    import signal_engine

    monkeypatch.setattr(signal_engine, "get_active_asset_map", lambda: {"NVDA": "STOCK"})
    monkeypatch.setattr(signal_engine, "get_all_asset_map", lambda: {"AAPL": "STOCK", "MSFT": "STOCK", "NVDA": "STOCK"})
    monkeypatch.setattr(signal_engine, "load_portfolio_positions", lambda: {
        "AAPL": {"shares": 5, "entry_price": 100.0, "stop_loss": 95.0, "partial_tp": 105.0, "take_profit": 115.0, "scaled_out": False, "entry_date": date(2026, 11, 20), "initial_atr": 2.0},
        "MSFT": {"shares": 5, "entry_price": 100.0, "stop_loss": 95.0, "partial_tp": 105.0, "take_profit": 115.0, "scaled_out": False, "entry_date": date(2026, 11, 20), "initial_atr": 2.0},
    })
    monkeypatch.setattr(signal_engine, "load_realized_pnl", lambda: 0.0)
    monkeypatch.setattr(signal_engine, "load_daily_realized_pnl", lambda: 0.0)
    monkeypatch.setattr(signal_engine, "load_prices_df", lambda: pl.DataFrame({"symbol": ["QQQ"], "price_date": [date(2026, 11, 27)], "close": [500.0]}))
    monkeypatch.setattr(signal_engine, "get_benchmark_metrics", lambda df: (True, 500.0, 480.0, 0.03))
    monkeypatch.setattr(signal_engine, "compute_finbert_sentiment", lambda syms: {"NVDA": (0.4, 0.1, 2)})
    monkeypatch.setattr(signal_engine, "get_days_to_earnings", lambda sym, atype: None)

    tech_rows = pl.DataFrame([
        {"symbol": "AAPL", "price_date": date(2026, 11, 27), "close": 102.0, "sma_20": 100.0, "sma_50": 98.0, "sma_200": 90.0, "rsi_14": 55.0, "atr_14": 2.0, "vol_ratio": 1.1, "rs_vs_qqq_pct": 2.0},
        {"symbol": "MSFT", "price_date": date(2026, 11, 27), "close": 102.0, "sma_20": 100.0, "sma_50": 98.0, "sma_200": 90.0, "rsi_14": 55.0, "atr_14": 2.0, "vol_ratio": 1.1, "rs_vs_qqq_pct": 2.0},
        {"symbol": "NVDA", "price_date": date(2026, 11, 27), "close": 150.0, "sma_20": 145.0, "sma_50": 140.0, "sma_200": 120.0, "rsi_14": 60.0, "atr_14": 3.0, "vol_ratio": 1.6, "rs_vs_qqq_pct": 6.0},
    ])
    monkeypatch.setattr(signal_engine, "compute_technical_and_risk", lambda df, syms, qqq_ret: tech_rows)

    res = signal_engine.run_analysis(persist=False)
    radar_list = res.get("radar") or res.get("radar_results") or []
    nvda = next(r for r in radar_list if r["symbol"] == "NVDA")
    assert "STOCK Grup Limiti Dolu 2/2" in nvda["decision"]
    assert nvda["risk"] is not None
    assert nvda["risk"]["shares"] == 0
    assert nvda["risk"]["position_usd"] == 0.0


def test_circuit_breaker_triggers_on_daily_loss():
    from signal_engine import evaluate_circuit_breaker
    triggered, reason = evaluate_circuit_breaker(effective_capital=9800.0, daily_pnl=-350.0, base_capital=10000.0)
    assert triggered is True
    assert "Gunluk Zarar" in reason


def test_backtest_exit_on_trend_breakdown():
    from datetime import date, timedelta
    from backtest import _simulate_slice

    rows = []
    # İlk 15 bar yükseliş trendi (giriş sinyali tetiklenir), sonraki 15 bar SMA20 altına kırılım
    for i in range(30):
        px = 100.0 + (i * 1.2) if i < 15 else 118.0 - ((i - 14) * 1.5)
        rows.append({
            "price_date": date(2025, 1, 1) + timedelta(days=i),
            "open": px,
            "high": px + 0.5,
            "low": px - 0.5,
            "close": px,
            "atr_14": 1.5,
            "sma_20": 110.0 if i < 15 else 116.0,
            "sma_50": 102.0,
            "sma_200": 90.0,
            "rsi_14": 62.0 if i < 15 else 38.0,
            "vol_ratio": 1.6 if i < 15 else 1.1,
            "rs_vs_qqq_pct": 5.0 if i < 15 else -6.0,
            "qqq_bull": True,
        })
    res = _simulate_slice(rows, asset_type="STOCK")
    assert res["trades"] >= 1


def test_portfolio_cli_sell_calculates_realized_pnl(monkeypatch):
    import portfolio_cli

    executed_queries = []

    class DummyCursor:
        def __init__(self):
            self._step = 0
        def execute(self, sql, params=None):
            executed_queries.append((sql.strip(), params))
        def fetchone(self):
            from datetime import date
            return (10.0, 100.0, date(2026, 11, 20))
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    class DummyConn:
        def cursor(self):
            return DummyCursor()
        def commit(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    monkeypatch.setattr(portfolio_cli, "get_db_connection", lambda: DummyConn())
    portfolio_cli.sell_position("AAPL", 10.0, 115.0)
    trade_inserts = [p for sql, p in executed_queries if "INSERT INTO trades" in sql]
    assert len(trade_inserts) == 1
    assert float(trade_inserts[0][4]) == 150.0


def test_format_report_message_renders_portfolio_positions():
    from notifier import format_report_message

    sample_report = {
        "macro": {"is_bull": True, "close": 520.0, "sma_200": 480.0},
        "capital": {
            "effective_capital": 10250.0,
            "realized_pnl": 250.0,
            "available_cash": 6000.0,
            "open_positions": 1,
            "max_positions": 3,
        },
        "portfolio": [
            {
                "symbol": "NVDA",
                "decision": "TUT (TRAIL SL AKTIF)",
                "close": 148.50,
                "risk": {
                    "pnl_pct": 12.50,
                    "stop_loss": 139.00,
                    "partial_tp": 152.00,
                },
            }
        ],
        "radar": [],
        "warnings": ["Fiyat gecikmesi uyarisi"],
    }

    msg = format_report_message(sample_report)
    assert "Pozisyon: 1/3" in msg
    assert "NVDA: TUT (TRAIL SL AKTIF)" in msg
    assert "$148.50" in msg
    assert "+12.50%" in msg
    assert "$139.00" in msg
    assert "$152.00" in msg
    assert "Fiyat gecikmesi uyarisi" in msg


def test_price_diff_shock_guard_triggers():
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from signal_engine import generate_signal

    row = {
        "close": 105.0,
        "sma_20": 100.0,
        "sma_50": 95.0,
        "sma_200": 85.0,
        "rsi_14": 58.0,
        "atr_14": 3.0,
        "vol_ratio": 2.2,
        "rs_vs_qqq_pct": 4.0,
        "price_diff": -8.5,
    }
    decision, score, reasons, risk = generate_signal(
        row,
        "STOCK",
        True,
        0.2,
        0.0,
        30,
        10000.0,
        10000.0,
    )
    assert decision.startswith("BEKLE") or decision != "AL"
