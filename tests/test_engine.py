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
