from datetime import date, timedelta
import polars as pl
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
