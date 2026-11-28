import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import polars as pl
import backtest
import signal_engine
from config import get_active_asset_map
from signal_engine import load_prices_df


def _patch_scoring(
    disable_rs: bool = False,
    disable_vol: bool = False,
    disable_rsi: bool = False,
    disable_shock: bool = False,
    disable_macro_filter: bool = False,
    min_vol_ratio: float | None = None,
    min_rs_pct: float | None = None,
):
    orig_score_factors = signal_engine._score_factors
    orig_generate_signal = signal_engine.generate_signal
    orig_profiles = {k: dict(v) for k, v in signal_engine.ASSET_PROFILES.items()}
    orig_shock_vol = signal_engine.SHOCK_VOL_RATIO

    if disable_macro_filter:
        for prof in signal_engine.ASSET_PROFILES.values():
            prof["require_macro_bull"] = False

    if disable_shock:
        signal_engine.SHOCK_VOL_RATIO = 1e9

    def patched_score_factors(row: dict, asset_type: str, sentiment: float):
        modified = dict(row)
        if disable_rs:
            modified["rs_vs_qqq_pct"] = 0.0
        elif min_rs_pct is not None:
            rs = float(modified.get("rs_vs_qqq_pct") or 0.0)
            if 0.0 < rs < min_rs_pct:
                modified["rs_vs_qqq_pct"] = 0.0

        if disable_vol:
            modified["vol_ratio"] = 1.0
        elif min_vol_ratio is not None:
            vr = float(modified.get("vol_ratio") or 1.0)
            if vr < min_vol_ratio:
                modified["vol_ratio"] = 1.0

        if disable_rsi:
            modified["rsi_14"] = 50.0

        return orig_score_factors(modified, asset_type, sentiment)

    def patched_generate_signal(*args, **kwargs):
        if disable_shock and len(args) >= 6:
            args = list(args)
            args[5] = None  # days_to_earnings devre disi
        return orig_generate_signal(*args, **kwargs)

    signal_engine._score_factors = patched_score_factors
    signal_engine.generate_signal = patched_generate_signal
    backtest.generate_signal = patched_generate_signal

    def restore():
        signal_engine._score_factors = orig_score_factors
        signal_engine.generate_signal = orig_generate_signal
        backtest.generate_signal = orig_generate_signal
        signal_engine.SHOCK_VOL_RATIO = orig_shock_vol
        for k, v in orig_profiles.items():
            if k in signal_engine.ASSET_PROFILES:
                signal_engine.ASSET_PROFILES[k].clear()
                signal_engine.ASSET_PROFILES[k].update(v)

    return restore


def main() -> None:
    asset_map = get_active_asset_map()
    raw_df: pl.DataFrame = load_prices_df()
    prepared, _ = backtest._prepare_all_symbols(raw_df, asset_map)

    scenarios = [
        ("01_Baseline (Mevcut)", {}),
        ("02_RS_Kapali", {"disable_rs": True}),
        ("03_Hacim_Skor_Kapali", {"disable_vol": True}),
        ("04_RSI_Skor_Kapali", {"disable_rsi": True}),
        ("05_Sok_Kilidi_Kapali", {"disable_shock": True}),
        ("06_Makro_Filtre_Kapali", {"disable_macro_filter": True}),
        ("07_Siki_Hacim_1.5x", {"min_vol_ratio": 1.5}),
        ("08_Siki_RS_4.0pct", {"min_rs_pct": 4.0}),
    ]

    print("\n" + "=" * 108)
    print(f"{'SENARYO':<26} | {'GETIRI':>8} | {'PF':>5} | {'SHARPE':>6} | {'MAX_DD':>8} | {'ISLEM':>5} | {'WIN%':>6}")
    print("=" * 108)

    for label, opts in scenarios:
        restore = _patch_scoring(**opts)
        try:
            res = backtest.run_portfolio_simulation(prepared, asset_map)
        finally:
            restore()

        print(
            f"{label:<26} | "
            f"{res['total_return_pct']:+7.2f}% | "
            f"{res['profit_factor']:5.2f} | "
            f"{res['sharpe']:6.2f} | "
            f"{res['max_dd_pct']:7.2f}% | "
            f"{res['total_trades']:5d} | "
            f"{res['win_rate']:5.1f}%"
        )
    print("=" * 108)


if __name__ == "__main__":
    main()
