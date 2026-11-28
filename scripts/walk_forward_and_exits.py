import argparse
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import polars as pl
import backtest
import signal_engine
from config import get_active_asset_map, ASSET_PROFILES
from signal_engine import load_prices_df


DEFAULT_EXPANDED_UNIVERSE = {
    "AAPL": "STOCK",
    "MSFT": "STOCK",
    "NVDA": "STOCK",
    "AMZN": "STOCK",
    "GOOGL": "STOCK",
    "META": "STOCK",
    "TSLA": "STOCK",
    "AMD": "STOCK",
    "AVGO": "STOCK",
    "JPM": "STOCK",
    "V": "STOCK",
    "MA": "STOCK",
    "UNH": "STOCK",
    "COST": "STOCK",
    "HD": "STOCK",
    "PG": "STOCK",
    "XOM": "STOCK",
    "CVX": "STOCK",
    "GLD": "COMMODITY",
    "SLV": "COMMODITY",
    "XLK": "SECTOR_ETF",
    "XLF": "SECTOR_ETF",
    "XLE": "SECTOR_ETF",
    "XLV": "SECTOR_ETF",
    "XLI": "SECTOR_ETF",
    "SMH": "SECTOR_ETF",
    "TLT": "SECTOR_ETF",
    "TQQQ": "LEVERAGED",
    "SOXL": "LEVERAGED",
    "IREX": "LEVERAGED",
}


def _split_prepared(prepared: dict[str, list[dict]], ratio: float = 0.70):
    train_map: dict[str, list[dict]] = {}
    test_map: dict[str, list[dict]] = {}
    for sym, rows in prepared.items():
        if len(rows) < 50:
            continue
        idx = int(len(rows) * ratio)
        if idx < 25 or (len(rows) - idx) < 20:
            continue
        train_map[sym] = rows[:idx]
        test_map[sym] = rows[idx:]
    return train_map, test_map


def _patch_entry_rules(disable_vol: bool = False, min_rs_pct: float | None = None):
    orig_score_factors = signal_engine._score_factors

    def patched_score_factors(row: dict, asset_type: str, sentiment: float):
        modified = dict(row)
        if min_rs_pct is not None:
            rs = float(modified.get("rs_vs_qqq_pct") or 0.0)
            if 0.0 < rs < min_rs_pct:
                modified["rs_vs_qqq_pct"] = 0.0
        if disable_vol:
            modified["vol_ratio"] = 1.0
        return orig_score_factors(modified, asset_type, sentiment)

    signal_engine._score_factors = patched_score_factors

    def restore():
        signal_engine._score_factors = orig_score_factors

    return restore


def _patch_exit_profiles(sl_mult: float, tp1_mult: float, tp2_mult: float, lev_max_hold: int):
    snapshots = {k: dict(v) for k, v in ASSET_PROFILES.items()}
    for k, prof in ASSET_PROFILES.items():
        if k == "LEVERAGED":
            prof["sl_atr"] = min(sl_mult, 1.5)
            prof["tp1_atr"] = min(tp1_mult, 1.5)
            prof["tp2_atr"] = max(tp2_mult, 2.0)
            prof["max_hold_days"] = lev_max_hold
        else:
            prof["sl_atr"] = sl_mult
            prof["tp1_atr"] = tp1_mult
            prof["tp2_atr"] = tp2_mult

    def restore():
        for k, snap in snapshots.items():
            ASSET_PROFILES[k].clear()
            ASSET_PROFILES[k].update(snap)

    return restore


def _print_row(label: str, res: dict) -> None:
    print(
        f"{label:<30} | "
        f"{res['total_return_pct']:+7.2f}% | "
        f"{res['profit_factor']:5.2f} | "
        f"{res['sharpe']:6.2f} | "
        f"{res['max_dd_pct']:7.2f}% | "
        f"{res['total_trades']:5d} | "
        f"{res['win_rate']:5.1f}%"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="70/30 OOS dogrulama ve cikis parametre taramasi")
    parser.add_argument(
        "--universe",
        choices=["active", "expanded"],
        default="active",
        help="active: DB izleme listesi | expanded: 30 likit hisse/ETF evreni",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=None,
        help="Ozel sembol listesi (orn. --symbols AAPL MSFT NVDA GLD XLE)",
    )
    args = parser.parse_args()

    if args.symbols:
        base_map = {**DEFAULT_EXPANDED_UNIVERSE, **get_active_asset_map()}
        asset_map = {s.strip().upper(): base_map.get(s.strip().upper(), "STOCK") for s in args.symbols}
    elif args.universe == "expanded":
        asset_map = dict(DEFAULT_EXPANDED_UNIVERSE)
    else:
        asset_map = get_active_asset_map()

    raw_df: pl.DataFrame = load_prices_df()
    available_syms = set(raw_df["symbol"].unique().to_list()) if not raw_df.is_empty() else set()
    missing = sorted(set(asset_map.keys()) - available_syms)
    if missing:
        print(f"[UYARI] DB'de fiyat verisi olmayan semboller atlandi: {', '.join(missing)}")
        asset_map = {k: v for k, v in asset_map.items() if k in available_syms}

    prepared, _ = backtest._prepare_all_symbols(raw_df, asset_map)
    train_map, test_map = _split_prepared(prepared, ratio=0.70)

    entry_candidates = [
        ("Baseline (Mevcut)", {}),
        ("Hacim Skoru Kapali", {"disable_vol": True}),
        ("Siki RS >= 4.0%", {"min_rs_pct": 4.0}),
        ("Hacim Kapali + RS >= 4.0%", {"disable_vol": True, "min_rs_pct": 4.0}),
    ]

    print(f"\nSembol Sayisi: {len(asset_map)} ({', '.join(sorted(asset_map.keys()))})")
    print("\n" + "=" * 112)
    print("1) GIRIS KURALLARI — ILK %70 (TRAIN) VS SON %30 (TEST / OOS)")
    print("=" * 112)
    print(f"{'KURAL / DÖNEM':<30} | {'GETIRI':>8} | {'PF':>5} | {'SHARPE':>6} | {'MAX_DD':>8} | {'ISLEM':>5} | {'WIN%':>6}")
    print("-" * 112)

    for name, opts in entry_candidates:
        restore = _patch_entry_rules(**opts)
        try:
            tr = backtest.run_portfolio_simulation(train_map, asset_map)
            te = backtest.run_portfolio_simulation(test_map, asset_map)
        finally:
            restore()
        _print_row(f"{name} [Train 70%]", tr)
        _print_row(f"{name} [Test  30%]", te)
        print("-" * 112)

    exit_grid = [
        (2.0, 1.5, 3.5, 10),
        (2.5, 2.0, 4.0, 10),
        (2.5, 2.0, 4.5, 15),
        (1.8, 1.2, 3.0, 7),
    ]

    print("\n" + "=" * 112)
    print("2) CIKIS PARAMETRE TARAMASI — TRAIN %70 vs TEST %30 vs TUMU %100")
    print("=" * 112)
    print(f"{'SL/TP1/TP2 / MAX_HOLD':<30} | {'GETIRI':>8} | {'PF':>5} | {'SHARPE':>6} | {'MAX_DD':>8} | {'ISLEM':>5} | {'WIN%':>6}")
    print("-" * 112)

    for sl_m, tp1_m, tp2_m, hold_d in exit_grid:
        restore = _patch_exit_profiles(sl_m, tp1_m, tp2_m, hold_d)
        try:
            tr = backtest.run_portfolio_simulation(train_map, asset_map)
            te = backtest.run_portfolio_simulation(test_map, asset_map)
            full = backtest.run_portfolio_simulation(prepared, asset_map)
        finally:
            restore()
        label = f"SL {sl_m} / TP {tp1_m}-{tp2_m} / {hold_d}g"
        _print_row(f"{label} [Train 70%]", tr)
        _print_row(f"{label} [Test  30%]", te)
        _print_row(f"{label} [Tumu 100%]", full)
        print("-" * 112)

    print("=" * 112)


if __name__ == "__main__":
    main()
