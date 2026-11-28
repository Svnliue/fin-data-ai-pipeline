import polars as pl


def add_technical_indicators(df: pl.DataFrame, qqq_ret_df: pl.DataFrame | None = None) -> pl.DataFrame:
    """Hem canli sinyal motoru hem de backtest icin ortak teknik gostergeler (Wilder Smoothing)."""
    sub = df.sort(["symbol", "price_date"]).with_columns([
        pl.col("close").rolling_mean(window_size=20).over("symbol").alias("sma_20"),
        pl.col("close").rolling_mean(window_size=50).over("symbol").alias("sma_50"),
        pl.col("close").rolling_mean(window_size=200).over("symbol").alias("sma_200"),
        pl.col("volume").cast(pl.Float64).rolling_mean(window_size=20).over("symbol").alias("vol_sma_20"),
        ((pl.col("close") / pl.col("close").shift(20).over("symbol")) - 1.0).alias("ret_20d"),
        pl.col("close").diff().over("symbol").alias("price_diff"),
        pl.col("close").shift(1).over("symbol").alias("prev_close"),
    ])

    sub = sub.with_columns([
        pl.when(pl.col("price_diff") > 0).then(pl.col("price_diff")).otherwise(0.0).alias("gain"),
        pl.when(pl.col("price_diff") < 0).then(-pl.col("price_diff")).otherwise(0.0).alias("loss"),
        pl.max_horizontal([
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("prev_close")).abs(),
            (pl.col("low") - pl.col("prev_close")).abs(),
        ]).alias("true_range"),
    ])

    # Wilder's Smoothing (RMA: alpha = 1 / 14)
    sub = sub.with_columns([
        pl.col("gain").ewm_mean(alpha=1.0 / 14.0, adjust=False, min_samples=14).over("symbol").alias("avg_gain"),
        pl.col("loss").ewm_mean(alpha=1.0 / 14.0, adjust=False, min_samples=14).over("symbol").alias("avg_loss"),
        pl.col("true_range").ewm_mean(alpha=1.0 / 14.0, adjust=False, min_samples=14).over("symbol").alias("atr_14"),
    ])

    sub = sub.with_columns([
        pl.when(pl.col("avg_loss") == 0)
        .then(100.0)
        .otherwise(100.0 - (100.0 / (1.0 + (pl.col("avg_gain") / pl.col("avg_loss")))))
        .alias("rsi_14"),
        pl.when(pl.col("vol_sma_20") > 0)
        .then(pl.col("volume").cast(pl.Float64) / pl.col("vol_sma_20"))
        .otherwise(1.0)
        .alias("vol_ratio"),
    ])

    if qqq_ret_df is not None and not qqq_ret_df.is_empty():
        sub = sub.join(qqq_ret_df, on="price_date", how="left").with_columns(
            ((pl.col("ret_20d") - pl.col("qqq_ret_20d").fill_null(0.0)) * 100.0).alias("rs_vs_qqq_pct")
        )
    return sub
