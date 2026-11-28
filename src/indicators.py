import polars as pl


def add_technical_indicators(df: pl.DataFrame) -> pl.DataFrame:
    """Hem canli sinyal motoru hem de backtest icin ortak teknik gostergeler (Wilder Smoothing)."""
    df = df.sort("price_date").with_columns([
        pl.col("close_price").rolling_mean(window_size=20).alias("sma_20"),
        pl.col("close_price").rolling_mean(window_size=50).alias("sma_50"),
        pl.col("close_price").rolling_mean(window_size=200).alias("sma_200"),
        pl.col("volume").cast(pl.Float64).rolling_mean(window_size=20).alias("vol_sma_20"),
        pl.col("close_price").diff().alias("price_diff"),
        pl.col("close_price").shift(1).alias("prev_close"),
    ])

    df = df.with_columns([
        pl.when(pl.col("price_diff") > 0).then(pl.col("price_diff")).otherwise(0.0).alias("gain"),
        pl.when(pl.col("price_diff") < 0).then(-pl.col("price_diff")).otherwise(0.0).alias("loss"),
        pl.max_horizontal([
            pl.col("high_price") - pl.col("low_price"),
            (pl.col("high_price") - pl.col("prev_close")).abs(),
            (pl.col("low_price") - pl.col("prev_close")).abs(),
        ]).alias("true_range"),
    ])

    # Wilder's Smoothing (RMA: alpha = 1 / 14)
    df = df.with_columns([
        pl.col("gain").ewm_mean(alpha=1.0 / 14.0, adjust=False, min_samples=14).alias("avg_gain"),
        pl.col("loss").ewm_mean(alpha=1.0 / 14.0, adjust=False, min_samples=14).alias("avg_loss"),
        pl.col("true_range").ewm_mean(alpha=1.0 / 14.0, adjust=False, min_samples=14).alias("atr_14"),
    ])

    df = df.with_columns([
        pl.when(pl.col("avg_loss") == 0)
        .then(100.0)
        .otherwise(100.0 - (100.0 / (1.0 + (pl.col("avg_gain") / pl.col("avg_loss")))))
        .alias("rsi_14")
    ])
    return df
