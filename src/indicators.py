import polars as pl


def add_technical_indicators(df: pl.DataFrame, qqq_ret_df: pl.DataFrame | None = None) -> pl.DataFrame:
    """
    Tum semboller icin ortak teknik indikatorleri (SMA20/50/200, ATR14, RSI14, Hacim Orani, RS vs QQQ) hesaplar.
    Hem signal_engine hem backtest tarafindan tek kaynak olarak kullanilir.
    """
    df = df.sort(["symbol", "price_date"])

    df = df.with_columns([
        pl.col("close").rolling_mean(window_size=20).over("symbol").alias("sma_20"),
        pl.col("close").rolling_mean(window_size=50).over("symbol").alias("sma_50"),
        pl.col("close").rolling_mean(window_size=200).over("symbol").alias("sma_200"),
        pl.col("volume").rolling_mean(window_size=20).over("symbol").alias("vol_sma_20"),
        ((pl.col("close") / pl.col("close").shift(20).over("symbol")) - 1.0).alias("ret_20d"),
        (pl.col("close") - pl.col("close").shift(1).over("symbol")).alias("delta"),
        pl.max_horizontal([
            pl.col("high") - pl.col("low"),
            (pl.col("high") - pl.col("close").shift(1).over("symbol")).abs(),
            (pl.col("low") - pl.col("close").shift(1).over("symbol")).abs(),
        ]).alias("tr"),
    ])

    df = df.with_columns([
        pl.when(pl.col("delta") > 0).then(pl.col("delta")).otherwise(0.0).alias("gain"),
        pl.when(pl.col("delta") < 0).then(-pl.col("delta")).otherwise(0.0).alias("loss"),
        pl.col("tr").rolling_mean(window_size=14).over("symbol").alias("atr_14"),
        (pl.col("volume") / (pl.col("vol_sma_20") + 1e-9)).alias("vol_ratio"),
    ])

    df = df.with_columns([
        pl.col("gain").rolling_mean(window_size=14).over("symbol").alias("avg_gain"),
        pl.col("loss").rolling_mean(window_size=14).over("symbol").alias("avg_loss"),
    ])

    df = df.with_columns(
        (100.0 - (100.0 / (1.0 + (pl.col("avg_gain") / (pl.col("avg_loss") + 1e-9))))).alias("rsi_14")
    )

    if qqq_ret_df is not None:
        df = df.join(qqq_ret_df, on="price_date", how="left").with_columns(
            ((pl.col("ret_20d") - pl.col("qqq_ret_20d").fill_null(0.0)) * 100.0).alias("rs_vs_qqq_pct")
        )
    else:
        df = df.with_columns(pl.lit(0.0).alias("rs_vs_qqq_pct"))

    return df
