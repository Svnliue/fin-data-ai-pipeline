import time
import fetch_prices
import fetch_news
import signal_engine
from logger import get_logger

log = get_logger("pipeline_runner")


def run_full_pipeline(skip_fetch: bool = False) -> dict:
    """
    Fiyat ETL + AI Haber Vektorizasyonu + Alpha Sinyal Motorunu tek akista calistirir.
    FastAPI BackgroundTasks ve CLI tarafindan dogrudan cagirilir.
    """
    t0 = time.perf_counter()
    log.info("Uçtan Uca Kantitatif Boru Hattı Başlatıldı...")

    if not skip_fetch:
        fetch_prices.main()
        fetch_news.main()

    signal_engine.main()
    elapsed = round(time.perf_counter() - t0, 2)
    log.info(f"Boru Hattı Tamamlandı ({elapsed} sn)")
    return {"status": "ok", "elapsed_seconds": elapsed}


if __name__ == "__main__":
    run_full_pipeline()
