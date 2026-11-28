import time
from migrate import run_migrations
import fetch_prices
import fetch_news
import signal_engine
from notifier import send_telegram_alert
from logger import get_logger

log = get_logger("pipeline_runner")


def run_full_pipeline(skip_fetch: bool = False, notify: bool = True) -> dict:
    t0 = time.perf_counter()
    log.info("Uçtan Uca Kantitatif Boru Hattı Başlatıldı...")
    run_migrations()

    if not skip_fetch:
        fetch_prices.main()
        fetch_news.main()

    report = signal_engine.run_analysis(persist=True)
    if notify:
        send_telegram_alert(report)

    elapsed = round(time.perf_counter() - t0, 2)
    log.info(f"Boru Hattı Tamamlandı ({elapsed} sn)")
    return {"status": "ok", "elapsed_seconds": elapsed, "report": report}


if __name__ == "__main__":
    run_full_pipeline()
