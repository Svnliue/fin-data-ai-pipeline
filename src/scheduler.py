import time
from datetime import datetime
from zoneinfo import ZoneInfo
import schedule
from logger import get_logger
from pipeline_runner import run_full_pipeline

log = get_logger("scheduler")
NY_TZ = ZoneInfo("America/New_York")


def _is_nyse_trading_day() -> bool:
    now_ny = datetime.now(NY_TZ)
    # Pazartesi (0) - Cuma (4) ABD seans gunu kontrolu
    return now_ny.weekday() < 5


def scheduled_job() -> None:
    if not _is_nyse_trading_day():
        log.info("ABD borsası kapalı (Hafta sonu). Zamanlanmış görev atlandı.")
        return
    log.info("NYSE seans kapanışı sonrası zamanlanmış pipeline tetiklendi.")
    run_full_pipeline()


def main():
    # NYSE 16:00 ET kapanisindan 15 dk sonra (16:15 ET) calisir -> yaz/kis saati otomatik uyumlu
    log.info("Scheduler başlatıldı: Her ABD işlem günü 16:15 America/New_York saatinde çalışacak.")
    schedule.every().day.at("16:15", "America/New_York").do(scheduled_job)

    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
