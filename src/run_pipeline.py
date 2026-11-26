import subprocess
import sys
import time
from datetime import datetime

STEPS = [
    ("1/4 - Fiyat & Endeks ETL (Polars -> PostgreSQL)", "src/fetch_prices.py"),
    ("2/4 - Haber & Vektör Boru Hattı (all-MiniLM-L6-v2 -> pgvector)", "src/fetch_news.py"),
    ("3/4 - Makro Rejim + Teknik + FinBERT + ATR Sinyal Motoru", "src/signal_engine.py"),
    ("4/4 - 1 Yıllık Walk-Forward Backtest Simülasyonu", "src/backtest.py"),
]


def main():
    pipeline_start = time.perf_counter()
    print("=" * 80)
    print(f"OTONOM FİNANSAL VERİ & AI PIPELINE BAŞLATILDI ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print("=" * 80)

    for title, script_path in STEPS:
        step_start = time.perf_counter()
        print(f"\n>>> ADIM {title} [{script_path}]")
        result = subprocess.run([sys.executable, script_path], check=False)
        elapsed = time.perf_counter() - step_start

        if result.returncode != 0:
            print(f"[HATA] {script_path} adımında hata oluştu (Kod: {result.returncode}). Boru hattı durduruldu.")
            sys.exit(result.returncode)

        print(f"[TAMAMLANDI] Süre: {elapsed:.2f} sn")

    total_elapsed = time.perf_counter() - pipeline_start
    print("\n" + "=" * 80)
    print(f"TÜM BORU HATTI BAŞARIYLA TAMAMLANDI (Toplam Süre: {total_elapsed:.2f} sn)")
    print("=" * 80)


if __name__ == "__main__":
    main()
