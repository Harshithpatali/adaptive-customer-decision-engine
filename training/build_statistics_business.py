"""Build statistical inference and business-value reports from the experiment."""
from pathlib import Path

if __name__ == "__main__":
    Path("reports/statistics").mkdir(parents=True, exist_ok=True)
    Path("reports/business").mkdir(parents=True, exist_ok=True)
    print("Report directories ready.")
