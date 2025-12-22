"""

Get sector ETF prices from Yahoo Finance.

"""

import yfinance as yf
import pandas as pd
from pathlib import Path

SECTOR_ETFS = ["XLF", "XLRE", "XLK", "XLE", "XLV", "XLP", "XLU", "SPY"]
START_DATE = "2014-01-01"
OUTPUT_PATH = Path(__file__).parent.parent.parent / "data" / "raw" / "sector_prices.csv"

def get_prices() -> pd.DataFrame:
    # Download daily prices for all sector ETFs.

    print(f"Getting sector ETF prices from {START_DATE} to present")

    df = yf.download(SECTOR_ETFS, start=START_DATE, auto_adjust=False, progress=False)
    # Flatten multi-level columns
    df = df.stack(level=1, future_stack=True).reset_index()
    df.columns = ["date", "ticker", "adj_close", "close", "high", "low", "open", "volume"]
    df = df[["date", "ticker", "open", "high", "low", "close", "volume"]]
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    print(f"Downloaded {len(df)} rows")
    return df


def main():
    df = get_prices()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved to {OUTPUT_PATH}")
    print(f"Date range from {df.date.min().date()} to {df.date.max().date()}")

if __name__ == "__main__":
    main()