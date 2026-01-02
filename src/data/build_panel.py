# Build the event panel for FOMC announcement analysis

import pandas as pd
from pathlib import Path

RAW_PATH = Path(__file__).parent.parent.parent / "data" / "raw"
PROCESSED_PATH = Path(__file__).parent.parent.parent / "data" / "processed"

EVENT_WINDOW = [-2, -1, 0, 1, 2]

def load_data():
    # Load prices and FOMC dates computing returns.

    prices = pd.read_csv(RAW_PATH / "sector_prices.csv", parse_dates=["date"])
    fomc = pd.read_csv(RAW_PATH / "fomc_dates.csv", parse_dates=["date"])

    prices = prices.sort_values(["ticker", "date"])
    prices["return"] = prices.groupby("ticker")["close"].pct_change(fill_method=None)
    prices["abs_return"] = prices["return"].abs()

    return prices, fomc

def build_panel(prices, fomc):
    # Create panel of returns around each FOMC date.

    # Map dates to trading day index
    trading_dates = prices[["date"]].drop_duplicates().sort_values("date").reset_index(drop=True)
    trading_dates["date_idx"] = trading_dates.index

    prices = prices.merge(trading_dates,  on="date")
    fomc = fomc.merge(trading_dates, on="date", how="left")

    # Expand FOMC dates to event window
    events = []
    for offset in EVENT_WINDOW:
        temp = fomc.copy()
        temp["event_day"] = offset
        temp["date_idx"] = temp["date_idx"] + offset
        events.append(temp)

    events = pd.concat(events, ignore_index=True)

    # Merge with prices
    panel = events.merge(
        prices[["date_idx", "ticker", "date", "return", "abs_return"]],
        on="date_idx",
        suffixes=("_fomc", "")
    )

    # Clean up columns
    panel = panel.rename(columns={"date_fomc": "fomc_date"})
    panel = panel[["fomc_date", "is_emergency", "ticker", "event_day", "date", "return", "abs_return"]]
    panel = panel.sort_values(["fomc_date", "ticker", "event_day"]).reset_index(drop=True)
    return panel

def main():
    prices, fomc = load_data()
    panel = build_panel(prices, fomc)

    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    panel.to_csv(PROCESSED_PATH / "event_panel.csv", index=False)

    print(f"FOMC events: {panel['fomc_date'].nunique()}")
    print(f"Tickers: {panel['ticker'].nunique()}")
    print(f"Total rows: {len(panel)}")

if __name__ == "__main__":
    main()

