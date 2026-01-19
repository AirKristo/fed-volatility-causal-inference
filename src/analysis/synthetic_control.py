"""

Synthetic control analysis for March 2020 emergency Fed cuts.

Constructs a synthetic version of XLF, which is financials, using other sectors,
then compares actual vs synthetic to estimate the causal effect.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from pathlib import Path

RAW_PATH = Path(__file__).parent.parent.parent / "data" / "raw"
OUTPUT_PATH = Path(__file__).parent.parent.parent / "outputs"

EVENT_DATE = "2020-03-03"
TREATED_TICKER = "XLF"
DONOR_TICKERS = ["XLP", "XLV", "XLK", "XLE", "XLU"]

PRE_START = "2019-06-01"
PRE_END = "2020-02-28"

POST_END = "2020-04-30"

def load_returns():
    # Load prices and compute daily returns

    prices = pd.read_csv(RAW_PATH / "sector_prices.csv", parse_dates=["date"])
    prices = prices.sort_values(["ticker", "date"])
    prices["return"] = prices.groupby("ticker")["close"].pct_change(fill_method=None)

    return prices

def prepare_data(prices):
    # Pivot returns to wide format for synthetic control.

    prices = prices[(prices["date"] >= PRE_START) & (prices["date"] <= POST_END)]
    returns = prices.pivot(index="date", columns="ticker", values="return").dropna()

    return returns