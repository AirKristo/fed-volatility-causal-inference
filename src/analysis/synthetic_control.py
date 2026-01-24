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
TREATED = "XLF"
DONORS = ["XLP", "XLV", "XLK", "XLE", "XLU"]

def load_returns():
    # Load prices and compute daily returns

    prices = pd.read_csv(RAW_PATH / "sector_prices.csv", parse_dates=["date"])
    prices = prices.sort_values(["ticker", "date"])
    prices["return"] = prices.groupby("ticker")["close"].pct_change(fill_method=None)

    prices = prices[(prices["date"] >= "2019-06-01") & (prices["date"] <= "2020-04-30")]
    returns = prices.pivot(index="date", columns="ticker", values="return").dropna()

    return returns

def find_weights(pre_treated, pre_donors):
    # Find donor weights that best match treated unit in pre-period

    n_donors = len(DONORS)

    initial_weights = np.ones(n_donors) / n_donors

    bounds = []
    for i in range(n_donors):
        bounds.append((0, 1))

    def weights_sum_to_one(w):
        return np.sum(w) - 1

    constraints = {"type": "eq", "fun": weights_sum_to_one}

    def objective(weights):
        synthetic = pre_donors @ weights
        squared_errors = (pre_treated - synthetic) ** 2
        return np.sum(squared_errors)

    result = minimize(
        objective,
        x0 = initial_weights,
        method="SLSQP",
        constraints = constraints,
        bounds = bounds,
    )

    return result.x

def main():
    returns = load_returns()

    pre_mask = returns.index < EVENT_DATE
    pre_returns = returns[pre_mask]

    pre_treated = pre_returns[TREATED].values
    pre_donors = pre_returns[DONORS].values
    weights = find_weights(pre_treated, pre_donors)

    print("Synthetic control weights:")
    for i, ticker in enumerate(DONORS):
        if weights[i] > 0.01:
            print(f"  {ticker}: {weights[i]:.3f}")

    synthetic = returns[DONORS] @ weights

    cum_actual = (1 + returns[TREATED]).cumprod() - 1
    cum_synthetic = (1 + synthetic).cumprod() - 1
    gap = cum_actual - cum_synthetic

    post_mask = returns.index >= EVENT_DATE
    final_actual = cum_actual[post_mask].iloc[-1]
    final_synthetic = cum_synthetic[post_mask].iloc[-1]
    final_gap = gap[post_mask].iloc[-1]

    print(f"\nPost-period cumulative returns:")
    print(f"  Actual XLF: {final_actual:.2%}")
    print(f"  Synthetic:  {final_synthetic:.2%}")
    print(f"  Gap:        {final_gap:.2%}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    ax1.plot(cum_actual, label="Actual XLF", linewidth=2)
    ax1.plot(cum_synthetic, label="Synthetic", linewidth=2, linestyle="--")
    ax1.axvline(pd.Timestamp(EVENT_DATE), color="red", linestyle="--", alpha=0.7, label="Emergency Cut")
    ax1.set_ylabel("Cumulative Return")
    ax1.set_title("Synthetic Control: XLF vs Synthetic")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(gap, linewidth=2, color="green")
    ax2.axvline(pd.Timestamp(EVENT_DATE), color="red", linestyle="--", alpha=0.7)
    ax2.axhline(0, color="black", alpha=0.3)
    ax2.fill_between(gap.index, gap, 0, alpha=0.3, color="green")
    ax2.set_ylabel("Gap (Actual - Synthetic)")
    ax2.set_title("Estimated Causal Effect")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH / "synthetic_control.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved to {OUTPUT_PATH / 'synthetic_control.png'}")

if __name__ == "__main__":
    main()
