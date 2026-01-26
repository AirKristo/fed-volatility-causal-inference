
#Parallel trends visualization to validate DiD assumption.

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROCESSED_PATH = Path(__file__).parent.parent.parent / "data" / "processed"
OUTPUT_PATH = Path(__file__).parent.parent.parent / "outputs"

TREATMENT_TICKERS = ["XLF", "XLRE"]
CONTROL_TICKERS = ["XLP", "XLV"]


def load_panel():
    """Load event panel and compute group averages."""

    panel = pd.read_csv(PROCESSED_PATH / "event_panel.csv", parse_dates=["fomc_date", "date"])
    panel = panel[panel["ticker"].isin(TREATMENT_TICKERS + CONTROL_TICKERS)].copy()
    panel = panel.dropna(subset=["abs_return"])

    # Assign groups
    panel["group"] = "Control"
    panel.loc[panel["ticker"].isin(TREATMENT_TICKERS), "group"] = "Treatment"

    return panel


def compute_trends(panel):
    """Compute average volatility by event day and group."""

    trends = panel.groupby(["event_day", "group"])["abs_return"].agg(["mean", "std", "count"])
    trends = trends.reset_index()

    # Compute standard error
    trends["se"] = trends["std"] / (trends["count"] ** 0.5)

    return trends


def main():
    panel = load_panel()
    trends = compute_trends(panel)

    # Separate treatment and control
    treat = trends[trends["group"] == "Treatment"].set_index("event_day")
    ctrl = trends[trends["group"] == "Control"].set_index("event_day")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Treatment with confidence band
    ax.plot(treat.index, treat["mean"], marker="o", linewidth=2, label="Treatment (XLF, XLRE)")
    ax.fill_between(
        treat.index,
        treat["mean"] - 1.96 * treat["se"],
        treat["mean"] + 1.96 * treat["se"],
        alpha=0.2
    )

    # Control with confidence band
    ax.plot(ctrl.index, ctrl["mean"], marker="s", linewidth=2, label="Control (XLP, XLV)")
    ax.fill_between(
        ctrl.index,
        ctrl["mean"] - 1.96 * ctrl["se"],
        ctrl["mean"] + 1.96 * ctrl["se"],
        alpha=0.2
    )

    # Event line
    ax.axvline(x=-0.5, color="red", linestyle="--", alpha=0.7, label="FOMC Announcement")

    ax.set_xlabel("Days Relative to FOMC Announcement")
    ax.set_ylabel("Average Absolute Return")
    ax.set_title("Parallel Trends Check: Pre-FOMC Volatility by Group")
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add annotation about parallel trends
    pre_treat = treat.loc[[-2, -1], "mean"].values
    pre_ctrl = ctrl.loc[[-2, -1], "mean"].values
    treat_slope = pre_treat[1] - pre_treat[0]
    ctrl_slope = pre_ctrl[1] - pre_ctrl[0]

    print("Parallel Trends Check:")
    print(f"  Treatment slope (day -2 to -1): {treat_slope:.6f}")
    print(f"  Control slope (day -2 to -1):   {ctrl_slope:.6f}")
    print(f"  Difference in slopes:           {abs(treat_slope - ctrl_slope):.6f}")

    if abs(treat_slope - ctrl_slope) < 0.001:
        print("  Result: Slopes are similar — parallel trends assumption appears valid.")
    else:
        print("  Result: Some difference in pre-trends, interpret DiD with caution.")

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH / "parallel_trends.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved to {OUTPUT_PATH / 'parallel_trends.png'}")


if __name__ == "__main__":
    main()