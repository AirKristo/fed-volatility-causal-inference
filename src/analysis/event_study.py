# Event study visualization for FOMC annoucements

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROCESSED_PATH = Path(__file__).parent.parent.parent / "data" / "processed"
OUTPUT_PATH = Path(__file__).parent.parent.parent / "outputs"

# Treatment = rate-sensitive, Control = defensive

TREATMENT_TICKERS = ["XLF", "XLRE"]
CONTROL_TICKERS = ["XLP", "XLV",]

def load_panel():
    # Load event panel and add treatment indicator.

    panel = pd.read_csv(PROCESSED_PATH / "event_panel.csv", parse_dates=["fomc_date", "date"])

    panel["group"] = None
    panel.loc[panel["ticker"].isin(TREATMENT_TICKERS), "group"] = "Treatment"
    panel.loc[panel["ticker"].isin(CONTROL_TICKERS), "group"] = "Control"

    # Keep only treatment and control for this analysis
    panel = panel[panel["group"].notna()]

    return panel

def compute_average_by_group(panel):
    # Compute average absolute return by event day and group.

    avg = panel.groupby(["event_day", "group"])["abs_return"].mean().reset_index()
    avg = avg.pivot(index="event_day", columns="group", values="abs_return")

    return avg

def plot_event_study(avg):
    # Plot avg volatility

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.plot(avg.index, avg["Treatment"], marker="o", linewidth=2, label="Treatment (XLF, XLRE)")
    ax.plot(avg.index, avg["Control"], marker="s", linewidth=2, label="Control (XLP, XLV)")

    ax.axvline(x=-0.5, color="red", linestyle="--", alpha=0.7, label="FOMC Announcement")

    ax.set_xlabel("Days Relative to FOMC Announcement")
    ax.set_ylabel("Average Absolute Return")
    ax.set_title("Sector Volatility Around FOMC Announcements (2014-2025)")
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig

def main():
    panel = load_panel()
    avg = compute_average_by_group(panel)

    print("Average absolute returns by event day:")
    print(avg.round(5))

    fig = plot_event_study(avg)

    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH / "event_study.png", dpi=150, bbox_inches="tight")

    print(f"\nSaved plot to {OUTPUT_PATH / 'event_study.png'}")

if __name__ == "__main__":
    main()