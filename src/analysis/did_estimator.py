#Difference-in-differences estimation for FOMC announcement effects.

import pandas as pd
import statsmodels.formula.api as smf
from pathlib import Path

PROCESSED_PATH = Path(__file__).parent.parent.parent / "data" / "processed"
OUTPUT_PATH = Path(__file__).parent.parent.parent / "outputs"

TREATMENT_TICKERS = ["XLF", "XLRE"]
CONTROL_TICKERS = ["XLP", "XLV"]


def load_panel():
    # Load event panel and prepare for DiD.

    panel = pd.read_csv(PROCESSED_PATH / "event_panel.csv", parse_dates=["fomc_date", "date"])

    # Assign treatment/control
    panel = panel[panel["ticker"].isin(TREATMENT_TICKERS + CONTROL_TICKERS)].copy()
    panel["treated"] = panel["ticker"].isin(TREATMENT_TICKERS).astype(int)

    # Post = announcement day and after
    panel["post"] = (panel["event_day"] >= 0).astype(int)

    # Drop rows with missing returns
    panel = panel.dropna(subset=["abs_return"])

    return panel


def run_did(panel):
    """
    Run DiD regression.

    Model: abs_return = β0 + β1*treated + β2*post + β3*treated*post + ε

    β3 is the DiD estimate: the additional effect of FOMC announcements
    on treatment sectors relative to control sectors.
    """

    model = smf.ols("abs_return ~ treated + post + treated:post", data=panel)
    results = model.fit(cov_type="cluster", cov_kwds={"groups": panel["fomc_date"]})

    return results


def main():
    panel = load_panel()

    print("=" * 60)
    print("DIFFERENCE-IN-DIFFERENCES: FOMC IMPACT ON SECTOR VOLATILITY")
    print("=" * 60)

    print(f"\nTreatment: {TREATMENT_TICKERS}")
    print(f"Control: {CONTROL_TICKERS}")
    print(f"Observations: {len(panel)}")
    print(f"FOMC events: {panel['fomc_date'].nunique()}")

    # Basic DiD
    print("\n" + "-" * 60)
    print("Basic DiD Model")
    print("-" * 60)
    results = run_did(panel)
    print(results.summary().tables[1])

    # Interpret
    did_coef = results.params["treated:post"]
    did_pval = results.pvalues["treated:post"]

    print(f"\nDiD Estimate: {did_coef:.6f}")
    print(f"P-value: {did_pval:.4f}")
    print(f"Interpretation: FOMC announcements cause an additional {did_coef * 100:.3f}% ")
    print(f"               volatility in rate-sensitive sectors vs defensive sectors.")

    if did_pval < 0.05:
        print("               This effect is statistically significant at the 5% level.")
    else:
        print("               This effect is NOT statistically significant at the 5% level.")

    # Save results
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH / "did_results.txt", "w") as f:
        f.write("Basic DiD Model\n")
        f.write(str(results.summary()))

    print(f"\nFull results saved to {OUTPUT_PATH / 'did_results.txt'}")


if __name__ == "__main__":
    main()