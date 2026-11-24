"""Run the study from the command line and print the comparison tables.

A quick check without Jupyter. The notebook does the same thing with plots and a
calibration step; this uses forward runs at ``BASE_PARAMS`` by default so it finishes in
under a minute.

    pixi run cea                  # forward run, no calibration
    pixi run cea --calibrate      # short local chain first (slow, and not converged)
"""

from __future__ import annotations

import argparse

import pandas as pd

from vacf.economics import analysis, costing, study


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Fit the model first. Short chain; check convergence before trusting it.",
    )
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--chains", type=int, default=4)
    args = parser.parse_args()

    pd.set_option("display.width", 200)

    print(study.describe())
    warning = study.provisional_warning()
    if warning:
        print(f"\n*** {warning} ***")
    print(
        f"\nDiscounting {costing.DISCOUNT_RATE:.0%} to {costing.REFERENCE_YEAR}; "
        f"horizon {study.HORIZON[0]}-{study.HORIZON[1]}\n"
    )

    print("Screening sensitivity by disease state")
    print(
        pd.DataFrame(study.sensitivity_table()).T[list(study.DISEASE_STATES)].to_string()
    )

    calibrated = None
    if args.calibrate:
        print(f"\nCalibrating: {args.chains} chains, {args.warmup} warmup, {args.samples} draws")
        idata = analysis.calibrate(
            num_warmup=args.warmup,
            num_samples=args.samples,
            num_chains=args.chains,
            progress_bar=False,
        )
        calibrated = analysis.posterior_medians(idata)
        print(f"  estimated {len(calibrated)} parameters")
    else:
        print("\nForward run at BASE_PARAMS (no calibration; pass --calibrate to fit).")

    runs = analysis.run_all(calibrated=calibrated)

    print("\n=== Epidemiological comparison ===")
    epi = analysis.epi_comparison(runs)
    print(
        epi[
            [
                "cum_incidence",
                "cum_deaths",
                "cum_incidence_averted_pct",
                "cum_deaths_averted_pct",
            ]
        ].to_string(float_format=lambda v: f"{v:,.1f}")
    )

    print("\n=== Economic comparison ===")
    econ = analysis.economic_comparison(runs)
    print(
        econ[["n_screened", "cost_provider", "cost_patient", "cost", "dalys"]].to_string(
            float_format=lambda v: f"{v:,.0f}"
        )
    )

    print("\n=== Incremental analysis ===")
    print(
        analysis.incremental(econ)[
            ["cost", "dalys", "status", "inc_cost", "inc_dalys_averted", "icer"]
        ].to_string(float_format=lambda v: f"{v:,.1f}")
    )


if __name__ == "__main__":
    main()
