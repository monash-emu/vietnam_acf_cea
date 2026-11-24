"""Summaries of a stored posterior run, on small synthetic tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vacf.economics import analysis, study


def _economic_summary() -> pd.DataFrame:
    rows = []
    for sample, (base_cost, base_dalys) in enumerate([(100.0, 50.0), (200.0, 80.0)]):
        rows.append((sample, study.BASELINE.name, "Baseline", base_cost, base_dalys))
        rows.append(
            (sample, "campaign", "Campaign", base_cost + 10.0 * (sample + 1), base_dalys - 5.0)
        )
    return pd.DataFrame(rows, columns=["sample", "arm", "label", "cost", "dalys"])


def test_incremental_vs_baseline_pairs_within_draws():
    out = analysis.incremental_vs_baseline(_economic_summary())
    assert list(out["arm"]) == ["campaign", "campaign"]
    np.testing.assert_allclose(out["inc_cost"], [10.0, 20.0])
    np.testing.assert_allclose(out["dalys_averted"], [5.0, 5.0])


def test_expected_incremental_uses_mean_costs_and_dalys():
    out = analysis.expected_incremental(_economic_summary())
    assert out.loc["Campaign", "cost"] == pytest.approx(165.0)
    assert out.loc["Campaign", "icer"] == pytest.approx(15.0 / 5.0)


def test_interval_summary_keeps_group_order_and_brackets_the_median():
    table = pd.DataFrame({"label": ["b", "b", "a", "a"], "x": [1.0, 3.0, 10.0, 30.0]})
    out = analysis.interval_summary(table, ["x"])
    assert list(out.index) == ["b", "a"]
    assert out.loc["b", ("x", "median")] == pytest.approx(2.0)
    assert (out[("x", "lower")] <= out[("x", "median")]).all()
    assert (out[("x", "median")] <= out[("x", "upper")]).all()


def test_prevalence_decline_is_one_at_the_base_survey():
    years = np.arange(2000.0, 2011.0)
    fit = pd.concat(
        pd.DataFrame({"sample": s, "year": years, "prevalence": scale * np.exp(-0.05 * years)})
        for s, scale in enumerate([1.0, 2.0])
    )
    out = analysis.with_prevalence_decline(fit, base_time=2005.5)
    for _, group in out.groupby("sample"):
        assert np.interp(2005.5, group["year"], group["decline"]) == pytest.approx(1.0, rel=1e-3)
    # Relative trajectories do not depend on the level of prevalence.
    np.testing.assert_allclose(
        out[out["sample"] == 0]["decline"], out[out["sample"] == 1]["decline"]
    )


def test_distribution_table_reports_median_iqr_and_95_interval():
    values = np.arange(101.0)
    table = pd.DataFrame({"arm": "a", "label": "A", "cost": values})
    out = analysis.distribution_table(table, ("cost",), ["arm", "label"])
    row = out.iloc[0]
    assert row["quantity"] == "cost" and row["unit"] == analysis.UNITS["cost"]
    assert (row["median"], row["iqr_low"], row["iqr_high"]) == (50.0, 25.0, 75.0)
    assert (row["ci95_low"], row["ci95_high"]) == pytest.approx((2.5, 97.5))


def test_every_exported_quantity_has_a_unit():
    exported = (
        analysis.EPI_OUTCOMES
        + analysis.ECONOMIC_OUTCOMES
        + analysis.INCREMENTAL_OUTCOMES
        + analysis.TRAJECTORY_INDICATORS
    )
    assert set(exported) <= set(analysis.UNITS)


def test_run_and_store_will_not_replace_a_stored_posterior(tmp_path):
    (tmp_path / "idata.nc").write_bytes(b"")
    with pytest.raises(FileExistsError):
        analysis.run_and_store(tmp_path, n_runs=1, n_chains=1, n_draws=1)
