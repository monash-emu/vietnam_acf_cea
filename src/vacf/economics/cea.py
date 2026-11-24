"""Cost-effectiveness analysis.

Works with DALYs rather than QALYs, so health outcomes are losses: lower is better. An
option is preferred when it has both lower cost and fewer DALYs.

Provides the standard sequence — strong dominance, extended dominance, incremental
cost-effectiveness ratios along the efficiency frontier — plus net monetary benefit and
acceptability curves for probabilistic results.
"""

import numpy as np
import pandas as pd


def _strongly_dominated(table: pd.DataFrame) -> list:
    """Options beaten by another on both cost and DALYs."""
    dominated = []
    for name, row in table.iterrows():
        others = table.drop(index=name)
        beaten = (
            (others["cost"] <= row["cost"])
            & (others["dalys"] <= row["dalys"])
            & ((others["cost"] < row["cost"]) | (others["dalys"] < row["dalys"]))
        )
        if beaten.any():
            dominated.append(name)
    return dominated


def efficiency_frontier(results: pd.DataFrame) -> pd.DataFrame:
    """Incremental analysis across scenarios.

    Parameters
    ----------
    results
        Indexed by scenario name, with columns ``cost`` and ``dalys``, both totals over the
        analytic horizon and already discounted.

    Returns
    -------
    pd.DataFrame
        The input with added columns ``status`` (``frontier``, ``dominated`` or
        ``extended_dominated``), ``inc_cost``, ``inc_dalys_averted`` and ``icer``.
        Incremental values compare each frontier option with the next cheapest frontier
        option; the cheapest has no incremental values.
    """
    for col in ("cost", "dalys"):
        if col not in results.columns:
            raise KeyError(f"results must have a '{col}' column")

    table = results.copy()
    table["status"] = "frontier"
    table.loc[_strongly_dominated(table), "status"] = "dominated"

    # Iteratively drop options whose ICER exceeds that of the next more costly option.
    while True:
        frontier = table[table["status"] == "frontier"].sort_values("cost")
        if len(frontier) < 3:
            break
        icers = _sequential_icers(frontier)
        increasing = icers.dropna()
        offenders = increasing.index[:-1][increasing.values[:-1] > increasing.values[1:]]
        if len(offenders) == 0:
            break
        table.loc[offenders[0], "status"] = "extended_dominated"

    frontier = table[table["status"] == "frontier"].sort_values("cost")
    table["inc_cost"] = np.nan
    table["inc_dalys_averted"] = np.nan
    table["icer"] = np.nan
    table.loc[frontier.index, "inc_cost"] = frontier["cost"].diff()
    table.loc[frontier.index, "inc_dalys_averted"] = -frontier["dalys"].diff()
    table.loc[frontier.index, "icer"] = _sequential_icers(frontier)
    return table


def _sequential_icers(frontier: pd.DataFrame) -> pd.Series:
    """ICER of each frontier option against the next cheapest one."""
    inc_cost = frontier["cost"].diff()
    inc_averted = -frontier["dalys"].diff()
    with np.errstate(divide="ignore", invalid="ignore"):
        icers = inc_cost / inc_averted
    return icers.replace([np.inf, -np.inf], np.nan)


def net_monetary_benefit(cost, dalys, threshold: float):
    """Net monetary benefit at a willingness-to-pay threshold per DALY averted.

    Expressed in absolute form, ``-(cost + threshold * dalys)``, so that it can be compared
    across scenarios without nominating a comparator. Higher is better; differences between
    scenarios equal the usual incremental net benefit.
    """
    return -(np.asarray(cost) + threshold * np.asarray(dalys))


def acceptability_curve(draws: dict, thresholds) -> pd.DataFrame:
    """Probability that each scenario is optimal, across willingness-to-pay thresholds.

    Parameters
    ----------
    draws
        Scenario name to a DataFrame of probabilistic draws with columns ``cost`` and
        ``dalys``, one row per draw. All scenarios must have the same number of draws, and
        draw *i* must correspond to the same posterior sample in every scenario — otherwise
        the correlation between scenarios is destroyed and the curve is meaningless.
    thresholds
        Willingness-to-pay values per DALY averted.

    Returns
    -------
    pd.DataFrame
        Threshold-indexed, one column per scenario, each row summing to one.
    """
    names = list(draws)
    sizes = {name: len(draws[name]) for name in names}
    if len(set(sizes.values())) != 1:
        raise ValueError(f"Scenarios have differing numbers of draws: {sizes}")

    rows = {}
    for threshold in thresholds:
        nmb = np.column_stack(
            [net_monetary_benefit(draws[n]["cost"], draws[n]["dalys"], threshold) for n in names]
        )
        best = np.argmax(nmb, axis=1)
        rows[threshold] = [np.mean(best == i) for i in range(len(names))]

    return pd.DataFrame(rows, index=names).T.rename_axis("threshold")
