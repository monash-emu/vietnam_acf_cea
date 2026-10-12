"""Plotly helpers for the summer3-style notebooks under ``notebooks/summer3/``.

Same scientific panels as summer3 ``plotting.py`` on ``tweak_cases_calib``,
built from summer4 ``run`` / ``outputs`` / ``mixing`` results (no ManagedArray).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from tb_macro.calibration import MIXING_TARGET_YEAR
from tb_macro.constants import AGE_STRATA, DISEASE_STATES, INFECT_COMPS, INF_STRATA, CLIN_STRATA
from tb_macro.inputs import get_norm_conmat, load_demography
from tb_macro.mixing import build_s_matrix_single_age_components, canberra_distance
from tb_macro.model import calendar_years, prepare_run_params, spline_mixing
from tb_macro.outputs import (
    _data,
    age_population,
    disease_state_totals,
    flow_total,
)
from tb_macro.parameters import PARAM_ABBREVS, PARAM_BOUNDS, PARAM_NAMES
from tb_macro.static_data import load_static

_INFECT_SOURCE_LABELS = {
    "mtb_naive": "never-infected",
    "contained": "contained",
    "cleared": "cleared",
    "recovered": "recovered",
}

COUNT_TITLES = {
    "prevalence": "prevalent cases",
    "incidence": "incident cases per year",
    "notifications": "notified cases per year",
    "deaths": "deaths per year",
    "latent": "adult infected population number",
    "pulm_prev": "adult pulmonary bacteriologically-detectable cases",
}
RATE_TITLES = {
    "prevalence": "prevalence per 100,000",
    "incidence": "incidence per 100,000 per year",
    "notifications": "notifications per 100,000 per year",
    "deaths": "deaths per 100,000 per year",
    "latent": "percentage of adults with latent infection",
    "pulm_prev": "adult pulmonary bacteriologically-detectable prevalence per 100,000",
}


def _years() -> np.ndarray:
    return np.asarray(calendar_years())


def _series_frame(by_age: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame({age: np.asarray(values) for age, values in by_age.items()}, index=_years())


def _slice(frame: pd.DataFrame, start: float, end: float) -> pd.DataFrame:
    return frame.loc[(frame.index >= start) & (frame.index <= end)]


def _area(fig: go.Figure, frame: pd.DataFrame, row: int, col: int, *, stack: bool) -> None:
    group = "stack" if stack else None
    for name in frame.columns:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame[name],
                name=str(name),
                mode="lines",
                stackgroup=group,
                line=dict(width=0.5) if stack else dict(width=1.5),
                showlegend=(row == 1 and col == 1),
            ),
            row=row,
            col=col,
        )


def _partial_stratum_props(result: Any, key: str, categories: tuple[str, ...]) -> pd.DataFrame:
    """Share of ``active`` population in each ragged stratum over time."""
    from tb_macro.model import PMAP

    data = _data(result["compartments"])
    years = _years()
    totals = np.zeros(len(years))
    parts = {cat: np.zeros(len(years)) for cat in categories}
    for row, traits in enumerate(PMAP.to_dicts()):
        if traits["disease_state"] != "active" or key not in traits:
            continue
        column = np.asarray(data[:, row])
        totals = totals + column
        parts[traits[key]] = parts[traits[key]] + column
    denom = np.maximum(totals, 1e-32)
    return pd.DataFrame({cat: parts[cat] / denom for cat in categories}, index=years)


def plot_comp_distributions(
    result: Any,
    group_popsize: pd.DataFrame,
    start: float,
    end: float,
) -> go.Figure:
    """Population, age, disease, clinical and infectiousness panels."""
    years = _years()
    age_pop = _series_frame(age_population(result))
    total = age_pop.sum(axis=1)
    age_props = age_pop.div(total.replace(0, np.nan), axis=0)
    dstate = pd.DataFrame(
        {state: np.asarray(values) for state, values in disease_state_totals(result).items()},
        index=years,
    )
    dstate = dstate[[s for s in DISEASE_STATES if s in dstate.columns]]
    dstate_props = dstate.div(dstate.sum(axis=1).replace(0, np.nan), axis=0)
    clin = _partial_stratum_props(result, "clinical", CLIN_STRATA)
    infect = _partial_stratum_props(result, "infectious", INF_STRATA)
    target = group_popsize.sum(axis=1)

    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "total population versus target data",
            "age group sizes",
            "age distribution",
            "disease state distribution",
            "clinical status distribution",
            "infectiousness status distribution",
        ),
        shared_xaxes=True,
    )
    window_total = _slice(pd.DataFrame({"modelled": total}), start, end)
    fig.add_trace(
        go.Scatter(
            x=window_total.index, y=window_total["modelled"], name="modelled", stackgroup="one"
        ),
        row=1,
        col=1,
    )
    t_win = target.loc[(target.index >= start) & (target.index <= end)]
    fig.add_trace(
        go.Scatter(
            x=t_win.index,
            y=t_win.values,
            name="target",
            mode="markers",
            marker=dict(size=4, color="black"),
        ),
        row=1,
        col=1,
    )
    _area(fig, _slice(age_pop, start, end).clip(lower=0), 1, 2, stack=True)
    _area(fig, _slice(age_props, start, end).clip(lower=0, upper=1), 1, 3, stack=True)
    _area(fig, _slice(dstate_props, start, end).clip(lower=0, upper=1), 2, 1, stack=True)
    _area(fig, _slice(clin, start, end).clip(lower=0, upper=1), 2, 2, stack=True)
    _area(fig, _slice(infect, start, end).clip(lower=0, upper=1), 2, 3, stack=True)
    fig.update_layout(
        height=700, width=1100, template="plotly_white", title_text="Population structure"
    )
    return fig


def plot_age_population_comparison(
    result: Any,
    group_popsize: pd.DataFrame,
    years: range,
) -> go.Figure:
    """Modelled vs UN age population bars at selected calendar years."""
    age_pop = _series_frame(age_population(result))
    year_list = list(years)
    n = len(year_list)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f"population by age in year {int(y)}" for y in year_list],
    )
    ages = [str(a) for a in AGE_STRATA]
    for i, year in enumerate(year_list):
        row, col = divmod(i, cols)
        modelled = age_pop.loc[float(year)].reindex(ages)
        target = group_popsize.loc[float(year)]
        target.index = target.index.astype(str)
        target = target.reindex(ages)
        fig.add_trace(
            go.Bar(
                x=ages, y=target.values, name="target", marker_color="#444", showlegend=(i == 0)
            ),
            row=row + 1,
            col=col + 1,
        )
        fig.add_trace(
            go.Bar(
                x=ages,
                y=modelled.values,
                name="modelled",
                marker_color="#1f77b4",
                showlegend=(i == 0),
            ),
            row=row + 1,
            col=col + 1,
        )
    fig.update_layout(barmode="group", height=200 * rows + 100, width=1000, template="plotly_white")
    return fig


def plot_infection_source_props(result: Any, start: float, end: float) -> go.Figure:
    """Stacked share of new infections by source compartment."""
    years = _years()
    total = np.asarray(sum(np.asarray(v) for v in age_population(result).values()))
    rates = {}
    for comp in INFECT_COMPS:
        infections = np.asarray(flow_total(result, f"infect_{comp}"))
        rates[_INFECT_SOURCE_LABELS[comp]] = infections / np.maximum(total, 1e-32) * 1e5
    df = pd.DataFrame(rates, index=years)
    df = _slice(df, start, end).clip(lower=0.0)
    props = df.div(df.sum(axis=1).replace(0, np.nan), axis=0).dropna()
    fig = go.Figure()
    for name in props.columns:
        fig.add_trace(
            go.Scatter(
                x=props.index,
                y=props[name],
                name=name,
                mode="lines",
                stackgroup="one",
                line=dict(width=0.5),
            )
        )
    fig.update_layout(
        title="infection source by compartment",
        yaxis_title="proportion",
        xaxis_title="year",
        template="plotly_white",
        height=450,
        width=900,
    )
    return fig


def _mixing_at(params: dict[str, Any], year: float) -> np.ndarray:
    prepared = prepare_run_params(params)
    return np.asarray(spline_mixing(year, prepared))


def plot_dynamic_mixing_matrix(
    params: dict[str, Any],
    start: float = 1970.0,
    interval: float = 10.0,
    n_cols: int = 3,
) -> go.Figure:
    """Spline mixing heatmaps at ``start``, ``start+interval``, …."""
    years = [start + i * interval for i in range(2 * n_cols)]
    matrices = [_mixing_at(params, year) for year in years]
    vmax = max(float(m.max()) for m in matrices)
    fig = make_subplots(rows=2, cols=n_cols, subplot_titles=[str(int(y)) for y in years])
    ages = [str(a) for a in AGE_STRATA]
    for i, (year, matrix) in enumerate(zip(years, matrices)):
        row, col = divmod(i, n_cols)
        fig.add_trace(
            go.Heatmap(
                z=matrix,
                x=ages,
                y=ages,
                zmin=0.0,
                zmax=vmax,
                colorscale="Viridis",
                showscale=(i == len(years) - 1),
                colorbar=dict(title="contact"),
            ),
            row=row + 1,
            col=col + 1,
        )
    fig.update_layout(
        title_text="Mixing matrix over time (folded spline)",
        height=700,
        width=1100,
        template="plotly_white",
    )
    return fig


def plot_mixing_target_comparison(
    params: dict[str, Any],
    iso3: str,
    year: float = MIXING_TARGET_YEAR,
) -> go.Figure:
    """Modelled spline matrix vs spectral-normalised synthetic conmat."""
    modelled = _mixing_at(params, year)
    target = np.asarray(get_norm_conmat(iso3))
    distance = float(canberra_distance(modelled, target))
    diff = modelled - target
    vmax = max(float(modelled.max()), float(target.max()))
    dmax = float(np.max(np.abs(diff))) or 1.0
    ages = [str(a) for a in AGE_STRATA]
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            f"modelled, {int(year)}",
            "synthetic contact matrix",
            "modelled minus target",
        ),
    )
    panels = (
        (modelled, "Viridis", 0.0, vmax),
        (target, "Viridis", 0.0, vmax),
        (diff, "RdBu_r", -dmax, dmax),
    )
    for i, (data, scale, zmin, zmax) in enumerate(panels):
        fig.add_trace(
            go.Heatmap(
                z=data,
                x=ages,
                y=ages,
                colorscale=scale,
                zmin=zmin,
                zmax=zmax,
                showscale=(i == 2),
            ),
            row=1,
            col=i + 1,
        )
    fig.update_layout(
        title_text=f"mixing target comparison, Canberra distance {distance:.2f}",
        height=420,
        width=1100,
        template="plotly_white",
    )
    return fig


def plot_single_age_s_matrix_components(
    params: dict[str, Any],
    year: float = MIXING_TARGET_YEAR,
) -> go.Figure:
    """Assortative / background / parent-child / overall single-age kernels."""
    static = load_static()
    background, assortative, parent_child, overall = build_s_matrix_single_age_components(
        static.fert,
        static.fert_ends,
        year,
        float(params["bg_mixing"]),
        float(params["a_spread"]),
        float(params["pc_strength"]),
    )
    panels = (
        (assortative, "assortative"),
        (background, "background"),
        (parent_child, "parent-child"),
        (overall, "overall"),
    )
    fig = make_subplots(rows=2, cols=2, subplot_titles=[title for _, title in panels])
    for i, (matrix, title) in enumerate(panels):
        row, col = divmod(i, 2)
        fig.add_trace(
            go.Heatmap(z=np.asarray(matrix), colorscale="Viridis", zmin=0.0, showscale=(i == 3)),
            row=row + 1,
            col=col + 1,
        )
    fig.update_layout(
        title_text=f"single-age mixing kernel, {int(year)}",
        height=800,
        width=900,
        template="plotly_white",
    )
    return fig


def plot_mean_contacts_by_age(
    params: dict[str, Any],
    iso3: str,
    year: float = MIXING_TARGET_YEAR,
) -> go.Figure:
    """Row sums of modelled vs synthetic normalised matrices."""
    modelled = _mixing_at(params, year)
    target = np.asarray(get_norm_conmat(iso3))
    ages = [str(a) for a in AGE_STRATA]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ages, y=modelled.sum(axis=1), name="modelled"))
    fig.add_trace(go.Bar(x=ages, y=target.sum(axis=1), name="target"))
    fig.update_layout(
        barmode="group",
        title=f"mean contacts by age, {int(year)}",
        xaxis_title="age group",
        yaxis_title="mean contacts",
        template="plotly_white",
        height=450,
        width=800,
    )
    return fig


def plot_outputs_panels(
    prev: pd.DataFrame,
    inc: pd.DataFrame,
    notif: pd.DataFrame,
    notif_target: pd.Series,
    tb_death: pd.DataFrame,
    death_target: pd.Series | None,
    latent: pd.DataFrame,
    latent_target: pd.Series | None,
    total_pop: pd.DataFrame,
    plot_start: float,
    end_time: float,
    mode: str,
    pulm_prev: pd.DataFrame,
    pulm_prev_target: pd.Series | None,
    adult_pop: pd.DataFrame,
    prev_decline_target: pd.Series | None = None,
) -> go.Figure:
    """Count or rate panels matching summer3 ``plot_outputs``."""
    titles = COUNT_TITLES if mode == "count" else RATE_TITLES
    if mode == "count":
        data = {
            "prevalence": prev,
            "incidence": inc,
            "notifications": notif,
            "deaths": tb_death,
            "latent": latent,
            "pulm_prev": pulm_prev,
        }
    elif mode == "rate":
        data = {
            "prevalence": prev.div(total_pop, axis=0) * 1e5,
            "incidence": inc.div(total_pop, axis=0) * 1e5,
            "notifications": notif.div(total_pop, axis=0) * 1e5,
            "deaths": tb_death.div(total_pop, axis=0) * 1e5,
            "latent": latent.div(adult_pop, axis=0) * 1e2,
            "pulm_prev": pulm_prev.div(adult_pop, axis=0) * 1e5,
        }
    else:
        raise ValueError(f"Unknown mode {mode!r}")

    keys = ("prevalence", "incidence", "notifications", "deaths", "latent", "pulm_prev")
    fig = make_subplots(rows=3, cols=2, subplot_titles=[titles[k] for k in keys], shared_xaxes=True)
    for i, key in enumerate(keys):
        row, col = divmod(i, 2)
        frame = data[key]
        window = frame.loc[(frame.index > plot_start) & (frame.index <= end_time - 1)]
        for sample in window.columns:
            fig.add_trace(
                go.Scatter(
                    x=window.index,
                    y=window[sample],
                    mode="lines",
                    line=dict(width=1, color="steelblue"),
                    opacity=0.35,
                    showlegend=False,
                ),
                row=row + 1,
                col=col + 1,
            )
    if mode == "count" and notif_target is not None:
        fig.add_trace(
            go.Scatter(
                x=notif_target.index,
                y=notif_target.values,
                mode="markers",
                marker=dict(color="black", size=7),
                name="notif target",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        if death_target is not None:
            fig.add_trace(
                go.Scatter(
                    x=death_target.index,
                    y=death_target.values,
                    mode="markers",
                    marker=dict(color="black", size=7),
                    showlegend=False,
                ),
                row=2,
                col=2,
            )
    if mode == "rate":
        if latent_target is not None:
            fig.add_trace(
                go.Scatter(
                    x=latent_target.index,
                    y=latent_target.values,
                    mode="markers",
                    marker=dict(color="black", size=7),
                    showlegend=False,
                ),
                row=3,
                col=1,
            )
        if pulm_prev_target is not None:
            fig.add_trace(
                go.Scatter(
                    x=pulm_prev_target.index,
                    y=pulm_prev_target.values,
                    mode="markers",
                    marker=dict(color="black", size=7),
                    showlegend=False,
                ),
                row=3,
                col=2,
            )
            if prev_decline_target is not None and len(prev_decline_target) >= 2:
                later_year = float(pulm_prev_target.index[0])
                later_prev = float(pulm_prev_target.iloc[0])
                decline = prev_decline_target.sort_index()
                base_year = float(decline.index[0])
                implied = later_prev * float(decline.iloc[0]) / float(decline.iloc[1])
                fig.add_trace(
                    go.Scatter(
                        x=[later_year, base_year],
                        y=[later_prev, implied],
                        mode="lines+markers",
                        line=dict(color="black", width=1.5),
                        marker=dict(size=8, color="black", symbol=["circle", "circle-open"]),
                        showlegend=False,
                    ),
                    row=3,
                    col=2,
                )
    fig.update_layout(
        height=900,
        width=1000,
        template="plotly_white",
        title_text=f"Calibration outputs ({mode})",
    )
    fig.update_yaxes(rangemode="tozero")
    return fig


def plot_prior_post(idata: Any, var_names: list[str], n_cols: int = 3) -> go.Figure:
    """Posterior histograms with uniform prior bands from ``PARAM_BOUNDS``."""
    n = len(var_names)
    n_rows = int(np.ceil(n / n_cols))
    titles = [(PARAM_NAMES | PARAM_ABBREVS).get(name, name) for name in var_names]
    fig = make_subplots(rows=n_rows, cols=n_cols, subplot_titles=titles)
    posterior = idata.posterior
    dataset = posterior.dataset if hasattr(posterior, "dataset") else posterior
    for i, name in enumerate(var_names):
        row, col = divmod(i, n_cols)
        samples = np.asarray(dataset[name]).ravel()
        low, high = PARAM_BOUNDS[name]
        fig.add_trace(
            go.Histogram(
                x=samples, histnorm="probability density", name=name, showlegend=False, opacity=0.7
            ),
            row=row + 1,
            col=col + 1,
        )
        height = 1.0 / max(high - low, 1e-12)
        fig.add_trace(
            go.Scatter(
                x=[low, high, high, low, low],
                y=[0.0, 0.0, height, height, 0.0],
                fill="toself",
                fillcolor="rgba(0,0,0,0.15)",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row + 1,
            col=col + 1,
        )
    fig.update_layout(
        height=280 * n_rows, width=1000, template="plotly_white", title_text="Prior vs posterior"
    )
    return fig


def load_group_popsize(iso3: str) -> pd.DataFrame:
    """UN age-group population used as the inspect-run target."""
    group_popsize, _death_rates, _age_weights = load_demography(iso3)
    return group_popsize
