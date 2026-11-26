"""Beamer slides on the transmission model, its calibration and epidemiological projections.

:func:`write_slides` writes a ``.tex`` deck and its figures for a stored run. The deck is
meant to be compiled in the folder of the earlier Kiribati talk, which provides the
``Unipd`` theme and the model diagrams it reuses (``images/TB Model.pdf``,
``images/ICE_TB.png``, ``images/Classif.pdf``). Figures made here go under
``images_2/``. Every number on the slides is read from the run or from
:mod:`vacf.economics.study`, so rebuilding against a new run updates the deck.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from vacf.calibration import observed_targets
from vacf.constants import AGE_STRATA
from vacf.economics import analysis, study
from vacf.parameters import BASE_PARAMS, HOMOGENEOUS_PARAM_BOUNDS
from vacf.targets import PREV_DECLINE_TARGET

FIGURE_DIR = Path("images_2")
TEX_NAME = "vietnam_model_slides.tex"

ARM_COLOURS = {"baseline": "#c0392b", "acf_xpert": "#27ae60", "acf_cxr_xpert": "#2e6fb5"}
SHORT_LABELS = {"baseline": "No ACF", "acf_xpert": "Xpert", "acf_cxr_xpert": "CXR +\nXpert"}
STYLE = {
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# (column, title, y-axis label, display scale)
FIT_PANELS = [
    ("notifications", "TB notifications", "Per year", 1.0),
    ("deaths", "TB deaths", "Per year", 1.0),
    ("latent", "Ever infected, adults", "%", 100.0),
    ("prevalence", "Pulmonary TB prevalence, adults", "Per 100,000", 1e5),
    ("decline", "Prevalence relative to {base} survey", "Ratio", 1.0),
    ("infectiousness", "More-infectious share of prevalent TB", "%", 100.0),
]
IMPACT_PANELS = [
    ("incidence", "TB incidence (/100,000/year)"),
    ("deaths", "TB mortality (/100,000/year)"),
    ("prevalence", "TB prevalence, incl. on treatment (/100,000)"),
    ("notifications", "TB notifications (/100,000/year)"),
]


# ---------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------


def _bands(table: pd.DataFrame, column: str, by: list[str]) -> pd.DataFrame:
    """Median, interquartile range and central 95% interval across draws."""
    wide = analysis.quantile_bands(table, column, by, level=0.95)
    iqr = analysis.quantile_bands(table, column, by, level=0.5)
    return pd.DataFrame(
        {
            "median": wide["median"],
            "iqr_low": iqr["lower"],
            "iqr_high": iqr["upper"],
            "ci_low": wide["lower"],
            "ci_high": wide["upper"],
        }
    )


def _plot_bands(ax: Any, bands: pd.DataFrame, colour: str, label: str | None = None) -> None:
    x = bands.index
    ax.fill_between(x, bands["ci_low"], bands["ci_high"], color=colour, alpha=0.15, lw=0)
    ax.fill_between(x, bands["iqr_low"], bands["iqr_high"], color=colour, alpha=0.35, lw=0)
    ax.plot(x, bands["median"], color=colour, lw=1.3, label=label)


def _legend_handles(colour: str) -> list[Any]:
    return [
        Line2D([], [], marker="o", color="black", linestyle="none", markersize=3.5),
        Line2D([], [], color=colour, lw=1.3),
        Patch(color=colour, alpha=0.5, lw=0),
        Patch(color=colour, alpha=0.2, lw=0),
    ]


def fit_figure(posterior: dict[str, Any], start: int = 2000, end: int = 2025) -> plt.Figure:
    """Model against each calibration target: median, IQR and 95% interval across draws."""
    observed = observed_targets()
    fit = analysis.with_prevalence_decline(
        posterior["calibration_fit"], base_time=observed["decline"].index[0]
    )
    fit = fit[fit["year"].between(start, end)]
    colour = ARM_COLOURS["baseline"]
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(7.6, 3.9), sharex=True)
        for ax, (name, title, ylabel, scale) in zip(axes.flat, FIT_PANELS):
            _plot_bands(ax, _bands(fit, name, ["year"]) * scale, colour)
            target = observed[name].loc[start:end]
            ax.scatter(target.index, target * scale, color="black", s=9, zorder=3)
            ax.set_title(title.format(base=int(observed["decline"].index[0])))
            ax.set_ylabel(ylabel)
            ax.set_ylim(bottom=0)
        for ax in axes[-1]:
            ax.set_xlabel("Year")
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.legend(
            _legend_handles(colour),
            ["Observed", "Model (median)", "Model (IQR)", "Model (95% CrI)"],
            frameon=False,
            loc="lower center",
            ncol=4,
        )
    return fig


def _rates(epi: pd.DataFrame) -> pd.DataFrame:
    """Per-100,000 rates, computed within each draw before summarising."""
    out = epi.copy()
    for column, _ in IMPACT_PANELS:
        out[column] = epi[column] / epi["population"] * 1e5
    return out


def impact_figure(
    posterior: dict[str, Any], arm: str, start: int = 2015, end: int | None = None
) -> plt.Figure:
    """One campaign against no ACF, as rates per 100,000, with the campaign shaded."""
    end = end or study.HORIZON[1]
    epi = posterior["epi"]
    epi = _rates(epi[epi["year"].between(start, end)])
    labels = {a.name: a.label for a in study.SCENARIOS}
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.0), sharex=True)
        for ax, (column, title) in zip(axes.flat, IMPACT_PANELS):
            for name in (study.BASELINE.name, arm):
                bands = _bands(epi[epi["arm"] == name], column, ["year"])
                _plot_bands(ax, bands, ARM_COLOURS[name], labels[name])
            ax.axvspan(
                study.ACF_START, study.ACF_START + study.ACF_DURATION, color="grey", alpha=0.1, lw=0
            )
            ax.set_title(title)
            ax.set_ylim(bottom=0)
        for ax in axes[-1]:
            ax.set_xlabel("Year")
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            [f"{label}: median, IQR, 95% CrI" for label in labels],
            frameon=False,
            loc="lower center",
            ncol=2,
        )
    return fig


def cumulative_figure(posterior: dict[str, Any]) -> plt.Figure:
    """Cumulative TB incidence and deaths over the horizon: box IQR, whiskers 95% interval."""
    summary = posterior["epi_summary"]
    arms = [a.name for a in study.SCENARIOS]
    start, end = study.HORIZON
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(4.6, 3.2))
        for ax, (column, title) in zip(
            axes, [("cum_incidence", "TB cases"), ("cum_deaths", "TB deaths")]
        ):
            data = [summary.loc[summary["arm"] == a, column].to_numpy() / 1e3 for a in arms]
            boxes = ax.boxplot(
                data,
                whis=(2.5, 97.5),
                showfliers=False,
                patch_artist=True,
                widths=0.6,
                medianprops={"color": "black", "lw": 1.2},
            )
            for patch, arm in zip(boxes["boxes"], arms):
                patch.set_facecolor(ARM_COLOURS[arm])
                patch.set_alpha(0.6)
            ax.set_xticks(range(1, len(arms) + 1), [SHORT_LABELS[a] for a in arms])
            ax.set_title(f"{title}, {start}–{end}")
            ax.set_ylabel("Thousands")
            ax.set_ylim(bottom=0)
        fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------------------
# Numbers quoted on the slides
# ---------------------------------------------------------------------------------------


def _interval(values: pd.Series, scale: float = 1.0, unit: str = "") -> str:
    """``median (2.5th--97.5th percentile)`` with thousands separators, for LaTeX."""
    low, mid, high = (values.quantile(q) / scale for q in (0.025, 0.5, 0.975))
    return f"{mid:,.0f}{unit} ({low:,.0f}--{high:,.0f}{unit})"


def impact_rows(posterior: dict[str, Any]) -> str:
    """Table rows of cases and deaths averted per campaign, absolute and relative."""
    summary = posterior["epi_summary"]
    rows = []
    for arm in study.SCENARIOS[1:]:
        g = summary[summary["arm"] == arm.name]
        cases = _interval(g["cum_incidence_averted"], 1e3, "k")
        cases_pct = _interval(g["cum_incidence_averted_pct"], unit=r"\%")
        deaths = _interval(g["cum_deaths_averted"], 1e3, "k")
        deaths_pct = _interval(g["cum_deaths_averted_pct"], unit=r"\%")
        rows.append(
            f"{arm.label} & \\begin{{tabular}}[t]{{@{{}}c@{{}}}}{cases}\\\\{cases_pct}\\end{{tabular}}"
            f" & \\begin{{tabular}}[t]{{@{{}}c@{{}}}}{deaths}\\\\{deaths_pct}\\end{{tabular}} \\\\[1.2em]"
        )
    return "\n".join(rows)


def sensitivity_rows() -> str:
    """Screening sensitivity by algorithm and disease state, as table rows."""
    rows = []
    for label, values in study.sensitivity_table().items():
        cells = " & ".join(f"{values[state]:.2g}" for state in study.DISEASE_STATES)
        rows.append(f"{label} & {cells} \\\\")
    return "\n".join(rows)


def _age_groups() -> str:
    lowers = list(AGE_STRATA)
    groups = [f"{lo}--{hi - 1}" for lo, hi in zip(lowers[:-1], lowers[1:])]
    return ", ".join(groups + [f"{lowers[-1]}+"])


def _years(series: pd.Series) -> str:
    return f"{int(series.index.min())}--{int(series.index.max())}"


# ---------------------------------------------------------------------------------------
# Deck
# ---------------------------------------------------------------------------------------

TEMPLATE = r"""\documentclass[aspectratio=169]{beamer}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[english]{babel}
\usepackage{lmodern}
\usepackage{graphicx}
\usepackage{booktabs}

\usetheme{Unipd}

\title[<<SHORT_TITLE>>]{<<TITLE>>}
\author[<<SHORT_AUTHOR>>]{<<AUTHOR>>}
\institute{<<INSTITUTE>>}
\date{<<DATE>>}

\begin{document}

\begin{frame}
  \titlepage
\end{frame}

% =======================================================================================
% Generated by vacf.economics.slides from run "<<RUN>>". Frames below can be copied into a
% combined deck; figures are under <<FIG_DIR>>/.
% =======================================================================================

\section{Model}
\begin{frame}{Model structure}
  \centering
  \includegraphics[width=.72\linewidth,trim=0 10 0 0,clip]{images/TB Model.pdf}

  \raggedright\scriptsize
  Not shown:
  \begin{itemize}
    \item Births and deaths, from UN population estimates and projections for Viet Nam.
    \item Reinfection may occur from the \textit{Contained}, \textit{Cleared} and \textit{Recovered} compartments.
    \item Self-clearance possible from subclinical TB.
    \item Stratified into <<N_AGES>> age groups (<<AGE_GROUPS>> years), mixing homogeneously.
  \end{itemize}
\end{frame}

\begin{frame}{TB classification framework}
\begin{columns}
    \column{0.5\textwidth}
        \includegraphics[width=\linewidth]{images/ICE_TB.png}

        \tiny
        Coussens A.K. et al. Classification of early tuberculosis states to guide research for improved care and prevention: an international Delphi consensus exercise. \textit{Lancet Respir Med.} 2024.

    \column{0.5\textwidth}
        \includegraphics[width=.9\linewidth]{images/Classif.pdf}
\end{columns}
\end{frame}

\begin{frame}{Main model assumptions}
\small
\begin{columns}[T]
  \column{0.5\textwidth}
    \textbf{Transmission and natural history}
    \begin{itemize}\setlength{\itemsep}{.3em}
      \item Frequency-dependent transmission, homogeneous mixing between age groups
      \item Partial protection against reinfection after containment, clearance or recovery (estimated)
      \item Age-specific progression and containment rates, as in the Kiribati model
      \item <<PCT_HIGH>>\% of incident TB starts as more infectious
      \item Asymptomatic TB <<REL_SUBCLIN>>\% as infectious as symptomatic; less-infectious TB <<REL_LOW>>\% as infectious as more-infectious
      \item TB mortality from symptomatic TB only
    \end{itemize}
  \column{0.5\textwidth}
    \textbf{Care cascade}
    \begin{itemize}\setlength{\itemsep}{.3em}
      \item Passive detection of symptomatic TB only, scaled up from 1957 to current levels by 2014
      \item Reduced detection during COVID-19 (2021)
      \item Treatment outcomes from WHO treatment-outcome data
      \item Active case finding reaches TB in all four disease states, according to the sensitivity of the screening algorithm
    \end{itemize}
\end{columns}
\end{frame}

\section{Calibration}
\begin{frame}{Calibration}
\footnotesize
Bayesian calibration (No-U-Turn Sampler) of <<N_PARAMS>> parameters: transmission rate, relative susceptibility (3), infection clearance and breakdown, progression to symptoms, gain of infectiousness, passive detection (4).

\vspace{.3em}
\centering\scriptsize
\begin{tabular}{lll}
  \toprule
  Target & Value & Source \\
  \midrule
  TB notifications, <<NOTIF_YEARS>> & <<NOTIF_RANGE>> per year & National TB Programme \\
  TB deaths, <<DEATH_YEARS>> & WHO estimates & WHO Global TB Programme \\
  Ever infected, adults, <<LATENT_YEAR>> & <<LATENT>>\% & Marks et al., IJTLD \\
  Pulmonary TB prevalence, adults, <<PREV_YEAR>> & <<PREV>> per 100,000 & Second national prevalence survey \\
  More-infectious share of prevalent TB, <<PREV_YEAR>> & <<INF_SHARE>>\% & Second national prevalence survey \\
  Prevalence decline, <<DECLINE_YEARS>> & <<DECLINE>> per 100,000 & Nguyen et al., EID \\
  \bottomrule
\end{tabular}

\vspace{.3em}
\raggedright
Results use <<N_DRAWS>> posterior draws.
\end{frame}

\begin{frame}{Model fits to observations}
\begin{center}
  \includegraphics[width=.84\textwidth]{<<FIG_DIR>>/fits.pdf}
\end{center}
\end{frame}

\section{Projections}
\begin{frame}{Scenarios}
\small
\begin{itemize}\setlength{\itemsep}{.5em}
  \item \textbf{No ACF}: passive case finding continues at current levels
  \item \textbf{Active case finding, <<ACF_START>>--<<ACF_END>>}: each year, <<COVERAGE>>\% of people aged $\geq$<<MIN_AGE>> years are screened
  \begin{itemize}
    \item Xpert front-line
    \item CXR front-line, Xpert confirmation of CXR-positive
  \end{itemize}
\end{itemize}

\vspace{.5em}
\centering\scriptsize
Sensitivity by disease state\\[.3em]
\begin{tabular}{lcccc}
  \toprule
  & \multicolumn{2}{c}{Symptomatic} & \multicolumn{2}{c}{Asymptomatic} \\
  \cmidrule(lr){2-3}\cmidrule(lr){4-5}
  Algorithm & less inf. & more inf. & less inf. & more inf. \\
  \midrule
<<SENSITIVITY_ROWS>>
  \bottomrule
\end{tabular}

\vspace{.5em}
\raggedright\tiny
Symptomatic TB is detected with certainty because symptom screening is the first step. Asymptomatic CXR values lie within the ranges estimated in Kiribati (PEARL).
\end{frame}

\begin{frame}{Projected impact: Xpert front-line}
\begin{center}
  \includegraphics[width=.72\textwidth]{<<FIG_DIR>>/impact_acf_xpert.pdf}
\end{center}
\end{frame}

\begin{frame}{Projected impact: CXR front-line, Xpert confirmation}
\begin{center}
  \includegraphics[width=.72\textwidth]{<<FIG_DIR>>/impact_acf_cxr_xpert.pdf}
\end{center}
\end{frame}

\begin{frame}{Cumulative impact, <<HORIZON>>}
\begin{columns}[c]
  \column{0.45\textwidth}
    \includegraphics[width=\linewidth]{<<FIG_DIR>>/cumulative.pdf}

    \tiny Boxes: interquartile range; whiskers: 95\% credible interval.
  \column{0.55\textwidth}
    \centering\scriptsize
    Averted compared with no ACF\\[.3em]
    \resizebox{\linewidth}{!}{%
    \begin{tabular}{@{}lcc@{}}
      \toprule
      & TB cases & TB deaths \\
      \midrule
<<IMPACT_ROWS>>
      \bottomrule
    \end{tabular}}

    \vspace{.3em}
    \tiny Median (95\% credible interval)
\end{columns}
\end{frame}

\end{document}
"""


def write_slides(
    posterior: dict[str, Any],
    directory: Path | str,
    *,
    title: str,
    short_title: str,
    author: str,
    short_author: str,
    institute: str,
    date: str,
    run_name: str = "",
) -> Path:
    """Write the deck and its figures; returns the path of the ``.tex`` file."""
    directory = Path(directory)
    figures = directory / FIGURE_DIR
    figures.mkdir(parents=True, exist_ok=True)

    made = {"fits": fit_figure(posterior), "cumulative": cumulative_figure(posterior)}
    for arm in study.SCENARIOS[1:]:
        made[f"impact_{arm.name}"] = impact_figure(posterior, arm.name)
    for name, fig in made.items():
        fig.savefig(figures / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)

    observed = observed_targets()
    decline = observed["decline"]
    values = {
        "TITLE": title,
        "SHORT_TITLE": short_title,
        "AUTHOR": author,
        "SHORT_AUTHOR": short_author,
        "INSTITUTE": institute,
        "DATE": date,
        "RUN": run_name,
        "FIG_DIR": FIGURE_DIR.as_posix(),
        "N_AGES": str(len(AGE_STRATA)),
        "AGE_GROUPS": _age_groups(),
        "PCT_HIGH": f"{100 * BASE_PARAMS['progression_prop_infectious']:.0f}",
        "REL_SUBCLIN": f"{100 * BASE_PARAMS['rel_infectiousness_subclin']:.0f}",
        "REL_LOW": f"{100 * BASE_PARAMS['rel_infectiousness_lowinf']:.0f}",
        "N_PARAMS": str(len(HOMOGENEOUS_PARAM_BOUNDS)),
        "NOTIF_YEARS": _years(observed["notifications"]),
        "NOTIF_RANGE": (
            f"{observed['notifications'].min():,.0f}--{observed['notifications'].max():,.0f}"
        ),
        "DEATH_YEARS": _years(observed["deaths"]),
        "LATENT_YEAR": f"{int(observed['latent'].index[0])}",
        "LATENT": f"{100 * observed['latent'].iloc[0]:.1f}",
        "PREV_YEAR": f"{int(observed['prevalence'].index[0])}",
        "PREV": f"{1e5 * observed['prevalence'].iloc[0]:.0f}",
        "INF_SHARE": f"{100 * observed['infectiousness'].iloc[0]:.1f}",
        "DECLINE_YEARS": _years(decline),
        "DECLINE": r" $\rightarrow$ ".join(f"{v:.0f}" for v in PREV_DECLINE_TARGET.sort_index()),
        "N_DRAWS": str(len(posterior["draws"])),
        "ACF_START": f"{study.ACF_START:.0f}",
        "ACF_END": f"{study.ACF_START + study.ACF_DURATION:.0f}",
        "COVERAGE": f"{100 * study.ACF_COVERAGE:.0f}",
        "MIN_AGE": f"{study.ACF_MIN_AGE:.0f}",
        "SENSITIVITY_ROWS": sensitivity_rows(),
        "HORIZON": f"{study.HORIZON[0]}--{study.HORIZON[1]}",
        "IMPACT_ROWS": impact_rows(posterior),
    }
    tex = TEMPLATE
    for key, value in values.items():
        tex = tex.replace(f"<<{key}>>", value)
    if "<<" in tex:
        missing = sorted(set(part.split(">>")[0] for part in tex.split("<<")[1:]))
        raise ValueError(f"Unfilled slide fields: {missing}")
    path = directory / TEX_NAME
    path.write_text(tex)
    return path
