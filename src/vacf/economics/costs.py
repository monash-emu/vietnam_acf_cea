"""Costing of model outputs.

Each cost component pairs a unit cost with the derived output giving the number of units
occurring each year. Total cost in a year is the product, summed across components; the
stream is then discounted.

The ``applies_to`` column of the input spreadsheet describes in prose which scenario a unit
cost belongs to. That mapping is deliberately *not* parsed automatically — free-text parsing
would fail silently and misattribute costs. Instead a project declares its cost structure
explicitly by referencing spreadsheet rows, so that what is being costed is readable in the
project code and reviewable alongside the methods.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vacf.economics.discounting import DEFAULT_DISCOUNT_RATE, discount_series
from vacf.economics.inputs import UncertainQuantity, load_economic_inputs, VALUE_COL


@dataclass(frozen=True)
class CostComponent:
    """A unit cost applied to a quantity produced by the model.

    Parameters
    ----------
    name
        Label for this component in the output cost table.
    unit_cost
        Cost per unit, with its distribution for probabilistic sensitivity analysis.
    quantity_output
        Name of the derived output giving units per year, e.g. ``n_screened_cxr_xpert``.
    perspective
        ``"provider"`` or ``"patient"``. Keeping these separable allows provider-only and
        societal results to be produced from a single run.
    """

    name: str
    unit_cost: UncertainQuantity
    quantity_output: str
    perspective: str = "provider"


@dataclass
class CostStructure:
    """All cost components applying to one scenario."""

    scenario: str
    components: list = field(default_factory=list)

    def quantity_outputs(self) -> list[str]:
        return [c.quantity_output for c in self.components]

    def validate(self, outputs_df: pd.DataFrame) -> None:
        missing = [c.quantity_output for c in self.components if c.quantity_output not in outputs_df]
        if missing:
            raise KeyError(
                f"Scenario '{self.scenario}' costs reference derived outputs that do not "
                f"exist: {missing}"
            )


def cost_items_from_spreadsheet(sd_fraction: float = 0.20) -> dict:
    """Unit costs from the spreadsheet, keyed by ``(parameter, applies_to)``.

    Parameters
    ----------
    sd_fraction
        Standard deviation as a fraction of the mean, for the gamma distributions used in
        probabilistic sensitivity analysis. The supplied Vietnam spreadsheet states 20% for
        every cost row, so it is a single argument rather than a per-row column; if that
        ever varies by row, read it from the ``Uncertainty`` column instead.
    """
    df = load_economic_inputs()
    items = {}
    for _, row in df.iterrows():
        value = row[VALUE_COL]
        if not isinstance(value, (int, float)):
            # Age-specific DALY rows carry prose here; they are handled by the DALY module.
            continue
        key = (str(row["parameter"]).strip(), str(row["applies_to"]).strip())
        items[key] = UncertainQuantity(
            name=f"{row['parameter']} [{row['applies_to']}]",
            mean=float(value),
            sd=sd_fraction * float(value),
            kind="gamma",
        )
    if not items:
        raise ValueError("No numeric unit costs found in the economic inputs")
    return items


def compute_cost_streams(
    outputs_df: pd.DataFrame,
    structure: CostStructure,
    unit_costs: dict = None,
) -> pd.DataFrame:
    """Undiscounted cost per year, one column per component plus totals.

    Parameters
    ----------
    unit_costs
        Optional override mapping component name to unit cost, used to inject a single
        probabilistic sensitivity analysis draw. When omitted, point estimates are used.
    """
    structure.validate(outputs_df)
    streams = {}
    for component in structure.components:
        unit = (
            unit_costs[component.name]
            if unit_costs is not None
            else component.unit_cost.fixed()
        )
        streams[component.name] = outputs_df[component.quantity_output] * unit

    table = pd.DataFrame(streams, index=outputs_df.index)
    for perspective in {c.perspective for c in structure.components}:
        cols = [c.name for c in structure.components if c.perspective == perspective]
        table[f"total_{perspective}"] = table[cols].sum(axis=1)
    table["total"] = table[[c.name for c in structure.components]].sum(axis=1)
    return table


def total_costs(
    outputs_df: pd.DataFrame,
    structure: CostStructure,
    unit_costs: dict = None,
    discount_rate: float = DEFAULT_DISCOUNT_RATE,
    reference_year: int = None,
    start_year: int = None,
    end_year: int = None,
) -> pd.Series:
    """Discounted total cost by component over the analytic horizon."""
    streams = compute_cost_streams(outputs_df, structure, unit_costs)
    if start_year is not None:
        streams = streams[streams.index >= start_year]
    if end_year is not None:
        streams = streams[streams.index <= end_year]
    if reference_year is None:
        reference_year = float(streams.index.min())
    return streams.apply(lambda col: np.sum(discount_series(col, discount_rate, reference_year)))


def sample_unit_costs(structure: CostStructure, n_draws: int, rng: np.random.Generator) -> list:
    """Draw ``n_draws`` sets of unit costs for probabilistic sensitivity analysis.

    Each draw is a dict suitable for passing as ``unit_costs``. Components sharing a unit
    cost are sampled independently here; if correlated uncertainty is needed, sample the
    shared quantity once and map it onto both components.
    """
    draws = {c.name: c.unit_cost.sample(n_draws, rng) for c in structure.components}
    return [{name: values[i] for name, values in draws.items()} for i in range(n_draws)]
