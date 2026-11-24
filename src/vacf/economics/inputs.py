"""Loading and sampling of economic input parameters.

Unit costs and DALY weights live in ``data/economics/cost_inputs_vnm.xlsx``, on the same
principle as model parameters: the spreadsheet is the single source of truth and the code
never hard-codes a value.

Sheet layout, as supplied for Vietnam:

``Sheet3`` — one row per cost or DALY component, with columns ``category``, ``parameter``,
``applies_to``, ``Frequency``, ``value (2025 USD)``, ``Uncertainty``, ``Annual discounting``,
``source``, ``notes``.

``Sheet2`` — age-specific years of life lost and post-TB DALYs, by five-year age band.
"""

from dataclasses import dataclass
from pathlib import Path

from vacf.constants import DATA_PATH

import numpy as np
import pandas as pd



#: Derived from the model's own data root rather than by counting directory levels, so
#: moving this module cannot silently break the path.
COST_INPUTS: Path = DATA_PATH / "economics" / "cost_inputs_vnm.xlsx"

COST_SHEET = "Sheet3"
AGE_DALY_SHEET = "Sheet2"

VALUE_COL = "value (2025 USD)"


@dataclass(frozen=True)
class UncertainQuantity:
    """A point estimate with a distribution for probabilistic sensitivity analysis.

    ``kind`` is ``"gamma"`` for costs (non-negative, right-skewed) and ``"beta"`` for
    proportions such as disability weights. Both are parameterised by moment matching from
    the mean and standard deviation, so the spreadsheet only ever has to state those.
    """

    name: str
    mean: float
    sd: float
    kind: str = "gamma"

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        if self.sd == 0:
            return np.full(size, self.mean)
        if self.kind == "gamma":
            shape = (self.mean / self.sd) ** 2
            scale = self.sd**2 / self.mean
            return rng.gamma(shape, scale, size)
        if self.kind == "beta":
            variance = self.sd**2
            spread = self.mean * (1.0 - self.mean) / variance - 1.0
            if spread <= 0:
                raise ValueError(
                    f"{self.name}: sd {self.sd} is too large for a beta distribution with "
                    f"mean {self.mean}"
                )
            return rng.beta(self.mean * spread, (1.0 - self.mean) * spread, size)
        raise ValueError(f"Unsupported distribution kind '{self.kind}' for {self.name}")

    def fixed(self) -> float:
        return self.mean


def sd_from_ci(low: float, high: float) -> float:
    """Standard deviation implied by a 95% interval, assuming approximate normality."""
    return (high - low) / (2.0 * 1.96)


def load_economic_inputs(path: Path = COST_INPUTS) -> pd.DataFrame:
    """Read the raw cost and DALY component table."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No economic inputs at {path}. Economic analysis requires the unit cost "
            "spreadsheet."
        )
    df = pd.read_excel(path, sheet_name=COST_SHEET)
    return df.where(pd.notna(df), None)


def load_age_daly_table(path: Path = COST_INPUTS) -> pd.DataFrame:
    """Read age-specific YLLs and post-TB DALYs, indexed by age-band lower bound.

    The source table is labelled with text bands such as ``"25-29 years"`` and
    ``"85+ years"``. These are parsed to integer lower bounds so they can be remapped onto
    whatever age groups the model uses.
    """
    raw = pd.read_excel(Path(path), sheet_name=AGE_DALY_SHEET, index_col=0)
    raw.columns = [str(c).strip() for c in raw.columns]

    lower_bounds = [_parse_age_band(str(label)) for label in raw.index]
    table = pd.DataFrame(
        {
            "yll": raw.iloc[:, 0].astype(float).values,
            "post_tb_daly": raw.iloc[:, 1].astype(float).values,
        },
        index=pd.Index(lower_bounds, name="age_lower"),
    ).sort_index()

    if table.index.min() != 0:
        raise ValueError(
            f"Age-specific DALY table starts at age {table.index.min()}, "
            "but must cover from birth."
        )
    return table


def _parse_age_band(label: str) -> int:
    """Lower bound of an age band label such as '25-29 years' or '85+ years'."""
    cleaned = label.strip().lower().replace("years of age", "").replace("years", "").strip()
    if cleaned.endswith("+"):
        return int(cleaned[:-1])
    if "-" in cleaned:
        return int(cleaned.split("-")[0])
    raise ValueError(f"Could not parse age band label '{label}'")
