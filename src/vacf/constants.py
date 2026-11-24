"""Vietnam TB model constants from ``tweak_cases_calib`` (``9cc8cf8``)."""

from __future__ import annotations

from pathlib import Path

# Compartments for TB model. Order matches the summer3 stratifications.
DISEASE_STATES: tuple[str, ...] = (
    "mtb_naive",
    "incipient",
    "contained",
    "cleared",
    "active",
    "treatment",
    "recovered",
)
INFECT_COMPS: tuple[str, ...] = (
    "mtb_naive",
    "contained",
    "cleared",
    "recovered",
)
# ``wrap`` indexes these in order. Do not sort them.
INF_STRATA: tuple[str, ...] = ("low", "high")
CLIN_STRATA: tuple[str, ...] = ("clin", "subclin")
PREV_STATES: tuple[str, ...] = ("active", "treatment")
INFECTED_STATES: tuple[str, ...] = tuple(s for s in DISEASE_STATES if s != "mtb_naive")

# List, not tuple: the copied input loaders do ``AGE_STRATA + [MAX_AGE]``.
AGE_STRATA: list[int] = [0, 3, 5, 10, 15, 18, 40, 65]
AGES: tuple[str, ...] = tuple(str(age) for age in AGE_STRATA)
MAX_AGE: int = 120
YOUNG_END_AGE: int = 15
#: Age from which people are eligible for active case finding in this analysis.
ACF_MIN_AGE: int = 15
#: What the summer3 reference and the forward-run goldens used. Pinned in the ``acf``
#: fixture so those goldens keep reproducing after the eligibility floor changed.
ACF_MIN_AGE_SUMMER3: int = 10
TOP_AGE_BRACKET_INFLATION: float = 2.0
CANBERRA_EPS: float = 1e-10

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_PATH: Path = REPO_ROOT / "data"
# Only used by the unused conmat helpers copied with the loaders.
BASE_PATH: Path = REPO_ROOT

ISO3: str = "VNM"
START_TIME: float = 1800.0
END_TIME: float = 2101.0
CALENDAR_YEAR_MIDPOINT: float = 0.5
OUTPUT_TIME_STEP: float = 1.0
# ``arange(1800, 2101)`` has 301 points. summer3 integrates index ``0 … 300``.
N_TIMES: int = 301
# Occupied compartments agree to a few parts per million. The absolute floor
# sits above pre-seed roundoff (~1e-9); forward goldens leave the step uncapped.
PARITY_RTOL: float = 1e-5
PARITY_ATOL: float = 1e-8
POP_PER_AGE: float = 1000.0

# Remote / local calibration artefacts. Jobs write under ``OUTPUT_PATH/<task>/``.
OUTPUT_PATH: Path = REPO_ROOT / "outputs"
LOCAL_OUTPUT_PATH: Path = OUTPUT_PATH / "local"
N_CHAINS_REMOTE: int = 8
N_RUNS_LOCAL: int = 40
N_OUTPUT_SAMPLES: int = 40
SCENARIO_PARAMS: list[dict[str, float]] = [{}, {"acf_coverage": 0.8}]

# Flows whose age-summed instantaneous rates are golden columns.
FLOW_TOTALS: tuple[str, ...] = (
    "progression",
    "detection",
    "acf",
    "tb_mortality",
    "rx_death",
    "entry_births",
    "success",
    "relapse",
    "seed_peak",
    *(f"infect_{comp}" for comp in INFECT_COMPS),
    *(f"nontb_deaths_{age}" for age in AGES),
)
