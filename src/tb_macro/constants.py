"""Shared constants for the three-age TB macro model."""

from __future__ import annotations

DISEASE_STATES: tuple[str, ...] = (
    "mtb_naive",
    "incipient",
    "contained",
    "cleared",
    "active",
    "treatment",
    "recovered",
)
INFECT_COMPS: tuple[str, ...] = ("mtb_naive", "contained", "cleared", "recovered")
AGES: tuple[str, ...] = ("0", "5", "15")
INFECTIOUS_STRATA: tuple[str, ...] = ("low", "high")
CLINICAL_STRATA: tuple[str, ...] = ("subclin", "clin")

# Calendar years. summer3 integrates index time 0 … 199; year = 1800 + index.
START_YEAR: float = 1800.0
END_YEAR: float = 1999.0
N_TIMES: int = 200
POP_PER_AGE: float = 1000.0

# Notebook parameters. Time functions see the index, so seed_peak_time 30 is
# year 1830 and passive_detection_inflection 40 is year 1840.
BASE_PARAMS: dict[str, float] = {
    "contact_rate": 0.8,
    "freq_dens_exponent": 0.5392614891179714,
    "rel_sus_mtb_naive": 1.0,
    "rel_sus_contained": 0.3324848208129065,
    "rel_sus_cleared": 0.7101012998861186,
    "rel_sus_recovered": 0.7101012998861186,
    "contain": 2.0,
    "clearance_rate": 0.05643858771640714,
    "breakdown_rate": 0.5678500778834155,
    "progression": 0.1,
    "increase_infect": 2.799432998282645,
    "decrease_infect": 1.0,
    "clinical_development": 2.280456422265371,
    "clinical_regression": 1.0,
    "self_recovery": 0.4,
    "detection": 3.1435769682295085,
    "treatment_recovery": 0.5,
    "treatment_relapse": 0.0,
    "seed_peak_time": 30.0,
    "seed_peak_rate": 0.01,
    "seed_duration": 10.0,
    "recent_detection_rate": 0.7,
    "passive_detection_shape": 0.12804084477668393,
    "passive_detection_inflection": 40.0,
    "passive_detection_past_frac": 0.7117933876877114,
}

# Turns the flows that a zero-heavy base would leave dark. Applied on top of
# BASE_PARAMS; the keys are exactly those named in the port plan.
REINFECTION_OVERRIDES: dict[str, float] = {
    "rel_sus_mtb_naive": 0.5,
    "rel_sus_contained": 0.5,
    "rel_sus_cleared": 0.5,
    "rel_sus_recovered": 0.5,
    "contain": 0.1,
    "clearance_rate": 0.1,
    "breakdown_rate": 0.1,
    "freq_dens_exponent": 0.7,
    "detection": 0.3,
    "treatment_relapse": 0.05,
}

FIXTURES: dict[str, dict[str, float]] = {
    "base": BASE_PARAMS,
    "reinfection": BASE_PARAMS | REINFECTION_OVERRIDES,
}

# Rolling-sum positions the original notebook queries: [7:60:7].
TARGET_SLICE: slice = slice(7, 60, 7)
CALIBRATION_CONTACT_RATE: float = 0.25
