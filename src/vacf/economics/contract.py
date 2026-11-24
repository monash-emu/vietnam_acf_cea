"""The contract between the model and the economic layer.

Economic analysis is post-processing: it reads solved model output and never touches the
model. That only works if both sides agree on which quantities exist and what they mean.
This module is that agreement, stated once.

``vacf.economics.indicators`` produces these columns; ``vacf.economics.costs`` and ``vacf.economics.dalys`` consume
them. A mismatch surfaces here as a clear error rather than as a silent zero in a cost
total.
"""

from __future__ import annotations

import pandas as pd

#: People screened per year. Campaign throughput times the eligible population — *not* the
#: ACF hazard times population, which counts repeat screening of the same person and at 80%
#: coverage overstates volume by a factor of two. See ``vacf.economics.indicators``.
N_SCREENED = "n_screened"

#: Treatment starts per year by how the case was found. The split is essential rather than
#: cosmetic: patient costs differ roughly two-fold between actively and passively detected
#: cases, and that difference is a principal driver of cost-effectiveness. The model keeps
#: ``acf`` and ``detection`` as separate flows, so no apportionment assumption is needed.
TREATMENT_STARTS_ACTIVE = "treatment_starts_active"
TREATMENT_STARTS_PASSIVE = "treatment_starts_passive"

#: TB deaths per year by age: community deaths plus deaths on treatment. Drives YLLs.
TB_DEATHS_BY_AGE = "tb_deaths_age_{age}"

#: Successful treatment completions per year by age. Drives post-TB DALYs, which accrue to
#: survivors of TB treatment.
TREATMENT_COMPLETIONS_BY_AGE = "treatment_completions_age_{age}"

#: Person-time in symptomatic active TB, by age. Drives symptomatic-TB YLDs.
SYMPTOMATIC_PERSON_TIME = "person_time_symptomatic_age_{age}"

#: Person-time on TB treatment, by age. Drives on-treatment YLDs.
TREATMENT_PERSON_TIME = "person_time_treatment_age_{age}"


def required_columns(ages: list[str]) -> list[str]:
    """Every column the economic layer needs, for a given set of model age groups."""
    names = [N_SCREENED, TREATMENT_STARTS_ACTIVE, TREATMENT_STARTS_PASSIVE]
    for age in ages:
        names += [
            TB_DEATHS_BY_AGE.format(age=age),
            TREATMENT_COMPLETIONS_BY_AGE.format(age=age),
            SYMPTOMATIC_PERSON_TIME.format(age=age),
            TREATMENT_PERSON_TIME.format(age=age),
        ]
    return names


def validate(frame: pd.DataFrame, ages: list[str]) -> None:
    """Raise if an indicator frame is missing anything the cost engine needs."""
    missing = [name for name in required_columns(ages) if name not in frame.columns]
    if missing:
        raise KeyError(
            f"Indicator frame is missing {len(missing)} columns required for economic "
            f"analysis: {missing[:8]}{' ...' if len(missing) > 8 else ''}"
        )
