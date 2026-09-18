"""Slow NUTS recovery of the synthetic contact rate."""

from __future__ import annotations

import numpy as np
import pytest

from tb_macro.calibration import sample, synthetic_target
from tb_macro.constants import CALIBRATION_CONTACT_RATE


@pytest.mark.slow
def test_nuts_recovers_contact_rate() -> None:
    observed = synthetic_target(seed=0)
    mcmc = sample(observed, num_warmup=100, num_samples=100, num_chains=2, seed=0)
    samples = np.asarray(mcmc.get_samples()["contact_rate"])
    lo, hi = np.quantile(samples, [0.025, 0.975])
    assert lo <= CALIBRATION_CONTACT_RATE <= hi
    divergences = int(mcmc.get_extra_fields()["diverging"].sum())
    assert divergences == 0
