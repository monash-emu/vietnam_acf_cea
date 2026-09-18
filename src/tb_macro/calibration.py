"""Hand-written numpyro calibration of ``contact_rate``.

# GLUE(TM8): replaced by WP10 — BayesianModel, Uniform prior, Poisson target,
# sample(kind="nuts").
"""

from __future__ import annotations

from typing import Any

import numpy as np
from jax import numpy as jnp
from jax.random import PRNGKey
from jax.scipy.special import gammaln
import numpyro
from numpyro import distributions as dist
from numpyro import infer
from numpyro.infer import MCMC
from tb_macro.constants import BASE_PARAMS, CALIBRATION_CONTACT_RATE
from tb_macro.model import run
from tb_macro.outputs import infection_target


def synthetic_target(seed: int = 0) -> np.ndarray:
    """Rolling-sum series at contact_rate 0.25, fuzzed with seeded exp(N(0, 0.01))."""
    params = dict(BASE_PARAMS)
    params["contact_rate"] = CALIBRATION_CONTACT_RATE
    clean = np.asarray(infection_target(run(params)))
    fuzz = np.random.default_rng(seed).normal(scale=0.01, size=clean.shape)
    return clean * np.exp(fuzz)


def _poisson_log_prob(rate: np.ndarray, value: Any) -> Any:
    """Poisson log pmf via gammaln, without an integer-support check.

    The original notebook evaluates ``Poisson(target).log_prob(modelled)`` on
    the continuous rolling sum. Current numpyro rejects non-integer
    observations, which makes every NUTS init invalid.
    """
    rate_arr = jnp.asarray(rate)
    value_arr = jnp.asarray(value)
    log_term = jnp.where(
        rate_arr > 0,
        value_arr * jnp.log(rate_arr),
        jnp.where(value_arr == 0, 0.0, -jnp.inf),
    )
    return log_term - rate_arr - gammaln(value_arr + 1.0)


def model(observed: np.ndarray) -> None:
    """Poisson likelihood of the fuzzed target against the modelled rolling sum.

    Matches the original notebook: the target is the Poisson mean and the model
    output is the observation, not the other way around.
    """
    contact_rate = numpyro.sample("contact_rate", dist.Uniform(0.001, 0.5))
    params = dict(BASE_PARAMS)
    params["contact_rate"] = contact_rate
    modelled = infection_target(run(params))
    # Leading target entries are 0. log(0) poisons the gradient even inside
    # ``where``, and those windows carry no information while the model is
    # still at zero there.
    observed_np = np.asarray(observed)
    keep = np.flatnonzero(observed_np > 0)
    log_prob = _poisson_log_prob(observed_np[keep], modelled[jnp.asarray(keep)])
    numpyro.factor("ll", log_prob.sum())


def sample(
    observed: np.ndarray,
    *,
    num_warmup: int = 200,
    num_samples: int = 200,
    num_chains: int = 4,
    seed: int = 0,
    chain_method: str = "sequential",
) -> MCMC:
    """NUTS. Chains are sequential: ``CompiledModel.run`` is not vmap-safe.

    The plan asks for ``chain_method="vectorized"``. summer4 builds the save
    plan on the host, so vmap over ``run`` fails. Sequential chains are the
    workaround until WP10's sampler owns the solve.
    """
    kernel = infer.NUTS(model)
    mcmc = infer.MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method=chain_method,
    )
    mcmc.run(PRNGKey(seed), observed)
    return mcmc
