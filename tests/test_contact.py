"""Stepwise matches the baseline. Smooth and spline match it on 1 January and are C1."""

from __future__ import annotations

from typing import Any

import jax
import numpy as np
from jax import numpy as jnp

from vacf.contact import prepare_smooth, prepare_spline, prepare_stepwise
from vacf.contact.aggregate import band_ids, components, pair_layout
from vacf.contact.pchip import eval_jax, eval_np, pchip_slopes
from vacf.contact.perron import spectral_radius
from vacf.mixing import get_norm_c_matrix
from vacf.parameters import BASE_PARAMS
from vacf.static_data import load_static

# Well under PARITY_RTOL. The two paths sum the same products in a different order.
_RTOL = 1e-7
_ATOL = 1e-9


def _static() -> Any:
    return load_static()


def _baseline(static: Any, time: float, bg: float, a: float, pc: float) -> Any:
    return get_norm_c_matrix(
        static.age_weights,
        static.weight_ends,
        static.pops,
        static.pop_ends,
        static.fert,
        static.fert_ends,
        time,
        bg,
        a,
        pc,
    )


def test_lag_tensor_sums_to_the_background() -> None:
    """Every age-pair product lands in exactly one lag, so ``L`` sums to ``B``."""
    band = band_ids([0, 2], max_age=3)
    _, gap, _ = pair_layout(3)
    weights = np.array([0.25, 0.75, 0.4, 0.6])
    fertility = np.zeros((4, 4))
    background, _, lagged = components(weights, fertility, band, gap, n_groups=2)
    np.testing.assert_allclose(lagged.sum(axis=-1), background, rtol=0.0, atol=0.0)


def test_pchip_hits_knots_and_is_c1() -> None:
    x = np.linspace(0.0, 4.0, 5)
    y = np.stack([np.exp(-x / 3.0), np.tanh(x)], axis=1)
    slopes = pchip_slopes(x, y)
    for knot, value, slope in zip(x, y, slopes, strict=True):
        np.testing.assert_allclose(eval_np(knot, x, y, slopes), value, rtol=0.0, atol=1e-12)
        left = eval_np(knot - 1e-6, x, y, slopes)
        right = eval_np(knot + 1e-6, x, y, slopes)
        # End knots are flat outside the table, so only step inward.
        if knot == x[0]:
            numerical = (right - value) / 1e-6
        elif knot == x[-1]:
            numerical = (value - left) / 1e-6
        else:
            numerical = (right - left) / 2e-6
        np.testing.assert_allclose(numerical, slope, rtol=1e-4, atol=1e-5)


def test_pchip_flattens_a_sign_change() -> None:
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 0.0])
    slopes = pchip_slopes(x, y)
    assert slopes[1] == 0.0


def test_stepwise_matches_baseline_on_and_off_integers() -> None:
    static = _static()
    contact = prepare_stepwise(
        static.age_weights,
        static.weight_ends,
        static.pops,
        static.pop_ends,
        static.fert,
        static.fert_ends,
    )
    sets = (
        (BASE_PARAMS["bg_mixing"], BASE_PARAMS["a_spread"], BASE_PARAMS["pc_strength"]),
        (0.002, 3.0, 0.1),
        (0.018, 14.0, 0.9),
    )
    times = [float(year) + frac for year in range(1950, 2102, 10) for frac in (0.0, 0.3)]
    for bg, a, pc in sets:
        for time in times:
            got = contact.matrix(time, bg, a, pc)
            expected = _baseline(static, time, bg, a, pc)
            np.testing.assert_allclose(got, expected, rtol=_RTOL, atol=_ATOL)


def test_stepwise_matches_baseline_before_the_weight_table() -> None:
    """Years before 1950 still match: weights clip, fertility does not."""
    static = _static()
    bg = float(BASE_PARAMS["bg_mixing"])
    a = float(BASE_PARAMS["a_spread"])
    pc = float(BASE_PARAMS["pc_strength"])
    contact = prepare_stepwise(
        static.age_weights,
        static.weight_ends,
        static.pops,
        static.pop_ends,
        static.fert,
        static.fert_ends,
        years=np.arange(1800.0, 2102.0),
    )
    ready = contact.materialize(bg, a, pc)
    for time in (1800.0, 1850.4, 1949.9):
        got = ready.matrix(time)
        expected = _baseline(static, time, bg, a, pc)
        np.testing.assert_allclose(got, expected, rtol=_RTOL, atol=_ATOL)


def test_fixed_scalars_match_the_free_path() -> None:
    static = _static()
    bg = float(BASE_PARAMS["bg_mixing"])
    a = float(BASE_PARAMS["a_spread"])
    pc = float(BASE_PARAMS["pc_strength"])
    free = prepare_stepwise(
        static.age_weights,
        static.weight_ends,
        static.pops,
        static.pop_ends,
        static.fert,
        static.fert_ends,
    )
    fixed_a = prepare_stepwise(
        static.age_weights,
        static.weight_ends,
        static.pops,
        static.pop_ends,
        static.fert,
        static.fert_ends,
        a_spread=a,
    )
    all_fixed = prepare_stepwise(
        static.age_weights,
        static.weight_ends,
        static.pops,
        static.pop_ends,
        static.fert,
        static.fert_ends,
        bg_mixing=bg,
        a_spread=a,
        pc_strength=pc,
    )
    time = 2024.4
    expected = free.matrix(time, bg, a, pc)
    np.testing.assert_allclose(fixed_a.matrix(time, bg, pc_strength=pc), expected)
    np.testing.assert_allclose(all_fixed.matrix(time), expected)
    table = free.materialize(bg, a, pc)
    assert table.table.shape[0] == static.age_weights.shape[0]
    np.testing.assert_allclose(table.matrix(time), expected)


def test_smooth_matches_baseline_at_integer_years() -> None:
    static = _static()
    contact = prepare_smooth(
        static.age_weights,
        static.weight_ends,
        static.pops,
        static.pop_ends,
        static.fert,
        static.fert_ends,
    )
    bg = float(BASE_PARAMS["bg_mixing"])
    a = float(BASE_PARAMS["a_spread"])
    pc = float(BASE_PARAMS["pc_strength"])
    years = np.asarray(contact.inputs.weight_x)
    for time in years:
        got = contact.matrix(float(time), bg, a, pc)
        expected = _baseline(static, float(time), bg, a, pc)
        np.testing.assert_allclose(got, expected, rtol=_RTOL, atol=_ATOL)
    mid = np.asarray(contact.matrix(2000.5, bg, a, pc))
    january = np.asarray(_baseline(static, 2000.5, bg, a, pc))
    assert float(np.max(np.abs(mid - january))) > 1e-4


def test_smooth_knots_track_the_exact_aggregation() -> None:
    static = _static()
    contact = prepare_smooth(
        static.age_weights,
        static.weight_ends,
        static.pops,
        static.pop_ends,
        static.fert,
        static.fert_ends,
    )
    bg = float(BASE_PARAMS["bg_mixing"])
    a = float(BASE_PARAMS["a_spread"])
    pc = float(BASE_PARAMS["pc_strength"])
    rng = np.random.default_rng(0)
    times = rng.uniform(1950.2, 2100.8, size=50)
    worst = 0.0
    for time in times:
        got = np.asarray(contact.matrix(float(time), bg, a, pc))
        exact = np.asarray(contact.exact_matrix(float(time), bg, a, pc))
        scale = np.maximum(np.max(np.abs(exact)), 1e-12)
        worst = max(worst, float(np.max(np.abs(got - exact)) / scale))
    assert worst < 1e-3, worst


def test_smooth_radius_stays_one() -> None:
    static = _static()
    contact = prepare_smooth(
        static.age_weights,
        static.weight_ends,
        static.pops,
        static.pop_ends,
        static.fert,
        static.fert_ends,
    )
    bg = float(BASE_PARAMS["bg_mixing"])
    a = float(BASE_PARAMS["a_spread"])
    pc = float(BASE_PARAMS["pc_strength"])
    for time in (1960.25, 2000.5, 2088.7):
        matrix = contact.matrix(time, bg, a, pc)
        radius = jnp.max(jnp.abs(jnp.linalg.eigvals(matrix)))
        np.testing.assert_allclose(radius, 1.0, rtol=0.0, atol=1e-6)


def test_time_gradient_is_zero_for_stepwise_and_smooth_for_smooth() -> None:
    static = _static()
    stepwise = prepare_stepwise(
        static.age_weights,
        static.weight_ends,
        static.pops,
        static.pop_ends,
        static.fert,
        static.fert_ends,
    )
    smooth = prepare_smooth(
        static.age_weights,
        static.weight_ends,
        static.pops,
        static.pop_ends,
        static.fert,
        static.fert_ends,
    )
    bg = float(BASE_PARAMS["bg_mixing"])
    a = float(BASE_PARAMS["a_spread"])
    pc = float(BASE_PARAMS["pc_strength"])

    def step_entry(time: Any) -> Any:
        return stepwise.matrix(time, bg, a, pc)[1, 6]

    def smooth_entry(time: Any) -> Any:
        return smooth.matrix(time, bg, a, pc)[1, 6]

    def smooth_sum(time: Any) -> Any:
        return jnp.sum(smooth.matrix(time, bg, a, pc))

    assert abs(float(jax.grad(step_entry)(2000.5))) < 1e-12
    assert abs(float(jax.grad(smooth_entry)(2000.5))) > 1e-8
    assert abs(float(jax.grad(smooth_sum)(2000.5))) > 1e-6

    knot = 2000.0
    eps = 1e-4

    left = float(jax.grad(smooth_entry)(knot - eps))
    right = float(jax.grad(smooth_entry)(knot + eps))
    np.testing.assert_allclose(left, right, rtol=1e-3, atol=1e-5)


def test_stepwise_param_gradients_match_baseline_differences() -> None:
    static = _static()
    contact = prepare_stepwise(
        static.age_weights,
        static.weight_ends,
        static.pops,
        static.pop_ends,
        static.fert,
        static.fert_ends,
    )
    time = 2000.0
    bg = float(BASE_PARAMS["bg_mixing"])
    a = float(BASE_PARAMS["a_spread"])
    pc = float(BASE_PARAMS["pc_strength"])

    def entry(bg_v: Any, a_v: Any, pc_v: Any) -> Any:
        return contact.matrix(time, bg_v, a_v, pc_v)[2, 5]

    grads = {
        "bg": jax.grad(lambda v: entry(v, a, pc))(bg),
        "a": jax.grad(lambda v: entry(bg, v, pc))(a),
        "pc": jax.grad(lambda v: entry(bg, a, v))(pc),
    }
    eps = 1e-5
    shifts = {
        "bg": ((bg + eps, a, pc), (bg - eps, a, pc)),
        "a": ((bg, a + eps, pc), (bg, a - eps, pc)),
        "pc": ((bg, a, pc + eps), (bg, a, pc - eps)),
    }
    for name, (plus, minus) in shifts.items():
        numerical = (_baseline(static, time, *plus) - _baseline(static, time, *minus))[2, 5]
        numerical = numerical / (2.0 * eps)
        np.testing.assert_allclose(grads[name], numerical, rtol=1e-3, atol=1e-6)
        assert jnp.isfinite(grads[name])


def test_smooth_param_gradients_are_finite() -> None:
    static = _static()
    contact = prepare_smooth(
        static.age_weights,
        static.weight_ends,
        static.pops,
        static.pop_ends,
        static.fert,
        static.fert_ends,
    )
    time = 2000.5
    bg = float(BASE_PARAMS["bg_mixing"])
    a = float(BASE_PARAMS["a_spread"])
    pc = float(BASE_PARAMS["pc_strength"])

    def entry(bg_v: Any, a_v: Any, pc_v: Any) -> Any:
        return jnp.sum(contact.matrix(time, bg_v, a_v, pc_v))

    for grad in (
        jax.grad(lambda v: entry(v, a, pc))(bg),
        jax.grad(lambda v: entry(bg, v, pc))(a),
        jax.grad(lambda v: entry(bg, a, v))(pc),
    ):
        assert jnp.isfinite(grad)


def _spline(static: Any, **kwargs: Any) -> Any:
    return prepare_spline(
        static.age_weights,
        static.weight_ends,
        static.pops,
        static.pop_ends,
        static.fert,
        static.fert_ends,
        **kwargs,
    )


def test_perron_radius_matches_eigvals_across_the_parameter_box() -> None:
    """Repeated squaring beats ``eigvals`` at the worst corner of the box."""
    rng = np.random.default_rng(7)
    batch = rng.uniform(0.01, 1.0, size=(6, 8, 8))
    # A nearly diagonal matrix is the hard case: the two leading roots are close.
    batch[0] += 40.0 * np.eye(8)
    matrices = jnp.asarray(batch)
    expected = np.max(np.abs(np.linalg.eigvals(batch)), axis=-1)
    np.testing.assert_allclose(spectral_radius(matrices), expected, rtol=1e-12, atol=0.0)


def test_perron_radius_gradient_is_the_perturbation_formula() -> None:
    """``d rho = u dC v / u v`` for a simple dominant root."""
    base = np.asarray(np.random.default_rng(3).uniform(0.05, 1.0, size=(8, 8)))
    direction = np.asarray(np.random.default_rng(4).uniform(-1.0, 1.0, size=(8, 8)))

    def radius(scale: Any) -> Any:
        return spectral_radius(jnp.asarray(base) + scale * jnp.asarray(direction))

    analytic = float(jax.grad(radius)(0.0))
    step = 1e-6
    numerical = (float(radius(step)) - float(radius(-step))) / (2.0 * step)
    np.testing.assert_allclose(analytic, numerical, rtol=1e-6)


def test_spline_matches_baseline_at_every_measurement_year() -> None:
    """Knots are the measured years, so the curve passes through the baseline."""
    static = _static()
    bg = float(BASE_PARAMS["bg_mixing"])
    a = float(BASE_PARAMS["a_spread"])
    pc = float(BASE_PARAMS["pc_strength"])
    contact = _spline(static).materialize(bg, a, pc)
    for time in (1950.0, 1987.0, 2024.0, 2101.0):
        np.testing.assert_allclose(
            contact.matrix(time), _baseline(static, time, bg, a, pc), rtol=_RTOL, atol=_ATOL
        )


def test_spline_radius_stays_one_between_years() -> None:
    """The Rayleigh correction is what the interpolated matrix alone loses."""
    static = _static()
    bg = float(BASE_PARAMS["bg_mixing"])
    a = float(BASE_PARAMS["a_spread"])
    pc = float(BASE_PARAMS["pc_strength"])
    pinned = _spline(static).materialize(bg, a, pc)
    loose = _spline(static, pin_radius=False).materialize(bg, a, pc)
    drift = 0.0
    for time in (1963.5, 1999.25, 2050.8):
        radius = np.max(np.abs(np.linalg.eigvals(np.asarray(pinned.matrix(time)))))
        np.testing.assert_allclose(radius, 1.0, rtol=0.0, atol=1e-7)
        drift = max(
            drift, abs(np.max(np.abs(np.linalg.eigvals(np.asarray(loose.matrix(time))))) - 1.0)
        )
    # Without the correction it moves, which is why the correction is the default.
    assert drift > 1e-8


def test_spline_is_c1_in_time_and_tracks_the_smooth_aggregation() -> None:
    static = _static()
    bg = float(BASE_PARAMS["bg_mixing"])
    a = float(BASE_PARAMS["a_spread"])
    pc = float(BASE_PARAMS["pc_strength"])
    contact = _spline(static).materialize(bg, a, pc)
    slope = jax.jit(jax.grad(lambda time: contact.matrix(time)[1, 6]))
    for knot in (1975.0, 2000.0):
        step = 1e-5
        jump = float(slope(knot + step) - slope(knot - step))
        assert abs(jump) < 1e-6, f"derivative jumps by {jump} at {knot}"
    assert abs(float(slope(2000.4))) > 0.0
    # Between knots it sits near the aggregate of interpolated inputs, which is
    # a different convention on the same annual data, not a reference value.
    smooth = prepare_smooth(
        static.age_weights,
        static.weight_ends,
        static.pops,
        static.pop_ends,
        static.fert,
        static.fert_ends,
    )
    for time in (1970.5, 2033.25):
        gap = float(jnp.max(jnp.abs(contact.matrix(time) - smooth.exact_matrix(time, bg, a, pc))))
        assert gap < 5e-3, f"spline is {gap} from the interpolated-input aggregate at {time}"


def test_spline_free_path_is_the_same_function_as_the_curve() -> None:
    """Catmull-Rom is local, so four knots reproduce the fitted curve exactly."""
    static = _static()
    bg = float(BASE_PARAMS["bg_mixing"])
    a = float(BASE_PARAMS["a_spread"])
    pc = float(BASE_PARAMS["pc_strength"])
    free = _spline(static)
    curve = free.materialize(bg, a, pc)
    for time in (1950.0, 1961.3, 2044.75, 2101.0):
        np.testing.assert_allclose(
            free.matrix(time, bg, a, pc), curve.matrix(time), rtol=1e-12, atol=1e-14
        )


def test_spline_fixed_scalars_match_the_free_path() -> None:
    static = _static()
    bg = float(BASE_PARAMS["bg_mixing"])
    a = float(BASE_PARAMS["a_spread"])
    pc = float(BASE_PARAMS["pc_strength"])
    expected = _spline(static).materialize(bg, a, pc).matrix(1987.6)
    fixed_a = _spline(static, a_spread=a).materialize(bg_mixing=bg, pc_strength=pc)
    all_fixed = _spline(static, bg_mixing=bg, a_spread=a, pc_strength=pc)
    np.testing.assert_allclose(fixed_a.matrix(1987.6), expected, rtol=_RTOL, atol=_ATOL)
    np.testing.assert_allclose(all_fixed.matrix(1987.6), expected, rtol=_RTOL, atol=_ATOL)


def test_spline_param_gradients_match_finite_differences() -> None:
    """The fold is once per parameter set and every scalar stays differentiable."""
    static = _static()
    start = (
        float(BASE_PARAMS["bg_mixing"]),
        float(BASE_PARAMS["a_spread"]),
        float(BASE_PARAMS["pc_strength"]),
    )
    contact = _spline(static)

    def total(bg: Any, a: Any, pc: Any) -> Any:
        curve = contact.materialize(bg, a, pc)
        return curve.matrix(1980.3)[2, 5] + curve.matrix(2040.0)[6, 1]

    analytic = [float(value) for value in jax.grad(total, argnums=(0, 1, 2))(*start)]
    step = 1e-6
    for index, value in enumerate(analytic):
        moved = list(start)
        moved[index] = start[index] + step
        high = float(total(*moved))
        moved[index] = start[index] - step
        low = float(total(*moved))
        np.testing.assert_allclose(value, (high - low) / (2.0 * step), rtol=1e-5, atol=1e-8)


def test_pchip_gradient_reaches_the_knot_values() -> None:
    """Folding a scalar into the knots is how a cotangent gets in.

    The cubic is linear in ``y`` and ``slopes``. A JVP rule that only
    propagated ``t`` would make ``SmoothContact.materialize`` report a zero
    gradient for every mixing scalar, silently.
    """
    x = jnp.arange(5.0)
    base = jnp.asarray(np.linspace(1.0, 2.0, 5) ** 2)
    slopes = jnp.asarray(pchip_slopes(np.asarray(x), np.asarray(base)))

    def at(scale: Any) -> Any:
        return eval_jax(2.3, x, scale * base, scale * slopes)

    analytic = float(jax.grad(at)(1.0))
    step = 1e-6
    np.testing.assert_allclose(
        analytic, (float(at(1.0 + step)) - float(at(1.0 - step))) / (2.0 * step), rtol=1e-6
    )
    # And the value itself is still linear in that scale.
    np.testing.assert_allclose(float(at(2.0)), 2.0 * float(at(1.0)), rtol=1e-12)


def test_smooth_folded_gradients_match_the_free_path() -> None:
    """``materialize`` must stay traceable and differentiable, not host-only."""
    static = _static()
    start = jnp.asarray(
        [
            float(BASE_PARAMS["bg_mixing"]),
            float(BASE_PARAMS["a_spread"]),
            float(BASE_PARAMS["pc_strength"]),
        ]
    )
    contact = prepare_smooth(
        static.age_weights,
        static.weight_ends,
        static.pops,
        static.pop_ends,
        static.fert,
        static.fert_ends,
    )

    def folded(scalars: Any) -> Any:
        return contact.materialize(scalars[0], scalars[1], scalars[2]).matrix(1987.4)[1, 6]

    def free(scalars: Any) -> Any:
        return contact.matrix(1987.4, scalars[0], scalars[1], scalars[2])[1, 6]

    # Folding is a host-side shortcut for the same function; under jit too.
    np.testing.assert_allclose(jax.jit(folded)(start), free(start), rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(jax.grad(folded)(start)), np.asarray(jax.grad(free)(start)), rtol=1e-9
    )
    assert float(jnp.max(jnp.abs(jax.grad(folded)(start)))) > 0.0
