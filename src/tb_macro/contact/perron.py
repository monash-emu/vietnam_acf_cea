"""Spectral radius of a positive matrix without ``jnp.linalg.eig``.

``normalise_by_spectral_radius`` calls ``eigvals``. That is a general complex
eigensolver, it is the most expensive part of the baseline step, and in JAX it
only has a CPU lowering, so it pins the whole vector field to the host.

A positive contact matrix has a simple real dominant root (Perron–Frobenius),
so the pair of dominant left and right vectors is enough. They are found by
normalise-and-square: squaring a matrix squares the ratio ``|lambda_2 / lambda_1``,
so ``n`` squarings are worth ``2**n`` power iterations. The radius is then the
Rayleigh quotient ``u C v / u v``.

The quotient is stationary at the true pair, so the vectors carry
``stop_gradient`` and the cotangent is the textbook perturbation
``u (dC) v / u v``. That is the exact derivative of the dominant eigenvalue,
not an approximation of it.
"""

from __future__ import annotations

from typing import Any

from jax import lax, numpy as jnp

# 2**10 power iterations. The worst corner of the Vietnam parameter box has
# ``|lambda_2 / lambda_1|`` near 0.9, where 2**6 is only good to 1e-8.
DEFAULT_SQUARINGS = 10


def perron_pair(matrices: Any, squarings: int = DEFAULT_SQUARINGS) -> tuple[Any, Any]:
    """Dominant left and right vectors of each matrix in a batch.

    Args:
        matrices: ``(..., n, n)``, entrywise positive
        squarings: Repeated squarings, worth ``2**squarings`` power iterations

    Returns:
        ``(left, right)``, each ``(..., n)`` and unit norm. Both are positive,
        so they vary continuously with the matrix and never flip sign. Neither
        carries a tangent: they are an argument of a stationary quotient, so a
        cotangent through the iteration would cost ``squarings`` reverse matrix
        products to buy a second-order correction to zero.
    """
    matrices = lax.stop_gradient(matrices)

    def scaled(matrix: Any) -> Any:
        return matrix / jnp.max(jnp.abs(matrix), axis=(-2, -1), keepdims=True)

    def body(_: Any, matrix: Any) -> Any:
        half = scaled(matrix)
        return half @ half

    power = scaled(lax.fori_loop(0, squarings, body, matrices))
    # ``power`` is numerically rank one: every column is the right vector and
    # every row is the left vector, up to scale.
    right = power.sum(axis=-1)
    left = power.sum(axis=-2)
    right = right / jnp.linalg.norm(right, axis=-1, keepdims=True)
    left = left / jnp.linalg.norm(left, axis=-1, keepdims=True)
    return left, right


def rayleigh_radius(matrices: Any, left: Any, right: Any) -> Any:
    """``u C v / u v``, batched over leading axes. Scale-free in ``u`` and ``v``."""
    numerator = jnp.einsum("...i,...ij,...j->...", left, matrices, right)
    return numerator / jnp.einsum("...i,...i->...", left, right)


def spectral_radius(matrices: Any, squarings: int = DEFAULT_SQUARINGS) -> Any:
    """Dominant eigenvalue of each positive matrix in a batch."""
    left, right = perron_pair(matrices, squarings)
    return rayleigh_radius(matrices, left, right)
