"""Group-level pieces of the single-age mixing kernel.

The 121×121 kernel is never stored. Background, parent–child and the
assortative lag tensor are accumulated from one year's within-band weights
and a fertility lookup.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vacf.constants import AGE_STRATA, MAX_AGE


def year_index_np(ends: Any, time: Any) -> np.ndarray:
    """Clip ``time`` to ``ends`` and truncate toward zero.

    Same rule as ``mixing.get_year_index``. Age gaps are integers, so
    ``floor(t - k) = floor(t) - k`` and the kernel depends on ``floor(t)``
    only.
    """
    bounds = np.asarray(ends, dtype=np.float64).reshape(-1)
    clamped = np.clip(np.asarray(time, dtype=np.float64), bounds[0], bounds[-1])
    return (clamped - bounds[0]).astype(np.int32)


def unit_years(ends: Any, n_rows: int, name: str) -> np.ndarray:
    """Calendar year of each row, assuming a unit spacing from ``ends[0]``."""
    bounds = np.asarray(ends, dtype=np.float64).reshape(-1)
    start = float(bounds[0])
    stop = float(bounds[-1])
    if int(round(stop - start)) + 1 != n_rows:
        raise ValueError(f"{name} has {n_rows} rows but ends run {start}..{stop}")
    return start + np.arange(n_rows, dtype=np.float64)


def band_ids(age_strata: list[int] | None = None, max_age: int | None = None) -> np.ndarray:
    """Single-year age to age-band index, covering ``0 .. max_age``."""
    breaks = list(AGE_STRATA if age_strata is None else age_strata)
    last_age = MAX_AGE if max_age is None else max_age
    n_ages = last_age + 1
    band = np.full(n_ages, -1, dtype=np.int32)
    for group, lower in enumerate(breaks):
        upper = n_ages if group == len(breaks) - 1 else breaks[group + 1]
        if lower < 0 or upper > n_ages or lower >= upper:
            raise ValueError(f"age break {lower}:{upper} is outside 0..{last_age}")
        band[lower:upper] = group
    if np.any(band < 0):
        raise ValueError("age breaks do not cover every single year of age")
    return band


def pair_layout(max_age: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ages, absolute gap and younger age, each shaped ``(n_ages, n_ages)``."""
    ages = np.arange(max_age + 1, dtype=np.int32)
    gap = np.abs(ages[:, None] - ages[None, :])
    child = np.minimum(ages[:, None], ages[None, :])
    return ages, gap, child


def fertility_pairs(
    fert: np.ndarray,
    fert_ends: Any,
    time: float,
    child: np.ndarray,
    gap: np.ndarray,
) -> np.ndarray:
    """Fertility at birth year ``time - min(i, j)``, indexed by the age gap."""
    index = year_index_np(fert_ends, time - child.astype(np.float64))
    return np.asarray(fert, dtype=np.float64)[index, gap]


def components(
    weights: np.ndarray,
    fertility: np.ndarray,
    band: np.ndarray,
    gap: np.ndarray,
    n_groups: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Background outer product, parent–child matrix and assortative lag tensor.

    Args:
        weights: Within-band shares, length ``n_ages``. Not assumed to sum to 1
        fertility: ``(n_ages, n_ages)`` parent–child weight before ``pc_strength``
        band: Single-year age to group, from :func:`band_ids`
        gap: Absolute age gap
        n_groups: Number of age bands

    Returns:
        ``B`` of shape ``(n_groups, n_groups)``, ``S_pc`` of the same shape,
        and ``L`` of shape ``(n_groups, n_groups, n_ages)``
    """
    shares = np.asarray(weights, dtype=np.float64)
    n_ages = shares.shape[0]
    prod = (shares[:, None] * shares[None, :]).reshape(-1)
    # One flat bin per (group_i, group_j, lag). ``bincount`` is the same scatter
    # as ``np.add.at`` on those indices and is an order of magnitude faster.
    pair = (band[:, None] * n_groups + band[None, :]).reshape(-1)
    lagged = np.bincount(
        pair * n_ages + gap.reshape(-1), weights=prod, minlength=n_groups * n_groups * n_ages
    ).reshape(n_groups, n_groups, n_ages)
    parent = np.bincount(
        pair,
        weights=prod * np.asarray(fertility, dtype=np.float64).reshape(-1),
        minlength=n_groups * n_groups,
    ).reshape(n_groups, n_groups)
    totals = np.bincount(band, weights=shares, minlength=n_groups).astype(np.float64)
    background = np.outer(totals, totals)
    return background, parent, lagged


def renormalise_bands(weights: np.ndarray, band: np.ndarray, n_groups: int) -> np.ndarray:
    """Divide each age band by its sum. A zero band is left at zero."""
    out = np.array(weights, dtype=np.float64, copy=True)
    for group in range(n_groups):
        mask = band == group
        total = float(out[mask].sum())
        if total > 0.0:
            out[mask] /= total
    return out


def renormalise_rows(rows: np.ndarray) -> np.ndarray:
    """Divide each fertility row by its sum. A zero row is left at zero."""
    totals = rows.sum(axis=-1, keepdims=True)
    safe = np.where(totals > 0.0, totals, 1.0)
    return np.where(totals > 0.0, rows / safe, rows)
