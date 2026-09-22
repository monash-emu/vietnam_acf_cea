"""Logging, git provenance, and mid-year interpolation for saved runs."""

from __future__ import annotations

import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from tb_macro.constants import CALENDAR_YEAR_MIDPOINT, REPO_ROOT


def get_logger(log_file: Path) -> logging.Logger:
    """Root logger writing to ``log_file`` and stdout."""
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    root_logger = logging.getLogger()

    file_handler = logging.FileHandler(log_file, mode="w")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    root_logger.setLevel(logging.INFO)
    return root_logger


def get_git_provenance(repo: Path | None = None) -> dict[str, str]:
    """Return the current git commit, branch, and dirty flag."""
    root = Path(repo) if repo is not None else REPO_ROOT

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    dirty = "unknown" if status.returncode != 0 else str(bool(status.stdout.strip())).lower()
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": dirty,
    }


def write_run_log(path: Path, extra: dict[str, Any] | None = None) -> None:
    """Write a small provenance log next to a saved run."""
    lines = [f"written: {datetime.now(UTC).strftime('%Y-%m-%dT%H:%MZ')}"]
    for key, value in get_git_provenance().items():
        lines.append(f"{key}: {value}")
    if extra:
        for key, value in extra.items():
            lines.append(f"{key}: {value}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def interp_annual_to_times(
    output_df: pd.DataFrame | pd.Series,
    times: Any,
) -> pd.DataFrame | pd.Series:
    """Linearly interpolate an annually indexed series to query times."""
    index = pd.Index(times, dtype=float)
    base = output_df.copy()
    base.index = base.index.astype(float)
    out = base.reindex(base.index.union(index)).sort_index().interpolate(method="index")
    return out.reindex(index)


def add_midyear_points(df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Keep 1 January solver values and insert interpolated mid-year points.

    The last annual point has no following year, so no mid-year is added after it.
    """
    years = df.index.astype(float)
    return interp_annual_to_times(df, years.union(years[:-1] + CALENDAR_YEAR_MIDPOINT))
