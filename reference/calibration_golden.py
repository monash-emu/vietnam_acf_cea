"""Calibration parity golden paths and I/O (no summer4 imports)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

CALIBRATION_SEED = 20240922
N_RANDOM_DRAWS = 6

TERM_NAMES: tuple[str, ...] = (
    "latent",
    "notifications",
    "deaths",
    "prevalence",
    "decline",
    "infectiousness",
    "mixing",
)

# summer3 calibration notebook: Dopri5, rtol=atol=1e-5, dtmax=1, max_steps=4000.
SUMMER3_CALIBRATION_SOLVER: dict[str, Any] = {
    "rtol": 1e-5,
    "atol": 1e-5,
    "dtmax": 1.0,
    "max_steps": 4000,
}

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "tests" / "golden" / "calibration"
MANIFEST_PATH = GOLDEN_DIR / "manifest.json"
TERMS_PATH = GOLDEN_DIR / "terms.parquet"
PARAMS_PATH = GOLDEN_DIR / "params.json"


def write_manifest(*, seed: int, n_random: int, draws: list[str]) -> None:
    manifest = {
        "seed": seed,
        "n_random": n_random,
        "draws": draws,
        "solver": SUMMER3_CALIBRATION_SOLVER,
        "terms": [*TERM_NAMES, "total"],
    }
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def write_terms(rows: list[dict[str, Any]]) -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    expected = ["draw", *TERM_NAMES, "total"]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"terms frame missing columns {missing}")
    frame.to_parquet(TERMS_PATH, index=False)


def write_params(params_by_draw: dict[str, dict[str, float]]) -> None:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    PARAMS_PATH.write_text(json.dumps(params_by_draw, indent=2, sort_keys=True) + "\n")


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text())


def load_terms() -> pd.DataFrame:
    return pd.read_parquet(TERMS_PATH)


def load_params() -> dict[str, dict[str, float]]:
    raw = json.loads(PARAMS_PATH.read_text())
    return {
        name: {key: float(value) for key, value in params.items()} for name, params in raw.items()
    }
