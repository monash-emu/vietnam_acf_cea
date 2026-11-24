"""Provenance fields written next to a remote calibration run."""

from __future__ import annotations

from pathlib import Path

from vacf.demography import smooth_birth_rate_at
from vacf.utils import write_run_log


def test_write_run_log_includes_git_and_model_fields(tmp_path: Path) -> None:
    """Finished massive jobs are auditable from ``<stamp>.log`` alone."""
    path = tmp_path / "stamp.log"
    write_run_log(
        path,
        extra={
            "task": "demo",
            "n_runs": 10,
            "mixing": "spline",
            "entry": "smooth",
            "match_golden": False,
        },
    )
    text = path.read_text()
    for key in (
        "written",
        "commit",
        "branch",
        "dirty",
        "task",
        "n_runs",
        "mixing",
        "entry",
        "match_golden",
    ):
        assert f"{key}:" in text, key
    assert "mixing: spline" in text
    assert "entry: smooth" in text
    assert "match_golden: False" in text


def test_compile_case_accepts_explicit_entry() -> None:
    """Massive pins ``smooth_birth_rate_at`` so the logged entry cannot drift."""
    from vacf.calibration import build_mixing_cases, compile_case

    case = next(item for item in build_mixing_cases() if item.name == "spline")
    compiled = compile_case(case, match_golden=False, entry=smooth_birth_rate_at)
    assert compiled is not None
