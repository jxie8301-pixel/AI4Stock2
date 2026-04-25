from __future__ import annotations

import json
from pathlib import Path

from src.rolling_evaluate import _append_run_warning, _write_run_warnings


def test_append_run_warning_records_context() -> None:
    warnings: list[dict[str, object]] = []

    _append_run_warning(
        warnings,
        code="baseline_reconstruction_failed",
        message="baseline failed",
        baseline="rank_ic_weighted_factor",
        error_type="RuntimeError",
        error="boom",
    )

    assert warnings == [
        {
            "severity": "warning",
            "code": "baseline_reconstruction_failed",
            "message": "baseline failed",
            "baseline": "rank_ic_weighted_factor",
            "error_type": "RuntimeError",
            "error": "boom",
        }
    ]


def test_write_run_warnings_skips_empty_and_writes_json(tmp_path: Path) -> None:
    assert _write_run_warnings(tmp_path, []) is None

    path = _write_run_warnings(
        tmp_path,
        [
            {
                "severity": "warning",
                "code": "opportunity_label_derivation_failed",
                "message": "failed",
            }
        ],
    )

    assert path == tmp_path / "run_warnings.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload[0]["code"] == "opportunity_label_derivation_failed"
