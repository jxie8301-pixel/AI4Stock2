from __future__ import annotations

from run_candidate_pool_diagnostics import build_delegated_command


def test_candidate_pool_wrapper_delegates_to_rust() -> None:
    command = build_delegated_command(
        ["--candidate-root", "candidate/shortlist", "--no-sync-candidate-root"]
    )

    assert command[:5] == ["cargo", "run", "--bin", "ai4stock-diagnostics", "--"]
    assert command[5:] == [
        "candidate-pool",
        "--candidate-root",
        "candidate/shortlist",
        "--no-sync-candidate-root",
    ]


def test_candidate_pool_wrapper_honors_explicit_rust_binary(monkeypatch) -> None:
    monkeypatch.setenv("AI4STOCK_DIAGNOSTICS_BIN", "target/release/ai4stock-diagnostics --flag")

    command = build_delegated_command(["--preset", "latest_slim_b_topk15"])

    assert command == [
        "target/release/ai4stock-diagnostics",
        "--flag",
        "candidate-pool",
        "--preset",
        "latest_slim_b_topk15",
    ]
