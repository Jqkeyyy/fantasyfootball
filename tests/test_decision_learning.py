import polars as pl

from ffapp.evaluation.decision_learning import decision_calibration, personal_profile


def test_decision_learning_summarizes_choice_results() -> None:
    ledger = pl.DataFrame(
        {
            "decision_type": ["lineup", "lineup", "waiver"],
            "status": ["accepted", "rejected", "recommended"],
            "confidence": [0.8, 0.6, None],
            "expected_delta": [3.0, 2.0, 4.0],
            "realized_delta": [5.0, -1.0, None],
            "decision_regret": [0.0, 0.0, None],
        }
    )
    profile = personal_profile(ledger)
    calibration = decision_calibration(ledger)

    assert profile["choices"] == 2
    assert profile["follow_rate"] == 0.5
    assert profile["helped"] == 1.0
    assert calibration["decisions"].sum() == 2

