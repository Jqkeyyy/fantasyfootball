import polars as pl

from ffapp.evaluation.postgame import decision_autopsy, model_miss_autopsy


def _review(status: str = "Scored") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [2026, 2026],
            "week": [2, 2],
            "player_id": ["start", "sit"],
            "player": ["Starter", "Bench"],
            "position": ["WR", "WR"],
            "projected": [15.0, 12.0],
            "actual": [8.0, 18.0],
            "error": [7.0, -6.0],
            "absolute_error": [7.0, 6.0],
            "status": [status, "Scored"],
            "signal": ["Usage changed", "Multiple touchdowns"],
            "evidence": ["Targets 3 vs recent 8", "2 total touchdowns"],
        }
    )


def test_model_autopsy_classifies_available_evidence() -> None:
    result = model_miss_autopsy(_review())
    assert result.row(0, named=True)["category"] == "Role / opportunity changed"


def test_decision_autopsy_measures_recorded_choice_and_excludes_injury() -> None:
    ledger = pl.DataFrame(
        {
            "season": [2026],
            "week": [2],
            "subject_player_id": ["start"],
            "alternative_player_id": ["sit"],
            "recommended_action": ["Start Starter over Bench"],
            "expected_delta": [3.0],
            "realized_delta": [-10.0],
            "status": ["accepted"],
        }
    )
    result = decision_autopsy(ledger, _review(), season=2026, week=2)
    assert result["outcome"].item() == "Choice cost points"
    assert result["your_choice_value"].item() == -10.0
    excluded = decision_autopsy(ledger, _review("Injury excluded"), season=2026, week=2)
    assert excluded["outcome"].item() == "Injury excluded"
    assert excluded["counted"].item() is False
