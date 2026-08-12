import pytest

from ffapp.features.build import LeakageError, assert_inference_availability, assert_training_lag
from ffapp.features.registry import FeatureSpec


def _spec(**kwargs: object) -> FeatureSpec:
    fields: dict[str, object] = {
        "name": "target_share_ewm_4",
        "description": "targets / team pass attempts, ewm span 4",
        "positions": ["WR", "TE", "RB"],
        "window": "ewm_4",
        "source_table": "player_week_usage",
        "available_at_inference": True,
        "lag_weeks": 1,
    }
    fields.update(kwargs)
    return FeatureSpec(**fields)


# --- assert_training_lag ------------------------------------------------------------


def test_assert_training_lag_passes_for_a_correctly_lagged_feature() -> None:
    assert_training_lag([_spec(lag_weeks=1)])  # should not raise


def test_assert_training_lag_raises_for_a_zero_lag_feature() -> None:
    """The deliberately mis-specified feature TASKS.md 1.5 requires: a
    feature with lag_weeks=0 would see the target week's own data."""
    with pytest.raises(LeakageError, match="lag_weeks"):
        assert_training_lag([_spec(name="leaky_feature", lag_weeks=0)])


def test_assert_training_lag_raises_for_negative_lag() -> None:
    with pytest.raises(LeakageError):
        assert_training_lag([_spec(name="leaky_feature", lag_weeks=-1)])


def test_assert_training_lag_checks_every_spec_not_just_the_first() -> None:
    with pytest.raises(LeakageError, match="second_feature"):
        assert_training_lag(
            [_spec(name="first_feature", lag_weeks=1), _spec(name="second_feature", lag_weeks=0)]
        )


# --- assert_inference_availability ---------------------------------------------------


def test_assert_inference_availability_passes_for_an_available_feature() -> None:
    assert_inference_availability([_spec(available_at_inference=True)])  # should not raise


def test_assert_inference_availability_raises_for_a_training_only_feature() -> None:
    """The deliberately mis-specified feature TASKS.md 1.5 requires: a
    training-only feature (e.g. route participation, SPEC §10.5) must
    never be handed to an inference model."""
    with pytest.raises(LeakageError, match="available_at_inference"):
        assert_inference_availability(
            [_spec(name="route_participation", available_at_inference=False)]
        )


def test_assert_inference_availability_checks_every_spec_not_just_the_first() -> None:
    with pytest.raises(LeakageError, match="second_feature"):
        assert_inference_availability(
            [
                _spec(name="first_feature", available_at_inference=True),
                _spec(name="second_feature", available_at_inference=False),
            ]
        )
