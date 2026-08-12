import pytest

from ffapp.features.registry import FEATURE_REGISTRY, DuplicateFeatureError, FeatureSpec, register


def _spec(**kwargs: object) -> FeatureSpec:
    fields: dict[str, object] = {
        "name": "test_feature",
        "description": "a feature used only by this test suite",
        "positions": ["WR", "TE"],
        "window": "ewm_4",
        "source_table": "player_week_usage",
        "available_at_inference": True,
        "lag_weeks": 1,
    }
    fields.update(kwargs)
    return FeatureSpec(**fields)


def test_register_adds_the_spec_to_the_given_registry() -> None:
    registry: dict[str, FeatureSpec] = {}

    result = register(_spec(), registry=registry)

    assert registry["test_feature"] is result


def test_register_raises_on_a_duplicate_name() -> None:
    registry: dict[str, FeatureSpec] = {}
    register(_spec(), registry=registry)

    with pytest.raises(DuplicateFeatureError):
        register(_spec(), registry=registry)


def test_register_does_not_overwrite_the_existing_spec_on_a_duplicate() -> None:
    registry: dict[str, FeatureSpec] = {}
    original = register(_spec(source_table="player_week_usage"), registry=registry)

    with pytest.raises(DuplicateFeatureError):
        register(_spec(source_table="team_week_context"), registry=registry)

    assert registry["test_feature"] is original


def test_register_defaults_to_the_shared_feature_registry() -> None:
    assert "test_feature_shared" not in FEATURE_REGISTRY
    try:
        register(_spec(name="test_feature_shared"))
        assert "test_feature_shared" in FEATURE_REGISTRY
    finally:
        FEATURE_REGISTRY.pop("test_feature_shared", None)
