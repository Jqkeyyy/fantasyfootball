import polars as pl
import pytest

from ffapp.features import opponent

# --- add_opponent_features ------------------------------------------------------------


def _schedule_row(**kwargs: object) -> dict:
    row: dict[str, object] = {
        "season": 2025,
        "week": 1,
        "home_team": "KC",
        "away_team": "BAL",
    }
    row.update(kwargs)
    return row


def _grid_row(**kwargs: object) -> dict:
    row: dict[str, object] = {
        "player_id": "p1",
        "season": 2025,
        "week": 1,
        "team": "KC",
        "position": "WR",
    }
    row.update(kwargs)
    return row


def _dpa_row(**kwargs: object) -> dict:
    row: dict[str, object] = {
        "defteam": "BAL",
        "season": 2025,
        "week": 1,
        "position_group": "WR",
        "adj_epa_allowed": -0.1,
        "adj_success_allowed": -0.02,
        "adj_ypt_allowed": -0.5,
        "adj_td_rate_allowed": -0.01,
        "n_plays": 120,
    }
    row.update(kwargs)
    return row


def test_add_opponent_features_maps_a_wr_to_the_wr_group_via_the_real_opponent() -> None:
    schedule = pl.DataFrame([_schedule_row()])
    dpa = pl.DataFrame([_dpa_row(defteam="BAL", position_group="WR")])
    grid = pl.DataFrame([_grid_row(team="KC", position="WR")])  # KC's opponent is BAL

    result = opponent.add_opponent_features(grid, schedule, dpa)

    row = result.row(0, named=True)
    assert row["def_adj_epa_allowed_wr"] == pytest.approx(-0.1)
    assert row["def_adj_success_allowed_wr"] == pytest.approx(-0.02)
    assert row["def_adj_ypt_allowed_wr"] == pytest.approx(-0.5)
    assert row["def_adj_td_rate_allowed_wr"] == pytest.approx(-0.01)
    assert row["def_n_plays_wr"] == 120
    # not a TE -- the TE columns must stay null.
    assert row["def_adj_epa_allowed_te"] is None


def test_add_opponent_features_gives_a_rb_both_receiving_and_rushing_groups() -> None:
    schedule = pl.DataFrame([_schedule_row()])
    dpa = pl.DataFrame(
        [
            _dpa_row(defteam="BAL", position_group="RB_receiving", adj_epa_allowed=-0.05),
            _dpa_row(defteam="BAL", position_group="RB_rushing", adj_epa_allowed=0.08),
        ]
    )
    grid = pl.DataFrame([_grid_row(team="KC", position="RB")])

    result = opponent.add_opponent_features(grid, schedule, dpa)

    row = result.row(0, named=True)
    assert row["def_adj_epa_allowed_rb_receiving"] == pytest.approx(-0.05)
    assert row["def_adj_epa_allowed_rb_rushing"] == pytest.approx(0.08)
    # a RB never gets a WR/TE/QB_* value.
    assert row["def_adj_epa_allowed_wr"] is None
    assert row["def_adj_epa_allowed_qb_passing"] is None


def test_add_opponent_features_gives_a_qb_both_passing_and_rushing_groups() -> None:
    schedule = pl.DataFrame([_schedule_row()])
    dpa = pl.DataFrame(
        [
            _dpa_row(defteam="BAL", position_group="QB_passing", adj_epa_allowed=0.02),
            _dpa_row(defteam="BAL", position_group="QB_rushing", adj_epa_allowed=-0.03),
        ]
    )
    grid = pl.DataFrame([_grid_row(team="KC", position="QB")])

    result = opponent.add_opponent_features(grid, schedule, dpa)

    row = result.row(0, named=True)
    assert row["def_adj_epa_allowed_qb_passing"] == pytest.approx(0.02)
    assert row["def_adj_epa_allowed_qb_rushing"] == pytest.approx(-0.03)


def test_add_opponent_features_uses_the_real_opponent_not_the_players_own_team() -> None:
    """The away team's own defense_position_allowed row (KC as defteam)
    must never leak onto a KC player's row -- it's BAL (the real
    opponent) that should be looked up."""
    schedule = pl.DataFrame([_schedule_row(home_team="KC", away_team="BAL")])
    dpa = pl.DataFrame(
        [
            _dpa_row(defteam="BAL", position_group="WR", adj_epa_allowed=-0.1),
            _dpa_row(defteam="KC", position_group="WR", adj_epa_allowed=0.9),  # KC's own defense
        ]
    )
    grid = pl.DataFrame([_grid_row(team="KC", position="WR")])

    result = opponent.add_opponent_features(grid, schedule, dpa)

    assert result.row(0, named=True)["def_adj_epa_allowed_wr"] == pytest.approx(-0.1)


def test_add_opponent_features_is_null_on_a_bye_week() -> None:
    schedule = pl.DataFrame([_schedule_row(home_team="KC", away_team="BAL")])
    dpa = pl.DataFrame([_dpa_row(defteam="BAL", position_group="WR")])
    grid = pl.DataFrame([_grid_row(team="DAL", position="WR")])  # DAL has no game

    result = opponent.add_opponent_features(grid, schedule, dpa)

    assert result.row(0, named=True)["def_adj_epa_allowed_wr"] is None


def test_add_opponent_features_handles_a_position_group_absent_from_the_input() -> None:
    """A narrow fixture with no QB_passing/QB_rushing rows at all must
    still produce those columns, all-null -- the schema can't depend on
    what happens to be present in a given input."""
    schedule = pl.DataFrame([_schedule_row()])
    dpa = pl.DataFrame([_dpa_row(defteam="BAL", position_group="WR")])
    grid = pl.DataFrame([_grid_row(team="KC", position="WR")])

    result = opponent.add_opponent_features(grid, schedule, dpa)

    assert "def_adj_epa_allowed_qb_passing" in result.columns
    assert result.row(0, named=True)["def_adj_epa_allowed_qb_passing"] is None


# --- register_opponent_features --------------------------------------------------------


def test_register_opponent_features_registers_every_group_and_metric() -> None:
    registry: dict[str, object] = {}

    opponent.register_opponent_features(registry=registry)

    assert "def_adj_epa_allowed_wr" in registry
    assert "def_adj_epa_allowed_rb_receiving" in registry
    assert "def_adj_epa_allowed_rb_rushing" in registry
    assert "def_adj_epa_allowed_qb_passing" in registry
    assert "def_adj_epa_allowed_qb_rushing" in registry
    assert "def_n_plays_te" in registry
    for spec in registry.values():
        assert spec.lag_weeks == 1
        assert spec.available_at_inference is True
