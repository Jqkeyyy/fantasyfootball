import json

import polars as pl
import pytest

from ffapp.models import baselines

# --- add_b1_season_to_date_mean -------------------------------------------------------


def _row(**kwargs: object) -> dict:
    row: dict[str, object] = {
        "player_id": "p1",
        "season": 2025,
        "week": 1,
        "position": "WR",
        "target": 10.0,
    }
    row.update(kwargs)
    return row


def test_add_b1_is_null_for_a_players_first_tracked_week_of_a_season() -> None:
    df = pl.DataFrame([_row(week=1, target=10.0)])

    result = baselines.add_b1_season_to_date_mean(df)

    assert result.row(0, named=True)["b1_season_to_date_mean"] is None


def test_add_b1_is_the_mean_of_strictly_prior_weeks_only() -> None:
    df = pl.DataFrame(
        [_row(week=1, target=10.0), _row(week=2, target=20.0), _row(week=3, target=0.0)]
    )

    result = baselines.add_b1_season_to_date_mean(df)

    rows = {row["week"]: row["b1_season_to_date_mean"] for row in result.iter_rows(named=True)}
    assert rows[1] is None
    assert rows[2] == pytest.approx(10.0)  # mean of week 1 only
    assert rows[3] == pytest.approx(15.0)  # mean of weeks 1-2, NOT including week 3's own 0.0


def test_add_b1_resets_at_a_season_boundary() -> None:
    df = pl.DataFrame(
        [_row(season=2024, week=17, target=30.0), _row(season=2025, week=1, target=5.0)]
    )

    result = baselines.add_b1_season_to_date_mean(df)

    week1_2025 = result.filter((pl.col("season") == 2025) & (pl.col("week") == 1)).row(
        0, named=True
    )
    assert week1_2025["b1_season_to_date_mean"] is None  # not blended with 2024's own history


# --- add_b2_ewm_4 ----------------------------------------------------------------------


def test_add_b2_is_null_for_a_players_first_tracked_week_of_a_season() -> None:
    df = pl.DataFrame([_row(week=1, target=10.0)])

    result = baselines.add_b2_ewm_4(df)

    assert result.row(0, named=True)["b2_ewm_4"] is None


def test_add_b2_never_uses_the_target_weeks_own_outcome() -> None:
    """A huge week-3 outcome (100.0) must not shift week 3's own
    prediction -- only weeks 1-2 can inform it."""
    df = pl.DataFrame(
        [_row(week=1, target=5.0), _row(week=2, target=5.0), _row(week=3, target=100.0)]
    )

    result = baselines.add_b2_ewm_4(df)

    week3 = result.filter(pl.col("week") == 3).row(0, named=True)
    assert week3["b2_ewm_4"] == pytest.approx(5.0)  # unaffected by its own 100.0


# --- add_b0_positional_mean -------------------------------------------------------------


def test_add_b0_pools_across_all_players_at_the_position_not_per_player() -> None:
    df = pl.DataFrame(
        [
            _row(player_id="p1", position="WR", week=1, target=10.0),
            _row(player_id="p2", position="WR", week=1, target=20.0),
            _row(player_id="p1", position="WR", week=2, target=0.0),
            _row(player_id="p2", position="WR", week=2, target=0.0),
        ]
    )

    result = baselines.add_b0_positional_mean(df)

    week2 = result.filter(pl.col("week") == 2)
    # both players' week-2 prediction = mean of ALL WR week-1 targets (10, 20) = 15.0
    assert week2["b0_positional_mean"].to_list() == pytest.approx([15.0, 15.0])


def test_add_b0_does_not_mix_positions() -> None:
    df = pl.DataFrame(
        [
            _row(player_id="p1", position="WR", week=1, target=10.0),
            _row(player_id="p2", position="RB", week=1, target=100.0),
            _row(player_id="p1", position="WR", week=2, target=0.0),
        ]
    )

    result = baselines.add_b0_positional_mean(df)

    week2_wr = result.filter((pl.col("week") == 2) & (pl.col("position") == "WR")).row(
        0, named=True
    )
    assert week2_wr["b0_positional_mean"] == pytest.approx(10.0)  # not polluted by RB's 100.0


def test_add_b0_falls_back_to_the_prior_seasons_positional_mean_at_week_one() -> None:
    df = pl.DataFrame(
        [
            _row(player_id="p1", position="WR", season=2024, week=1, target=8.0),
            _row(player_id="p1", position="WR", season=2024, week=2, target=12.0),
            _row(player_id="p2", position="WR", season=2025, week=1, target=0.0),
        ]
    )

    result = baselines.add_b0_positional_mean(df)

    week1_2025 = result.filter((pl.col("season") == 2025) & (pl.col("week") == 1)).row(
        0, named=True
    )
    assert week1_2025["b0_positional_mean"] == pytest.approx(10.0)  # mean of 2024's 8.0 and 12.0


def test_add_b0_is_null_with_no_prior_season_and_no_current_season_trailing_data() -> None:
    df = pl.DataFrame([_row(season=2015, week=1, target=10.0)])  # first tracked season, week 1

    result = baselines.add_b0_positional_mean(df)

    assert result.row(0, named=True)["b0_positional_mean"] is None


# --- add_availability_base_rate (task 1.14) ----------------------------------------------


def test_add_availability_base_rate_pools_across_all_players_at_the_position() -> None:
    """Same shape as B0 (`pooled_rolling_mean`), applied to
    `availability_flag` instead of `target`."""
    df = pl.DataFrame(
        [
            _row(player_id="p1", position="WR", week=1, availability_flag=True),
            _row(player_id="p2", position="WR", week=1, availability_flag=False),
            _row(player_id="p1", position="WR", week=2, availability_flag=True),
            _row(player_id="p2", position="WR", week=2, availability_flag=True),
        ]
    )

    result = baselines.add_availability_base_rate(df)

    week2 = result.filter(pl.col("week") == 2)
    # both players' week-2 prediction = fraction of WR week-1 rows active (1/2) = 0.5
    assert week2["availability_base_rate"].to_list() == pytest.approx([0.5, 0.5])


def test_add_availability_base_rate_is_null_with_no_trailing_data() -> None:
    df = pl.DataFrame([_row(season=2015, week=1, availability_flag=True)])

    result = baselines.add_availability_base_rate(df)

    assert result.row(0, named=True)["availability_base_rate"] is None


# --- add_b3_fp_weekly_consensus ---------------------------------------------------------


def _players_dim_row(**kwargs: object) -> dict:
    row: dict[str, object] = {
        "full_name": "Joe Burrow",
        "normalized_name": "joe burrow",
        "position": "QB",
        "player_id": "00-0036442",
        "sleeper_id": "6770",
    }
    row.update(kwargs)
    return row


def _fp_weekly_row(**kwargs: object) -> dict:
    row: dict[str, object] = {
        "player_name": "Joe Burrow",
        "pos": "QB",
        "team": "CIN",
        "ecr": 1.4,
        "b3_points": 22.1,
        "season": 2023,
        "week": 5,
    }
    row.update(kwargs)
    return row


def test_add_b3_resolves_to_the_canonical_player_id() -> None:
    fp_weekly = pl.DataFrame([_fp_weekly_row()])
    players_dim = pl.DataFrame([_players_dim_row()])

    result = baselines.add_b3_fp_weekly_consensus(fp_weekly, players_dim)

    row = result.row(0, named=True)
    assert row["player_id"] == "00-0036442"
    assert row["season"] == 2023
    assert row["week"] == 5
    assert row["b3_points"] == pytest.approx(22.1)


def test_add_b3_drops_a_row_that_never_resolves_to_a_real_player() -> None:
    fp_weekly = pl.DataFrame([_fp_weekly_row(player_name="Totally Unknown Person", pos="QB")])
    players_dim = pl.DataFrame([_players_dim_row()])

    result = baselines.add_b3_fp_weekly_consensus(fp_weekly, players_dim)

    assert result.height == 0


def test_add_b3_does_not_cross_match_different_positions() -> None:
    """Same normalized name, different real position -- must not resolve
    to the wrong player_id."""
    fp_weekly = pl.DataFrame([_fp_weekly_row(player_name="Joe Burrow", pos="RB")])
    players_dim = pl.DataFrame([_players_dim_row(position="QB")])

    result = baselines.add_b3_fp_weekly_consensus(fp_weekly, players_dim)

    assert result.height == 0


# --- fetch_b3_for_week -------------------------------------------------------------------

_FP_WEEKLY_CSV = (
    '"page","player_name","pos","team","ecr","r2p_pts"\n"qb","Joe Burrow","QB","CIN",1.42,"22.1"\n'
)


def test_fetch_b3_for_week_resolves_the_real_commit_before_cutoff(tmp_path, monkeypatch) -> None:
    from ffapp.ingest import rankings

    commits_path = tmp_path / "fp_weekly_commits.json"
    commits_path.write_text(
        json.dumps({"commits": [{"sha": "abc123", "date": "2023-10-01T00:00:00Z"}]})
    )
    snapshot_path = tmp_path / "fp_weekly_abc123.csv"
    snapshot_path.write_text(_FP_WEEKLY_CSV)

    monkeypatch.setattr(rankings, "fetch_fp_weekly_commits", lambda **kwargs: commits_path)

    captured_sha: list[str] = []

    def fake_fetch_snapshot(sha: str, **kwargs: object):
        captured_sha.append(sha)
        return snapshot_path

    monkeypatch.setattr(rankings, "fetch_fp_weekly_snapshot", fake_fetch_snapshot)

    players_dim = pl.DataFrame([_players_dim_row()])
    result = baselines.fetch_b3_for_week(2023, 5, "2023-10-05T00:00:00Z", players_dim, offline=True)

    assert captured_sha == ["abc123"]
    row = result.row(0, named=True)
    assert row["player_id"] == "00-0036442"
    assert row["season"] == 2023
    assert row["week"] == 5
    assert row["b3_points"] == pytest.approx(22.1)


def test_fetch_b3_for_week_is_honestly_empty_when_no_commit_precedes_the_cutoff(
    tmp_path, monkeypatch
) -> None:
    from ffapp.ingest import rankings

    commits_path = tmp_path / "fp_weekly_commits.json"
    commits_path.write_text(
        json.dumps({"commits": [{"sha": "abc123", "date": "2023-10-01T00:00:00Z"}]})
    )
    monkeypatch.setattr(rankings, "fetch_fp_weekly_commits", lambda **kwargs: commits_path)

    def fail_if_called(sha: str, **kwargs: object):
        raise AssertionError("should not fetch a snapshot with no commit before cutoff")

    monkeypatch.setattr(rankings, "fetch_fp_weekly_snapshot", fail_if_called)

    players_dim = pl.DataFrame([_players_dim_row()])
    result = baselines.fetch_b3_for_week(2021, 1, "2021-08-01T00:00:00Z", players_dim, offline=True)

    assert result.is_empty()
    assert result.columns == ["player_id", "season", "week", "b3_points"]


# --- empirical_error_quantiles / apply_empirical_error_quantiles -------------------------


def test_empirical_error_quantiles_computes_the_real_per_position_distribution() -> None:
    rows = pl.DataFrame(
        [
            {"position": "RB", "target": 10.0, "b3_points": 8.0},  # error 2.0
            {"position": "RB", "target": 12.0, "b3_points": 8.0},  # error 4.0
            {"position": "RB", "target": 4.0, "b3_points": 8.0},  # error -4.0
            {"position": "WR", "target": 20.0, "b3_points": 10.0},  # error 10.0
        ]
    )

    result = baselines.empirical_error_quantiles(rows, "b3_points", [0.5])

    assert result["RB"][0.5] == pytest.approx(2.0)  # median of [2.0, 4.0, -4.0]
    assert result["WR"][0.5] == pytest.approx(10.0)


def test_empirical_error_quantiles_is_honestly_empty_for_a_position_with_no_rows() -> None:
    rows = pl.DataFrame([{"position": "RB", "target": 10.0, "b3_points": 8.0}])

    result = baselines.empirical_error_quantiles(rows, "b3_points", [0.5])

    assert result["WR"] == {}


def test_apply_empirical_error_quantiles_adds_the_real_offset() -> None:
    mean = pl.Series("mean", [8.0, 10.0])
    position = pl.Series("position", ["RB", "WR"])
    error_quantiles = {"RB": {0.5: 2.0}, "WR": {0.5: -3.0}}

    result = baselines.apply_empirical_error_quantiles(mean, position, error_quantiles, 0.5)

    assert result.to_list() == pytest.approx([10.0, 7.0])


def test_apply_empirical_error_quantiles_clips_at_zero() -> None:
    mean = pl.Series("mean", [1.0])
    position = pl.Series("position", ["RB"])
    error_quantiles = {"RB": {0.1: -50.0}}

    result = baselines.apply_empirical_error_quantiles(mean, position, error_quantiles, 0.1)

    assert result.to_list() == [0.0]


def test_apply_empirical_error_quantiles_is_null_for_a_position_with_no_recorded_quantiles() -> (
    None
):
    mean = pl.Series("mean", [8.0])
    position = pl.Series("position", ["TE"])
    error_quantiles: dict[str, dict[float, float]] = {"RB": {0.5: 2.0}}

    result = baselines.apply_empirical_error_quantiles(mean, position, error_quantiles, 0.5)

    assert result.to_list() == [None]


# --- pooled_rolling_mean (generalized, replaces _positional_rolling_rate) ---------


def test_pooled_rolling_mean_works_with_a_different_group_column() -> None:
    df = pl.DataFrame(
        {
            "position": ["TEAM_ENV", "TEAM_ENV", "TEAM_ENV", "TEAM_ENV"],
            "season": [2025, 2025, 2025, 2025],
            "week": [1, 1, 2, 2],
            "target": [60.0, 70.0, 65.0, 75.0],
        }
    )

    result = baselines.pooled_rolling_mean(df, "position", "target", "pooled_mean")

    week2 = result.filter(pl.col("week") == 2)
    # week 2's pooled mean is week 1's pooled average across both rows: (60+70)/2 = 65
    assert week2["pooled_mean"].to_list() == pytest.approx([65.0, 65.0])
