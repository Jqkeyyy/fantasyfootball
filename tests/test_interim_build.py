import polars as pl
import pytest

from ffapp.interim import build

# --- build_team_week_context -----------------------------------------------------


def _pbp_row(**kwargs: object) -> dict:
    row: dict[str, object] = {
        "season": 2025,
        "week": 1,
        "posteam": "KC",
        "defteam": "BAL",
        "play_type": "pass",
        "epa": 0.0,
        "success": 0,
        "yardline_100": 50,
        "receiver_player_id": None,
        "rusher_player_id": None,
    }
    row.update(kwargs)
    return row


def _pbp(rows: list[dict]) -> pl.DataFrame:
    """Real nflverse pbp always types receiver_player_id/rusher_player_id as
    Utf8 -- a small fixture where one of them is None in every row would
    otherwise get inferred as Null dtype, breaking the join against a real
    Utf8 column downstream. Force the real dtype explicitly."""
    return pl.DataFrame(
        rows, schema_overrides={"receiver_player_id": pl.Utf8, "rusher_player_id": pl.Utf8}
    )


def test_build_team_week_context_computes_plays_pass_rate_epa_success() -> None:
    pbp = _pbp(
        [
            _pbp_row(play_type="pass", epa=1.0, success=1),
            _pbp_row(play_type="pass", epa=-1.0, success=0),
            _pbp_row(play_type="run", epa=0.5, success=1),
            _pbp_row(play_type="punt", epa=0.0, success=0),  # not scrimmage -- excluded
        ]
    )

    result = build.build_team_week_context(pbp)

    kc = result.filter(pl.col("team") == "KC").row(0, named=True)
    assert kc["plays"] == 3  # punt excluded
    assert kc["pass_rate"] == pytest.approx(2 / 3)
    assert kc["epa_per_play_off"] == pytest.approx((1.0 - 1.0 + 0.5) / 3)
    assert kc["success_rate_off"] == pytest.approx(2 / 3)


def test_build_team_week_context_leaves_deferred_columns_null() -> None:
    pbp = _pbp([_pbp_row()])

    result = build.build_team_week_context(pbp)

    row = result.row(0, named=True)
    assert row["proe"] is None
    assert row["neutral_pace_sec"] is None
    assert row["implied_total"] is None
    assert row["spread"] is None


# --- _player_position_by_season ---------------------------------------------------


def test_player_position_by_season_takes_the_players_own_position() -> None:
    player_stats = pl.DataFrame(
        {
            "player_id": ["1", "1", "2"],
            "season": [2025, 2025, 2025],
            "week": [1, 2, 1],
            "position": ["WR", "WR", "RB"],
        }
    )

    result = build._player_position_by_season(player_stats)

    positions = {row["player_id"]: row["position"] for row in result.iter_rows(named=True)}
    assert positions == {"1": "WR", "2": "RB"}


# --- build_defense_position_allowed -----------------------------------------------


def test_build_defense_position_allowed_counts_plays_by_position_group() -> None:
    pbp = _pbp(
        [
            _pbp_row(play_type="pass", receiver_player_id="wr1", defteam="BAL"),
            _pbp_row(play_type="pass", receiver_player_id="te1", defteam="BAL"),
            _pbp_row(play_type="run", rusher_player_id="rb1", defteam="BAL"),
            _pbp_row(play_type="run", rusher_player_id="qb1", defteam="BAL"),
            _pbp_row(play_type="pass", receiver_player_id="rb1", defteam="BAL"),
        ]
    )
    player_stats = pl.DataFrame(
        {
            "player_id": ["wr1", "te1", "rb1", "qb1"],
            "season": [2025, 2025, 2025, 2025],
            "week": [1, 1, 1, 1],
            "position": ["WR", "TE", "RB", "QB"],
        }
    )

    result = build.build_defense_position_allowed(pbp, player_stats)

    groups = {row["position_group"]: row["n_plays"] for row in result.iter_rows(named=True)}
    assert groups == {"WR": 1, "TE": 1, "RB_rushing": 1, "QB_rushing": 1, "RB_receiving": 1}


def test_build_defense_position_allowed_leaves_adjusted_columns_null() -> None:
    pbp = _pbp([_pbp_row(play_type="pass", receiver_player_id="wr1")])
    player_stats = pl.DataFrame(
        {"player_id": ["wr1"], "season": [2025], "week": [1], "position": ["WR"]}
    )

    result = build.build_defense_position_allowed(pbp, player_stats)

    row = result.row(0, named=True)
    assert row["adj_epa_allowed"] is None
    assert row["adj_success_allowed"] is None
    assert row["adj_ypt_allowed"] is None
    assert row["adj_td_rate_allowed"] is None


def test_build_defense_position_allowed_drops_plays_with_no_position_match() -> None:
    """A play whose receiver/rusher isn't in player_stats (unresolvable
    position) contributes to no group rather than crashing or fabricating
    one -- the position_group-null filter drops it cleanly."""
    pbp = _pbp([_pbp_row(play_type="pass", receiver_player_id="unknown_guy")])
    player_stats = pl.DataFrame(
        {"player_id": ["wr1"], "season": [2025], "week": [1], "position": ["WR"]}
    )

    result = build.build_defense_position_allowed(pbp, player_stats)

    assert result.height == 0


# --- _snap_counts_by_player_id -----------------------------------------------------


def test_snap_counts_by_player_id_resolves_via_pfr_crosswalk() -> None:
    snap_counts = pl.DataFrame(
        {
            "pfr_player_id": ["MahoPa00"],
            "season": [2025],
            "week": [1],
            "offense_snaps": [65],
            "offense_pct": [1.0],
        }
    )
    players_dim = pl.DataFrame({"pfr_id": ["MahoPa00"], "player_id": ["00-0033873"]})

    result = build._snap_counts_by_player_id(snap_counts, players_dim)

    row = result.row(0, named=True)
    assert row["player_id"] == "00-0033873"
    assert row["offense_snaps"] == 65
    assert row["offense_snap_pct"] == 1.0


def test_snap_counts_by_player_id_leaves_unresolvable_rows_null() -> None:
    snap_counts = pl.DataFrame(
        {
            "pfr_player_id": ["NoMatch00"],
            "season": [2025],
            "week": [1],
            "offense_snaps": [10],
            "offense_pct": [0.2],
        }
    )
    players_dim = pl.DataFrame({"pfr_id": ["SomeoneElse00"], "player_id": ["00-1111111"]})

    result = build._snap_counts_by_player_id(snap_counts, players_dim)

    assert result.height == 1
    assert result.row(0, named=True)["player_id"] is None


# --- _red_zone_touch_counts ---------------------------------------------------------


def test_red_zone_touch_counts_filters_by_yardline() -> None:
    pbp = _pbp(
        [
            _pbp_row(play_type="pass", receiver_player_id="wr1", yardline_100=15),  # RZ
            _pbp_row(play_type="pass", receiver_player_id="wr1", yardline_100=50),  # not RZ
            _pbp_row(play_type="run", rusher_player_id="rb1", yardline_100=3),  # RZ + GZ
            _pbp_row(play_type="run", rusher_player_id="rb1", yardline_100=18),  # RZ only
        ]
    )

    result = build._red_zone_touch_counts(pbp)

    rows = {row["player_id"]: row for row in result.iter_rows(named=True)}
    assert rows["wr1"]["rz_targets"] == 1
    assert rows["rb1"]["rz_carries"] == 2
    assert rows["rb1"]["gz_carries"] == 1


# --- build_player_week_usage (integration) -----------------------------------------


def test_build_player_week_usage_pulls_share_columns_straight_from_player_stats() -> None:
    player_stats = pl.DataFrame(
        {
            "player_id": ["wr1"],
            "season": [2025],
            "week": [1],
            "team": ["KC"],
            "targets": [8],
            "target_share": [0.3],
            "receiving_air_yards": [80],
            "air_yards_share": [0.4],
            "wopr": [0.55],
            "carries": [0],
        }
    )
    snap_counts = pl.DataFrame(
        {"pfr_player_id": [], "season": [], "week": [], "offense_snaps": [], "offense_pct": []},
        schema={
            "pfr_player_id": pl.Utf8,
            "season": pl.Int64,
            "week": pl.Int64,
            "offense_snaps": pl.Float64,
            "offense_pct": pl.Float64,
        },
    )
    players_dim = pl.DataFrame({"pfr_id": ["x"], "player_id": ["y"]})
    pbp = _pbp([_pbp_row(play_type="pass", receiver_player_id="wr1", yardline_100=50)])

    result = build.build_player_week_usage(player_stats, snap_counts, pbp, players_dim)

    row = result.row(0, named=True)
    assert row["target_share"] == 0.3
    assert row["air_yards_share"] == 0.4
    assert row["wopr"] == 0.55
    assert row["adot"] == pytest.approx(10.0)  # 80 / 8
    assert row["route_participation"] is None
    assert row["xfp"] is None


def test_build_player_week_usage_carry_share_guards_against_zero_team_carries() -> None:
    player_stats = pl.DataFrame(
        {
            "player_id": ["qb1"],
            "season": [2025],
            "week": [1],
            "team": ["KC"],
            "targets": [0],
            "target_share": [0.0],
            "receiving_air_yards": [0],
            "air_yards_share": [0.0],
            "wopr": [0.0],
            "carries": [0],
        }
    )
    snap_counts = pl.DataFrame(
        {"pfr_player_id": [], "season": [], "week": [], "offense_snaps": [], "offense_pct": []},
        schema={
            "pfr_player_id": pl.Utf8,
            "season": pl.Int64,
            "week": pl.Int64,
            "offense_snaps": pl.Float64,
            "offense_pct": pl.Float64,
        },
    )
    players_dim = pl.DataFrame({"pfr_id": ["x"], "player_id": ["y"]})
    pbp = _pbp([_pbp_row(play_type="pass")])

    result = build.build_player_week_usage(player_stats, snap_counts, pbp, players_dim)

    row = result.row(0, named=True)
    assert row["carry_share"] is None
    assert row["adot"] is None  # 0 targets -- guarded, not 0/0
