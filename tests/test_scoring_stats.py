"""Tests for scoring/stats.py: assembling the per-player-week stat frame that
scoring.engine.score_stat_line consumes, from nflreadpy's player_stats/team_stats/
schedules (SPEC.md §8.4's golden test needs this; task 1.1 will supersede it with
the full interim/player_week_stats.parquet pipeline -- see HANDOFF.md §4).
"""

from __future__ import annotations

import polars as pl

from ffapp.scoring import stats


def _team_stats(**overrides: object) -> pl.DataFrame:
    base = {
        "season": [2025],
        "week": [1],
        "team": ["KC"],
        "opponent_team": ["BAL"],
        "game_id": ["2025_01_BAL_KC"],
        "def_sacks": [3],
        "def_interceptions": [1],
        "def_fumbles_forced": [2],
        "fumble_recovery_opp": [1],
        "fumble_recovery_tds": [0],
        "def_safeties": [0],
        "def_tds": [1],
        "special_teams_tds": [0],
        "fg_blocked": [0],
        "pat_blocked": [0],
        # load_team_stats is a full team box score, not DST-only -- these must
        # never leak into the built DST row (see build_dst_stat_frame docstring).
        "passing_yards": [310],
        "rushing_yards": [140],
        "receptions": [28],
    }
    base.update(overrides)
    return pl.DataFrame(base)


_PBP_SCHEMA = {
    "season": pl.Int64,
    "week": pl.Int64,
    "defteam": pl.Utf8,
    "td_team": pl.Utf8,
    "return_touchdown": pl.Int64,
    "play_type": pl.Utf8,
    "fumble": pl.Int64,
    "special_teams_play": pl.Int64,
    "forced_fumble_player_1_team": pl.Utf8,
    "fumbled_1_team": pl.Utf8,
    "fumble_recovery_1_team": pl.Utf8,
}


def _pbp(*rows: dict[str, object]) -> pl.DataFrame:
    """Zero or more play-by-play rows, each a dict of overrides on top of a
    harmless default (a plain pass play with no TD, no fumble). Pass no rows for
    an empty-but-correctly-typed frame."""
    if not rows:
        return pl.DataFrame(schema=_PBP_SCHEMA)
    default = {
        "season": 2025,
        "week": 1,
        "defteam": None,
        "td_team": None,
        "return_touchdown": 0,
        "play_type": "pass",
        "fumble": 0,
        "special_teams_play": 0,
        "forced_fumble_player_1_team": None,
        "fumbled_1_team": None,
        "fumble_recovery_1_team": None,
    }
    built = [{**default, **row} for row in rows]
    return pl.DataFrame(built, schema=_PBP_SCHEMA)


def _td_play(
    *, week: int = 1, defteam: str, td_team: str, play_type: str = "pass"
) -> dict[str, object]:
    return {
        "week": week,
        "defteam": defteam,
        "td_team": td_team,
        "return_touchdown": 1,
        "play_type": play_type,
    }


def _fumble_play(
    *,
    week: int = 1,
    forced_by: str,
    fumbled_by: str,
    recovered_by: str,
    special_teams: bool,
) -> dict[str, object]:
    return {
        "week": week,
        "fumble": 1,
        "special_teams_play": 1 if special_teams else 0,
        "play_type": "kickoff" if special_teams else "run",
        "forced_fumble_player_1_team": forced_by,
        "fumbled_1_team": fumbled_by,
        "fumble_recovery_1_team": recovered_by,
    }


def _schedules(**overrides: object) -> pl.DataFrame:
    base = {
        "game_id": ["2025_01_BAL_KC"],
        "season": [2025],
        "week": [1],
        "home_team": ["KC"],
        "away_team": ["BAL"],
        "home_score": [27],
        "away_score": [17],
    }
    base.update(overrides)
    return pl.DataFrame(base)


def test_build_dst_stat_frame_derives_points_allowed_from_opponent_score() -> None:
    team_stats = _team_stats()
    schedules = _schedules()

    dst = stats.build_dst_stat_frame(team_stats, schedules, _pbp())

    kc = dst.filter(pl.col("player_id") == "KC").row(0, named=True)
    assert kc["points_allowed"] == 17  # KC is home; allowed the away team's score


def test_build_dst_stat_frame_credits_the_blocking_defense_not_the_kicking_team() -> None:
    """Sleeper's blk_kick scores the defense that forced the block. nflreadpy
    records a blocked kick on the *kicking* team's own row (fg_blocked/pat_blocked),
    so BAL's blocked kick must show up as KC's opponent_blocked_kicks, not BAL's."""
    team_stats = pl.concat(
        [
            _team_stats(team=["KC"], opponent_team=["BAL"], fg_blocked=[0]),
            _team_stats(team=["BAL"], opponent_team=["KC"], fg_blocked=[1], pat_blocked=[0]),
        ]
    )
    schedules = _schedules()

    dst = stats.build_dst_stat_frame(team_stats, schedules, _pbp())

    kc = dst.filter(pl.col("player_id") == "KC").row(0, named=True)
    bal = dst.filter(pl.col("player_id") == "BAL").row(0, named=True)
    assert kc["opponent_blocked_kicks"] == 1  # KC's defense forced BAL's blocked FG
    assert bal["opponent_blocked_kicks"] == 0


def test_build_dst_stat_frame_sets_player_id_to_team_abbreviation_and_position_dst() -> None:
    dst = stats.build_dst_stat_frame(_team_stats(), _schedules(), _pbp())

    row = dst.row(0, named=True)
    assert row["player_id"] == "KC"
    assert row["position"] == "DST"


def test_build_dst_stat_frame_passes_through_core_defense_stats() -> None:
    dst = stats.build_dst_stat_frame(_team_stats(), _schedules(), _pbp())

    row = dst.row(0, named=True)
    assert row["def_sacks"] == 3
    assert row["def_interceptions"] == 1
    assert row["special_teams_tds"] == 0
    # def_fumbles_forced / fumble_recovery_opp are PBP-derived (see below), not
    # passed through from team_stats -- zero here since this fixture's PBP is empty.
    assert row["def_fumbles_forced"] == 0
    assert row["fumble_recovery_opp"] == 0


def test_build_dst_stat_frame_credits_a_genuine_defensive_return_touchdown() -> None:
    """A play-by-play row where the scoring team (`td_team`) was on defense
    (`defteam`) that play -- an interception return or a recovery of the
    *opponent's* fumble -- is a genuine defensive score and must credit the DST.
    Confirmed live: ARI's and BAL's real week-2-2025 sack-fumble-return TDs."""
    pbp = _pbp(_td_play(defteam="KC", td_team="KC"))

    dst = stats.build_dst_stat_frame(_team_stats(), _schedules(), pbp)

    kc = dst.filter(pl.col("player_id") == "KC").row(0, named=True)
    assert kc["def_return_tds"] == 1


def test_build_dst_stat_frame_excludes_punt_and_kickoff_return_touchdowns() -> None:
    """Regression test: nflreadpy's `return_touchdown` flags punt/kickoff returns
    too, not just interception/fumble returns during a scrimmage play. Sleeper
    scores those as special-teams credit (def_st_td/st_td), not def_td -- without
    this exclusion, a punt/kickoff return TD gets double-counted against both.
    Confirmed live: NE's real week-4-2025 87-yard punt-return TD."""
    pbp = _pbp(_td_play(defteam="KC", td_team="KC", play_type="punt"))

    dst = stats.build_dst_stat_frame(_team_stats(), _schedules(), pbp)

    kc = dst.filter(pl.col("player_id") == "KC").row(0, named=True)
    assert kc["def_return_tds"] == 0


def test_build_dst_stat_frame_excludes_offense_recovering_its_own_fumble() -> None:
    """Regression test: an offensive player recovering their OWN team's fumble
    and scoring is not a defensive score. `td_team` won't equal `defteam` on that
    play (the offense was on offense, not defense), so it must not credit the
    DST -- confirmed live: nflreadpy's team-level `fumble_recovery_tds` column
    conflates this with genuine defensive scores (real case: HOU week 15 2025,
    RB W.Marks recovering QB C.Stroud's own fumbled snap), which is why that
    column was dropped from DST credit entirely in favour of this PBP derivation."""
    pbp = _pbp(_td_play(defteam="BAL", td_team="KC"))  # KC's offense scored; BAL was on defense

    dst = stats.build_dst_stat_frame(_team_stats(), _schedules(), pbp)

    kc = dst.filter(pl.col("player_id") == "KC").row(0, named=True)
    assert kc["def_return_tds"] == 0


def test_build_dst_stat_frame_excludes_team_offensive_stat_columns() -> None:
    """load_team_stats carries the team's own full offensive box score alongside
    its defensive columns -- without narrowing the selection, a DST row would also
    pick up the team's passing/rushing/receiving yards and get scored as if the
    defense itself threw for 300 yards. Regression test for that bug."""
    dst = stats.build_dst_stat_frame(_team_stats(), _schedules(), _pbp())

    assert "passing_yards" not in dst.columns
    assert "rushing_yards" not in dst.columns
    assert "receptions" not in dst.columns


# --- scrimmage vs. special-teams fumble credit ------------------------------
# team_stats' own def_fumbles_forced/fumble_recovery_opp aggregate BOTH scrimmage
# and special-teams fumble events together, but Sleeper scores them via two
# entirely separate key pairs (ff/fum_rec for scrimmage; def_st_ff/def_st_fum_rec
# for special teams). Confirmed live: PIT's real week-1-2025 kickoff-return fumble
# (PIT forced it and recovered NYJ's loose ball) was already being counted under
# team_stats' general def_fumbles_forced/fumble_recovery_opp, inflating PIT's
# `ff`/`fum_rec` credit by exactly the amount PIT was over-scored by -- so simply
# adding separate special-teams credit on top would have double-counted the same
# event. Both `ff`/`fum_rec` and `def_st_ff`/`def_st_fum_rec` are derived here
# from play-by-play's structured fumble columns, split by `special_teams_play`.
# "Recovered" only counts as a turnover (`fumble_recovery_1_team`) when it's the
# *opponent's* fumble -- a team recovering its own fumble isn't a defensive event.


def test_build_dst_stat_frame_credits_a_scrimmage_forced_fumble() -> None:
    pbp = _pbp(
        _fumble_play(forced_by="KC", fumbled_by="BAL", recovered_by="BAL", special_teams=False)
    )

    dst = stats.build_dst_stat_frame(_team_stats(), _schedules(), pbp)

    kc = dst.filter(pl.col("player_id") == "KC").row(0, named=True)
    assert kc["def_fumbles_forced"] == 1
    assert kc["fumble_recovery_opp"] == 0  # BAL recovered its own fumble -- not a turnover


def test_build_dst_stat_frame_credits_a_scrimmage_opponent_fumble_recovery() -> None:
    pbp = _pbp(
        _fumble_play(forced_by="KC", fumbled_by="BAL", recovered_by="KC", special_teams=False)
    )

    dst = stats.build_dst_stat_frame(_team_stats(), _schedules(), pbp)

    kc = dst.filter(pl.col("player_id") == "KC").row(0, named=True)
    assert kc["fumble_recovery_opp"] == 1


def test_build_dst_stat_frame_credits_special_teams_forced_fumble_and_recovery_separately() -> None:
    """Real case: PIT's week-1-2025 kickoff -- PIT forced NYJ's fumble and
    recovered it, both on a special-teams play. Must land in def_st_ff/
    def_st_fum_rec, not the scrimmage ff/fum_rec pool."""
    pbp = _pbp(
        _fumble_play(forced_by="PIT", fumbled_by="NYJ", recovered_by="PIT", special_teams=True)
    )

    dst = stats.build_dst_stat_frame(
        _team_stats(team=["PIT"], opponent_team=["NYJ"]), _schedules(), pbp
    )

    pit = dst.filter(pl.col("player_id") == "PIT").row(0, named=True)
    assert pit["special_teams_forced_fumbles"] == 1
    assert pit["special_teams_fumble_recoveries"] == 1
    assert pit["def_fumbles_forced"] == 0  # not double-counted into the scrimmage pool
    assert pit["fumble_recovery_opp"] == 0


def test_build_dst_stat_frame_excludes_a_team_recovering_its_own_special_teams_fumble() -> None:
    """The other half of PIT's real week-1-2025 game: NYJ forced PIT's own kickoff
    fumble, but PIT recovered its own ball -- not a turnover, so PIT gets no
    special_teams_fumble_recoveries credit, only NYJ gets special_teams_forced_fumbles."""
    pbp = _pbp(
        _fumble_play(forced_by="NYJ", fumbled_by="PIT", recovered_by="PIT", special_teams=True)
    )

    dst = stats.build_dst_stat_frame(
        pl.concat(
            [
                _team_stats(team=["PIT"], opponent_team=["NYJ"]),
                _team_stats(team=["NYJ"], opponent_team=["PIT"]),
            ]
        ),
        _schedules(),
        pbp,
    )

    pit = dst.filter(pl.col("player_id") == "PIT").row(0, named=True)
    nyj = dst.filter(pl.col("player_id") == "NYJ").row(0, named=True)
    assert pit["special_teams_fumble_recoveries"] == 0
    assert nyj["special_teams_forced_fumbles"] == 1


def test_build_player_stat_frame_sets_player_id_and_keeps_position() -> None:
    player_stats = pl.DataFrame(
        {
            "player_id": ["00-0031234"],
            "season": [2025],
            "week": [1],
            "position": ["QB"],
            "passing_yards": [250],
        }
    )

    built = stats.build_player_stat_frame(player_stats)

    row = built.row(0, named=True)
    assert row["player_id"] == "00-0031234"
    assert row["position"] == "QB"
    assert row["passing_yards"] == 250


def test_build_stat_frame_concatenates_player_and_dst_rows_without_dropping_either() -> None:
    player_stats = pl.DataFrame(
        {
            "player_id": ["00-0031234"],
            "season": [2025],
            "week": [1],
            "position": ["QB"],
            "passing_yards": [250],
        }
    )
    team_stats = _team_stats()
    schedules = _schedules()

    combined = stats.build_stat_frame(player_stats, team_stats, schedules, _pbp())

    player_ids = set(combined["player_id"].to_list())
    assert player_ids == {"00-0031234", "KC"}
    assert combined.height == 2
