import polars as pl
import pytest

from ffapp.projections import aggregate

NON_PPR_SCORING = {
    "pass_yd": 0.04,
    "pass_td": 4,
    "pass_int": -1,
    "rush_yd": 0.1,
    "rush_td": 6,
    "rec": 0,  # non-PPR: receptions themselves are worth nothing
    "rec_yd": 0.1,
    "rec_td": 6,
}

PPR_SCORING = {**NON_PPR_SCORING, "rec": 1}


def _source_row(**kwargs: object) -> dict:
    row = {
        "source": "fixture",
        "season": 2026,
        "player_name": "Test Player",
        "position": "RB",
        "team": "KC",
    }
    row.update(kwargs)
    return row


# --- apply_league_scoring -----------------------------------------------------


def test_apply_league_scoring_adds_points_column() -> None:
    df = pl.DataFrame([_source_row(rushing_yards=1000.0, rushing_tds=10.0)])

    scored = aggregate.apply_league_scoring(df, NON_PPR_SCORING)

    # 1000*0.1 + 10*6 = 160
    assert scored["points"][0] == pytest.approx(160.0)


def test_apply_league_scoring_does_not_crash_on_scoring_keys_the_source_lacks() -> None:
    """A real league's scoring_settings includes keys (blk_kick, def_st_td,
    ...) that a projections source never publishes -- score_stat_line's
    DirectStat path does `stats[column]`, which raises ColumnNotFoundError
    on an absent column, not just a null one. apply_league_scoring must fill
    in every column any STAT_KEY_MAP entry could reference before scoring,
    so an unpublished stat contributes 0 instead of crashing the source."""
    df = pl.DataFrame([_source_row(rushing_yards=500.0)])
    scoring_with_unpublished_keys = {
        **NON_PPR_SCORING,
        "blk_kick": 2,  # opponent_blocked_kicks -- no projections source has this
        "def_st_td": 6,  # special_teams_tds, DST-gated -- not in a skill-position row
    }

    scored = aggregate.apply_league_scoring(df, scoring_with_unpublished_keys)

    assert scored["points"][0] == pytest.approx(50.0)  # 500 * 0.1, everything else 0


def test_apply_league_scoring_rescales_ppr_vs_non_ppr_differently() -> None:
    """SPEC §9.2 point 2: rescale per source BEFORE aggregating -- proven
    here by the same raw row producing different points under PPR vs
    non-PPR scoring."""
    df = pl.DataFrame([_source_row(receiving_yards=0.0, receptions=80.0)])

    non_ppr = aggregate.apply_league_scoring(df, NON_PPR_SCORING)
    ppr = aggregate.apply_league_scoring(df, PPR_SCORING)

    assert non_ppr["points"][0] == pytest.approx(0.0)
    assert ppr["points"][0] == pytest.approx(80.0)


# --- add_join_key --------------------------------------------------------------


def test_add_join_key_normalizes_name_and_keeps_position() -> None:
    df = pl.DataFrame(
        [
            _source_row(player_name="A.J. Brown", position="WR"),
            _source_row(player_name="AJ Brown", position="WR"),
        ]
    )

    keyed = aggregate.add_join_key(df)

    assert keyed["join_key"][0] == keyed["join_key"][1]


def test_add_join_key_distinguishes_same_name_different_position() -> None:
    df = pl.DataFrame(
        [
            _source_row(player_name="Josh Allen", position="QB"),
            _source_row(player_name="Josh Allen", position="LB"),
        ]
    )

    keyed = aggregate.add_join_key(df)

    assert keyed["join_key"][0] != keyed["join_key"][1]


# --- add_join_key: DST cross-source name canonicalization -----------------------
#
# Confirmed live against every real per-stat source's own actual output
# (2026 season, this project's own data): ESPN spells DST "{Nickname}
# D/ST" ("Texans D/ST"), FantasySharks/FFToday spell it the full "{City}
# {Nickname}" ("Houston Texans"), CBS spells it just the city ("Houston"),
# or a disambiguated "L.A."/"N.Y." prefix for the two shared-city pairs.
# Before DST_TEAM_ABBREVIATIONS existed, none of these four spellings
# normalized to the same join_key, so a real DST never aggregated across
# sources at all -- every one of the 32 real teams showed n_sources=1,
# dispersion=0.0 on the real board.


def test_add_join_key_merges_the_same_real_dst_spelled_four_different_ways() -> None:
    df = pl.DataFrame(
        [
            _source_row(player_name="Texans D/ST", position="DST"),  # ESPN
            _source_row(player_name="Houston Texans", position="DST"),  # FantasySharks/FFToday
            _source_row(player_name="Houston", position="DST"),  # CBS
        ]
    )

    keyed = aggregate.add_join_key(df)

    assert keyed["join_key"].n_unique() == 1
    assert keyed["player_name"].to_list() == ["Houston Texans"] * 3


def test_add_join_key_disambiguates_the_two_shared_city_dst_pairs() -> None:
    df = pl.DataFrame(
        [
            _source_row(player_name="L.A. Rams", position="DST"),
            _source_row(player_name="L.A. Chargers", position="DST"),
            _source_row(player_name="N.Y. Giants", position="DST"),
            _source_row(player_name="N.Y. Jets", position="DST"),
        ]
    )

    keyed = aggregate.add_join_key(df)

    assert keyed["join_key"].n_unique() == 4  # all four stay genuinely distinct
    assert keyed["player_name"].to_list() == [
        "Los Angeles Rams",
        "Los Angeles Chargers",
        "New York Giants",
        "New York Jets",
    ]


def test_add_join_key_leaves_an_unrecognised_dst_name_uncanonicalised_not_crashed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    df = pl.DataFrame([_source_row(player_name="Mystery Team FC", position="DST")])

    with caplog.at_level("WARNING"):
        keyed = aggregate.add_join_key(df)

    assert keyed["player_name"][0] == "Mystery Team FC"
    assert any("unrecognised" in record.message for record in caplog.records)


def test_add_join_key_does_not_touch_non_dst_rows_that_share_a_dst_style_name() -> None:
    """A real skill player who happens to share text with a DST spelling
    (unlikely, but the guard is `position == "DST"`, not a name match)
    must not get rewritten."""
    df = pl.DataFrame([_source_row(player_name="Houston", position="WR")])

    keyed = aggregate.add_join_key(df)

    assert keyed["player_name"][0] == "Houston"


def test_dst_canonical_names_and_abbreviations_cover_all_32_real_teams() -> None:
    assert len(aggregate.DST_CANONICAL_NAMES) == 32
    # 4 real spellings per team (ESPN / FantasySharks-FFToday / CBS / ADP),
    # all 32 teams.
    assert len(aggregate.DST_TEAM_ABBREVIATIONS) == 32 * 4


def test_add_join_key_recognises_the_adp_sources_defense_spelling() -> None:
    df = pl.DataFrame(
        [
            _source_row(player_name="Houston Defense", position="DST"),
            _source_row(player_name="Houston Texans", position="DST"),
        ]
    )

    keyed = aggregate.add_join_key(df)

    assert keyed["join_key"].n_unique() == 1
    assert keyed["player_name"].to_list() == ["Houston Texans", "Houston Texans"]


# --- build_reference_curve / map_ranks_to_points -------------------------------


def test_build_reference_curve_is_the_median_points_at_each_positional_rank() -> None:
    source_a = pl.DataFrame(
        [
            _source_row(player_name="P1", position="WR", points=100.0),
            _source_row(player_name="P2", position="WR", points=80.0),
        ]
    )
    source_b = pl.DataFrame(
        [
            _source_row(player_name="P1", position="WR", points=90.0),
            _source_row(player_name="P2", position="WR", points=70.0),
        ]
    )

    curve = aggregate.build_reference_curve([source_a, source_b])

    rank1 = curve.filter((pl.col("position") == "WR") & (pl.col("rank") == 1))
    rank2 = curve.filter((pl.col("position") == "WR") & (pl.col("rank") == 2))
    assert rank1["ref_points"][0] == pytest.approx(95.0)  # median(100, 90)
    assert rank2["ref_points"][0] == pytest.approx(75.0)  # median(80, 70)


def test_map_ranks_to_points_looks_up_position_and_rounded_rank() -> None:
    reference_curve = pl.DataFrame(
        {"position": ["WR", "WR"], "rank": [1, 2], "ref_points": [95.0, 75.0]}
    )
    rank_only = pl.DataFrame([_source_row(player_name="Ranked Player", position="WR", rank=1.6)])

    mapped = aggregate.map_ranks_to_points(rank_only, reference_curve)

    assert mapped["points"][0] == pytest.approx(75.0)  # rounds 1.6 -> rank 2


def test_map_ranks_to_points_leaves_points_null_when_no_curve_entry() -> None:
    """Thin coverage at the tail: SPEC §9.2 says flag, don't drop."""
    reference_curve = pl.DataFrame({"position": ["WR"], "rank": [1], "ref_points": [95.0]})
    rank_only = pl.DataFrame([_source_row(player_name="Deep Sleeper", position="WR", rank=99.0)])

    mapped = aggregate.map_ranks_to_points(rank_only, reference_curve)

    assert mapped["points"][0] is None


# --- aggregate_projections ------------------------------------------------------


def test_aggregate_projections_computes_trimmed_mean_dispersion_coverage() -> None:
    sources = [
        aggregate.add_join_key(
            pl.DataFrame([_source_row(player_name="Star Player", position="RB", points=p)])
        )
        for p in [100.0, 110.0, 90.0, 200.0, 10.0]  # 5 sources; trim drops 200 and 10
    ]

    result = aggregate.aggregate_projections(sources, n_sources=5)

    row = result.row(0, named=True)
    assert row["n_sources"] == 5
    assert row["coverage"] == pytest.approx(1.0)
    assert row["proj_points"] == pytest.approx(100.0)  # mean(90, 100, 110)


def test_aggregate_projections_coverage_reflects_thin_source_count() -> None:
    sources = [
        aggregate.add_join_key(
            pl.DataFrame([_source_row(player_name="Thin Player", position="RB", points=50.0)])
        )
        for _ in range(2)
    ]

    result = aggregate.aggregate_projections(sources, n_sources=4)

    row = result.row(0, named=True)
    assert row["n_sources"] == 2
    assert row["coverage"] == pytest.approx(0.5)


def test_aggregate_projections_excludes_null_points_rows() -> None:
    covered = aggregate.add_join_key(
        pl.DataFrame([_source_row(player_name="P1", position="RB", points=50.0)])
    )
    uncovered = aggregate.add_join_key(
        pl.DataFrame([_source_row(player_name="P1", position="RB", points=None)])
    )

    result = aggregate.aggregate_projections([covered, uncovered], n_sources=2)

    row = result.row(0, named=True)
    assert row["n_sources"] == 1  # the null-points row doesn't count as coverage


def test_add_join_key_merges_a_known_player_nickname_alias() -> None:
    """Real bug: one source spells Kenneth Walker III as "Ken Walker III",
    which normalizes to "ken walker" -- distinct from every other source's
    "kenneth walker" -- so he showed up twice on the board, each with only
    partial source coverage. `_PLAYER_NAME_ALIASES` rewrites the alias
    spelling to the canonical one before the join_key is built."""
    df = pl.DataFrame(
        [
            _source_row(player_name="Ken Walker III", position="RB"),
            _source_row(player_name="Kenneth Walker III", position="RB"),
        ]
    )

    keyed = aggregate.add_join_key(df)

    assert keyed["join_key"].n_unique() == 1
    assert keyed["player_name"].to_list() == ["Kenneth Walker III", "Kenneth Walker III"]


def test_aggregate_projections_merges_sources_that_spell_a_name_differently() -> None:
    """Real bug found via task 0.14's replay testing: one source spelling a
    player "James Cook" and another "James Cook III" both normalize to the
    same join_key, but the old groupby (join_key + literal player_name)
    split his real 3-source coverage into two separate rows. Grouping by
    join_key alone merges them into one player with full coverage."""
    plain = aggregate.add_join_key(
        pl.DataFrame([_source_row(player_name="James Cook", position="RB", points=200.0)])
    )
    suffixed = aggregate.add_join_key(
        pl.DataFrame([_source_row(player_name="James Cook III", position="RB", points=220.0)])
    )

    result = aggregate.aggregate_projections([plain, suffixed], n_sources=2)

    assert result.height == 1
    row = result.row(0, named=True)
    assert row["n_sources"] == 2
    assert row["coverage"] == pytest.approx(1.0)
    assert row["proj_points"] == pytest.approx(210.0)  # mean(200, 220) -- both sources counted


# --- full pipeline: SPEC §9.2 acceptance shape ----------------------------------


def test_aggregate_pipeline_end_to_end_with_four_point_sources_and_non_ppr_scoring() -> None:
    """Task 0.7's literal acceptance bar: a per-player table with
    proj_points/dispersion/n_sources/coverage, sourced from >=4 providers,
    with league scoring applied before aggregation -- proven here with a
    non-PPR fixture where a reception-heavy player's rank would differ from
    a generic-PPR ranking."""
    raw_stat_sources = {
        "espn": [(1000.0, 8.0, 60.0)],
        "fantasysharks": [(950.0, 9.0, 55.0)],
        "cbs": [(1050.0, 7.0, 65.0)],
        "fftoday": [(900.0, 10.0, 50.0)],
    }
    scored = []
    for source_name, rows in raw_stat_sources.items():
        df = pl.DataFrame(
            [
                _source_row(
                    source=source_name,
                    player_name="Bell Cow Back",
                    position="RB",
                    rushing_yards=rushing_yards,
                    rushing_tds=rushing_tds,
                    receptions=receptions,
                )
                for rushing_yards, rushing_tds, receptions in rows
            ]
        )
        scored_df = aggregate.apply_league_scoring(df, NON_PPR_SCORING)
        scored.append(aggregate.add_join_key(scored_df))

    result = aggregate.aggregate_projections(scored, n_sources=4)

    assert set(result.columns) >= {"proj_points", "dispersion", "n_sources", "coverage"}
    row = result.row(0, named=True)
    assert row["n_sources"] == 4
    assert row["coverage"] == pytest.approx(1.0)
    # Under non-PPR, receptions contribute 0 -- proj_points must reflect
    # ONLY rushing_yards*0.1 + rushing_tds*6 per source, not a PPR total.
    # trim=0.2 on n=4 sources drops int(4*0.2)=0 from each end, so this is
    # just the plain mean of all four sources' non-PPR points.
    expected_points = [
        ry * 0.1 + rt * 6 for ry, rt, _rec in (rows[0] for rows in raw_stat_sources.values())
    ]
    assert row["proj_points"] == pytest.approx(sum(expected_points) / len(expected_points))
    assert row["dispersion"] >= 0.0
