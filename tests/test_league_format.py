from ffapp.config import LeagueConfig
from ffapp.league_format import LeagueFormat, parse_league_format


def _league(
    roster_positions: list[str],
    total_rosters: int = 12,
    playoff_week_start: int = 15,
    waiver_budget: int | None = 100,
    overrides: dict[str, list[str]] | None = None,
) -> LeagueConfig:
    return LeagueConfig(
        slug="fixture-league",
        display_name="Fixture League",
        is_primary=True,
        league_id="1",
        season=2026,
        league_cache={
            "total_rosters": total_rosters,
            "roster_positions": roster_positions,
            "playoff_week_start": playoff_week_start,
            "waiver_budget": waiver_budget,
        },
        overrides=overrides or {},
    )


def test_parses_standard_12_team_single_flex_format() -> None:
    league = _league(
        [
            "QB",
            "RB",
            "RB",
            "WR",
            "WR",
            "TE",
            "FLEX",
            "K",
            "DEF",
            "BN",
            "BN",
            "BN",
            "BN",
            "BN",
            "BN",
        ],
        overrides={"flex_eligible": ["RB", "WR", "TE"]},
    )

    fmt = parse_league_format(league)

    assert fmt == LeagueFormat(
        n_teams=12,
        starters={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1},
        flex_slots={"FLEX": 1, "SUPER_FLEX": 0, "REC_FLEX": 0},
        flex_eligible={"FLEX": ["RB", "WR", "TE"]},
        bench=6,
        ir=0,
        playoff_week_start=15,
        waiver_budget=100,
    )


def test_parses_two_flex_format() -> None:
    league = _league(
        [
            "QB",
            "RB",
            "RB",
            "WR",
            "WR",
            "TE",
            "FLEX",
            "FLEX",
            "K",
            "DEF",
            "BN",
            "BN",
            "BN",
            "BN",
            "BN",
            "BN",
        ],
        total_rosters=10,
        overrides={"flex_eligible": ["RB", "WR", "TE"]},
    )

    fmt = parse_league_format(league)

    assert fmt.n_teams == 10
    assert fmt.flex_slots == {"FLEX": 2, "SUPER_FLEX": 0, "REC_FLEX": 0}
    assert fmt.flex_eligible == {"FLEX": ["RB", "WR", "TE"]}
    assert fmt.starters == {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1}


def test_parses_superflex_format_with_ir() -> None:
    league = _league(
        [
            "QB",
            "QB",
            "RB",
            "RB",
            "WR",
            "WR",
            "TE",
            "SUPER_FLEX",
            "K",
            "DEF",
            "BN",
            "BN",
            "BN",
            "BN",
            "IR",
        ],
        overrides={"superflex_eligible": ["QB", "RB", "WR", "TE"]},
    )

    fmt = parse_league_format(league)

    assert fmt.starters == {"QB": 2, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DST": 1}
    assert fmt.flex_slots == {"FLEX": 0, "SUPER_FLEX": 1, "REC_FLEX": 0}
    assert fmt.flex_eligible == {"SUPER_FLEX": ["QB", "RB", "WR", "TE"]}
    assert fmt.bench == 4
    assert fmt.ir == 1


def test_parses_rec_flex_slot() -> None:
    league = _league(
        ["QB", "RB", "RB", "WR", "WR", "TE", "REC_FLEX", "K", "DEF", "BN"],
        overrides={"rec_flex_eligible": ["WR", "TE"]},
    )

    fmt = parse_league_format(league)

    assert fmt.flex_slots == {"FLEX": 0, "SUPER_FLEX": 0, "REC_FLEX": 1}
    assert fmt.flex_eligible == {"REC_FLEX": ["WR", "TE"]}


def test_flex_eligible_falls_back_to_default_when_override_absent() -> None:
    league = _league(["QB", "RB", "WR", "TE", "FLEX", "K", "DEF", "BN"], overrides={})

    fmt = parse_league_format(league)

    assert fmt.flex_eligible == {"FLEX": ["RB", "WR", "TE"]}


def test_taxi_slots_are_not_counted_as_bench_or_starters() -> None:
    league = _league(["QB", "RB", "WR", "TE", "K", "DEF", "BN", "TAXI", "TAXI"])

    fmt = parse_league_format(league)

    assert fmt.bench == 1
    assert "TAXI" not in fmt.starters
    assert sum(fmt.starters.values()) == 6


def test_waiver_budget_none_when_no_faab() -> None:
    league = _league(["QB", "RB", "WR", "TE", "K", "DEF"], waiver_budget=None)

    fmt = parse_league_format(league)

    assert fmt.waiver_budget is None


def test_n_teams_and_playoff_week_start_default_to_zero_when_key_present_but_none() -> None:
    """cache/registry.py writes total_rosters/playoff_week_start with value None
    (not an absent key) whenever Sleeper's own response omits the field."""
    league = _league(["QB", "RB", "WR", "TE", "K", "DEF"])
    league.league_cache["total_rosters"] = None
    league.league_cache["playoff_week_start"] = None

    fmt = parse_league_format(league)

    assert fmt.n_teams == 0
    assert fmt.playoff_week_start == 0
