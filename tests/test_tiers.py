import polars as pl
import pytest

from ffapp.tools import tiers


def _players(position: str, values: list[float]) -> list[dict]:
    return [
        {"position": position, "player_name": f"{position}{i}", "vor": v}
        for i, v in enumerate(values, start=1)
    ]


# --- gap method --------------------------------------------------------------


def test_gap_method_breaks_a_tier_at_a_single_large_gap() -> None:
    """4 gaps of 1, one huge gap of 20, then 9 more gaps of 1 -- the huge gap
    dominates its local window regardless of centered/trailing windowing
    convention, since the surrounding gaps are uniformly tiny."""
    values = [
        100.0,
        99.0,
        98.0,
        97.0,
        96.0,
        76.0,
        75.0,
        74.0,
        73.0,
        72.0,
        71.0,
        70.0,
        69.0,
        68.0,
        67.0,
    ]
    df = pl.DataFrame(_players("WR", values))

    result = tiers.assign_tiers(df, method="gap")

    high = result.filter(pl.col("vor") >= 96.0)
    low = result.filter(pl.col("vor") <= 76.0)
    assert high["tier"].n_unique() == 1
    assert low["tier"].n_unique() == 1
    assert high["tier"][0] != low["tier"][0]


def test_gap_method_never_produces_a_tier_smaller_than_two_players() -> None:
    """A lone player (100) between two huge gaps must be merged into a
    neighbouring tier rather than standing alone."""
    values = [100.0, 80.0, 79.0, 78.0, 50.0, 49.0, 48.0, 47.0, 46.0]
    df = pl.DataFrame(_players("RB", values))

    result = tiers.assign_tiers(df, method="gap")

    tier_sizes = result.group_by("tier").len()["len"].to_list()
    assert all(size >= 2 for size in tier_sizes)


def test_gap_method_caps_at_twelve_tiers() -> None:
    """15 well-isolated clusters of 10 players each -- comfortably more than
    12 natural gap breaks -- must still collapse to at most 12 tiers."""
    values = []
    for cluster in range(15):
        top = 100_000.0 - cluster * 500.0
        values.extend(top - offset for offset in range(10))
    df = pl.DataFrame(_players("RB", values))

    result = tiers.assign_tiers(df, method="gap")

    assert result["tier"].n_unique() <= 12
    tier_sizes = result.group_by("tier").len()["len"].to_list()
    assert all(size >= 2 for size in tier_sizes)


def test_gap_method_single_player_position_gets_tier_one() -> None:
    df = pl.DataFrame(_players("K", [42.0]))

    result = tiers.assign_tiers(df, method="gap")

    assert result["tier"].to_list() == [1]


# --- shared behaviour across methods ------------------------------------------


@pytest.mark.parametrize("method", ["gap", "kmeans", "gmm"])
def test_every_method_respects_min_tier_size_and_max_tiers(method: str) -> None:
    values = []
    for cluster in range(15):
        top = 100_000.0 - cluster * 500.0
        values.extend(top - offset for offset in range(10))
    df = pl.DataFrame(_players("RB", values))

    result = tiers.assign_tiers(df, method=method)

    tier_sizes = result.group_by("tier").len()["len"].to_list()
    assert all(size >= 2 for size in tier_sizes)
    assert result["tier"].n_unique() <= 12


@pytest.mark.parametrize("method", ["gap", "kmeans", "gmm"])
def test_every_method_assigns_tier_one_to_the_highest_vor_player(method: str) -> None:
    values = [100.0, 99.0, 98.0, 50.0, 49.0, 48.0, 10.0, 9.0, 8.0]
    df = pl.DataFrame(_players("TE", values))

    result = tiers.assign_tiers(df, method=method)

    best = result.sort("vor", descending=True).row(0, named=True)
    assert best["tier"] == 1


def test_tiers_are_computed_independently_per_position() -> None:
    df = pl.DataFrame(
        _players("QB", [400.0, 399.0, 398.0, 200.0, 199.0, 198.0])
        + _players("K", [20.0, 19.0, 18.0, 5.0, 4.0, 3.0])
    )

    result = tiers.assign_tiers(df, method="gap")

    assert (result.filter(pl.col("position") == "QB")["tier"] == 1).any()
    assert (result.filter(pl.col("position") == "K")["tier"] == 1).any()


def test_unknown_method_raises() -> None:
    df = pl.DataFrame(_players("RB", [10.0, 9.0, 8.0]))

    with pytest.raises(ValueError, match="unknown"):
        tiers.assign_tiers(df, method="not-a-real-method")


def test_null_vor_raises() -> None:
    df = pl.DataFrame({"position": ["RB", "RB"], "player_name": ["A", "B"], "vor": [10.0, None]})

    with pytest.raises(ValueError, match="RB"):
        tiers.assign_tiers(df, method="gap")
