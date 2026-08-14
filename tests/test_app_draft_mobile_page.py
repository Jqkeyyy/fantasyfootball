import polars as pl

from ffapp.app import draft_mobile_page
from ffapp.draft import live


def _pool() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player": ["Elite RB", "Good WR", "Deep Bench TE", "Backup QB"],
            "position": ["RB", "WR", "TE", "QB"],
            "team": ["DET", "CIN", "KC", "BUF"],
            "bye_week": [6, 10, 12, 7],
            "tier": [1, 1, 3, 2],
            "vor": [150.0, 90.0, 10.0, 40.0],
            "p_avail_next": [0.1, 0.6, None, 0.71],
        }
    )


# --- filter_pool_by_position -------------------------------------------------------


def test_filter_pool_by_position_filters_to_one_position() -> None:
    result = draft_mobile_page.filter_pool_by_position(_pool(), "RB")

    assert result["player"].to_list() == ["Elite RB"]


def test_filter_pool_by_position_all_returns_everything() -> None:
    result = draft_mobile_page.filter_pool_by_position(_pool(), "ALL")

    assert result.height == 4


def test_filter_pool_by_position_none_returns_everything() -> None:
    result = draft_mobile_page.filter_pool_by_position(_pool(), None)

    assert result.height == 4


# --- format_why_line ---------------------------------------------------------------


def test_format_why_line_includes_tier_remaining_and_survival_probability() -> None:
    row = {"position": "QB", "tier": 2, "p_avail_next": 0.71}

    result = draft_mobile_page.format_why_line(row, {("QB", 2): 3})

    assert result == "Tier 2 · 3 left · falls to you 71%"


def test_format_why_line_omits_remaining_when_tier_not_in_lookup() -> None:
    row = {"position": "TE", "tier": 3, "p_avail_next": None}

    result = draft_mobile_page.format_why_line(row, {})

    assert result == "Tier 3"


def test_format_why_line_omits_survival_probability_when_null() -> None:
    row = {"position": "TE", "tier": 3, "p_avail_next": None}

    result = draft_mobile_page.format_why_line(row, {("TE", 3): 5})

    assert result == "Tier 3 · 5 left"


# --- build_cards ---------------------------------------------------------------------


def test_build_cards_returns_the_top_n_players_with_every_card_field() -> None:
    pool = _pool()
    tier_depth = live.tier_depth_remaining(pool)

    cards = draft_mobile_page.build_cards(pool, tier_depth, n=2)

    assert len(cards) == 2
    assert cards[0]["player"] == "Elite RB"
    assert cards[0]["position"] == "RB"
    assert cards[0]["team"] == "DET"
    assert cards[0]["bye_week"] == 6
    assert cards[0]["tier"] == 1
    assert cards[0]["vor"] == 150.0
    assert "Tier 1" in cards[0]["why_line"]


def test_build_cards_respects_the_default_card_count_range() -> None:
    assert 20 <= draft_mobile_page.DEFAULT_CARD_COUNT <= 30


def test_build_cards_on_an_empty_pool_returns_no_cards() -> None:
    empty = _pool().head(0)
    tier_depth = live.tier_depth_remaining(empty)

    cards = draft_mobile_page.build_cards(empty, tier_depth)

    assert cards == []


# --- top_line_summary ----------------------------------------------------------------


def test_top_line_summary_reports_the_best_available_player_and_its_survival_probability() -> None:
    pool = _pool()
    tier_summary = live.current_tier_summary(pool)

    result = draft_mobile_page.top_line_summary(pool, tier_summary)

    assert result["best_player"] == "Elite RB"
    assert result["best_vor"] == 150.0
    assert result["best_p_avail_next"] == 0.1


def test_top_line_summary_includes_tier_depth_per_position_sorted_by_position() -> None:
    pool = _pool()
    tier_summary = live.current_tier_summary(pool)

    result = draft_mobile_page.top_line_summary(pool, tier_summary)

    positions = [row["position"] for row in result["tier_depth_by_position"]]
    assert positions == sorted(positions)
    assert set(positions) == {"RB", "WR", "TE", "QB"}


def test_top_line_summary_on_an_empty_pool_returns_nulls_not_an_error() -> None:
    empty = _pool().head(0)
    tier_summary = live.current_tier_summary(empty)

    result = draft_mobile_page.top_line_summary(empty, tier_summary)

    assert result["best_player"] is None
    assert result["tier_depth_by_position"] == []
