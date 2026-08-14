import polars as pl

from ffapp.ingest import manual_rankings as mr

# --- normalize_manual_cbs -------------------------------------------------------


def test_normalize_manual_cbs_maps_columns_and_leaves_team_null() -> None:
    raw = pl.DataFrame(
        {
            "Rank": [1, 2],
            "Player": ["J. Gibbs", "B. Robinson"],
            "Position": ["RB", "RB"],
            "Auction Value": [34, 33],
            "Bye Week": [6, 11],
        }
    )

    result = mr.normalize_manual_cbs(raw)

    assert result.columns == ["source", "player_name", "position", "team", "rank"]
    assert result["player_name"].to_list() == ["J. Gibbs", "B. Robinson"]
    assert result["rank"].to_list() == [1.0, 2.0]
    assert result["team"].to_list() == [None, None]
    assert result["source"].to_list() == ["cbs", "cbs"]


def test_normalize_manual_cbs_repairs_the_roman_numeral_suffix_export_bug() -> None:
    """Real bug in CBS's own xlsx export, confirmed live: a player whose
    name ends in a roman-numeral suffix loses the last character of that
    suffix into the *next* cell -- "K. Walker II" / "IRB" instead of
    "K. Walker III" / "RB". Every real occurrence follows the same shape:
    Position starts with "I" followed by a real position code."""
    raw = pl.DataFrame(
        {
            "Rank": [17, 47, 185],
            "Player": ["K. Walker II", "L. Burden II", "O. Gadsden I"],
            "Position": ["IRB", "IWR", "ITE"],
            "Auction Value": [19, 11, 0],
            "Bye Week": [5, 10, 7],
        }
    )

    result = mr.normalize_manual_cbs(raw)

    assert result["player_name"].to_list() == ["K. Walker III", "L. Burden III", "O. Gadsden II"]
    assert result["position"].to_list() == ["RB", "WR", "TE"]


def test_normalize_manual_cbs_leaves_an_ordinary_position_untouched() -> None:
    """The I-prefix repair is gated on Position actually starting with "I"
    followed by a real position code -- a real position that happens to
    start with a different letter must not be touched."""
    raw = pl.DataFrame(
        {
            "Rank": [1],
            "Player": ["Josh Allen"],
            "Position": ["QB"],
            "Auction Value": [13],
            "Bye Week": [7],
        }
    )

    result = mr.normalize_manual_cbs(raw)

    assert result["player_name"].to_list() == ["Josh Allen"]
    assert result["position"].to_list() == ["QB"]


def test_normalize_manual_cbs_drops_positions_outside_the_six_fantasy_relevant() -> None:
    raw = pl.DataFrame(
        {
            "Rank": [1, 2],
            "Player": ["Josh Allen", "Some Kicker"],
            "Position": ["QB", "FB"],
            "Auction Value": [13, 0],
            "Bye Week": [7, 7],
        }
    )

    result = mr.normalize_manual_cbs(raw)

    assert result.height == 1
    assert result["player_name"].to_list() == ["Josh Allen"]


# --- normalize_manual_espn -------------------------------------------------------


def test_normalize_manual_espn_maps_columns() -> None:
    raw = pl.DataFrame(
        {
            "Overall Rank": [1, 2],
            "Position": ["RB", "WR"],
            "Position Rank": ["RB1", "WR1"],
            "Player": ["Jahmyr Gibbs", "Ja'Marr Chase"],
            "Team": ["DET", "CIN"],
            "Salary Cap Value": [57, 56],
            "Bye Week": [6, 6],
        }
    )

    result = mr.normalize_manual_espn(raw)

    assert result["player_name"].to_list() == ["Jahmyr Gibbs", "Ja'Marr Chase"]
    assert result["team"].to_list() == ["DET", "CIN"]
    assert result["rank"].to_list() == [1.0, 2.0]


# --- normalize_manual_fantasypros -------------------------------------------------


def test_normalize_manual_fantasypros_maps_columns() -> None:
    raw = pl.DataFrame(
        {
            "RK": [1],
            "Tier": [1],
            "Player": ["Ja'Marr Chase"],
            "Team": ["CIN"],
            "Position": ["WR"],
            "Pos Rank": ["WR1"],
            "Bye": [6],
            "Consensus Rating (1-5)": [4],
            "ECR vs ADP": [2],
        }
    )

    result = mr.normalize_manual_fantasypros(raw)

    assert result["player_name"].to_list() == ["Ja'Marr Chase"]
    assert result["rank"].to_list() == [1.0]


# --- normalize_manual_fftoday ------------------------------------------------------


def test_normalize_manual_fftoday_normalises_the_slash_dst_spelling() -> None:
    raw = pl.DataFrame(
        {
            "Rank": [1, 150],
            "Position": ["RB", "D/ST"],
            "Pos Rank": ["RB1", "DST1"],
            "Player": ["Jahmyr Gibbs", "Denver"],
            "Age": [24.4, None],
            "Team": ["DET", "DEN"],
            "Bye": [6, 12],
        }
    )

    result = mr.normalize_manual_fftoday(raw)

    assert result["position"].to_list() == ["RB", "DST"]


# --- normalize_manual_footballguys -------------------------------------------------


def test_normalize_manual_footballguys_normalises_pk_and_td_and_ignores_consensus_rank() -> None:
    raw = pl.DataFrame(
        {
            "Rank": [1, 2],
            "Player": ["Some Kicker", "Denver Broncos"],
            "Team": ["SF", "DEN"],
            "Team Code #": ["1", "2"],
            "Position": ["PK", "TD"],
            "Pos Rank": ["K1", "DST1"],
            "Proj Points": [100.0, 90.0],
            "PPG": [10.0, 9.0],
            "Age": ["30", None],
            "Exp": ["5", None],
            "Bye": ["9", "12"],
            "Consensus Rank": ["50", "60"],  # deliberately different from Rank -- must be ignored
            "Rank Change": ["-", "-"],
            "Metric 1": [0.5, 0.5],
            "Metric 2": [None, None],
        }
    )

    result = mr.normalize_manual_footballguys(raw)

    assert result["position"].to_list() == ["K", "DST"]
    assert result["rank"].to_list() == [1.0, 2.0]  # Rank, not Consensus Rank


# --- normalize_manual_draftsharks -------------------------------------------------


def test_normalize_manual_draftsharks_normalises_the_def_spelling() -> None:
    raw = pl.DataFrame(
        {
            "RK": [1, 200],
            "Tier": [1, 8],
            "Player": ["Puka Nacua", "Houston Texans"],
            "Team": ["LAR", "HOU"],
            "Position": ["WR", "DEF"],
            "Pos Rank": ["WR1", "DST1"],
            "Games": [17, 17],
            "ADP": ["1.04", "150"],
            "Bye": [11, 8],
            "SOS": ["-3.1%", "0%"],
            "Injury Risk": ["51%", None],
            "Floor Proj": [268, 80],
            "Consensus Proj": [326, 90],
            "DS Proj": [357, 95],
            "Ceiling Proj": [397, 100],
            "3D Value": [100, 10],
        }
    )

    result = mr.normalize_manual_draftsharks(raw)

    assert result["position"].to_list() == ["WR", "DST"]


# --- normalize_manual_fantasysharks -------------------------------------------------


def test_normalize_manual_fantasysharks_joins_first_and_last_name_and_normalises_d() -> None:
    raw = pl.DataFrame(
        {
            "Rank": [1, 157, 200],
            "ID": [15281, 515, 999],
            "Last Name": ["Chase", "Seahawks", "Campbell"],
            "First Name": ["Ja'Marr", "Seattle", "Jack"],
            "Position": ["WR", "D", "LB"],
            "Team": ["CIN", "SEA", "SEA"],
            "Points": [368.2, 231.9, 100.0],
            "VBD": [460.8, 189.6, 50.0],
            "Bye Week": [6, 11, 11],
        }
    )

    result = mr.normalize_manual_fantasysharks(raw)

    assert result.height == 2  # the LB (IDP, not fantasy-relevant) row is dropped
    assert result["player_name"].to_list() == ["Ja'Marr Chase", "Seattle Seahawks"]
    assert result["position"].to_list() == ["WR", "DST"]
