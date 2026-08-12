import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from ffapp.cache.offline import OfflineCacheMiss, StaleCacheError, sidecar_path, write_sidecar
from ffapp.config import CacheSettings, Settings
from ffapp.ingest import rankings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_root=tmp_path,
        sleeper_username="fixture_user",
        cache=CacheSettings(
            root=tmp_path / "raw",
            offline_default=True,
            staleness_hours={"rankings_espn": 24, "rankings_adp": 24},
            warn_on_stale=True,
        ),
    )


def _age_stamp(hours: float) -> str:
    return (datetime.now(UTC) - timedelta(hours=hours)).isoformat()


FIXTURE_PAYLOAD = {"players": [{"player": {"fullName": "Fixture Player"}}]}


# --- fetch_espn -----------------------------------------------------------


def test_fetch_espn_online_writes_raw_json_and_sidecar(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    monkeypatch.setattr(rankings, "_get_espn_json", lambda season: FIXTURE_PAYLOAD)

    path = rankings.fetch_espn(2026, offline=False, settings=settings)

    assert path.exists()
    assert json.loads(path.read_text()) == FIXTURE_PAYLOAD
    meta = json.loads(sidecar_path(path).read_text())
    assert meta["source"] == "rankings"
    assert meta["cache_key"] == "rankings_espn"
    assert meta["rows"] == 1


def test_fetch_espn_offline_with_fresh_cache_does_not_call_network(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    def _boom(season: int) -> dict:
        raise AssertionError("network should not be called offline")

    monkeypatch.setattr(rankings, "_get_espn_json", _boom)
    path = settings.cache.root / "rankings" / "espn_2026.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(FIXTURE_PAYLOAD))
    write_sidecar(path, source="rankings", call="x", cache_key="rankings_espn")

    result = rankings.fetch_espn(2026, offline=True, settings=settings)

    assert result == path


def test_fetch_espn_offline_without_cache_raises_offline_cache_miss(
    settings: Settings,
) -> None:
    with pytest.raises(OfflineCacheMiss) as exc_info:
        rankings.fetch_espn(2026, offline=True, settings=settings)

    message = str(exc_info.value)
    assert "rankings" in message
    assert "season=2026" in message


def test_fetch_espn_offline_with_stale_cache_logs_warning(
    monkeypatch: pytest.MonkeyPatch, settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.delenv("FFAPP_CACHE_STRICT", raising=False)
    path = settings.cache.root / "rankings" / "espn_2026.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(FIXTURE_PAYLOAD))
    sidecar_path(path).write_text(
        json.dumps(
            {
                "source": "rankings",
                "fetched_at_utc": _age_stamp(200),
                "cache_key": "rankings_espn",
            }
        )
    )

    with caplog.at_level(logging.WARNING):
        result = rankings.fetch_espn(2026, offline=True, settings=settings)

    assert result == path
    assert any("stale" in record.message.lower() for record in caplog.records)


def test_fetch_espn_offline_with_stale_cache_and_strict_env_raises(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    monkeypatch.setenv("FFAPP_CACHE_STRICT", "1")
    path = settings.cache.root / "rankings" / "espn_2026.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(FIXTURE_PAYLOAD))
    sidecar_path(path).write_text(
        json.dumps(
            {
                "source": "rankings",
                "fetched_at_utc": _age_stamp(200),
                "cache_key": "rankings_espn",
            }
        )
    )

    with pytest.raises(StaleCacheError):
        rankings.fetch_espn(2026, offline=True, settings=settings)


# --- normalize_espn ---------------------------------------------------------


def _espn_stat_row(
    *, scoring_period_id: int, stat_source_id: int, stat_split_type_id: int, stats: dict[str, float]
) -> dict:
    return {
        "scoringPeriodId": scoring_period_id,
        "statSourceId": stat_source_id,
        "statSplitTypeId": stat_split_type_id,
        "seasonId": 2026,
        "stats": stats,
    }


def _espn_player(
    *, full_name: str, default_position_id: int, pro_team_id: int, stats: list[dict]
) -> dict:
    return {
        "player": {
            "fullName": full_name,
            "defaultPositionId": default_position_id,
            "proTeamId": pro_team_id,
            "stats": stats,
        }
    }


def test_normalize_espn_extracts_season_total_projection_row() -> None:
    payload = {
        "players": [
            _espn_player(
                full_name="Jahmyr Gibbs",
                default_position_id=2,
                pro_team_id=8,
                stats=[
                    # A weekly actual row that must be ignored.
                    _espn_stat_row(
                        scoring_period_id=1,
                        stat_source_id=0,
                        stat_split_type_id=1,
                        stats={"24": 80.0},
                    ),
                    # The season-total projection row.
                    _espn_stat_row(
                        scoring_period_id=0,
                        stat_source_id=1,
                        stat_split_type_id=0,
                        stats={"24": 1200.0, "25": 12.0, "41": 50.0},
                    ),
                ],
            )
        ]
    }

    df = rankings.normalize_espn(payload, season=2026)

    assert df.height == 1
    row = df.row(0, named=True)
    assert row["player_name"] == "Jahmyr Gibbs"
    assert row["position"] == "RB"
    assert row["team"] == "DET"
    assert row["season"] == 2026
    assert row["rushing_yards"] == 1200.0
    assert row["rushing_tds"] == 12.0
    assert row["receptions"] == 50.0


def test_normalize_espn_skips_players_with_no_season_total_row() -> None:
    payload = {
        "players": [
            _espn_player(
                full_name="No Projection Player",
                default_position_id=2,
                pro_team_id=8,
                stats=[
                    _espn_stat_row(
                        scoring_period_id=1,
                        stat_source_id=0,
                        stat_split_type_id=1,
                        stats={"24": 80.0},
                    ),
                ],
            )
        ]
    }

    df = rankings.normalize_espn(payload, season=2026)

    assert df.height == 0


def test_normalize_espn_maps_verified_real_position_ids() -> None:
    """cwendt94/espn-api's community PLAYER POSITION_MAP (0=QB, 4=WR, 6=TE,
    17=K) does not match ESPN's current live payload -- confirmed against a
    real 500-player leaguedefaults response: Josh Allen (unambiguously QB)
    has defaultPositionId=1, real WRs have 3, George Kittle/Travis Kelce
    (unambiguously TE) have 4, and a real kicker (Brandon Aubrey) has 5.
    Under the stale community map, every real QB/WR/K was silently dropped
    and every TE was mislabeled as WR."""
    payload = {
        "players": [
            _espn_player(
                full_name="Verified QB",
                default_position_id=1,
                pro_team_id=2,
                stats=[
                    _espn_stat_row(
                        scoring_period_id=0,
                        stat_source_id=1,
                        stat_split_type_id=0,
                        stats={"3": 4000.0},
                    )
                ],
            ),
            _espn_player(
                full_name="Verified WR",
                default_position_id=3,
                pro_team_id=2,
                stats=[
                    _espn_stat_row(
                        scoring_period_id=0,
                        stat_source_id=1,
                        stat_split_type_id=0,
                        stats={"53": 90.0},
                    )
                ],
            ),
            _espn_player(
                full_name="Verified TE",
                default_position_id=4,
                pro_team_id=2,
                stats=[
                    _espn_stat_row(
                        scoring_period_id=0,
                        stat_source_id=1,
                        stat_split_type_id=0,
                        stats={"53": 70.0},
                    )
                ],
            ),
            _espn_player(
                full_name="Verified K",
                default_position_id=5,
                pro_team_id=2,
                stats=[
                    _espn_stat_row(
                        scoring_period_id=0,
                        stat_source_id=1,
                        stat_split_type_id=0,
                        stats={"83": 30.0},
                    )
                ],
            ),
        ]
    }

    df = rankings.normalize_espn(payload, season=2026)

    positions = dict(zip(df["player_name"], df["position"], strict=True))
    assert positions == {
        "Verified QB": "QB",
        "Verified WR": "WR",
        "Verified TE": "TE",
        "Verified K": "K",
    }
    assert df.filter(pl.col("player_name") == "Verified WR")["receptions"].item() == 90.0


def test_normalize_espn_does_not_drop_a_stat_column_that_first_appears_late() -> None:
    """pl.DataFrame(list[dict]) only infers schema from the first 100 rows by
    default -- confirmed live: in a real 500-player ESPN sample sorted by
    overall rank, kickers rank low enough that a naive DataFrame(rows) call
    silently dropped fg_made/pat_made entirely (no error, just a missing
    column) because no kicker appeared in the first 100 rows. Reproduced
    here with >100 QB rows (no fg_made key) ahead of one kicker row."""
    payload = {
        "players": [
            _espn_player(
                full_name=f"Filler QB {i}",
                default_position_id=1,
                pro_team_id=2,
                stats=[
                    _espn_stat_row(
                        scoring_period_id=0,
                        stat_source_id=1,
                        stat_split_type_id=0,
                        stats={"3": 3000.0},
                    )
                ],
            )
            for i in range(120)
        ]
        + [
            _espn_player(
                full_name="Late Kicker",
                default_position_id=5,
                pro_team_id=2,
                stats=[
                    _espn_stat_row(
                        scoring_period_id=0,
                        stat_source_id=1,
                        stat_split_type_id=0,
                        stats={"83": 30.0},
                    )
                ],
            )
        ]
    }

    df = rankings.normalize_espn(payload, season=2026)

    assert "fg_made" in df.columns
    assert df.filter(pl.col("player_name") == "Late Kicker")["fg_made"].item() == 30.0


def test_normalize_espn_skips_unrostered_positions() -> None:
    payload = {
        "players": [
            _espn_player(
                full_name="Some Head Coach",
                default_position_id=19,  # HC -- not fantasy-relevant
                pro_team_id=8,
                stats=[
                    _espn_stat_row(
                        scoring_period_id=0,
                        stat_source_id=1,
                        stat_split_type_id=0,
                        stats={"158": 300.0},
                    ),
                ],
            )
        ]
    }

    df = rankings.normalize_espn(payload, season=2026)

    assert df.height == 0


# --- fetch_fantasypros -------------------------------------------------------

FANTASYPROS_FIXTURE_CSV = (
    "fp_page,page_type,ecr_type,player,id,pos,team,ecr,sd,best,worst,sportsdata_id,"
    "player_filename,yahoo_id,cbs_id,player_owned_avg,player_owned_espn,player_owned_yahoo,"
    "player_image_url,player_square_image_url,rank_delta,bye,mergename,scrape_date,tm\n"
    "/nfl/rankings/qb-cheatsheets.php,redraft-qb,rp,Josh Allen,17298,QB,BUF,1.02,0.15,1,2,"
    "NA,josh-allen-qb.php,NA,NA,99.5,NA,NA,NA,NA,7,BUF,josh allen,2026-08-07,BUF\n"
)


def test_fetch_fantasypros_online_writes_raw_csv_and_sidecar(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    monkeypatch.setattr(rankings, "_get_fantasypros_csv", lambda: FANTASYPROS_FIXTURE_CSV)

    path = rankings.fetch_fantasypros(offline=False, settings=settings)

    assert path.exists()
    assert path.read_text() == FANTASYPROS_FIXTURE_CSV
    meta = json.loads(sidecar_path(path).read_text())
    assert meta["source"] == "rankings"
    assert meta["cache_key"] == "rankings_fantasypros"
    assert meta["rows"] == 1


def test_fetch_fantasypros_offline_with_fresh_cache_does_not_call_network(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    def _boom() -> str:
        raise AssertionError("network should not be called offline")

    monkeypatch.setattr(rankings, "_get_fantasypros_csv", _boom)
    path = settings.cache.root / "rankings" / "fantasypros.csv"
    path.parent.mkdir(parents=True)
    path.write_text(FANTASYPROS_FIXTURE_CSV)
    write_sidecar(path, source="rankings", call="x", cache_key="rankings_fantasypros")

    result = rankings.fetch_fantasypros(offline=True, settings=settings)

    assert result == path


def test_fetch_fantasypros_offline_without_cache_raises_offline_cache_miss(
    settings: Settings,
) -> None:
    with pytest.raises(OfflineCacheMiss) as exc_info:
        rankings.fetch_fantasypros(offline=True, settings=settings)

    assert "rankings" in str(exc_info.value)
    assert "fantasypros" in str(exc_info.value)


# --- normalize_fantasypros ---------------------------------------------------


def _fantasypros_row(
    *, page_type: str, ecr_type: str, player: str, pos: str, team: str, ecr: float
) -> str:
    return (
        f"/nfl/rankings/{page_type}-cheatsheets.php,{page_type},{ecr_type},{player},1,{pos},"
        f"{team},{ecr},1.0,1,2,NA,x.php,NA,NA,99.0,NA,NA,NA,NA,5,{team},{player.lower()},"
        "2026-08-07,TM\n"
    )


FANTASYPROS_HEADER = (
    "fp_page,page_type,ecr_type,player,id,pos,team,ecr,sd,best,worst,sportsdata_id,"
    "player_filename,yahoo_id,cbs_id,player_owned_avg,player_owned_espn,player_owned_yahoo,"
    "player_image_url,player_square_image_url,rank_delta,bye,mergename,scrape_date,tm\n"
)


def test_normalize_fantasypros_extracts_redraft_positional_rank_rows() -> None:
    csv_text = FANTASYPROS_HEADER + _fantasypros_row(
        page_type="redraft-qb", ecr_type="rp", player="Josh Allen", pos="QB", team="BUF", ecr=1.02
    )

    df = rankings.normalize_fantasypros(csv_text, season=2026)

    assert df.height == 1
    row = df.row(0, named=True)
    assert row["player_name"] == "Josh Allen"
    assert row["position"] == "QB"
    assert row["team"] == "BUF"
    assert row["season"] == 2026
    assert row["rank"] == 1.02


def test_normalize_fantasypros_excludes_overall_and_dynasty_rows() -> None:
    csv_text = (
        FANTASYPROS_HEADER
        + _fantasypros_row(
            page_type="redraft-overall",
            ecr_type="ro",
            player="Ja'Marr Chase",
            pos="WR",
            team="CIN",
            ecr=1.8,
        )
        + _fantasypros_row(
            page_type="dynasty-qb",
            ecr_type="dp",
            player="Josh Allen",
            pos="QB",
            team="BUF",
            ecr=1.0,
        )
    )

    df = rankings.normalize_fantasypros(csv_text, season=2026)

    assert df.height == 0


def test_normalize_fantasypros_handles_na_literal_in_numeric_bye_column() -> None:
    """This repo's CSVs use the literal string "NA" for missing values, not
    empty cells -- same gotcha task 0.3's crosswalk fetch already hit
    (HANDOFF.md §5). Confirmed live with a real row (Tyreek Hill, whose `bye`
    column is "NA"): polars' schema inference tries to parse the whole
    numeric `bye` column as an int and crashes on that literal string,
    unless null_values=["NA"] is passed -- reproduced verbatim here rather
    than a hand-built row, since a hand-built fixture is exactly the kind of
    "narrow enough to be convenient" case this project has been burned by
    missing real-data quirks before (see HANDOFF.md's team_stats gotcha)."""
    real_na_bye_row = (
        "/nfl/rankings/best-ball-overall.php,best-overall,bo,Tyreek Hill,15802,WR,FA,238,48.26,"
        "181,299,NA,tyreek-hill.php,NA,NA,28.8,NA,NA,NA,NA,-10,NA,Tyreek Hill,2026-08-07,FA\n"
    )
    csv_text = (
        FANTASYPROS_HEADER
        + real_na_bye_row
        + _fantasypros_row(
            page_type="redraft-qb",
            ecr_type="rp",
            player="Josh Allen",
            pos="QB",
            team="BUF",
            ecr=1.02,
        )
    )

    df = rankings.normalize_fantasypros(csv_text, season=2026)

    assert df.height == 1
    assert df.row(0, named=True)["player_name"] == "Josh Allen"


def test_normalize_fantasypros_excludes_idp_positions() -> None:
    csv_text = FANTASYPROS_HEADER + _fantasypros_row(
        page_type="redraft-lb",
        ecr_type="rp",
        player="Some Linebacker",
        pos="LB",
        team="BUF",
        ecr=1.0,
    )

    df = rankings.normalize_fantasypros(csv_text, season=2026)

    assert df.height == 0


# --- fetch_fantasysharks -----------------------------------------------------

FANTASYSHARKS_FIXTURE_PAYLOAD = {
    "ALL": [
        {
            "Rank": 1,
            "ADP": 5,
            "ID": "13589",
            "Name": "Allen, Josh",
            "Pos": "QB",
            "Team": "BUF",
            "Bye": "7",
            "Comp": "304",
            "PassYards": "3600",
            "PassTD": 30,
            "Int": "9",
            "Att": "107",
            "RushYards": "514",
            "RushTD": 12,
            "Fum": "3",
            "Rec": "0",
            "RecYards": "0",
            "RecTD": 0,
            "FantasyPoints": 408,
        }
    ],
    "PK": [
        {
            "Rank": 1,
            "ADP": 140,
            "ID": "12860",
            "Name": "Fairbairn, Ka'imi",
            "Team": "HOU",
            "Bye": "8",
            "XP": "30",
            "FG": 41,
            "Miss": "5",
            "FantasyPoints": 167,
        }
    ],
    "D": [
        {
            "Rank": 1,
            "ADP": 103,
            "ID": "0515",
            "Name": "Seahawks, Seattle",
            "Team": "SEA",
            "Bye": "11",
            "TD": "8",
            "Int": "17",
            "Fum": "10",
            "Sack": "49",
            "PtsAllow": "255",
            "YdsAllow": "4841",
            "FantasyPoints": 251,
        }
    ],
}


def test_fetch_fantasysharks_online_writes_raw_json_and_sidecar(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    monkeypatch.setattr(
        rankings,
        "_get_fantasysharks_json",
        lambda pos: FANTASYSHARKS_FIXTURE_PAYLOAD[pos],
    )

    path = rankings.fetch_fantasysharks(offline=False, settings=settings)

    assert path.exists()
    assert json.loads(path.read_text()) == FANTASYSHARKS_FIXTURE_PAYLOAD
    meta = json.loads(sidecar_path(path).read_text())
    assert meta["source"] == "rankings"
    assert meta["cache_key"] == "rankings_fantasysharks"
    assert meta["rows"] == 3


def test_fetch_fantasysharks_offline_with_fresh_cache_does_not_call_network(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    def _boom(pos: str) -> list:
        raise AssertionError("network should not be called offline")

    monkeypatch.setattr(rankings, "_get_fantasysharks_json", _boom)
    path = settings.cache.root / "rankings" / "fantasysharks.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(FANTASYSHARKS_FIXTURE_PAYLOAD))
    write_sidecar(path, source="rankings", call="x", cache_key="rankings_fantasysharks")

    result = rankings.fetch_fantasysharks(offline=True, settings=settings)

    assert result == path


def test_fetch_fantasysharks_offline_without_cache_raises_offline_cache_miss(
    settings: Settings,
) -> None:
    with pytest.raises(OfflineCacheMiss) as exc_info:
        rankings.fetch_fantasysharks(offline=True, settings=settings)

    assert "rankings" in str(exc_info.value)
    assert "fantasysharks" in str(exc_info.value)


# --- normalize_fantasysharks --------------------------------------------------


def test_normalize_fantasysharks_maps_skill_position_stats() -> None:
    df = rankings.normalize_fantasysharks(FANTASYSHARKS_FIXTURE_PAYLOAD, season=2026)

    qb = df.filter(pl.col("position") == "QB").row(0, named=True)
    assert qb["player_name"] == "Josh Allen"
    assert qb["team"] == "BUF"
    assert qb["season"] == 2026
    assert qb["passing_yards"] == 3600.0
    assert qb["passing_tds"] == 30.0
    assert qb["rushing_yards"] == 514.0
    assert qb["rushing_tds"] == 12.0


def test_normalize_fantasysharks_reverses_last_first_name_order() -> None:
    df = rankings.normalize_fantasysharks(FANTASYSHARKS_FIXTURE_PAYLOAD, season=2026)

    qb = df.filter(pl.col("position") == "QB").row(0, named=True)
    assert qb["player_name"] == "Josh Allen"


def test_normalize_fantasysharks_labels_kicker_rows_k_with_fg_and_pat_stats() -> None:
    df = rankings.normalize_fantasysharks(FANTASYSHARKS_FIXTURE_PAYLOAD, season=2026)

    k = df.filter(pl.col("position") == "K").row(0, named=True)
    assert k["player_name"] == "Ka'imi Fairbairn"
    assert k["pat_made"] == 30.0
    assert k["fg_made"] == 41.0
    assert k["fg_missed"] == 5.0


def test_normalize_fantasysharks_labels_defense_rows_dst() -> None:
    df = rankings.normalize_fantasysharks(FANTASYSHARKS_FIXTURE_PAYLOAD, season=2026)

    dst = df.filter(pl.col("position") == "DST").row(0, named=True)
    assert dst["player_name"] == "Seattle Seahawks"
    assert dst["def_sacks"] == 49.0
    assert dst["def_interceptions"] == 17.0
    assert dst["def_return_tds"] == 8.0
    assert dst["fumble_recovery_opp"] == 10.0


# --- fetch_cbs / normalize_cbs ------------------------------------------------
#
# Fixtures are minimal reproductions of CBS's real table structure (thead/tbody
# with a CellPlayerName--long span for the clean player name, td[0] holding a
# duplicated short+long name blob) rather than a full page dump.

CBS_TABLE_TEMPLATE = """
<table>
<thead><tr class="TableBase-headTr">{header_ths}</tr></thead>
<tbody>{body_trs}</tbody>
</table>
"""


def _cbs_th(label: str) -> str:
    return f"<th>{label}</th>"


def _cbs_player_row(name: str, team: str, position: str, other_cells: list[str]) -> str:
    name_td = (
        f'<td><span class="CellPlayerName--short"><span class="">'
        f'<a href="#">X. Y</a><span class="CellPlayerName-position">{position}</span>'
        f'<span class="CellPlayerName-team">{team}</span></span></span>'
        f'<span class="CellPlayerName--long"><span class="">'
        f'<a href="#">{name}</a><span class="CellPlayerName-position">{position}</span>'
        f'<span class="CellPlayerName-team">{team}</span></span></span></td>'
    )
    cells = "".join(f"<td>{c}</td>" for c in other_cells)
    return f'<tr class="TableBase-bodyTr">{name_td}{cells}</tr>'


def _cbs_team_row(team: str, other_cells: list[str]) -> str:
    cells = "".join(f"<td>{c}</td>" for c in other_cells)
    return f'<tr class="TableBase-bodyTr"><td>{team}</td>{cells}</tr>'


def _cbs_table(headers: list[str], body_rows: list[str]) -> str:
    return CBS_TABLE_TEMPLATE.format(
        header_ths="".join(_cbs_th(h) for h in headers), body_trs="".join(body_rows)
    )


CBS_QB_HTML = _cbs_table(
    [
        "Player",
        "gp",
        "att",
        "cmp",
        "yds",
        "yds/g",
        "td",
        "int",
        "rate",
        "att",
        "yds",
        "avg",
        "td",
        "fl",
        "fpts",
        "fppg",
    ],
    [
        _cbs_player_row(
            "Josh Allen",
            "BUF",
            "QB",
            [
                "17",
                "479",
                "326",
                "3787",
                "222.8",
                "26",
                "9",
                "95.0",
                "113",
                "567",
                "5.0",
                "12",
                "4",
                "384.2",
                "22.6",
            ],
        )
    ],
)

CBS_K_HTML = _cbs_table(
    [
        "Player",
        "gp",
        "fgm",
        "fga",
        "lng",
        "1-19",
        "1-19a",
        "20-29",
        "20-29a",
        "30-39",
        "30-39a",
        "40-49",
        "40-49a",
        "50+",
        "50+a",
        "xpm",
        "xpa",
        "fpts",
        "fppg",
    ],
    [
        _cbs_player_row(
            "Jason Myers",
            "SEA",
            "K",
            [
                "17",
                "39",
                "44",
                "—",
                "0.3",
                "0.3",
                "8.3",
                "9.2",
                "12.5",
                "13.2",
                "11.4",
                "12.9",
                "6.4",
                "6.4",
                "50",
                "51",
                "167",
                "9.8",
            ],
        )
    ],
)

CBS_DST_HTML = _cbs_table(
    [
        "Team",
        "int",
        "sfty",
        "sck",
        "tk",
        "frec",
        "fum",
        "dtd",
        "pts",
        "ppg",
        "pass",
        "rush",
        "total",
        "avg",
        "fpts",
        "fppg",
    ],
    [
        _cbs_team_row(
            "Denver",
            [
                "15",
                "0",
                "72.6",
                "649",
                "10",
                "14",
                "2.9",
                "315",
                "18.5",
                "0",
                "1763",
                "4243",
                "249.6",
                "276",
                "16.2",
            ],
        )
    ],
)


def test_fetch_cbs_online_writes_raw_html_per_position_and_sidecar(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    fixture_by_pos = {pos: f"<html>{pos}</html>" for pos in rankings.CBS_POSITIONS}
    monkeypatch.setattr(rankings, "_get_cbs_html", lambda pos, season: fixture_by_pos[pos])

    path = rankings.fetch_cbs(2026, offline=False, settings=settings)

    assert path.exists()
    assert json.loads(path.read_text()) == fixture_by_pos
    meta = json.loads(sidecar_path(path).read_text())
    assert meta["source"] == "rankings"
    assert meta["cache_key"] == "rankings_cbs"


def test_fetch_cbs_offline_without_cache_raises_offline_cache_miss(settings: Settings) -> None:
    with pytest.raises(OfflineCacheMiss) as exc_info:
        rankings.fetch_cbs(2026, offline=True, settings=settings)

    assert "rankings" in str(exc_info.value)
    assert "cbs" in str(exc_info.value)


def test_normalize_cbs_maps_qb_passing_and_rushing_stats() -> None:
    df = rankings.normalize_cbs({"QB": CBS_QB_HTML}, season=2026)

    qb = df.row(0, named=True)
    assert qb["player_name"] == "Josh Allen"
    assert qb["team"] == "BUF"
    assert qb["position"] == "QB"
    assert qb["season"] == 2026
    assert qb["passing_yards"] == 3787.0
    assert qb["passing_tds"] == 26.0
    assert qb["passing_interceptions"] == 9.0
    assert qb["rushing_yards"] == 567.0
    assert qb["rushing_tds"] == 12.0
    assert qb["fumbles_lost_total"] == 4.0


def test_normalize_cbs_maps_kicker_stats() -> None:
    df = rankings.normalize_cbs({"K": CBS_K_HTML}, season=2026)

    k = df.row(0, named=True)
    assert k["player_name"] == "Jason Myers"
    assert k["position"] == "K"
    assert k["fg_made"] == 39.0
    assert k["pat_made"] == 50.0


def test_normalize_cbs_maps_dst_stats_from_team_name_cell() -> None:
    df = rankings.normalize_cbs({"DST": CBS_DST_HTML}, season=2026)

    dst = df.row(0, named=True)
    assert dst["player_name"] == "Denver"
    assert dst["position"] == "DST"
    assert dst["def_interceptions"] == 15.0
    assert dst["def_sacks"] == 72.6
    assert dst["def_return_tds"] == 2.9


def test_normalize_cbs_raises_if_a_position_page_layout_changes() -> None:
    """Column mapping for CBS is positional (repeated header text like "yds"
    and "td" means different stats on different position pages), so a
    silent column-count mismatch would silently mis-map every stat on that
    page -- this must fail loudly instead (CLAUDE.md rule 4's spirit)."""
    malformed_html = _cbs_table(
        ["Player", "gp", "att"],  # far fewer columns than the real QB page
        [_cbs_player_row("Josh Allen", "BUF", "QB", ["17", "479"])],
    )

    with pytest.raises(rankings.UnexpectedColumnLayoutError):
        rankings.normalize_cbs({"QB": malformed_html}, season=2026)


# --- fetch_fftoday / normalize_fftoday ---------------------------------------
#
# FFToday's real player-data table is nested inside a malformed outer <td>
# alongside several small layout tables (confirmed live) -- fixtures
# reproduce that shape (a small decoy table plus the real, bigger one) to
# prove the "pick the table with the most rows" strategy actually
# discriminates, rather than trivially working because there's only one
# table in the fixture.


def _fftoday_row(cells: list[str]) -> str:
    return "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"


def _fftoday_page(data_rows: list[str]) -> str:
    decoy_table = "<table><tr><td>decoy</td></tr></table>"
    data_table = "<table>" + "".join(data_rows) + "</table>"
    return f"<html><body><table><tr><td>{decoy_table}{data_table}</td></tr></table></body></html>"


FFTODAY_QB_HTML = _fftoday_page(
    [
        _fftoday_row(["", "Passing", "Rushing", "Fantasy"]),
        _fftoday_row(
            [
                "Chg",
                "Player",
                "Tm",
                "Bye",
                "Cmp",
                "Att",
                "Yds",
                "TD",
                "INT",
                "Att",
                "Yds",
                "TD",
                "FPts",
            ]
        ),
        _fftoday_row(
            [
                "",
                "Josh Allen",
                "BUF",
                "7",
                "326",
                "479",
                "3,787",
                "26",
                "9",
                "113",
                "567",
                "12",
                "384.2",
            ]
        ),
    ]
)

FFTODAY_K_HTML = _fftoday_page(
    [
        _fftoday_row(["", "", "Fantasy"]),
        _fftoday_row(["Chg", "Player", "Tm", "Bye", "FGM", "FGA", "FG%", "EPM", "EPA", "FPts"]),
        _fftoday_row(["", "Brandon Aubrey", "DAL", "14", "35", "40", "87.5%", "45", "46", "150.0"]),
    ]
)

FFTODAY_DST_HTML = _fftoday_page(
    [
        _fftoday_row(["", "", "Fantasy"]),
        _fftoday_row(
            [
                "Chg",
                "Team",
                "Bye",
                "Sack",
                "FR",
                "INT",
                "DefTD",
                "PA",
                "PaYd/G",
                "RuYd/G",
                "S",
                "KickTD",
                "FPts",
            ]
        ),
        _fftoday_row(
            [
                "",
                "Houston Texans",
                "8",
                "48",
                "10",
                "16",
                "3",
                "325",
                "220.9",
                "107.5",
                "1",
                "1",
                "126.0",
            ]
        ),
    ]
)


def test_fetch_fftoday_online_writes_raw_html_per_position_and_sidecar(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    fixture_by_pos = {pos: f"<html>{pos}</html>" for pos in rankings.FFTODAY_POSITIONS}
    monkeypatch.setattr(rankings, "_get_fftoday_html", lambda pos, season: fixture_by_pos[pos])

    path = rankings.fetch_fftoday(2026, offline=False, settings=settings)

    assert path.exists()
    assert json.loads(path.read_text()) == fixture_by_pos
    meta = json.loads(sidecar_path(path).read_text())
    assert meta["source"] == "rankings"
    assert meta["cache_key"] == "rankings_fftoday"


def test_fetch_fftoday_offline_without_cache_raises_offline_cache_miss(
    settings: Settings,
) -> None:
    with pytest.raises(OfflineCacheMiss) as exc_info:
        rankings.fetch_fftoday(2026, offline=True, settings=settings)

    assert "rankings" in str(exc_info.value)
    assert "fftoday" in str(exc_info.value)


def test_normalize_fftoday_maps_qb_passing_and_rushing_stats() -> None:
    df = rankings.normalize_fftoday({"QB": FFTODAY_QB_HTML}, season=2026)

    qb = df.row(0, named=True)
    assert qb["player_name"] == "Josh Allen"
    assert qb["team"] == "BUF"
    assert qb["position"] == "QB"
    assert qb["season"] == 2026
    assert qb["passing_yards"] == 3787.0
    assert qb["passing_tds"] == 26.0
    assert qb["passing_interceptions"] == 9.0
    assert qb["rushing_yards"] == 567.0
    assert qb["rushing_tds"] == 12.0


def test_normalize_fftoday_maps_kicker_stats() -> None:
    df = rankings.normalize_fftoday({"K": FFTODAY_K_HTML}, season=2026)

    k = df.row(0, named=True)
    assert k["player_name"] == "Brandon Aubrey"
    assert k["position"] == "K"
    assert k["fg_made"] == 35.0
    assert k["pat_made"] == 45.0


def test_normalize_fftoday_maps_dst_stats_from_team_name_cell() -> None:
    df = rankings.normalize_fftoday({"DST": FFTODAY_DST_HTML}, season=2026)

    dst = df.row(0, named=True)
    assert dst["player_name"] == "Houston Texans"
    assert dst["position"] == "DST"
    assert dst["def_sacks"] == 48.0
    assert dst["fumble_recovery_opp"] == 10.0
    assert dst["def_interceptions"] == 16.0
    assert dst["def_return_tds"] == 3.0
    assert dst["def_safeties"] == 1.0


def test_normalize_fftoday_raises_if_a_position_page_layout_changes() -> None:
    malformed_html = _fftoday_page(
        [
            _fftoday_row(["", "Passing", "Fantasy"]),
            _fftoday_row(["Chg", "Player", "Tm"]),  # far fewer columns than the real QB page
            _fftoday_row(["", "Josh Allen", "BUF"]),
        ]
    )

    with pytest.raises(rankings.UnexpectedColumnLayoutError):
        rankings.normalize_fftoday({"QB": malformed_html}, season=2026)


# --- fetch_adp / normalize_adp (FantasyFootballCalculator) -----------------

FFC_FIXTURE_PAYLOAD = {
    "status": "Success",
    "meta": {"type": "PPR", "teams": 10, "rounds": 15},
    "players": [
        {
            "player_id": 1,
            "name": "Jahmyr Gibbs",
            "position": "RB",
            "team": "DET",
            "adp": 1.8,
            "adp_formatted": "1.02",
            "times_drafted": 1176,
            "high": 1,
            "low": 4,
            "stdev": 0.8,
            "bye": 6,
        },
        {
            "player_id": 2,
            "name": "Denver Defense",
            "position": "DEF",
            "team": "DEN",
            "adp": 94.3,
            "adp_formatted": "10.04",
            "times_drafted": 243,
            "high": 68,
            "low": 114,
            "stdev": 9.8,
            "bye": 10,
        },
        {
            "player_id": 3,
            "name": "Some Kicker",
            "position": "PK",
            "team": "SF",
            "adp": 150.0,
            "adp_formatted": "15.10",
            "times_drafted": 50,
            "high": 130,
            "low": 170,
            "stdev": 8.0,
            "bye": 9,
        },
    ],
}


def test_fetch_adp_online_writes_raw_json_and_sidecar(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    calls = []
    monkeypatch.setattr(
        rankings,
        "_get_ffc_json",
        lambda season, teams, scoring: (
            calls.append((season, teams, scoring)) or FFC_FIXTURE_PAYLOAD
        ),
    )

    path = rankings.fetch_adp(2026, teams=10, offline=False, settings=settings)

    assert calls == [(2026, 10, "ppr")]
    assert path.name == "adp_2026_10_ppr.json"
    assert json.loads(path.read_text()) == FFC_FIXTURE_PAYLOAD
    meta = json.loads(sidecar_path(path).read_text())
    assert meta["source"] == "rankings"
    assert meta["cache_key"] == "rankings_adp"
    assert meta["rows"] == 3


def test_fetch_adp_offline_without_cache_raises_offline_cache_miss(
    settings: Settings,
) -> None:
    with pytest.raises(OfflineCacheMiss) as exc_info:
        rankings.fetch_adp(2026, teams=10, offline=True, settings=settings)

    message = str(exc_info.value)
    assert "rankings" in message
    assert "season=2026" in message
    assert "teams=10" in message


def test_normalize_adp_maps_positions_and_carries_spread_columns() -> None:
    df = rankings.normalize_adp(FFC_FIXTURE_PAYLOAD, season=2026)

    assert df.height == 3
    gibbs = df.filter(pl.col("player_name") == "Jahmyr Gibbs").row(0, named=True)
    assert gibbs["position"] == "RB"
    assert gibbs["team"] == "DET"
    assert gibbs["season"] == 2026
    assert gibbs["adp"] == 1.8
    assert gibbs["adp_sd"] == 0.8
    assert gibbs["adp_high"] == 1
    assert gibbs["adp_low"] == 4
    assert gibbs["times_drafted"] == 1176

    dst = df.filter(pl.col("player_name") == "Denver Defense").row(0, named=True)
    assert dst["position"] == "DST"  # DEF -> DST

    kicker = df.filter(pl.col("player_name") == "Some Kicker").row(0, named=True)
    assert kicker["position"] == "K"  # PK -> K


def test_normalize_adp_drops_unknown_positions() -> None:
    payload = {
        "players": [
            *FFC_FIXTURE_PAYLOAD["players"],
            {
                "player_id": 4,
                "name": "Some Punter",
                "position": "PN",
                "team": "SF",
                "adp": 200.0,
                "times_drafted": 5,
                "high": 190,
                "low": 210,
                "stdev": 3.0,
            },
        ]
    }

    df = rankings.normalize_adp(payload, season=2026)

    assert df.height == 3
    assert "Some Punter" not in df["player_name"].to_list()
