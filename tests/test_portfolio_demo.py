"""Public demo navigation and session-only trade interaction."""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "portfolio" / "app.py"


def test_demo_navigation_and_empty_filters():
    app = AppTest.from_file(str(APP)).run()
    assert not app.exception
    for page in ("Rankings", "Start / sit", "Trade explorer"):
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception
    # Hawks: Mercer + Vale + Reed + Wells + Brooks' former backup Ellis.
    # Foxes: Archer + Brooks + Lane + Quinn + West.
    assert [m.value for m in app.metric] == ["82.3 pts", "81.8 pts"]
    assert [m.delta for m in app.metric] == ["+1.8 pts", "+2.1 pts"]
    app.multiselect[0].set_value([]).run()
    assert not app.exception
    assert any("Choose at least" in message.value for message in app.info)
    app.sidebar.radio[0].set_value("Rankings").run()
    app.multiselect[0].set_value([]).run()
    assert not app.exception
    assert app.dataframe[0].value.empty


def test_demo_trade_does_not_mutate_another_session():
    first = AppTest.from_file(str(APP)).run()
    first.sidebar.radio[0].set_value("Trade explorer").run()
    first.multiselect[0].set_value(["Alex Mercer"]).run()
    assert not first.exception
    assert any("cannot fill" in message.value for message in first.warning)
    second = AppTest.from_file(str(APP)).run()
    second.sidebar.radio[0].set_value("Trade explorer").run()
    assert not second.exception
    assert second.multiselect[0].value == ["Theo Brooks"]
