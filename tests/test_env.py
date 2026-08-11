import os
from pathlib import Path

import pytest

from ffapp.env import load_env


def test_load_env_sets_environment_variables_from_dotenv_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("FFAPP_OFFLINE_TEST_VAR", raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text("FFAPP_OFFLINE_TEST_VAR=hello\n")

    load_env(env_file)

    assert os.environ["FFAPP_OFFLINE_TEST_VAR"] == "hello"


def test_load_env_does_not_raise_when_file_missing(tmp_path: Path) -> None:
    load_env(tmp_path / "does-not-exist.env")
