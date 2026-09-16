from __future__ import annotations

from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

from ffapp.tools.automation import RefreshSchedule, task_xml


def test_task_xml_uses_project_executable_and_weekly_label(tmp_path: Path) -> None:
    executable = tmp_path / ".venv" / "Scripts" / "ffapp.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"")
    xml = task_xml(
        tmp_path,
        RefreshSchedule("thursday", "Thursday", 9),
        start=datetime(2026, 9, 16, 12),
    )
    root = ET.fromstring(xml)
    values = [node.text for node in root.iter()]
    assert str(executable) in values
    assert str(tmp_path) in values
    assert "refresh weekly --all-leagues --run-label thursday --no-offline" in values
    assert any(node.tag.endswith("Thursday") for node in root.iter())
    assert "PT5M" in values
    assert "3" in values
