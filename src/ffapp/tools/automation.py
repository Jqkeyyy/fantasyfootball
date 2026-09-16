"""Windows Task Scheduler integration for unattended league refreshes."""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from xml.etree import ElementTree as ET


@dataclass(frozen=True)
class RefreshSchedule:
    run_label: str
    weekday: str
    hour: int
    minute: int = 0

    @property
    def task_name(self) -> str:
        return f"FantasyFootball\\{self.run_label.title()} Refresh"


DEFAULT_SCHEDULES = (
    RefreshSchedule("tuesday", "Tuesday", 9),
    RefreshSchedule("thursday", "Thursday", 9),
    RefreshSchedule("sunday", "Sunday", 9),
)


def task_xml(
    project_root: Path, schedule: RefreshSchedule, *, start: datetime | None = None
) -> str:
    """Build a Task Scheduler definition without shell-quoting user paths."""
    executable = project_root / ".venv" / "Scripts" / "ffapp.exe"
    if not executable.exists():
        raise FileNotFoundError(f"Expected project executable at {executable}")
    base = start or datetime.now().astimezone()
    boundary = (base + timedelta(days=1)).replace(
        hour=schedule.hour, minute=schedule.minute, second=0, microsecond=0
    )
    ns = "http://schemas.microsoft.com/windows/2004/02/mit/task"
    ET.register_namespace("", ns)
    task = ET.Element(f"{{{ns}}}Task", version="1.4")
    triggers = ET.SubElement(task, f"{{{ns}}}Triggers")
    calendar = ET.SubElement(triggers, f"{{{ns}}}CalendarTrigger")
    ET.SubElement(calendar, f"{{{ns}}}StartBoundary").text = boundary.isoformat(timespec="seconds")
    ET.SubElement(calendar, f"{{{ns}}}Enabled").text = "true"
    weekly = ET.SubElement(calendar, f"{{{ns}}}ScheduleByWeek")
    ET.SubElement(weekly, f"{{{ns}}}WeeksInterval").text = "1"
    days = ET.SubElement(weekly, f"{{{ns}}}DaysOfWeek")
    ET.SubElement(days, f"{{{ns}}}{schedule.weekday}")
    principals = ET.SubElement(task, f"{{{ns}}}Principals")
    principal = ET.SubElement(principals, f"{{{ns}}}Principal", id="Author")
    ET.SubElement(principal, f"{{{ns}}}LogonType").text = "InteractiveToken"
    ET.SubElement(principal, f"{{{ns}}}RunLevel").text = "LeastPrivilege"
    settings = ET.SubElement(task, f"{{{ns}}}Settings")
    ET.SubElement(settings, f"{{{ns}}}MultipleInstancesPolicy").text = "IgnoreNew"
    ET.SubElement(settings, f"{{{ns}}}StartWhenAvailable").text = "true"
    ET.SubElement(settings, f"{{{ns}}}ExecutionTimeLimit").text = "PT2H"
    actions = ET.SubElement(task, f"{{{ns}}}Actions", Context="Author")
    action = ET.SubElement(actions, f"{{{ns}}}Exec")
    ET.SubElement(action, f"{{{ns}}}Command").text = str(executable)
    ET.SubElement(
        action, f"{{{ns}}}Arguments"
    ).text = f"refresh weekly --all-leagues --run-label {schedule.run_label} --no-offline"
    ET.SubElement(action, f"{{{ns}}}WorkingDirectory").text = str(project_root)
    return ET.tostring(task, encoding="unicode")


def install_tasks(project_root: Path) -> list[str]:
    installed: list[str] = []
    for schedule in DEFAULT_SCHEDULES:
        definition = task_xml(project_root, schedule)
        with tempfile.NamedTemporaryFile(
            "w", suffix=".xml", encoding="utf-16", delete=False
        ) as file:
            file.write(definition)
            path = Path(file.name)
        try:
            result = subprocess.run(
                ["schtasks.exe", "/Create", "/TN", schedule.task_name, "/XML", str(path), "/F"],
                capture_output=True,
                text=True,
                check=False,
            )
        finally:
            path.unlink(missing_ok=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        installed.append(schedule.task_name)
    return installed


def task_statuses() -> dict[str, bool]:
    statuses: dict[str, bool] = {}
    for schedule in DEFAULT_SCHEDULES:
        result = subprocess.run(
            ["schtasks.exe", "/Query", "/TN", schedule.task_name],
            capture_output=True,
            text=True,
            check=False,
        )
        statuses[schedule.run_label] = result.returncode == 0
    return statuses


def remove_tasks() -> list[str]:
    removed: list[str] = []
    for schedule in DEFAULT_SCHEDULES:
        result = subprocess.run(
            ["schtasks.exe", "/Delete", "/TN", schedule.task_name, "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0:
            removed.append(schedule.task_name)
    return removed


__all__ = [
    "DEFAULT_SCHEDULES",
    "RefreshSchedule",
    "install_tasks",
    "remove_tasks",
    "task_statuses",
    "task_xml",
]
