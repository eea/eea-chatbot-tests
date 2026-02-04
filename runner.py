"""Test runner for chatbot Playwright tests."""

import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional

import pytest

from chatbot_tests.config import get_settings
from chatbot_tests.plugin import ResultCollectorPlugin, set_current_plugin, TestEvent
from chatbot_tests.output import JSONLWriter, generate_output_filename
from chatbot_tests import console


class TestStatus(str, Enum):
    """Test run status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TestRunSummary:
    """Summary of a test run."""
    test_id: str
    status: TestStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    total: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    duration_seconds: float = 0
    current_test: Optional[str] = None
    progress: int = 0
    output_file: Optional[str] = None


def _print_event(event: TestEvent, run: TestRunSummary):
    """Print test event to console."""
    if event.event_type == "test_start":
        run.current_test = event.name
        console.writeln(f"\n{console.label('TEST:')} {console.info(event.name)}")

    elif event.event_type == "step":
        passed = event.outcome == "passed"
        failed = event.outcome == "failed"
        if passed:
            icon = console.success("+")
        elif failed:
            icon = console.error("x")
        duration = console.dim(f" ({event.duration_ms}ms)") if event.duration_ms else ""
        console.writeln(f"  [{icon}] {event.step_name}{duration}")
        if event.message and failed:
            console.writeln(f"{' ' * 6}{console.error(event.message)}")

    elif event.event_type == "test_end":
        run.total += 1
        if event.outcome == "passed":
            run.passed += 1
            status = console.success("PASSED")
        elif event.outcome == "skipped":
            run.skipped += 1
            run.current_test = None
            return
        elif event.outcome == "failed":
            run.failed += 1
            status = console.error("FAILED")
        else:
            status = console.warn(event.outcome.upper())
        duration = console.dim(f" ({event.duration_seconds:.2f}s)") if event.duration_seconds else ""
        console.writeln(f"  => {status}{duration}")
        if event.message:
            console.writeln(f"{' ' * 5}{console.error('Error:')} {event.message}")
        run.current_test = None


class TestRunner:
    """Orchestrates test execution."""

    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self._active_runs: dict[str, TestRunSummary] = {}

    def create_run(self) -> str:
        """Create a new test run and return its ID."""
        test_id = str(uuid.uuid4())
        self._active_runs[test_id] = TestRunSummary(
            test_id=test_id,
            status=TestStatus.PENDING,
            started_at=datetime.now(),
        )
        return test_id

    def get_run(self, test_id: str) -> Optional[TestRunSummary]:
        """Get a test run by ID."""
        return self._active_runs.get(test_id)

    def get_all_runs(self) -> list[TestRunSummary]:
        """Get all test runs."""
        return list(self._active_runs.values())

    def run_tests(
        self,
        test_id: str,
        markers: Optional[List[str]] = None,
        headless: bool = True,
        output_path: Optional[Path] = None,
        limit: Optional[int] = None,
    ) -> TestRunSummary:
        """Run tests and return summary."""
        run = self._active_runs.get(test_id)
        if not run:
            raise ValueError(f"Test run {test_id} not found")

        run.status = TestStatus.RUNNING
        run.started_at = datetime.now()

        # Determine output path
        if output_path is None:
            output_path = self.settings.reports_path / generate_output_filename()
        run.output_file = str(output_path)

        # Build pytest args
        tests_dir = Path(__file__).parent / "tests"
        pytest_args = [
            # "--video=retain-on-failure",
            # "--output=./reports/videos",
            str(tests_dir),
        ]

        if markers:
            pytest_args.extend(["-m", " or ".join(markers)])
        if not headless:
            pytest_args.append("--headed")
        if limit:
            pytest_args.extend(["--limit", str(limit)])

        # Run tests with JSONL output
        with JSONLWriter(output_path) as writer:
            def on_event(event: TestEvent):
                writer.write_event(event)
                _print_event(event, run)

            plugin = ResultCollectorPlugin(on_event=on_event)
            set_current_plugin(plugin)
            exit_code = pytest.main(pytest_args, plugins=[plugin])
            set_current_plugin(None)

        # Finalize
        run.completed_at = datetime.now()
        run.duration_seconds = (run.completed_at - run.started_at).total_seconds()
        run.status = TestStatus.COMPLETED if exit_code == 0 else TestStatus.FAILED
        return run


def run_tests_sync(
    markers: Optional[List[str]] = None,
    headless: bool = True,
    output_path: Optional[Path] = None,
    limit: Optional[int] = None,
) -> TestRunSummary:
    """Synchronous helper to run tests."""
    runner = TestRunner()
    test_id = runner.create_run()
    return runner.run_tests(
        test_id=test_id,
        markers=markers,
        headless=headless,
        output_path=output_path,
        limit=limit,
    )
