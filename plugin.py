from dataclasses import dataclass, field, asdict
from typing import Callable, Optional, List
from datetime import datetime
import time
import pytest


@dataclass
class TestEvent:
    event_type: str
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat())
    nodeid: str = None
    name: str = None
    location: str = None
    outcome: str = None
    duration_seconds: float = None
    duration_ms: int = None
    message: str = None
    step_name: str = None
    markers: List[str] = None
    step_type: str = None  # "action", "info", or "wait"

    def to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}


class ResultCollectorPlugin:
    """Pytest plugin that collects results and emits events via callback."""

    def __init__(self, on_event: Callable[[TestEvent], None] = None):
        self.on_event = on_event
        self.results: List[TestEvent] = []
        self._current_test_start: float = None
        self._current_nodeid: str = None
        self._current_test_name: str = None
        self._current_markers: List[str] = None

    def _emit(self, event: TestEvent):
        self.results.append(event)
        if self.on_event:
            self.on_event(event)

    def pytest_sessionstart(self, session):
        self._emit(TestEvent("session_start"))

    def pytest_runtest_logstart(self, nodeid, location):
        self._current_test_start = time.time()
        self._current_nodeid = nodeid
        self._current_test_name = nodeid.split("::")[-1]

    def pytest_runtest_setup(self, item):
        # Capture markers from test item
        self._current_markers = [
            mark.name for mark in item.iter_markers()
            if mark.name not in ('parametrize', 'usefixtures')
        ]
        self._emit(TestEvent(
            "test_start",
            nodeid=item.nodeid,
            name=item.name,
            location=item.location[0],
            markers=self._current_markers if self._current_markers else None
        ))

    def pytest_runtest_makereport(self, item, call):
        duration = time.time() - self._current_test_start if self._current_test_start else None
        if call.excinfo:
            message = str(call.excinfo.value).replace("\n", "\n" + " " * 5)
            outcome = "failed"
            if call.excinfo.errisinstance(pytest.skip.Exception):
                outcome = "skipped"
            self._emit(TestEvent(
                "test_end",
                nodeid=item.nodeid,
                name=item.name,
                outcome=outcome,
                duration_seconds=round(duration, 3) if duration else None,
                message=message
            ))
        elif call.when == "call":
            self._emit(TestEvent(
                "test_end",
                nodeid=item.nodeid,
                name=item.name,
                outcome="passed",
                duration_seconds=round(duration, 3) if duration else None,
            ))
        if call.excinfo or call.when == "call":
            # Clear test context
            self._current_nodeid = None
            self._current_test_name = None
            self._current_markers = None

    def pytest_sessionfinish(self, session, exitstatus):
        self._emit(
            TestEvent(
                "session_end", outcome="passed" if exitstatus == 0 else "failed"
            )
        )


# For step logging within tests
_current_plugin: Optional[ResultCollectorPlugin] = None


def set_current_plugin(plugin: ResultCollectorPlugin):
    global _current_plugin
    _current_plugin = plugin


def log_step(name: str, outcome: str = "passed", message: str = None, duration_ms: int = None, start_time: int = None, step_type: str = "action"):
    """Log a test step event.

    Args:
        name: Step description
        outcome: "passed" or "failed"
        message: Optional error message
        duration_ms: Duration in milliseconds (if not using start_time)
        start_time: Start timestamp to calculate duration from
        step_type: Type of step - "action" (timed), "info" (no timing), or "wait" (timed)
    """
    ms = int((time.time() - start_time) * 1000) if start_time else duration_ms
    if _current_plugin:
        _current_plugin._emit(
            TestEvent(
                "step",
                step_name=name,
                outcome=outcome,
                message=message,
                duration_ms=ms,
                nodeid=_current_plugin._current_nodeid,
                name=_current_plugin._current_test_name,
                step_type=step_type
            ))
