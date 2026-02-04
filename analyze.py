"""Analysis module for JSONL test results with comprehensive metrics and insights."""

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from chatbot_tests import console


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class StepResult:
    """Individual step result."""
    step_name: str
    outcome: str
    duration_ms: Optional[int] = None
    message: Optional[str] = None
    timestamp: Optional[str] = None
    step_type: Optional[str] = None  # "action", "info", or "wait"


@dataclass
class TestResult:
    """Individual test result with steps."""
    nodeid: str
    name: str
    outcome: str
    duration_seconds: Optional[float] = None
    message: Optional[str] = None
    markers: List[str] = field(default_factory=list)
    steps: List[StepResult] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Complete analysis of a test run with comprehensive metrics."""

    # === Summary Statistics ===
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    skipped_tests: int = 0

    total_steps: int = 0
    passed_steps: int = 0
    failed_steps: int = 0

    # === Timing ===
    session_start: Optional[str] = None
    session_end: Optional[str] = None
    total_duration_seconds: float = 0.0

    # === Performance Metrics ===
    avg_test_duration: float = 0.0
    avg_step_duration_ms: float = 0.0
    slowest_tests: List[Tuple[str, float]] = field(default_factory=list)
    fastest_tests: List[Tuple[str, float]] = field(default_factory=list)
    avg_steps_per_test: float = 0.0

    # === Failure Analysis ===
    failure_categories: Dict[str, List[str]] = field(default_factory=dict)
    most_failing_steps: List[Tuple[str, int]] = field(default_factory=list)

    # === Detailed Results ===
    tests: List[TestResult] = field(default_factory=list)
    tests_by_marker: Dict[str, List[TestResult]] = field(default_factory=dict)
    failed_test_names: List[str] = field(default_factory=list)

    # === Source ===
    source_file: Optional[str] = None

    @property
    def pass_rate(self) -> float:
        """Calculate test pass rate as percentage."""
        if self.total_tests == 0:
            return 0.0
        return (self.passed_tests / (self.total_tests - self.skipped_tests)) * 100

    @property
    def step_pass_rate(self) -> float:
        """Calculate step pass rate as percentage."""
        if self.total_steps == 0:
            return 0.0
        return (self.passed_steps / self.total_steps) * 100

    @property
    def health_status(self) -> str:
        """Determine overall health based on pass rate."""
        if self.pass_rate >= 95:
            return "healthy"
        elif self.pass_rate >= 80:
            return "degraded"
        elif self.pass_rate >= 50:
            return "unstable"
        return "critical"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "summary": {
                "total_tests": self.total_tests,
                "passed_tests": self.passed_tests,
                "failed_tests": self.failed_tests,
                "skipped_tests": self.skipped_tests,
                "pass_rate": round(self.pass_rate, 2),
                "total_steps": self.total_steps,
                "passed_steps": self.passed_steps,
                "failed_steps": self.failed_steps,
                "step_pass_rate": round(self.step_pass_rate, 2),
                "total_duration_seconds": round(self.total_duration_seconds, 3),
                "health_status": self.health_status,
            },
            "performance": {
                "avg_test_duration_seconds": round(self.avg_test_duration, 3),
                "avg_step_duration_ms": round(self.avg_step_duration_ms, 1),
                "avg_steps_per_test": round(self.avg_steps_per_test, 1),
                "slowest_tests": [
                    {"name": name, "duration_seconds": round(dur, 3)}
                    for name, dur in self.slowest_tests
                ],
                "fastest_tests": [
                    {"name": name, "duration_seconds": round(dur, 3)}
                    for name, dur in self.fastest_tests
                ],
            },
            "failures": {
                "failed_tests": self.failed_test_names,
                "failure_categories": self.failure_categories,
                "most_failing_steps": [
                    {"step_name": name, "failure_count": count}
                    for name, count in self.most_failing_steps
                ],
            },
            "session": {
                "start": self.session_start,
                "end": self.session_end,
                "source_file": self.source_file,
            },
            "by_marker": {
                marker: [
                    {"name": t.name, "outcome": t.outcome, "duration_seconds": t.duration_seconds}
                    for t in tests
                ]
                for marker, tests in self.tests_by_marker.items()
            },
            "tests": [
                {
                    "nodeid": t.nodeid,
                    "name": t.name,
                    "outcome": t.outcome,
                    "duration_seconds": t.duration_seconds,
                    "message": t.message,
                    "markers": t.markers,
                    "steps": [
                        {
                            "step_name": s.step_name,
                            "outcome": s.outcome,
                            "duration_ms": s.duration_ms,
                            "message": s.message,
                            "step_type": s.step_type,
                        }
                        for s in t.steps
                    ],
                }
                for t in self.tests
            ],
        }


@dataclass
class ComparisonResult:
    """Result of comparing multiple test runs with trend analysis."""
    runs: List[AnalysisResult] = field(default_factory=list)
    pass_rate_trend: List[float] = field(default_factory=list)
    regressions: List[str] = field(default_factory=list)
    fixes: List[str] = field(default_factory=list)
    flaky: List[str] = field(default_factory=list)

    # NEW: Trend metrics
    trend_direction: str = "stable"  # "improving", "declining", "stable"
    avg_pass_rate: float = 0.0
    stability_score: int = 0  # 1-10

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "summary": {
                "run_count": len(self.runs),
                "trend_direction": self.trend_direction,
                "avg_pass_rate": round(self.avg_pass_rate, 2),
                "stability_score": self.stability_score,
            },
            "pass_rate_trend": [round(r, 2) for r in self.pass_rate_trend],
            "changes": {
                "regressions": self.regressions,
                "fixes": self.fixes,
                "flaky_tests": self.flaky,
            },
            "runs": [
                {
                    "source_file": r.source_file,
                    "pass_rate": round(r.pass_rate, 2),
                    "total_tests": r.total_tests,
                    "passed_tests": r.passed_tests,
                    "failed_tests": r.failed_tests,
                    "health_status": r.health_status,
                }
                for r in self.runs
            ],
        }


# =============================================================================
# Analysis Functions
# =============================================================================

def _categorize_failure(message: Optional[str]) -> str:
    """Categorize a failure by its error message."""
    if not message:
        return "Unknown"

    message_lower = message.lower()

    # Timeout-related
    if any(kw in message_lower for kw in ["timeout", "timed out", "exceeded"]):
        return "Timeout"

    # Assertion failures
    if "assert" in message_lower:
        if "visible" in message_lower or "hidden" in message_lower:
            return "UI Visibility"
        if "count" in message_lower or "length" in message_lower:
            return "Count Mismatch"
        if "text" in message_lower or "contain" in message_lower:
            return "Content Mismatch"
        return "Assertion Failed"

    # Network/API errors
    if any(kw in message_lower for kw in ["api", "network", "connection", "http", "status"]):
        return "API/Network Error"

    # Element not found
    if any(kw in message_lower for kw in ["not found", "no such element", "locator"]):
        return "Element Not Found"

    return "Other"


def _calculate_performance_metrics(result: AnalysisResult) -> None:
    """Calculate performance metrics from test results."""
    if not result.tests:
        return

    # Test duration metrics
    durations = [t.duration_seconds for t in result.tests if t.duration_seconds]
    if durations:
        result.avg_test_duration = sum(durations) / len(durations)

        sorted_tests = sorted(
            [(t.name, t.duration_seconds) for t in result.tests if t.duration_seconds],
            key=lambda x: x[1],
            reverse=True,
        )
        result.slowest_tests = sorted_tests[:5]
        result.fastest_tests = sorted_tests[-5:][::-1]

    # Step metrics
    all_step_durations = []
    step_failure_counts: Dict[str, int] = defaultdict(int)
    total_steps = 0

    for test in result.tests:
        total_steps += len(test.steps)
        for step in test.steps:
            if step.duration_ms:
                all_step_durations.append(step.duration_ms)
            if step.outcome == "failed":
                step_failure_counts[step.step_name] += 1

    if all_step_durations:
        result.avg_step_duration_ms = sum(all_step_durations) / len(all_step_durations)

    if result.tests:
        result.avg_steps_per_test = total_steps / len(result.tests)

    result.most_failing_steps = sorted(
        step_failure_counts.items(), key=lambda x: x[1], reverse=True
    )[:5]


def _analyze_failures(result: AnalysisResult) -> None:
    """Analyze and categorize failures."""
    categories: Dict[str, List[str]] = defaultdict(list)

    for test in result.tests:
        if test.outcome != "failed":
            continue

        # Categorize by test error message
        category = _categorize_failure(test.message)
        categories[category].append(test.name)

        # Also check step failures
        for step in test.steps:
            if step.outcome == "failed":
                step_category = _categorize_failure(step.message)
                if step_category != category:
                    categories[step_category].append(f"{test.name} > {step.step_name}")

    result.failure_categories = dict(categories)


def analyze_jsonl(file_path: Path) -> AnalysisResult:
    """Analyze a JSONL file and return comprehensive results.

    Args:
        file_path: Path to the JSONL file

    Returns:
        AnalysisResult with complete analysis including metrics
    """
    result = AnalysisResult()
    result.source_file = str(file_path)
    current_test: Optional[TestResult] = None
    tests_by_nodeid: Dict[str, TestResult] = {}
    marker_groups: Dict[str, List[TestResult]] = defaultdict(list)

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            event = json.loads(line)
            event_type = event.get("event_type")

            if event_type == "session_start":
                result.session_start = event.get("timestamp")

            elif event_type == "session_end":
                result.session_end = event.get("timestamp")
                if result.session_start and result.session_end:
                    start = datetime.fromisoformat(result.session_start)
                    end = datetime.fromisoformat(result.session_end)
                    result.total_duration_seconds = (end - start).total_seconds()

            elif event_type == "test_start":
                nodeid = event.get("nodeid", "")
                current_test = TestResult(
                    nodeid=nodeid,
                    name=event.get("name", ""),
                    outcome="unknown",
                    markers=event.get("markers", []) or [],
                )
                tests_by_nodeid[nodeid] = current_test

            elif event_type == "test_end":
                nodeid = event.get("nodeid", "")
                outcome = event.get("outcome", "unknown")

                if outcome == "skipped":
                    current_test = TestResult(
                        nodeid=nodeid,
                        name=event.get("name", ""),
                        outcome="unknown",
                        markers=event.get("markers", []) or [],
                    )
                    tests_by_nodeid[nodeid] = current_test

                test = tests_by_nodeid.get(nodeid)
                if test:
                    test.outcome = outcome
                    test.duration_seconds = event.get("duration_seconds")
                    test.message = event.get("message")

                    result.total_tests += 1
                    if outcome == "passed":
                        result.passed_tests += 1
                    elif outcome == "failed":
                        result.failed_tests += 1
                        result.failed_test_names.append(test.name)
                    elif outcome == "skipped":
                        result.skipped_tests += 1

                    for marker in test.markers:
                        marker_groups[marker].append(test)

                    result.tests.append(test)

                current_test = None

            elif event_type == "step":
                step = StepResult(
                    step_name=event.get("step_name", ""),
                    outcome=event.get("outcome", "unknown"),
                    duration_ms=event.get("duration_ms"),
                    message=event.get("message"),
                    timestamp=event.get("timestamp"),
                    step_type=event.get("step_type"),
                )

                result.total_steps += 1
                if step.outcome == "passed":
                    result.passed_steps += 1
                elif step.outcome == "failed":
                    result.failed_steps += 1

                nodeid = event.get("nodeid")
                if nodeid and nodeid in tests_by_nodeid:
                    tests_by_nodeid[nodeid].steps.append(step)
                elif current_test:
                    current_test.steps.append(step)

    result.tests_by_marker = dict(marker_groups)

    # Calculate additional metrics
    _calculate_performance_metrics(result)
    _analyze_failures(result)

    return result


def compare_runs(file_paths: List[Path]) -> ComparisonResult:
    """Compare multiple test runs with trend analysis.

    Args:
        file_paths: List of JSONL file paths to compare (in chronological order)

    Returns:
        ComparisonResult with trends and identified issues
    """
    comparison = ComparisonResult()

    for path in file_paths:
        analysis = analyze_jsonl(path)
        comparison.runs.append(analysis)
        comparison.pass_rate_trend.append(analysis.pass_rate)

    if not comparison.pass_rate_trend:
        return comparison

    # Calculate average pass rate
    comparison.avg_pass_rate = sum(comparison.pass_rate_trend) / len(comparison.pass_rate_trend)

    if len(comparison.runs) < 2:
        comparison.stability_score = 10 if comparison.avg_pass_rate >= 95 else 5
        return comparison

    # Determine trend direction
    first_half = comparison.pass_rate_trend[: len(comparison.pass_rate_trend) // 2]
    second_half = comparison.pass_rate_trend[len(comparison.pass_rate_trend) // 2 :]

    first_avg = sum(first_half) / len(first_half) if first_half else 0
    second_avg = sum(second_half) / len(second_half) if second_half else 0

    if second_avg > first_avg + 5:
        comparison.trend_direction = "improving"
    elif second_avg < first_avg - 5:
        comparison.trend_direction = "declining"
    else:
        comparison.trend_direction = "stable"

    # Track test outcomes across runs
    test_outcomes: Dict[str, List[str]] = defaultdict(list)
    for run in comparison.runs:
        run_tests = {t.name: t.outcome for t in run.tests}
        all_test_names = set(test_outcomes.keys()) | set(run_tests.keys())
        for name in all_test_names:
            outcome = run_tests.get(name, "missing")
            test_outcomes[name].append(outcome)

    # Identify regressions, fixes, and flaky tests
    flaky_count = 0
    for test_name, outcomes in test_outcomes.items():
        if len(outcomes) < 2:
            continue

        prev = outcomes[-2]
        curr = outcomes[-1]

        if prev == "passed" and curr == "failed":
            comparison.regressions.append(test_name)
        elif prev == "failed" and curr == "passed":
            comparison.fixes.append(test_name)

        unique_outcomes = set(o for o in outcomes if o != "missing")
        if len(unique_outcomes) > 1:
            comparison.flaky.append(test_name)
            flaky_count += 1

    # Calculate stability score (1-10)
    flaky_ratio = flaky_count / len(test_outcomes) if test_outcomes else 0
    stability_base = 10 - int(flaky_ratio * 10)
    regression_penalty = min(len(comparison.regressions), 3)
    comparison.stability_score = max(1, stability_base - regression_penalty)

    return comparison


# =============================================================================
# Formatting Helpers
# =============================================================================

def _format_outcome(outcome: str) -> str:
    """Format outcome with color."""
    if outcome == "passed":
        return console.success("PASS")
    elif outcome == "failed":
        return console.error("FAIL")
    elif outcome == "skipped":
        return console.dim("SKIP")
    return console.dim(outcome.upper())


def _format_percentage(value: float, good: float = 90, warn: float = 70) -> str:
    """Format percentage with color based on thresholds."""
    formatted = f"{value:.1f}%"
    if value >= good:
        return console.success(formatted)
    elif value >= warn:
        return console.warn(formatted)
    return console.error(formatted)


def _format_health(status: str) -> str:
    """Format health status with color."""
    status_upper = status.upper()
    if status == "healthy":
        return console.success(status_upper)
    elif status == "degraded":
        return console.warn(status_upper)
    return console.error(status_upper)


def _format_trend(direction: str) -> str:
    """Format trend direction with arrow and color."""
    if direction == "improving":
        return console.success("IMPROVING")
    elif direction == "declining":
        return console.error("DECLINING")
    return console.dim("STABLE")


def _format_duration(seconds: float) -> str:
    """Format duration in human-readable form."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.1f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.0f}s"


# =============================================================================
# Print Functions
# =============================================================================

def print_summary(analysis: AnalysisResult) -> None:
    """Print comprehensive summary with colors."""
    print("\n" + console.bold("=" * 60))
    print(console.bold("TEST RUN ANALYSIS"))
    print(console.bold("=" * 60))

    if analysis.source_file:
        print(f"\n{console.label('Source:')} {console.dim(analysis.source_file)}")

    # Session info
    print(f"{console.label('Session:')} {analysis.session_start or 'N/A'}")
    print(f"{console.label('Duration:')} {_format_duration(analysis.total_duration_seconds)}")
    print(f"{console.label('Health:')} {_format_health(analysis.health_status)}")

    # Test Summary
    print(f"\n{console.bold('--- Test Summary ---')}")
    print(f"  Total:     {analysis.total_tests}")
    print(f"  Passed:    {console.success(str(analysis.passed_tests))}")
    print(f"  Failed:    {console.error(str(analysis.failed_tests)) if analysis.failed_tests else '0'}")
    print(f"  Skipped:   {console.dim(str(analysis.skipped_tests))}")
    print(f"  Pass Rate: {_format_percentage(analysis.pass_rate)}")

    # Step Summary
    if analysis.total_steps > 0:
        print(f"\n{console.bold('--- Step Summary ---')}")
        print(f"  Total:     {analysis.total_steps}")
        print(f"  Passed:    {console.success(str(analysis.passed_steps))}")
        print(f"  Failed:    {console.error(str(analysis.failed_steps)) if analysis.failed_steps else '0'}")
        print(f"  Pass Rate: {_format_percentage(analysis.step_pass_rate)}")

    print(console.bold("=" * 60))


def print_performance(analysis: AnalysisResult) -> None:
    """Print performance analysis."""
    print(f"\n{console.bold('--- Performance Analysis ---')}")

    print(f"\n{console.label('Timing:')}")
    print(f"  Avg test duration:  {_format_duration(analysis.avg_test_duration)}")
    print(f"  Avg step duration:  {analysis.avg_step_duration_ms:.0f}ms")
    print(f"  Avg steps per test: {analysis.avg_steps_per_test:.1f}")

    if analysis.slowest_tests:
        print(f"\n{console.label('Slowest Tests:')}")
        for name, duration in analysis.slowest_tests[:5]:
            print(f"  {console.warn(_format_duration(duration)):>8}  {name}")

    if analysis.fastest_tests:
        print(f"\n{console.label('Fastest Tests:')}")
        for name, duration in analysis.fastest_tests[:3]:
            print(f"  {console.success(_format_duration(duration)):>8}  {name}")


def print_failures(analysis: AnalysisResult) -> None:
    """Print detailed failure analysis."""
    failed_tests = [t for t in analysis.tests if t.outcome == "failed"]

    if not failed_tests:
        print(f"\n{console.success('No failures to report.')}")
        return

    print(f"\n{console.bold('--- Failure Analysis ---')}")
    print(f"Total failures: {console.error(str(len(failed_tests)))}")

    # Show by category
    if analysis.failure_categories:
        print(f"\n{console.label('By Category:')}")
        for category, tests in sorted(
            analysis.failure_categories.items(), key=lambda x: -len(x[1])
        ):
            print(f"  {console.error(category)}: {len(tests)} failure(s)")

    # Most failing steps
    if analysis.most_failing_steps:
        print(f"\n{console.label('Most Failing Steps:')}")
        for step_name, count in analysis.most_failing_steps[:5]:
            print(f"  {console.error(str(count))} failures: {step_name}")

    # Detailed failures
    print(f"\n{console.label('Failed Tests:')}")
    for test in failed_tests:
        print(f"\n  {console.error('FAILED')}: {test.name}")
        if test.message:
            # Truncate long messages
            msg = test.message[:200] + "..." if len(test.message) > 200 else test.message
            print(f"    Error: {console.dim(msg)}")

        failed_steps = [s for s in test.steps if s.outcome == "failed"]
        if failed_steps:
            print(f"    Failed steps:")
            for step in failed_steps:
                print(f"      - {step.step_name}")
                if step.message:
                    msg = step.message[:100] + "..." if len(step.message) > 100 else step.message
                    print(f"        {console.dim(msg)}")


def print_steps(analysis: AnalysisResult) -> None:
    """Print full step details for each test."""
    print(f"\n{console.bold('--- Step Details ---')}")
    print(f"Total: {len(analysis.tests)} tests\n")

    for test in analysis.tests:
        outcome_str = _format_outcome(test.outcome)
        duration_str = f" ({_format_duration(test.duration_seconds)})" if test.duration_seconds else ""
        print(f"[{outcome_str}] {test.name}{duration_str}")

        if test.steps:
            for step in test.steps:
                icon = console.success("+") if step.outcome == "passed" else console.error("x")
                dur = f" ({step.duration_ms}ms)" if step.duration_ms else ""
                print(f"    [{icon}] {step.step_name}{console.dim(dur)}")
                if step.message and step.outcome == "failed":
                    msg = step.message[:80] + "..." if len(step.message) > 80 else step.message
                    print(f"        {console.dim(msg)}")
        else:
            print(f"    {console.dim('(no steps)')}")
        print()


def print_by_marker(analysis: AnalysisResult) -> None:
    """Print results grouped by pytest marker."""
    if not analysis.tests_by_marker:
        print(f"\n{console.dim('No marker data available.')}")
        return

    print(f"\n{console.bold('--- Results by Marker ---')}")

    for marker, tests in sorted(analysis.tests_by_marker.items()):
        passed = sum(1 for t in tests if t.outcome == "passed")
        failed = sum(1 for t in tests if t.outcome == "failed")
        total = len(tests)
        rate = (passed / total * 100) if total > 0 else 0

        rate_str = _format_percentage(rate)
        print(f"\n  {console.label('@' + marker)}: {passed}/{total} passed ({rate_str})")

        if failed > 0:
            for test in tests:
                if test.outcome == "failed":
                    print(f"    {console.error('FAIL')}: {test.name}")


def print_insights(analysis: AnalysisResult) -> None:
    """Generate automatic insights from the data."""
    print(f"\n{console.bold('--- Insights & Recommendations ---')}")

    # Health assessment
    print(f"\n{console.label('Health Assessment:')}")
    health_msg = {
        "healthy": "Test suite is in good shape. All critical functionality working.",
        "degraded": "Some issues detected. Review failures before deployment.",
        "unstable": "Significant failures. Investigation required.",
        "critical": "Major issues. Do not deploy without fixing.",
    }
    print(f"  Status: {_format_health(analysis.health_status)}")
    print(f"  {health_msg.get(analysis.health_status, 'Unknown status')}")

    # Key metrics
    print(f"\n{console.label('Key Metrics:')}")
    print(f"  Test pass rate: {_format_percentage(analysis.pass_rate)}")
    print(f"  Step pass rate: {_format_percentage(analysis.step_pass_rate)}")

    if analysis.avg_test_duration > 30:
        print(f"  {console.warn('Warning')}: Average test duration ({_format_duration(analysis.avg_test_duration)}) is high")

    # Recommendations
    recommendations = []

    if analysis.failed_tests > 0:
        recommendations.append(f"Fix {analysis.failed_tests} failing test(s) before deployment")

    if analysis.most_failing_steps:
        step_name, count = analysis.most_failing_steps[0]
        recommendations.append(f"Investigate '{step_name}' - most common failure point ({count} failures)")

    if analysis.pass_rate < 95:
        recommendations.append(f"Target 95%+ pass rate (currently {analysis.pass_rate:.1f}%)")

    if analysis.slowest_tests:
        slowest_name, slowest_dur = analysis.slowest_tests[0]
        if slowest_dur > 60:
            recommendations.append(f"Optimize slow test '{slowest_name}' ({_format_duration(slowest_dur)})")

    if "Timeout" in analysis.failure_categories:
        timeout_count = len(analysis.failure_categories["Timeout"])
        recommendations.append(f"Address {timeout_count} timeout issue(s) - may indicate performance problems")

    if recommendations:
        print(f"\n{console.label('Recommendations:')}")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    else:
        print(f"\n{console.success('No immediate actions required.')}")


def print_comparison(comparison: ComparisonResult) -> None:
    """Print enhanced comparison results."""
    print("\n" + console.bold("=" * 60))
    print(console.bold("TEST RUN COMPARISON"))
    print(console.bold("=" * 60))

    print(f"\n{console.label('Runs analyzed:')} {len(comparison.runs)}")
    print(f"{console.label('Trend:')} {_format_trend(comparison.trend_direction)}")
    print(f"{console.label('Avg pass rate:')} {_format_percentage(comparison.avg_pass_rate)}")
    print(f"{console.label('Stability score:')} {comparison.stability_score}/10")

    # Pass rate trend
    print(f"\n{console.bold('--- Pass Rate Trend ---')}")
    for i, (run, rate) in enumerate(zip(comparison.runs, comparison.pass_rate_trend)):
        filename = Path(run.source_file).name if run.source_file else f"Run {i + 1}"
        bar_len = int(rate / 5)
        bar = console.success("#" * bar_len) if rate >= 90 else (
            console.warn("#" * bar_len) if rate >= 70 else console.error("#" * bar_len)
        )
        rate_str = _format_percentage(rate)
        print(f"  {filename:30} {rate_str:>8} {bar}")

    # Regressions
    if comparison.regressions:
        print(f"\n{console.bold('--- Regressions ---')} ({console.error(str(len(comparison.regressions)))})")
        print(f"{console.dim('Tests that went from PASS to FAIL:')}")
        for name in comparison.regressions:
            print(f"  {console.error('-')} {name}")

    # Fixes
    if comparison.fixes:
        print(f"\n{console.bold('--- Fixes ---')} ({console.success(str(len(comparison.fixes)))})")
        print(f"{console.dim('Tests that went from FAIL to PASS:')}")
        for name in comparison.fixes:
            print(f"  {console.success('+')} {name}")

    # Flaky tests
    if comparison.flaky:
        print(f"\n{console.bold('--- Flaky Tests ---')} ({console.warn(str(len(comparison.flaky)))})")
        print(f"{console.dim('Tests with inconsistent results:')}")
        for name in comparison.flaky:
            print(f"  {console.warn('~')} {name}")

    # Summary recommendations
    print(f"\n{console.bold('--- Summary ---')}")
    if comparison.trend_direction == "declining":
        print(f"  {console.error('Warning')}: Quality is declining. Review recent changes.")
    elif comparison.regressions:
        print(f"  {console.warn('Note')}: {len(comparison.regressions)} regression(s) detected. Investigate before release.")
    elif comparison.trend_direction == "improving":
        print(f"  {console.success('Good')}: Quality is improving!")
    else:
        print(f"  Quality is stable.")

    if comparison.flaky:
        print(f"  {console.warn('Note')}: {len(comparison.flaky)} flaky test(s) need attention.")

    print(console.bold("=" * 60))
