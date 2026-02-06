"""LLM-based analysis for chatbot responses and test reports.

This module provides intelligent analysis using external LLM services
for deeper insights into test results and chatbot response quality.
"""

import json
from typing import Optional, Union
from pydantic import BaseModel, Field

try:
    import litellm
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False


# =============================================================================
# Structured Output Models
# =============================================================================

class ResponseVerification(BaseModel):
    """Combined verification results from a single LLM call."""

    lack_information: bool = Field(
        description="Whether the response lacks sufficient information to answer the question"
    )
    lack_information_explanation: str = Field(
        description="Brief explanation for the lack_information verdict"
    )

    answers_question: bool = Field(
        description="Whether the response directly answers the question asked"
    )
    answers_question_explanation: str = Field(
        description="Brief explanation for the answers_question verdict"
    )

    not_vague: bool = Field(
        description="Whether the response is specific and informative (not vague)"
    )
    not_vague_explanation: str = Field(
        description="Brief explanation for the not_vague verdict"
    )

    has_citations: bool = Field(
        description="Whether the response properly cites sources"
    )
    has_citations_explanation: str = Field(
        description="Brief explanation for the has_citations verdict"
    )


# =============================================================================
# LLM Prompts
# =============================================================================

CHATBOT_CONTEXT = """The system being tested is an AI chatbot for the European Environment Agency (EEA).

Key characteristics:
- Answers questions about environmental topics (climate change, air quality, water, biodiversity, pollution)
- Uses RAG (Retrieval-Augmented Generation) with source citations from EEA documents
- Features: Halloumi fact-checking (quality scores 0-100%), related question generation, user feedback
- Expected behavior: Accurate, well-cited responses based on EEA content

Testing context:
- Tests use a STEP-BASED structure: each test contains multiple steps (10-30+ steps per test)
- LLM responses are STREAMED progressively in the browser UI (not instant)
- Expected test durations: 20-60 seconds is NORMAL for tests involving real LLM calls
- High step counts and streaming cause legitimate long test times - this is NOT a latency issue
- Only flag performance as a concern if actual FAILURES occur due to timeouts"""


REPORT_ANALYSIS_PROMPT = f"""{CHATBOT_CONTEXT}

You are a senior QA automation engineer analyzing test results. Provide a thorough analysis.

## Step Types

Steps have a `step_type` field indicating their timing expectations:

- **"action"**: Browser/API actions that SHOULD have timing measured (click, wait, send, etc.)
- **"info"**: Informational logging that does NOT have timing - null/0 duration is EXPECTED and correct
- **"wait"**: Async waits that SHOULD have timing measured

**IMPORTANT**: Do NOT flag missing or zero timing on "info" type steps as a bug. These are
intentionally untimed informational messages (e.g., "LLM analysis: answer on-topic").

## Step Timing Analysis Guidelines

When analyzing step durations, understand the expected timing patterns:

**Expected to be slow (5-30+ seconds):**
- "Wait for assistant response" - LLM generates and streams response
- "Send message" / "Send predefined message" - Includes full LLM round-trip
- "Wait for response" - Streaming completion
- "Verify Halloumi quality fact-check" - External API call
- "Wait for related questions" - Additional LLM call

**Should be fast (<500ms):**
- "Verify X is visible/hidden" - Simple DOM check
- "Click X button" - User interaction
- "Type message" / "Fill textarea" - Input simulation
- "Verify textarea contains" - Value assertion

**No timing expected (info steps):**
- "INFO: ..." messages
- "LLM analysis: ..." results
- Status/result logging

**Analyze step timing to identify:**
1. UI steps taking >1s may indicate rendering issues or flaky selectors
2. Multiple slow "verify visible" steps in sequence may indicate poor page load
3. If "Wait for response" steps are consistently >30s, backend may need optimization
4. If fast steps (clicks, verifications) are slow, suspect frontend performance issues
5. Info steps with null/0 timing are CORRECT - do not flag these

## Instructions

Analyze the test report and provide:

### Summary
Brief overview (2-3 sentences) of what was tested and overall health status.

### Key Findings

#### Working Well
- List specific features/tests that passed consistently
- Note any performance improvements if comparing runs

#### Issues Detected
- List specific failures with test names
- Identify patterns in failures (e.g., all timeout issues, all UI failures)

### Step Timing Analysis
Analyze the step-level timing data:
- Identify steps that are slower than expected for their type
- Flag UI verification steps taking >1 second
- Note if LLM response steps are within acceptable range (10-30s)
- Highlight any patterns (e.g., all "verify visible" steps slow = page load issue)

### Risk Assessment

Rate one of: Critical, High, Medium, or Low

Explain why based on:
- Impact on core functionality (sending messages, receiving responses, citations)
- Number and severity of failures
- Whether failures affect user-facing features
- Step timing anomalies that could affect user experience

### Root Cause Analysis
For each issue category, suggest probable causes:
- Timeout failures → Check which step timed out (LLM response vs UI element)
- Slow UI steps → Frontend rendering, heavy components, network latency for assets
- Slow LLM steps → Expected if 10-30s, investigate if consistently >45s
- Assertion failures → Backend changes, data format changes, race conditions

### Recommended Actions

Prioritized list (most important first). For each action, state the priority level (Critical, High, Medium, Low) followed by the action and reference specific steps/tests.

### Quality Assessment
What this report indicates about the chatbot's production readiness.
Consider both functional correctness AND performance characteristics.

---

## Formatting Guidelines

- Use proper heading hierarchy (h2, h3, h4) - avoid bold text as pseudo-headings
- Keep bullet points concise (under 100 characters per line when possible)
- Use numbered lists for sequential steps or ranked items
- Avoid inline formatting like `**Bold:** text` - use subheadings instead
- Add blank lines between sections for readability
- Keep paragraphs short (2-3 sentences max)

Be specific. Reference test names AND step names with their durations. Avoid generic advice."""


COMPARISON_ANALYSIS_PROMPT = f"""{CHATBOT_CONTEXT}

You are analyzing multiple test runs to identify trends and changes.

## Instructions

Analyze the comparison data and provide:

### Trend Summary

State the direction: Improving, Declining, or Stable.

Explain based on pass rate changes between runs.

### Regression Analysis

For each regression (tests that started failing), provide:

1. Test name - What broke
2. Probable cause - Based on test name and context
3. Severity - Critical, High, Medium, or Low
4. Suggested action - Specific fix to investigate

### Improvements
Tests that started passing - what likely got fixed and why it matters.

### Stability Assessment

#### Flaky Tests
Tests with inconsistent results across runs:
- For each flaky test, explain possible causes (timing issues, external dependencies, race conditions)
- Recommend whether to fix, quarantine, or investigate further

#### Stability Score Interpretation
- 8-10: Reliable test suite
- 5-7: Some instability, address flaky tests
- 1-4: Significant reliability issues

### Recommendations

Prioritized actions based on the comparison. Group by priority:

1. Urgent - Critical regressions to fix
2. Important - Flaky tests to stabilize
3. Consider - Improvements to maintain

---

## Formatting Guidelines

- Use proper heading hierarchy (h2, h3, h4) - avoid bold text as pseudo-headings
- Keep bullet points concise (under 100 characters per line when possible)
- Use numbered lists for sequential steps or ranked items
- Avoid inline formatting like `**Bold:** text` - use subheadings instead
- Add blank lines between sections for readability
- Keep paragraphs short (2-3 sentences max)

Be specific about test names and changes between runs."""


VERIFY_ANSWER_PROMPT = """You are a QA analyst verifying chatbot responses for the European Environment Agency.

Evaluate the response against these three criteria:

1. **Lack Information**: Does the response lack sufficient information to answer the question?
   - true if it doesn't provide enough information to answer the question
   - false if it provides sufficient information to answer the question

2. **Answers Question**: Does the response directly address the question asked?
   - true if it provides relevant information that answers what was asked
   - false if it's off-topic, vague, or doesn't address the question

3. **Not Vague**: Is the response specific and informative?
   - true if it contains concrete facts, data, or actionable details
   - false if it uses generic statements like "it depends" or lacks substance

4. **Has Citations**: Does the response properly cite sources?
   - true if claims are attributed to specific documents/sources
   - false if it makes claims without citing where the information comes from

Return your evaluation as JSON with boolean verdicts and brief explanations (1-2 sentences each)."""


# =============================================================================
# LLM Analyzer Class
# =============================================================================

class LLMAnalyzer:
    """Analyzes chatbot responses and test reports using an LLM."""

    def __init__(
        self,
        model: str = "Inhouse-LLM/gpt-oss-120b",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        if not LITELLM_AVAILABLE:
            raise ImportError(
                "litellm is not installed. Install with: pip install litellm"
            )

        self.model = model
        self.base_url = base_url
        self.api_key = api_key

    # =========================================================================
    # Private Methods
    # =========================================================================

    def _build_kwargs(self, response_format: Optional[type] = None) -> dict:
        """Build kwargs dict for litellm.completion calls."""
        model = self.model
        if self.base_url and not model.startswith("openai/"):
            model = f"openai/{model}"

        kwargs = {
            "model": model,
            "api_key": self.api_key,
        }

        if self.base_url:
            kwargs["api_base"] = self.base_url

        if response_format:
            kwargs["response_format"] = response_format

        return kwargs

    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[type] = None
    ) -> str:
        """Execute LLM completion.

        Args:
            system_prompt: System message for the LLM
            user_prompt: User message/query
            response_format: Optional Pydantic model for structured output

        Returns:
            LLM response content as string
        """
        kwargs = self._build_kwargs(response_format)

        response = litellm.completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            **kwargs
        )

        return response.choices[0].message.content

    def _analyze(self, system_prompt: str, user_prompt: str) -> str:
        """Execute analysis returning text response.

        Args:
            system_prompt: Analysis instructions
            user_prompt: Content to analyze

        Returns:
            Analysis text or error message
        """
        try:
            return self._call_llm(system_prompt, user_prompt)
        except Exception as e:
            return f"Analysis failed: {str(e)}"

    # =========================================================================
    # Verification Methods
    # =========================================================================

    def verify_answer(
        self,
        question: str,
        response_message: str,
        final_documents: Optional[list[dict]] = None,
    ) -> ResponseVerification:
        """Verify all quality criteria in a single LLM call.

        Args:
            question: The original question
            response_message: The chatbot's response
            final_documents: Optional list of source documents

        Returns:
        ResponseVerification with all verdicts and explanations
        """
        user_prompt = f"Question: {question}\n\nResponse: {response_message}"
        if final_documents:
            user_prompt += f"\n\nCited sources available: {len(final_documents)}"
            for doc in final_documents:
                citation_num = doc.get("citation", {}).get("citation_num")
                content = doc.get("content") or doc.get("blurb", "")
                user_prompt += f"\n- [{citation_num}]: {content}"

        try:
            result = self._call_llm(
                VERIFY_ANSWER_PROMPT,
                user_prompt,
                response_format=ResponseVerification
            )
            # litellm with response_format returns parsed object or string
            if isinstance(result, str):
                data = json.loads(result)
                return ResponseVerification(**data)
            return result
        except Exception as e:
            # Return conservative failure on error
            raise Exception(f"Verification failed: {e}")

    # =========================================================================
    # Analysis Methods
    # =========================================================================

    def analyze_test_report(self, report_data: Union[str, dict]) -> str:
        """Analyze a test report and provide comprehensive insights.

        Args:
            report_data: Either formatted text or dict from AnalysisResult.to_dict()

        Returns:
            Detailed analysis with recommendations
        """
        if isinstance(report_data, dict):
            report_text = self._format_report_for_llm(report_data)
        else:
            report_text = report_data

        print(report_text)

        return self._analyze(
            REPORT_ANALYSIS_PROMPT,
            f"Test Report:\n\n{report_text}"
        )

    def analyze_test_comparison(self, comparison_data: Union[str, dict]) -> str:
        """Analyze a comparison of multiple test runs.

        Args:
            comparison_data: Either formatted text or dict from ComparisonResult.to_dict()

        Returns:
            Trend analysis with recommendations
        """
        if isinstance(comparison_data, dict):
            comparison_text = self._format_comparison_for_llm(comparison_data)
        else:
            comparison_text = comparison_data

        return self._analyze(
            COMPARISON_ANALYSIS_PROMPT,
            f"Comparison Data:\n\n{comparison_text}"
        )

    # =========================================================================
    # Formatting Helpers
    # =========================================================================

    # Step patterns for timing classification
    SLOW_STEP_PATTERNS = [
        "wait for assistant response",
        "wait for response",
        "send message",
        "send predefined message",
        "send a message",
        "send a predefined message",
        "wait for fact-check",
        "halloumi quality fact-check",
        "wait for related questions",
        "click halloumi",
        "verify halloumi",
        "after a short period",
        "verify loading states",
        "wait for",
    ]

    FAST_STEP_PATTERNS = [
        "verify",
        "click",
        "type",
        "fill",
        "press",
        "check",
        "capture",
        "clean up",
        "remove",
        "ensure"
    ]

    def _classify_step_timing(self, step_name: str, duration_ms: int) -> tuple[str, bool]:
        """Classify a step and determine if its timing is anomalous.

        Returns:
            Tuple of (expected_type: 'slow'|'fast'|'unknown', is_anomalous: bool)
        """
        name_lower = step_name.lower()

        # Check if it's an expected slow step
        for pattern in self.SLOW_STEP_PATTERNS:
            if pattern in name_lower:
                # Slow steps: anomalous if > 60s (very slow) or < 1s (suspiciously fast)
                is_anomalous = duration_ms > 60000 or duration_ms < 1000
                return ("slow", is_anomalous)

        # Check if it's an expected fast step
        for pattern in self.FAST_STEP_PATTERNS:
            if name_lower.startswith(pattern):
                # Fast steps: anomalous if > 2s
                is_anomalous = duration_ms > 2000
                return ("fast", is_anomalous)

        return ("unknown", False)

    def _format_report_for_llm(self, report_data: dict) -> str:
        """Format report data as structured text for LLM analysis.

        Args:
            report_data: Dict from AnalysisResult.to_dict()

        Returns:
            Formatted text representation
        """
        lines = []

        # Summary section
        summary = report_data.get("summary", {})
        lines.append("## Summary Statistics")
        lines.append(f"- Total tests: {summary.get('total_tests', 0)}")
        lines.append(f"- Passed: {summary.get('passed_tests', 0)}")
        lines.append(f"- Failed: {summary.get('failed_tests', 0)}")
        lines.append(f"- Skipped: {summary.get('skipped_tests', 0)}")
        lines.append(f"- Pass rate: {summary.get('pass_rate', 0):.1f}%")
        lines.append(f"- Health status: {summary.get('health_status', 'unknown')}")
        lines.append(f"- Total duration: {summary.get('total_duration_seconds', 0):.1f}s")
        lines.append("")

        # Step summary
        lines.append("## Step Summary")
        lines.append(f"- Total steps: {summary.get('total_steps', 0)}")
        lines.append(f"- Steps passed: {summary.get('passed_steps', 0)}")
        lines.append(f"- Steps failed: {summary.get('failed_steps', 0)}")
        lines.append(f"- Step pass rate: {summary.get('step_pass_rate', 0):.1f}%")
        lines.append("")

        # Performance metrics
        performance = report_data.get("performance", {})
        if performance:
            lines.append("## Performance")
            lines.append(f"- Avg test duration: {performance.get('avg_test_duration_seconds', 0):.2f}s")
            lines.append(f"- Avg step duration: {performance.get('avg_step_duration_ms', 0):.0f}ms")

            slowest = performance.get("slowest_tests", [])
            if slowest:
                lines.append("- Slowest tests:")
                for test in slowest[:3]:
                    lines.append(f"  - {test['name']}: {test['duration_seconds']:.1f}s")
            lines.append("")

        # Step Timing Analysis
        tests = report_data.get("tests", [])
        slow_steps_expected = []  # LLM steps that are slow (expected)
        slow_steps_anomalous = []  # Fast steps that are slow (unexpected)
        fast_steps_all = []
        info_steps = []  # Informational steps (no timing expected)

        for test in tests:
            for step in test.get("steps", []):
                step_name = step.get("step_name", "")
                step_type = step.get("step_type")
                dur = step.get("duration_ms")

                # Info steps don't have timing - skip timing analysis
                if step_type == "info":
                    info_steps.append((step_name, test.get("name", "")))
                    continue

                if dur is None:
                    continue

                expected_type, is_anomalous = self._classify_step_timing(step_name, dur)

                if expected_type == "slow":
                    slow_steps_expected.append((step_name, dur, test.get("name", "")))
                elif expected_type == "fast":
                    fast_steps_all.append((step_name, dur, test.get("name", "")))
                    if is_anomalous:
                        slow_steps_anomalous.append((step_name, dur, test.get("name", "")))

        lines.append("## Step Timing Analysis")
        lines.append("")

        # Info steps (no timing expected)
        if info_steps:
            lines.append("### Info Steps (no timing expected)")
            lines.append(f"- Count: {len(info_steps)}")
            lines.append("- These are informational logging messages - null/0 duration is CORRECT")
            lines.append("")

        # LLM/API steps (expected to be slow)
        if slow_steps_expected:
            avg_llm = sum(s[1] for s in slow_steps_expected) / len(slow_steps_expected)
            max_llm = max(s[1] for s in slow_steps_expected)
            min_llm = min(s[1] for s in slow_steps_expected)
            lines.append("### LLM/API Steps (expected slow)")
            lines.append(f"- Count: {len(slow_steps_expected)}")
            lines.append(f"- Avg duration: {avg_llm/1000:.1f}s")
            lines.append(f"- Range: {min_llm/1000:.1f}s - {max_llm/1000:.1f}s")
            if max_llm > 45000:
                lines.append("- WARNING: Some LLM steps exceed 45s")
            lines.append("")

        # UI steps timing
        if fast_steps_all:
            avg_ui = sum(s[1] for s in fast_steps_all) / len(fast_steps_all)
            lines.append("### UI Steps (expected fast)")
            lines.append(f"- Count: {len(fast_steps_all)}")
            lines.append(f"- Avg duration: {avg_ui:.0f}ms")

            if slow_steps_anomalous:
                lines.append(f"- ANOMALIES: {len(slow_steps_anomalous)} UI steps took >2s:")
                for step_name, dur, test_name in slow_steps_anomalous[:5]:
                    lines.append(f"  - '{step_name}' took {dur}ms in {test_name}")
            lines.append("")

        # Failures section
        failures = report_data.get("failures", {})
        failed_tests = failures.get("failed_tests", [])
        if failed_tests:
            lines.append("## Failed Tests")
            for name in failed_tests:
                lines.append(f"- {name}")
            lines.append("")

            # Failure categories
            categories = failures.get("failure_categories", {})
            if categories:
                lines.append("## Failure Categories")
                for category, tests in categories.items():
                    lines.append(f"- {category}: {len(tests)} failure(s)")
                lines.append("")

            # Most failing steps
            failing_steps = failures.get("most_failing_steps", [])
            if failing_steps:
                lines.append("## Most Failing Steps")
                for step in failing_steps[:5]:
                    lines.append(f"- {step['step_name']}: {step['failure_count']} failure(s)")
                lines.append("")

        # Detailed test results
        tests = report_data.get("tests", [])
        if tests:
            lines.append("## Detailed Test Results")
            lines.append("")

            for test in tests:
                if test["outcome"] == "passed":
                    icon = "PASS"
                elif test["outcome"] == "skipped":
                    icon = "SKIP"
                else:
                    icon = "FAIL"
                lines.append(f"### [{icon}] {test['name']}")

                if test.get("duration_seconds"):
                    lines.append(f"Duration: {test['duration_seconds']:.2f}s")

                if test.get("markers"):
                    lines.append(f"Markers: {', '.join(test['markers'])}")

                if test.get("message"):
                    msg = test["message"][:300]
                    if icon == "FAIL":
                        lines.append(f"Error: {msg}")
                    elif icon == "SKIP":
                        lines.append(f"Reason: {msg}")

                steps = test.get("steps", [])
                if steps:
                    lines.append("Steps:")
                    for step in steps:
                        s_icon = "PASS" if step["outcome"] == "passed" else "FAIL"
                        step_type = step.get("step_type", "action")
                        # Info steps don't have timing - show [info] tag instead of duration
                        if step_type == "info":
                            dur = " [info]"
                        elif step.get('duration_ms') is not None:
                            dur = f" ({step.get('duration_ms', 0)}ms)"
                        else:
                            dur = ""
                        lines.append(f"  [{s_icon}] {step['step_name']}{dur}")
                        if step.get("message") and step["outcome"] == "failed":
                            msg = step["message"][:150]
                            lines.append(f"    Error: {msg}")

                lines.append("")

        return "\n".join(lines)

    def _format_comparison_for_llm(self, comparison_data: dict) -> str:
        """Format comparison data as structured text for LLM analysis.

        Args:
            comparison_data: Dict from ComparisonResult.to_dict()

        Returns:
            Formatted text representation
        """
        lines = []

        # Summary
        summary = comparison_data.get("summary", {})
        lines.append("## Comparison Summary")
        lines.append(f"- Runs compared: {summary.get('run_count', 0)}")
        lines.append(f"- Trend direction: {summary.get('trend_direction', 'unknown')}")
        lines.append(f"- Average pass rate: {summary.get('avg_pass_rate', 0):.1f}%")
        lines.append(f"- Stability score: {summary.get('stability_score', 0)}/10")
        lines.append("")

        # Pass rate trend
        trend = comparison_data.get("pass_rate_trend", [])
        if trend:
            lines.append("## Pass Rate Trend")
            for i, rate in enumerate(trend, 1):
                lines.append(f"- Run {i}: {rate:.1f}%")
            lines.append("")

        # Individual runs
        runs = comparison_data.get("runs", [])
        if runs:
            lines.append("## Individual Runs")
            for i, run in enumerate(runs, 1):
                source = run.get("source_file", "Unknown")
                lines.append(f"### Run {i}: {source}")
                lines.append(f"- Pass rate: {run.get('pass_rate', 0):.1f}%")
                lines.append(f"- Total: {run.get('total_tests', 0)}")
                lines.append(f"- Passed: {run.get('passed_tests', 0)}")
                lines.append(f"- Failed: {run.get('failed_tests', 0)}")
                lines.append(f"- Health: {run.get('health_status', 'unknown')}")
                lines.append("")

        # Changes
        changes = comparison_data.get("changes", {})

        regressions = changes.get("regressions", [])
        if regressions:
            lines.append("## Regressions (PASS → FAIL)")
            for name in regressions:
                lines.append(f"- {name}")
            lines.append("")

        fixes = changes.get("fixes", [])
        if fixes:
            lines.append("## Fixes (FAIL → PASS)")
            for name in fixes:
                lines.append(f"- {name}")
            lines.append("")

        flaky = changes.get("flaky_tests", [])
        if flaky:
            lines.append("## Flaky Tests (Inconsistent)")
            for name in flaky:
                lines.append(f"- {name}")
            lines.append("")

        return "\n".join(lines)


# =============================================================================
# Factory Function
# =============================================================================

def create_analyzer_from_settings(settings) -> Optional[LLMAnalyzer]:
    """Create an LLMAnalyzer from settings if enabled.

    Args:
        settings: Settings instance with LLM configuration

    Returns:
        LLMAnalyzer if enabled, None otherwise

    Raises:
        ImportError: If LLM analysis is enabled but litellm unavailable
    """
    if not settings.enable_llm_analysis:
        return None

    if not LITELLM_AVAILABLE:
        raise ImportError(
            "LLM analysis is enabled but litellm is not available. "
            "Install litellm or disable LLM analysis."
        )

    return LLMAnalyzer(
        model=settings.llm_model,
        base_url=settings.llm_url,
        api_key=settings.llm_api_key
    )
