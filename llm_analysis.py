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
- Only flag performance as a concern if actual FAILURES occur due to timeouts

Test parametrization:
- `test_question_response` is the main test - dynamically parametrized from fixture JSON files
- Each fixture question becomes a separate test instance: test_question_response[Q-XXX-chromium]
- Q-XXX is the question ID from the fixture; the actual question text appears in the "Send question" step
- Each test instance runs 7 validation phases per question:
  1. Send question and verify response received
  2. LLM-based quality verification (if enabled)
  3. Source citation validation and related questions validation
  4. Response content validation (length, keywords)
  5. Halloumi quality fact-check (if configured)
  6. Feedback functionality (if enabled)
  7. Chat cleanup

LLM quality verdicts:
- Steps with `step_type="llm_verdict"` are external LLM quality assessments of chatbot responses
- These evaluate three dimensions: relevance (on-topic vs off-topic), specificity (not vague vs too vague), citations (has vs missing)
- They do NOT have timing - null/0 duration is expected and correct
- A test may be skipped with "LLM analysis: answer lacks information" - this means the LLM evaluator determined the chatbot response lacked sufficient information to evaluate

LLMException errors:
- `LLMException` means the chatbot's backend LLM failed to generate a response
- Typically: "LLMException: Final answer is empty. Inference provider likely failed to provide content packets."
- This is a backend/infrastructure issue, not a test framework issue"""


REPORT_ANALYSIS_PROMPT = f"""{CHATBOT_CONTEXT}

You are a senior QA analyst writing a test report analysis for a PDF document. Your analysis
will be read by stakeholders, product owners, and developers. Provide professional, insightful
analysis that adds real value beyond what the raw numbers show.

## What Makes Good Analysis

Good analysis identifies PATTERNS and MEANING — not just what happened, but WHY it matters.

- Instead of "12 Halloumi timeouts occurred" (the reader can count), explain what this pattern
  reveals about the Halloumi service reliability and its impact on the user experience.
- Instead of listing every passing test, highlight what the passing tests collectively demonstrate
  about the chatbot's strengths.
- Instead of "add retry logic", recommend specific thresholds, architectural changes, or
  investigation steps tied to the evidence.

## Important Context

- Skipped tests are EXCLUDED from the pass rate denominator.
  Pass rate = passed / (total - skipped). Always mention skips separately.
- Steps with `step_type` "info" or "llm_verdict" intentionally have no timing — do not flag this.
- Test durations of 20-60s are normal (real LLM calls with streaming). Only flag timing if
  actual failures occurred due to timeouts.

## Report Structure

### Executive Summary

A concise overview (3-4 sentences) suitable for a report introduction. State the overall
health status with pass rate and skip count. Identify the most significant finding and its
business impact.

### Chatbot Strengths

What does this test run demonstrate is working well? Focus on capabilities, not individual
test names. Examples of good insights:
- "The chatbot consistently provides well-cited, on-topic responses across all tested domains"
- "Error recovery works reliably — the UI handles network failures and API errors gracefully"

Keep this to 3-5 bullet points of meaningful observations.

### Issues and Root Causes

This is the most important section. Group related failures by their underlying cause,
not by individual test. For each issue group:

1. Describe the issue pattern and scope (how many tests affected, which ones)
2. Analyze the root cause based on the error messages and step details
3. Assess the user impact — does this affect the core Q&A experience or auxiliary features?
4. Note any correlations (e.g., failures clustered by topic, question length, or marker)

Do NOT list the same root cause multiple times for different test IDs.

### Response Quality Assessment

Analyze the LLM quality verdicts and Halloumi fact-check scores to assess the chatbot's
response quality as a whole:

- What do the verdict pass rates reveal about the chatbot's knowledge coverage?
- Are there topic areas where the chatbot struggles (look at markers, question content)?
- What do skipped tests (insufficient information) tell us about knowledge base gaps?
- How do Halloumi fact-check scores distribute? Are low scores concentrated in specific areas?

This section should read as a quality report on the chatbot's responses, not on the test framework.

### Risk Assessment

Rate: Critical, High, Medium, or Low

Provide a clear rationale in 2-3 sentences. Focus on:
- Whether core functionality (asking questions and getting useful answers) is at risk
- Whether the issues found would be visible to end users
- Whether the issues are intermittent or systematic

### Recommended Actions

A numbered list of 3-5 concrete actions, ordered by priority. Each action must:
- Start with a priority tag: [Critical], [High], [Medium], or [Low]
- Be specific enough that someone can start working on it
- Reference the evidence from this report (test IDs, error patterns)
- Explain what success looks like

Bad: "Improve Halloumi reliability"
Good: "Increase the Halloumi fact-check timeout from 90s to 150s for tests with long responses
(Q-007, Q-009, Q-018 all had response lengths >6000 chars), or implement async polling
instead of synchronous waiting"

### Production Readiness

A brief assessment (1 paragraph) of the chatbot's readiness for production use based on
this test run. Address:
- Is the core Q&A experience reliable enough for users?
- What is the biggest risk to user satisfaction?
- What should be resolved before a wider rollout?

## Formatting

- Use markdown headings (##, ###) — the output will be rendered in a PDF
- Keep paragraphs short (2-3 sentences)
- Use bullet points for lists, numbered lists for ordered/prioritized items
- Reference test IDs (e.g., Q-029) and step names when citing evidence
- Write in a professional, analytical tone suitable for a formal report"""


COMPARISON_ANALYSIS_PROMPT = f"""{CHATBOT_CONTEXT}

You are a senior QA analyst writing a multi-run comparison analysis for a PDF report.
The reader already has the raw comparison data (pass rates, regressions, fixes, flaky tests).
Your job is to interpret the data and explain what the trends MEAN for the chatbot's health.

## Important Context

- Skipped tests are EXCLUDED from the pass rate denominator.
  Pass rate = passed / (total - skipped). Always mention skips separately.
- Test durations of 20-60s are normal (real LLM calls with streaming). Only flag duration
  changes if they correlate with new failures.

## Report Structure

### Executive Summary

Present a markdown table comparing the first and last runs:

| Metric | First Run | Last Run | Delta |
|--------|-----------|----------|-------|

Include: Total tests, Passed, Failed, Skipped, Pass rate (delta in "pp"), Total duration,
Avg test duration. Use "+" prefix for increases, "-" for decreases.

After the table, write 2-3 sentences interpreting the direction. Focus on what changed
and whether it matters — not just restating the deltas.

### Trend Interpretation

Go beyond "Improving/Declining/Stable". Analyze:
- Is the trend consistent across runs or volatile?
- Are pass rate changes driven by real fixes/regressions or by flaky tests oscillating?
- Is the chatbot getting better, or are the same problems persisting?

### Regressions and Fixes

For regressions (tests that started failing):
- Group by likely root cause if multiple regressions share one
- Assess severity: does this affect the core Q&A experience or auxiliary features?
- Suggest specific investigation steps

For fixes (tests that started passing):
- What capability was restored?
- Is the fix stable (consistent across recent runs) or potentially fragile?

Skip this section if there are no regressions or fixes.

### Stability Analysis

Analyze flaky tests (inconsistent results across runs):
- Are flaky tests caused by external dependencies (Halloumi, LLM backend) or test issues?
- Which flaky tests pose the highest risk to production confidence?
- Are there patterns (e.g., all flaky tests involve the same step or service)?

Interpret the stability score in context — what does it mean for this specific chatbot,
not as a generic scale explanation.

### Warning Trends

If per-run test details are available, analyze tests that gained or lost warnings between runs:
- Which tests went from clean pass to passed-with-warnings (or vice versa)?
- Are new warnings concentrated in specific markers or question types?
- Do warnings correlate with specific failed steps (e.g., Halloumi timeouts, citation checks)?

Skip this section if there are no warning changes between runs.

### Per-Test Outcome Analysis

The cross-run test outcome matrix shows the outcome progression for every test that changed.
Use it together with the Regressions, Fixes, and Flaky sections to identify:
- Outcome trajectories (e.g., passed -> skipped vs failed -> passed)
- Tests that were added or removed between runs (appearing as "missing")
- Clusters of tests that changed outcome together (may share a root cause)

Do NOT duplicate the flaky tests list — instead, add interpretation of the patterns.
Skip this section if the outcome matrix is absent.

### Response Quality Trends

If LLM verdict trend data is available:
- How is the chatbot's response quality evolving across runs?
- Are quality improvements/declines correlated with pass rate changes?
- Which quality dimensions are consistently strong or weak?
- Do knowledge base gaps (information sufficiency failures) persist across runs?
- Use per-run LLM verdict details to identify specific tests that consistently fail
  particular quality dimensions across runs

This section should assess the chatbot's answer quality trajectory, not the test framework.
Skip this section if no verdict trend data is available.

### Recommended Actions

A numbered list of 3-5 concrete actions, ordered by priority. Each action must:
- Start with a priority tag: [Critical], [High], [Medium], or [Low]
- Be tied to specific evidence from the comparison data
- Be actionable — someone can start working on it immediately

### Overall Assessment

One paragraph: Based on the trend across runs, is the chatbot's reliability improving,
stable, or degrading? What is the single most important thing to address before the
next test run?

## Formatting

- Use markdown headings (##, ###) — the output will be rendered in a PDF
- Keep paragraphs short (2-3 sentences)
- Use bullet points for lists, numbered lists for ordered/prioritized items
- Reference specific test IDs when citing evidence
- Write in a professional, analytical tone suitable for a formal report"""


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

        # print(report_text)

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

        print(comparison_text)

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
        total = summary.get('total_tests', 0)
        passed = summary.get('passed_tests', 0)
        passed_with_warnings = summary.get('passed_with_warnings', 0)
        failed = summary.get('failed_tests', 0)
        skipped = summary.get('skipped_tests', 0)
        evaluated = total - skipped

        lines.append("## Summary Statistics")
        lines.append(f"- Total tests: {total}")
        lines.append(f"- Passed: {passed}")
        lines.append(f"- Failed: {failed}")
        lines.append(f"- Skipped: {skipped}")
        lines.append(f"- Evaluated (non-skipped): {evaluated}")
        lines.append(f"- Pass rate: {summary.get('pass_rate', 0):.1f}% ({passed} passed / {evaluated} evaluated)")
        lines.append(f"- Passed with warnings: {passed_with_warnings}")
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

                # Info and llm_verdict steps don't have timing - skip timing analysis
                if step_type in ("info", "llm_verdict"):
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
            failing_steps = failures.get("failing_steps", [])
            if failing_steps:
                lines.append("## Failed Steps")
                for step in failing_steps:
                    lines.append(f"- {step['step_name']}: {step['failure_count']} failure(s)")
                lines.append("")

        # LLM Quality Verdict Summary
        llm_verdicts = report_data.get("llm_verdicts", {})
        verdict_summary = llm_verdicts.get("summary", {})
        if verdict_summary:
            lines.append("## LLM Quality Verdict Summary")

            dimension_labels = {
                "relevance": "Relevance (on-topic)",
                "specificity": "Specificity (not vague)",
                "citations": "Citations",
                "information": "Information sufficiency",
            }
            total_passed = 0
            total_checked = 0

            for dimension in ["relevance", "specificity", "citations", "information"]:
                counts = verdict_summary.get(dimension)
                if not counts:
                    continue
                passed = counts.get("passed", 0)
                failed = counts.get("failed", 0)
                total = passed + failed
                rate = (passed / total * 100) if total > 0 else 0
                total_passed += passed
                total_checked += total

                label = dimension_labels.get(dimension, dimension.capitalize())
                lines.append(f"- {label}: {passed}/{total} passed ({rate:.0f}%)")

            if total_checked > 0:
                overall_rate = (total_passed / total_checked * 100)
                lines.append("")
                lines.append(f"Overall quality rate: {overall_rate:.1f}%")

            verdict_tests = llm_verdicts.get("per_test", {})
            failed_tests = {
                name: verdicts
                for name, verdicts in verdict_tests.items()
                if any(not v for v in verdicts.values())
            }
            if failed_tests:
                lines.append("")
                lines.append("Tests with failed verdicts:")
                for test_name, verdicts in sorted(failed_tests.items()):
                    failed_dims = [dim for dim, passed in verdicts.items() if not passed]
                    lines.append(f"- {_clean_test_name(test_name)}: failed on {', '.join(failed_dims)}")

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
                    msg = test["message"]
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
                        # Info and llm_verdict steps don't have timing
                        if step_type == "info":
                            dur = " [info]"
                        elif step_type == "llm_verdict":
                            dur = " [llm_verdict]"
                        elif step.get('duration_ms') is not None:
                            dur = f" ({step.get('duration_ms', 0)}ms)"
                        else:
                            dur = ""
                        lines.append(f"  [{s_icon}] {step['step_name']}{dur}")
                        if step.get("message") and step["outcome"] == "failed":
                            msg = step["message"]
                            lines.append(f"    Error: {msg}")

                lines.append("")

        return "\n".join(lines)

    def _format_test_change(
        self,
        name: str,
        test_context: dict,
        lines: list,
        test_outcomes: Optional[dict] = None,
    ) -> None:
        """Format a regression/fix/flaky test with context details."""
        ctx = test_context.get(name)
        if not ctx:
            lines.append(f"- {name}")
            return

        markers = ctx.get("markers", [])
        marker_str = f" [{', '.join(markers)}]" if markers else ""
        outcome = ctx.get("outcome", "")
        progression = test_outcomes.get(name) if test_outcomes else None
        prog_str = f" ({' -> '.join(progression)})" if progression else f" (last: {outcome})"
        lines.append(f"- {name}{marker_str}{prog_str}")

        if ctx.get("error"):
            lines.append(f"  Error: {ctx['error']}")
        if ctx.get("previous_error"):
            lines.append(
                f"  Previous error: {ctx['previous_error']}"
            )
        for fs in ctx.get("failed_steps", []):
            step_line = f"  Failed step: {fs['step']}"
            if fs.get("error"):
                step_line += f" — {fs['error']}"
            lines.append(step_line)

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

        # Executive summary data (first vs last run)
        exec_summary = comparison_data.get("executive_summary", {})
        if exec_summary:
            lines.append("## Executive Summary Data (First vs Last Run)")
            lines.append(f"- First run: {exec_summary.get('first_run', 'N/A')}")
            lines.append(f"- Last run: {exec_summary.get('last_run', 'N/A')}")
            for metric in ["total_tests", "passed", "passed_with_warnings", "failed", "skipped"]:
                data = exec_summary.get(metric, {})
                lines.append(f"- {metric}: first={data.get('first', 'N/A')}, last={data.get('last', 'N/A')}, delta={data.get('delta', 0)}")
            pr = exec_summary.get("pass_rate", {})
            lines.append(f"- pass_rate: first={pr.get('first', 'N/A')}%, last={pr.get('last', 'N/A')}%, delta={pr.get('delta_pp', 0)} pp")
            dur = exec_summary.get("total_duration_seconds", {})
            lines.append(f"- total_duration_seconds: first={dur.get('first', 'N/A')}s, last={dur.get('last', 'N/A')}s, delta={dur.get('delta', 0)}s")
            avg = exec_summary.get("avg_test_duration", {})
            lines.append(f"- avg_test_duration: first={avg.get('first', 'N/A')}s, last={avg.get('last', 'N/A')}s, delta={avg.get('delta', 0)}s")
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
                total = run.get('total_tests', 0)
                passed = run.get('passed_tests', 0)
                passed_with_warnings = run.get('passed_with_warnings', 0)
                failed = run.get('failed_tests', 0)
                skipped = run.get('skipped_tests', 0)
                evaluated = total - skipped
                lines.append(f"### Run {i}: {source}")
                lines.append(f"- Total tests: {total}")
                lines.append(f"- Passed: {passed}")
                lines.append(f"- Failed: {failed}")
                lines.append(f"- Skipped: {skipped}")
                lines.append(f"- Evaluated (non-skipped): {evaluated}")
                lines.append(f"- Pass rate: {run.get('pass_rate', 0):.1f}% ({passed} passed / {evaluated} evaluated)")
                lines.append(f"- Passed with warnings: {passed_with_warnings}")
                lines.append(f"- Health status: {run.get('health_status', 'unknown')}")
                lines.append(f"- Total duration: {run.get('total_duration_seconds', 0):.1f}s")
                lines.append("")

        # Per-run test details (non-trivial tests only)
        if runs:
            lines.append("## Per-Run Test Details")
            lines.append("(Only showing failed, skipped, or passed-with-warnings tests)")
            lines.append("")
            for i, run in enumerate(runs, 1):
                run_tests = run.get("tests", [])
                interesting = [
                    t for t in run_tests
                    if t.get("outcome") in ("failed", "skipped")
                    or t.get("warn")
                ]
                if not interesting:
                    continue
                source = run.get("source_file", f"Run {i}")
                lines.append(f"### Run {i}: {source}")
                for t in interesting:
                    outcome = t.get("outcome", "unknown")
                    warn_tag = " [WARN]" if t.get("warn") else ""
                    markers = t.get("markers", [])
                    marker_str = f" [{', '.join(markers)}]" if markers else ""
                    lines.append(f"- {t['name']}: {outcome}{warn_tag}{marker_str}")
                    for fs in t.get("failed_steps", []):
                        lines.append(f"  Failed step: {fs}")
                    verdicts = t.get("llm_verdicts", {})
                    if verdicts:
                        failed_dims = [d for d, v in verdicts.items() if not v]
                        passed_dims = [d for d, v in verdicts.items() if v]
                        if failed_dims:
                            lines.append(f"  LLM verdicts failed: {', '.join(failed_dims)}")
                        if passed_dims:
                            lines.append(f"  LLM verdicts passed: {', '.join(passed_dims)}")
                lines.append("")

        # Changes with failure context (needed early for cross-run matrix filtering)
        changes = comparison_data.get("changes", {})
        test_context = comparison_data.get("test_context", {})

        # Cross-run test outcome matrix (all tests whose outcome changed)
        test_outcomes = comparison_data.get("test_outcomes", {})
        if test_outcomes:
            changed = {
                name: outcomes
                for name, outcomes in test_outcomes.items()
                if len(set(outcomes)) > 1
            }
            if changed:
                lines.append("## Cross-Run Test Outcome Matrix")
                lines.append("(Outcome progression for every test that changed across runs)")
                lines.append("")
                for name, outcomes in sorted(changed.items()):
                    lines.append(f"- {name}: {' -> '.join(outcomes)}")
                lines.append("")

        # Per-run LLM verdict summary
        if runs:
            has_verdicts = any(
                t.get("llm_verdicts")
                for run in runs
                for t in run.get("tests", [])
            )
            if has_verdicts:
                lines.append("## Per-Run LLM Verdict Details")
                lines.append("")
                for i, run in enumerate(runs, 1):
                    run_tests = run.get("tests", [])
                    # Aggregate per-dimension stats for this run
                    dim_stats: dict = {}
                    dim_failures: dict = {}
                    for t in run_tests:
                        verdicts = t.get("llm_verdicts", {})
                        for dim, passed in verdicts.items():
                            if dim not in dim_stats:
                                dim_stats[dim] = {"passed": 0, "failed": 0}
                                dim_failures[dim] = []
                            if passed:
                                dim_stats[dim]["passed"] += 1
                            else:
                                dim_stats[dim]["failed"] += 1
                                dim_failures[dim].append(t["name"])
                    if not dim_stats:
                        continue
                    source = run.get("source_file", f"Run {i}")
                    lines.append(f"### Run {i}: {source}")
                    for dim in ["relevance", "specificity", "citations", "information"]:
                        counts = dim_stats.get(dim)
                        if not counts:
                            continue
                        total = counts["passed"] + counts["failed"]
                        rate = (counts["passed"] / total * 100) if total > 0 else 0
                        lines.append(f"- {dim}: {counts['passed']}/{total} ({rate:.0f}%)")
                        if dim_failures.get(dim):
                            for tname in dim_failures[dim]:
                                lines.append(f"  Failed: {tname}")
                    lines.append("")

        regressions = changes.get("regressions", [])
        if regressions:
            lines.append("## Regressions (PASS → FAIL)")
            for name in regressions:
                self._format_test_change(
                    name, test_context, lines, test_outcomes
                )
            lines.append("")

        fixes = changes.get("fixes", [])
        if fixes:
            lines.append("## Fixes (FAIL → PASS)")
            for name in fixes:
                self._format_test_change(
                    name, test_context, lines, test_outcomes
                )
            lines.append("")

        flaky = changes.get("flaky_tests", [])
        if flaky:
            lines.append("## Flaky Tests (Inconsistent)")
            for name in flaky:
                self._format_test_change(
                    name, test_context, lines, test_outcomes
                )
            lines.append("")

        # LLM Quality Verdict Trends
        verdict_trends = comparison_data.get("llm_verdict_trends", {})
        if verdict_trends:
            dimension_labels = {
                "relevance": "Relevance (on-topic)",
                "specificity": "Specificity (not vague)",
                "citations": "Citations",
                "information": "Information sufficiency",
            }

            lines.append("## LLM Quality Verdict Trends")
            for dimension in ["relevance", "specificity", "citations", "information"]:
                trend = verdict_trends.get(dimension)
                if not trend:
                    continue

                label = dimension_labels.get(dimension, dimension.capitalize())
                rates = []
                for counts in trend:
                    total = counts.get("passed", 0) + counts.get("failed", 0)
                    rate = (counts.get("passed", 0) / total * 100) if total > 0 else 0
                    rates.append(f"{rate:.0f}%")

                lines.append(f"- {label}: {' -> '.join(rates)}")
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
