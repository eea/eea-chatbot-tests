# Writing End-to-End Tests for volto-eea-chatbot

This guide covers how to write Playwright-based end-to-end tests for the EEA Chatbot frontend using the chatbot_tests framework.

## Table of Contents

1. [Understanding the Chatbot Frontend](#understanding-the-chatbot-frontend)
2. [Test Framework Overview](#test-framework-overview)
3. [Writing Basic UI Tests](#writing-basic-ui-tests)
4. [Testing LLM Responses](#testing-llm-responses)
5. [Working with Fixtures](#working-with-fixtures)
6. [Using Markers](#using-markers)
7. [Step-Based Logging](#step-based-logging)
8. [Page Object Reference](#page-object-reference)
9. [LLM-Based Verification](#llm-based-verification)
10. [Running and Analyzing Tests](#running-and-analyzing-tests)

---

## Understanding the Chatbot Frontend

The volto-eea-chatbot is a React-based chat interface built as a Volto addon. Key UI elements:

### DOM Structure

```
.chat-window                        # Main container (with data-playwright-block-id)
├── .messages                       # Message display area
│   ├── .conversation              # Scrollable message list
│   │   ├── .comment:has(.circle.user)       # User messages
│   │   └── .comment:has(.circle.assistant)  # AI messages
│   │       ├── .comment-tabs              # Tab navigation
│   │       │   └── .ui.menu               # Answer / Sources tabs
│   │       ├── .answer-tab .answer-content # Answer text
│   │       ├── .answer-tab .sources        # Inline sources
│   │       ├── .multi-tool-renderer        # Processing steps (deep research)
│   │       ├── .halloumi-feedback-button   # Fact-check trigger
│   │       ├── .claim-message              # Halloumi verdict
│   │       └── .message-actions            # Like/Dislike buttons
│   ├── .empty-state               # Landing page (shown when no messages)
│   │   ├── h2                     # Assistant name
│   │   ├── p                      # Assistant description
│   │   └── .starter-messages-container
│   │       └── .starter-message   # Clickable starter prompts
│   └── [aria-label='Clear chat']  # Clear chat button
├── .chat-form                      # Input area
│   ├── .textarea-wrapper
│   │   └── textarea               # Message input
│   └── .chat-right-actions
│       └── [aria-label='Send']    # Send button
└── .chat-controls
    ├── .quality-check-toggle      # Fact-check toggle
    └── .deep-research-toggle      # Deep research toggle
```

### Key Selectors

| Element | Selector |
|---------|----------|
| Chat window | `.chat-window[data-playwright-block-id]` |
| Message input | `form .textarea-wrapper textarea` |
| Send button | `form .chat-right-actions [aria-label='Send']` |
| User messages | `.comment:has(.circle.user)` |
| Assistant messages | `.comment:has(.circle.assistant:not(.placeholder))` |
| Clear chat | `[aria-label='Clear chat']` |
| Sources container | `.answer-tab .sources` |
| Source items | `.source` |
| Show all sources | `.show-all-sources-btn` |
| Related questions | `.chat-related-questions` |
| Related question buttons | `.relatedQuestionButton` |
| Like button | `.message-actions [aria-label='Like']` |
| Dislike button | `.message-actions [aria-label='Dislike']` |
| Feedback modal | `.ui.modal, .feedback-modal` |
| Feedback toast | `.message-actions .feedback-toast` |
| Fact-check button | `.halloumi-feedback-button .claims-btn` |
| Halloumi message | `.claim-message .content` |
| Claims | `span.claim` |
| Answer loader | `.loader-container .loader` |
| Fact-check toggle | `.quality-check-toggle` |
| Deep research toggle | `.deep-research-toggle` |
| Copy button | `[aria-label='Copy']` |

### Response Streaming

The chatbot uses server-sent events (SSE) for streaming responses. Each response consists of JSONL chunks:

```json
{"user_message_id": 123, "reserved_assistant_message_id": 456}
{"ind": 0, "obj": {"type": "message_delta", "message": "Air quality..."}}
{"ind": 0, "obj": {"type": "message_start", "documents": [...]}}
{"ind": 0, "obj": {"type": "citation", "document_id": "...", ...}}
```

---

## Test Framework Overview

### Architecture

```
CLI (main.py)
    │
    ▼
run_tests_sync() in runner.py
    │
    ├── JSONLWriter opens output file
    │
    ▼
pytest.main() with ResultCollectorPlugin
    │
    ├── Pytest hooks → TestEvent objects
    │         │
    │         ├── on_event callback:
    │         │     ├── writer.write_event() → JSONL file
    │         │     └── _print_event() → console output
    │         │
    │         └── step() context manager → log_step() → plugin._emit()
    │
    ▼
TestRunSummary returned
```

### Directory Structure

All source files live at the project root (the `chatbot_tests.` import prefix is configured in `pyproject.toml`):

```
eea-chatbot-tests/
├── tests/
│   ├── conftest.py          # Fixtures and hooks
│   ├── test_basic.py        # UI/integration tests (@pytest.mark.always)
│   └── test_questions.py    # Data-driven question validation
├── fixtures/
│   └── golden_dataset.json  # Test data (v2.0 format)
├── page_objects/
│   ├── chatbot_page.py      # ChatbotPage + ChatbotPageSelectors
│   └── response.py          # StreamedResponse model
└── reports/                  # Generated JSONL, JSON, and PDF files
```

---

## Writing Basic UI Tests

### Test File Structure

Create test files in the `tests/` directory:

```python
# tests/test_ui.py
import pytest
from playwright.sync_api import expect
from chatbot_tests.step import step, info
from chatbot_tests.page_objects import ChatbotPage


@pytest.mark.basic
class TestChatbotUI:
    """UI tests that verify the chatbot interface renders correctly."""

    def test_chatbot_loads(self, chatbot_page: ChatbotPage):
        """Verify the chatbot UI loads with all essential elements."""

        with step("Verify chat window is visible"):
            expect(chatbot_page.chat_window).to_be_visible()

        with step("Verify message input is visible"):
            expect(chatbot_page.textarea).to_be_visible()

        with step("Verify send button is visible"):
            expect(chatbot_page.submit_button).to_be_visible()
```

### Testing User Interactions

```python
@pytest.mark.basic
class TestUserInteractions:

    def test_starter_prompts_clickable(self, chatbot_page: ChatbotPage):
        """Verify clicking a starter prompt sends the message."""

        if not chatbot_page.show_starter_messages:
            info("INFO: Starter messages not configured")
            return

        with step("Verify empty state is visible"):
            expect(chatbot_page.empty_state).to_be_visible()

        with step("Get first starter prompt"):
            starter = chatbot_page.starter_messages.first
            prompt_text = starter.text_content()

        with step("Click starter prompt"):
            starter.click()

        with step("Verify user message appears"):
            expect(chatbot_page.user_messages.first).to_be_visible()

    def test_send_button_interaction(self, chatbot_page: ChatbotPage):
        """Verify send button sends message."""

        with step("Type a message"):
            chatbot_page.textarea.fill("What is climate change?")

        with step("Click send button"):
            chatbot_page.submit_button.click()

        with step("Verify message was sent"):
            expect(chatbot_page.user_messages.first).to_be_visible()

    def test_clear_chat(self, chatbot_page: ChatbotPage):
        """Verify clear chat button resets the conversation."""

        with step("Send a message first"):
            with chatbot_page.send_message("Hello") as response:
                pass
            chatbot_page.verify_answer(response.value)

        with step("Click clear chat button"):
            chatbot_page.clear_chat_button.click()

        with step("Verify conversation is cleared"):
            chatbot_page.verify_empty_conversation()
```

### Testing UI Components

```python
@pytest.mark.feedback
class TestFeedbackUI:

    def test_feedback_buttons_visible(self, chatbot_page: ChatbotPage):
        """Verify feedback buttons appear after response."""
        feedback_enabled = chatbot_page.block_config.get("enableFeedback")
        if not feedback_enabled:
            info("INFO: Feedback not enabled in block config")
            return

        with step("Send message and wait for response"):
            with chatbot_page.send_message("What is biodiversity?") as response:
                pass
            chatbot_page.verify_answer(response.value)

        with step("Verify like button is visible"):
            expect(chatbot_page.like_button).to_be_visible()

        with step("Verify dislike button is visible"):
            expect(chatbot_page.dislike_button).to_be_visible()
```

---

## Testing LLM Responses

### Basic Response Verification

```python
@pytest.mark.basic
class TestLLMResponses:

    def test_receives_response(self, chatbot_page: ChatbotPage):
        """Verify chatbot returns a response to a question."""

        with step("Send question"):
            with chatbot_page.send_message(
                "What is the current state of air quality in Europe?"
            ) as response:
                pass
            chatbot_page.verify_answer(response.value)

        with step("Verify response has content"):
            message = response.value.get_message()
            assert len(message) > 0

    def test_response_has_sources(self, chatbot_page: ChatbotPage):
        """Verify response includes source documents."""

        with step("Send question about a specific topic"):
            with chatbot_page.send_message(
                "What policies exist for reducing plastic pollution?"
            ) as response:
                pass
            chatbot_page.verify_answer(response.value)

        with step("Verify sources are present"):
            documents = response.value.get_final_documents()
            assert len(documents) > 0, "Expected at least one source document"

        with step("Verify citations are present"):
            citations = response.value.get_citations()
            assert len(citations) > 0, "Expected at least one citation"
```

### Using the StreamedResponse Model

The `StreamedResponse` class provides methods for extracting data from streamed JSONL responses:

```python
def test_response_analysis(self, chatbot_page: ChatbotPage):
    with chatbot_page.send_message("What is climate change?") as response:
        pass
    chatbot_page.verify_answer(response.value)
    r = response.value

    # Get the full message text
    message = r.get_message()

    # Get reasoning (if deep research enabled)
    reasoning = r.get_reasoning()

    # Get search tool results
    search_tools = r.get_search_tools()

    # Get final documents/sources
    documents = r.get_final_documents()

    # Get inline citations
    citations = r.get_citations()

    # Get related questions
    related = r.get_related_questions()

    # Get specific chunk types
    deltas = r.get_by_type("message_delta")
```

---

## Working with Fixtures

### Fixture File Format (v2.0)

Fixtures use a minimal format with `default_validation` at the file level. Each test case inherits defaults and can override specific fields via deep merge.

```json
{
  "version": "2.0.0",
  "validation_thresholds": {
    "min_response_length": 100,
    "min_sources": 1,
    "min_quality_score": 60,
    "min_related_questions": 2
  },
  "default_validation": {
    "response": { "min_length": 100 },
    "sources": { "min_count": 1 },
    "quality_check": { "min_score": 60 },
    "related_questions": { "min_count": 2 },
    "llm": {
      "verify_lack_information": true,
      "verify_answers_question": true,
      "verify_not_vague": true,
      "verify_citations": true
    }
  },
  "default_feedback": true,
  "test_cases": [
    {
      "id": "Q-001",
      "priority": "high",
      "question": "What is the current state of air quality in Europe?",
      "markers": ["comprehensive", "air_quality"],
      "validation": {
        "response": {
          "expected_keywords": ["air", "pollution", "Europe"]
        }
      }
    },
    {
      "id": "Q-002",
      "priority": "medium",
      "question": "How does air pollution affect human health?",
      "markers": ["health"]
    }
  ]
}
```

### Validation Sections

| Section | Options | Description |
|---------|---------|-------------|
| `response` | `min_length`, `expected_keywords` | Content validation |
| `sources` | `min_count` | Citation requirements |
| `quality_check` | `min_score` | Halloumi fact-check threshold (0-100) |
| `related_questions` | `min_count` | Minimum generated follow-up questions |
| `llm` | `verify_lack_information`, `verify_answers_question`, `verify_not_vague`, `verify_citations` | LLM-based quality checks |

### Using the `data` Fixture

The `data` fixture automatically parametrizes tests with each test case from `fixtures/*.json`. Limit with `--limit N`.

```python
@pytest.mark.question
class TestQuestionValidation:

    def test_question_response(self, chatbot_page: ChatbotPage, data: dict, settings: Settings):
        question = data["question"]
        test_id = data["id"]
        validation = data.get("validation", {})

        with step(f"Send question [{test_id}]: '{question}'"):
            with chatbot_page.send_message(question) as response:
                pass
            chatbot_page.verify_answer(response.value)

        # Validate response content
        response_config = validation.get("response", {})
        min_length = response_config.get("min_length", 50)
        message = response.value.get_message()

        with step(f"Verify response length ({len(message)} >= {min_length})"):
            assert len(message) >= min_length
```

### Adding New Questions

1. Add test case to `fixtures/golden_dataset.json` or create a new topic fixture file
2. Include appropriate `markers` for filtering
3. Set `priority` (high/medium/low) for selective runs
4. Configure `validation` sections (inherits from `default_validation`)
5. Run tests: `chatbot_tests run`

---

## Using Markers

### Defined Markers (from `pytest.ini`)

```python
@pytest.mark.always        # Tests that always run regardless of marker filter
@pytest.mark.basic         # Basic chatbot functionality
@pytest.mark.halloumi      # Halloumi fact-check tests
@pytest.mark.feedback      # Feedback functionality
@pytest.mark.follow_up     # Follow-up query tests
```

### Priority Markers (auto-added from fixture `priority` field)

```python
@pytest.mark.high          # High priority
@pytest.mark.medium        # Medium priority
@pytest.mark.low           # Low priority
```

### Topic Markers (from fixture `markers` arrays)

Custom markers are dynamically applied from test case data:

```json
{
  "id": "Q-001",
  "markers": ["satellite", "security", "copernicus", "ai"]
}
```

### Running with Markers

```bash
# Run only basic tests
chatbot_tests run -m basic

# Run high priority tests
chatbot_tests run -m high

# Run multiple markers (OR logic)
chatbot_tests run -m "basic,sources"

# Run with marker expression (AND logic)
chatbot_tests run -m "basic and high"

# Exclude markers
chatbot_tests run -m "not slow"

# Limit fixture-driven tests
chatbot_tests run --limit 5
```

### The `always` Marker

Tests marked with `@pytest.mark.always` run regardless of the `-m` filter. This is useful for core UI verification tests that should always execute:

```python
@pytest.mark.always
@pytest.mark.basic
class TestCoreUI:
    def test_chatbot_loads(self, chatbot_page):
        # Runs even with -m "questions"
        pass
```

---

## Step-Based Logging

### Using the `step()` Context Manager

Every logical test action should be wrapped in a `step()`:

```python
from chatbot_tests.step import step, info, llm_verdict

def test_example(self, chatbot_page):
    with step("Verify page loaded"):
        expect(chatbot_page.chat_window).to_be_visible()

    with step("Send message"):
        with chatbot_page.send_message("Hello") as response:
            pass

    with step("Verify response"):
        assert response.value is not None
```

### Informational Messages

Use `info()` for non-timed logging (configuration state, feature detection):

```python
if not chatbot_page.block_config.get("enableFeedback"):
    info("INFO: Feedback not enabled in block config")
    return
```

### LLM Verdicts

Use `llm_verdict()` to log LLM quality assessments:

```python
llm_verdict("LLM analysis: answer on-topic", explanation)
llm_verdict("LLM analysis: answer too vague", explanation, "failed")
```

### Continue on Failure

For non-critical checks that shouldn't fail the whole test:

```python
# Critical check - test fails if this fails
with step("Verify response exists"):
    assert response is not None

# Optional check - logged but doesn't fail the test
with step("Check for related questions", continue_on_failure=True):
    rq = chatbot_page.related_question_buttons
    assert rq.count() > 0, "No related questions found"
```

### Step Output

Each step produces a JSONL event:

```json
{
  "event_type": "step",
  "nodeid": "tests/test_basic.py::TestChatbot::test_example[chromium]",
  "name": "test_example[chromium]",
  "outcome": "passed",
  "duration_ms": 1234,
  "step_name": "Send message",
  "step_type": "action"
}
```

---

## Page Object Reference

### ChatbotPage Properties (Locators)

```python
# Containers
chatbot_page.chat_window              # Main container (scoped by block ID)
chatbot_page.messages                 # Message area
chatbot_page.conversation             # Scrollable message list
chatbot_page.empty_state              # Landing page
chatbot_page.starter_container        # Starter messages container
chatbot_page.chat_form                # Input form area
chatbot_page.chat_controls            # Toggle controls area

# Input
chatbot_page.textarea                 # Message input
chatbot_page.submit_button            # Send button
chatbot_page.clear_chat_button        # Clear chat button

# Messages
chatbot_page.user_messages            # All user messages
chatbot_page.assistant_messages       # All assistant messages

# Sources
chatbot_page.sources_container        # Sources section
chatbot_page.source_items             # Individual source items
chatbot_page.show_all_sources_button  # "See all sources" button

# Related Questions
chatbot_page.related_questions_container
chatbot_page.related_question_buttons
chatbot_page.related_questions_loader

# Feedback
chatbot_page.like_button
chatbot_page.dislike_button
chatbot_page.feedback_modal
chatbot_page.feedback_textarea
chatbot_page.feedback_toast

# Halloumi (Fact-Check)
chatbot_page.fact_check_button
chatbot_page.halloumi_message
chatbot_page.halloumi_claims
chatbot_page.verify_claims_loading

# Toggles
chatbot_page.fact_check_toggle
chatbot_page.deep_research_toggle

# Loading
chatbot_page.answer_loader

# Starter Messages
chatbot_page.starter_messages

# Other
chatbot_page.copy_button
chatbot_page.tabs
```

### ChatbotPage Methods

```python
# Messaging (context manager pattern)
with chatbot_page.send_message(message) as response:
    # response.value is a StreamedResponse after the context exits
    pass
chatbot_page.verify_answer(response.value)

# Validation
chatbot_page.verify_empty_conversation()
chatbot_page.verify_interactions_disabled()
chatbot_page.validate_predefined_messages_in_ui()

# Response parsing
chatbot_page.parse_response(playwright_response)  # Parse raw response → StreamedResponse

# Feedback (context manager)
with chatbot_page.send_feedback() as response:
    pass

# Configuration
chatbot_page.block_config      # Block configuration dict
chatbot_page.assistant         # Assistant model (id, name, description, starter_messages)
chatbot_page.show_starter_messages  # Whether starter messages are configured
```

### StreamedResponse Methods

```python
response.get_message()           # Full message text
response.get_reasoning()         # Reasoning data (dict by ind)
response.get_search_tools()      # Search tool results
response.get_final_documents()   # Source documents
response.get_citations()         # Inline citation references
response.get_related_questions() # Related question suggestions
response.get_by_type(type, ind)  # Filter chunks by type

# Properties
response.user_message_id         # User message ID
response.assistant_message_id    # Assistant message ID
response.grouped_chunks          # Chunks grouped by ind
response.chunks                  # Raw chunk list
response.error                   # Error message if any
```

---

## LLM-Based Verification

### Setup

Enable LLM analysis in `config.json`:

```json
{
  "enable_llm_analysis": true,
  "llm_model": "Inhouse-LLM/gpt-oss-120b",
  "llm_url": "https://llmgw.eea.europa.eu",
  "llm_api_key": "your_api_key"
}
```

### Using verify_answer() in Tests

The `verify_answer()` method performs a single LLM call that evaluates all four quality dimensions, returning a `ResponseVerification` object:

```python
from chatbot_tests.config import Settings
from chatbot_tests.llm_analysis import create_analyzer_from_settings
from chatbot_tests.step import llm_verdict

def test_response_quality(self, chatbot_page: ChatbotPage, data: dict, settings: Settings):
    question = data["question"]
    llm_config = data.get("validation", {}).get("llm", {})

    with step("Send question"):
        with chatbot_page.send_message(question) as response:
            pass
        chatbot_page.verify_answer(response.value)

    if not settings.enable_llm_analysis:
        return

    analyzer = create_analyzer_from_settings(settings)
    if not analyzer:
        return

    message = response.value.get_message()
    documents = response.value.get_final_documents()
    citations = response.value.get_citations()

    # Single LLM call for all verifications
    verification = analyzer.verify_answer(question, message, documents)

    # Log verdicts for each dimension
    if llm_config.get("verify_lack_information") and verification.lack_information:
        llm_verdict("LLM analysis: answer lacks information", verification.lack_information_explanation)
        pytest.skip(f"LLM analysis: answer lacks information - {verification.lack_information_explanation}")
    else:
        llm_verdict("LLM analysis: answer has sufficient information", verification.lack_information_explanation)

    if llm_config.get("verify_answers_question") and not verification.answers_question:
        llm_verdict("LLM analysis: answer off-topic", verification.answers_question_explanation, "failed")
    else:
        llm_verdict("LLM analysis: answer on-topic", verification.answers_question_explanation)

    if llm_config.get("verify_not_vague") and not verification.not_vague:
        llm_verdict("LLM analysis: answer too vague", verification.not_vague_explanation, "failed")
    else:
        llm_verdict("LLM analysis: answer not vague", verification.not_vague_explanation)

    if llm_config.get("verify_citations") and not verification.has_citations:
        llm_verdict("LLM analysis: answer missing citations", verification.has_citations_explanation, "failed")
    else:
        llm_verdict("LLM analysis: answer has citations", verification.has_citations_explanation)
```

### ResponseVerification Fields

```python
verification.lack_information              # bool - lacks info to answer
verification.lack_information_explanation  # str
verification.answers_question              # bool - directly answers question
verification.answers_question_explanation  # str
verification.not_vague                     # bool - specific, not vague
verification.not_vague_explanation         # str
verification.has_citations                 # bool - properly cites sources
verification.has_citations_explanation     # str
```

### LLM Verdict Conventions

Verdicts follow a specific naming pattern used by the analysis engine:

| Dimension | Pass Verdict | Fail Verdict |
|-----------|-------------|--------------|
| Information | `answer has sufficient information` | `answer lacks information` |
| Relevance | `answer on-topic` | `answer off-topic` |
| Specificity | `answer not vague` | `answer too vague` |
| Citations | `answer has citations` | `answer missing citations` |

---

## Running and Analyzing Tests

### Running Tests

```bash
# Run all tests
chatbot_tests run

# Run with specific markers
chatbot_tests run -m basic
chatbot_tests run -m "basic and high"

# Run in headed mode (see browser)
chatbot_tests run --headed

# Custom output file
chatbot_tests run -o ./results.jsonl

# Force color output
chatbot_tests run --color

# Limit number of fixture test cases
chatbot_tests run --limit 10

# Use different config
chatbot_tests run -c production.json
```

### Analyzing Results

```bash
# Summary
chatbot_tests analyze ./results.jsonl

# Show failures
chatbot_tests analyze ./results.jsonl --failures

# Show all steps
chatbot_tests analyze ./results.jsonl --steps

# Group by marker
chatbot_tests analyze ./results.jsonl --by-marker

# Show performance metrics (durations, slowest tests)
chatbot_tests analyze ./results.jsonl --performance

# Show auto-generated insights (health, recommendations)
chatbot_tests analyze ./results.jsonl --insights

# Show all analysis sections at once
chatbot_tests analyze ./results.jsonl --all

# Include LLM analysis (generates PDF report)
chatbot_tests analyze ./results.jsonl --llm
```

### Analysis Output Sections

| Flag | Description |
|------|-------------|
| (default) | Summary with pass/fail counts, pass rate, health status |
| `--failures` | Detailed failure breakdown with error messages, grouped by category |
| `--steps` | Full step-by-step breakdown for each test |
| `--by-marker` | Tests grouped by pytest markers |
| `--performance` | Duration metrics: avg test/step time, slowest/fastest tests |
| `--insights` | Auto-generated insights: health assessment, patterns, recommendations |
| `--all` | Show performance, failures, and insights together |
| `--llm` | LLM-powered analysis with risk assessment, generates JSON + PDF |

### Comparing Runs

```bash
# Compare multiple runs
chatbot_tests compare ./run1.jsonl ./run2.jsonl ./run3.jsonl

# With LLM comparison (generates PDF)
chatbot_tests compare ./run1.jsonl ./run2.jsonl --llm
```

---

## Test Files Overview

### `test_basic.py` — Always-Run UI Tests

Contains UI/integration tests marked with `@pytest.mark.always`. These tests:
- Verify core chatbot functionality (UI elements, interactions, navigation)
- Test feature-specific UI (feedback, Halloumi, sources, deep research)
- Run regardless of fixture data or `-m` filter
- Are organized by feature in separate test classes

### `test_questions.py` — Data-Driven Question Tests

Contains comprehensive question validation parametrized from `fixtures/*.json`. For each question:

1. **Pre-phase**: Configure quality check toggle based on block config
2. **Phase 1**: Send question, capture response + related questions
3. **Phase 2**: LLM-based quality verification (if enabled)
4. **Phase 3**: Source citation validation + related questions validation
5. **Phase 4**: Response content validation (length, keywords)
6. **Phase 5**: Halloumi quality check validation (if enabled)
7. **Phase 6**: Feedback functionality validation (if enabled)
8. **Cleanup**: Clear chat for next test

### Block Config Feature Detection

Features are conditionally tested based on the chatbot block configuration (`window.__EEA_CHATBOT_TEST_CONFIG__`):

| Feature | Block Config Key | Values |
|---------|-----------------|--------|
| Quality Check | `qualityCheck` | `enabled`, `ondemand`, `ondemand_toggle`, `disabled` |
| Related Questions | `enableQgen` | `true`/`false` |
| Feedback | `enableFeedback` | `true`/`false` |
| Deep Research | `deepResearch` | `always_on`, `user_on`, `user_off`, `disabled` |
| Starter Messages | `showAssistantPrompts` / `enableStarterPrompts` | `true`/`false` |

When a feature is not enabled, tests log an info message and skip that validation gracefully.

---

## Best Practices

1. **Always use steps**: Wrap every logical action in a `step()` for clear logging and JSONL output.

2. **Check block config**: Skip feature validations gracefully when not enabled in the chatbot configuration.

3. **Use `info()` for status**: Log configuration state and skip reasons with `info()` so they appear in reports.

4. **Use `llm_verdict()` for LLM results**: LLM quality assessments should use `llm_verdict()` for proper tracking in analysis.

5. **Handle LLM analysis conditionally**: Always check `settings.enable_llm_analysis` before using LLM verification.

6. **Use `continue_on_failure` for optional checks**: Non-critical validations shouldn't fail the entire test.

7. **Clean up state**: Use `clear_chat_button.click()` + `verify_empty_conversation()` when tests need a fresh conversation.

8. **Use descriptive step names**: Step names appear in reports and should be self-explanatory.

9. **Inherit fixture validation**: Use `default_validation` at the file level and only override specific fields per test case.

10. **Set appropriate timeouts**: LLM responses can be slow; the framework uses configurable `timeout` and `expect_timeout` settings.
