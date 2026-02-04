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
.chat-window                        # Main container
├── .messages                       # Message display area
│   ├── .conversation              # Scrollable message list
│   │   ├── .comment.user          # User messages (with .circle.user avatar)
│   │   └── .comment.assistant     # AI messages (with .circle.assistant avatar)
│   └── .empty-state               # Landing page (shown when no messages)
│       ├── .starter-message-heading
│       └── .starter-messages-container
│           └── .starter-message   # Clickable starter prompts
├── .chat-form                      # Input area
│   ├── .textarea-wrapper
│   │   └── textarea               # Message input
│   └── .chat-right-actions
│       └── .submit-btn            # Send button (aria-label="Send")
└── .chat-controls
    ├── #fact-check-toggle         # Quality check toggle
    └── #deep-research-toggle      # Deep research toggle
```

### Key Selectors

| Element | Selector |
|---------|----------|
| Chat window | `.chat-window` |
| Message input | `.textarea-wrapper textarea` |
| Send button | `[aria-label='Send']` |
| User message | `.comment:has(.circle.user)` |
| Assistant message | `.circle.assistant` (parent is the message) |
| Clear chat | `button.clear-chat` |
| Sources container | `.sources-container` |
| Source items | `.source` |
| Related questions | `.chat-related-questions` |
| Like button | `[aria-label='Like']` |
| Dislike button | `[aria-label='Dislike']` |
| Feedback modal | `.feedback-modal` |
| Fact-check button | `.halloumi-feedback-button .claims-btn` |
| Answer loader | `.loader-container .loader` |

### Response Streaming

The chatbot uses server-sent events (SSE) for streaming responses. Each response consists of JSONL chunks:

```json
{"user_message_id": 123, "reserved_assistant_message_id": 456}
{"ind": 0, "obj": {"type": "message_delta", "message": "Air quality..."}}
{"ind": 0, "obj": {"type": "message_start", "documents": [...]}}
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

```
chatbot_tests/
├── tests/
│   ├── conftest.py          # Fixtures and hooks
│   └── test_basic.py        # Test classes
├── fixtures/
│   └── golden_dataset.json  # Test data
├── page_objects/
│   ├── chatbot_page.py      # Page object
│   └── response.py          # Response model
└── reports/                  # Generated JSONL output
```

---

## Writing Basic UI Tests

### Test File Structure

Create test files in the `tests/` directory:

```python
# tests/test_ui.py
import pytest
from playwright.sync_api import expect
from chatbot_tests.step import step
from chatbot_tests.page_objects import ChatbotPage


@pytest.mark.ui
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
@pytest.mark.ui
class TestUserInteractions:

    def test_starter_prompts_clickable(self, chatbot_page: ChatbotPage):
        """Verify clicking a starter prompt sends the message."""

        with step("Verify empty state is visible"):
            expect(chatbot_page.empty_state).to_be_visible()

        with step("Get first starter prompt"):
            starter = chatbot_page.starter_messages.first
            prompt_text = starter.text_content()

        with step("Click starter prompt"):
            starter.click()

        with step("Verify user message appears"):
            expect(chatbot_page.user_message.first).to_be_visible()

    def test_send_button_interaction(self, chatbot_page: ChatbotPage):
        """Verify send button sends message."""

        with step("Type a message"):
            chatbot_page.textarea.fill("What is climate change?")

        with step("Click send button"):
            chatbot_page.submit_button.click()

        with step("Verify message was sent"):
            expect(chatbot_page.user_message.first).to_have_text("What is climate change?")

    def test_enter_key_sends_message(self, chatbot_page: ChatbotPage):
        """Verify pressing Enter sends the message."""

        with step("Type and press Enter"):
            chatbot_page.textarea.fill("Test question")
            chatbot_page.textarea.press("Enter")

        with step("Verify message appears in conversation"):
            expect(chatbot_page.user_message.first).to_be_visible()

    def test_clear_chat(self, chatbot_page: ChatbotPage):
        """Verify clear chat button resets the conversation."""

        with step("Send a message first"):
            chatbot_page.send_message_and_wait("Hello")

        with step("Click clear chat button"):
            chatbot_page.clear_chat_button.click()

        with step("Verify conversation is cleared"):
            expect(chatbot_page.empty_state).to_be_visible()
```

### Testing UI Components

```python
@pytest.mark.feedback
class TestFeedbackUI:

    def test_feedback_buttons_visible(self, chatbot_page: ChatbotPage):
        """Verify feedback buttons appear after response."""

        with step("Send message and wait for response"):
            chatbot_page.send_message_and_wait("What is biodiversity?")

        with step("Verify like button is visible"):
            expect(chatbot_page.like_button).to_be_visible()

        with step("Verify dislike button is visible"):
            expect(chatbot_page.dislike_button).to_be_visible()

    def test_feedback_modal_opens(self, chatbot_page: ChatbotPage):
        """Verify feedback modal opens when clicking dislike."""

        with step("Send message and wait for response"):
            chatbot_page.send_message_and_wait("Test question")

        with step("Click dislike button"):
            chatbot_page.dislike_button.click()

        with step("Verify feedback modal opens"):
            expect(chatbot_page.feedback_modal).to_be_visible()
```

---

## Testing LLM Responses

### Basic Response Verification

```python
@pytest.mark.llm
class TestLLMResponses:

    def test_receives_response(self, chatbot_page: ChatbotPage):
        """Verify chatbot returns a response to a question."""

        with step("Send question"):
            response = chatbot_page.send_message_and_wait(
                "What is the current state of air quality in Europe?"
            )

        with step("Verify response was received"):
            assert response is not None
            assert response.user_message_id is not None
            assert response.assistant_message_id is not None

        with step("Verify response has content"):
            message = response.get_message()
            assert len(message) > 0

    def test_response_has_sources(self, chatbot_page: ChatbotPage):
        """Verify response includes source documents."""

        with step("Send question about a specific topic"):
            response = chatbot_page.send_message_and_wait(
                "What policies exist for reducing plastic pollution?"
            )

        with step("Verify sources are present"):
            documents = response.get_final_documents()
            assert len(documents) > 0, "Expected at least one source document"

        with step("Verify source structure"):
            for doc in documents:
                assert "semantic_identifier" in doc
                assert "blurb" in doc
```

### Using the StreamedResponse Model

The `StreamedResponse` class provides methods for extracting data from responses:

```python
def test_response_analysis(self, chatbot_page: ChatbotPage):
    response = chatbot_page.send_message_and_wait("What is climate change?")

    # Get the full message text
    message = response.get_message()

    # Get reasoning (if deep research enabled)
    reasoning = response.get_reasoning()

    # Get search tool results
    search_tools = response.get_search_tools()

    # Get final documents/sources
    documents = response.get_final_documents()

    # Get specific chunk types
    deltas = response.get_by_type("message_delta")
```

### Testing Response Quality

```python
@pytest.mark.quality
class TestResponseQuality:

    def test_response_length(self, chatbot_page: ChatbotPage):
        """Verify response meets minimum length requirements."""

        with step("Send a complex question"):
            response = chatbot_page.send_message_and_wait(
                "Explain the causes and effects of ocean acidification"
            )

        with step("Verify response is substantial"):
            message = response.get_message()
            assert len(message) >= 100, f"Response too short: {len(message)} chars"

    def test_response_contains_keywords(self, chatbot_page: ChatbotPage):
        """Verify response contains expected topic keywords."""

        with step("Send question about specific topic"):
            response = chatbot_page.send_message_and_wait(
                "What are the main greenhouse gases?"
            )

        with step("Verify response mentions key terms"):
            message = response.get_message().lower()
            expected_terms = ["carbon dioxide", "co2", "methane", "greenhouse"]
            found = any(term in message for term in expected_terms)
            assert found, f"Response should mention greenhouse gases"
```

---

## Working with Fixtures

### Fixture File Format

Create JSON files in the `fixtures/` directory:

```json
{
  "version": "1.0.0",
  "metadata": {
    "name": "Air Quality Tests",
    "description": "Tests for air quality related questions"
  },
  "validation_thresholds": {
    "min_response_length": 100,
    "min_sources": 2
  },
  "test_cases": [
    {
      "id": "AQ-001",
      "priority": "high",
      "question": "What is the current state of air quality in Europe?",
      "markers": ["basic", "sources"],
      "validation": {
        "expect_sources": true,
        "min_sources": 2,
        "expect_response": true,
        "min_response_length": 100
      }
    },
    {
      "id": "AQ-002",
      "priority": "medium",
      "question": "How does air pollution affect human health?",
      "markers": ["health", "comprehensive"],
      "validation": {
        "expect_sources": true,
        "min_sources": 1,
        "expected_keywords": ["respiratory", "health", "pollution"]
      }
    }
  ]
}
```

### Using the `data` Fixture

The `data` fixture automatically parametrizes tests with each test case:

```python
@pytest.mark.sources
class TestSources:

    def test_sources_present(self, chatbot_page: ChatbotPage, data: dict):
        """Verify responses include required sources."""

        # Skip if this test case doesn't require sources
        validation = data.get("validation", {})
        if not validation.get("expect_sources"):
            pytest.skip("Test case doesn't require sources")

        with step(f"Send question: {data['id']}"):
            response = chatbot_page.send_message_and_wait(data["question"])

        with step("Verify minimum sources"):
            documents = response.get_final_documents()
            min_sources = validation.get("min_sources", 1)
            assert len(documents) >= min_sources, \
                f"Expected at least {min_sources} sources, got {len(documents)}"
```

### Creating Topic-Specific Fixtures

Organize fixtures by topic for better management:

```
fixtures/
├── golden_dataset.json      # Core test cases
├── air_quality.json         # Air quality questions
├── climate_change.json      # Climate change questions
├── biodiversity.json        # Biodiversity questions
└── water_quality.json       # Water quality questions
```

Example topic fixture:

```json
{
  "version": "1.0.0",
  "metadata": {
    "name": "Climate Change Tests",
    "topic": "climate"
  },
  "validation_thresholds": {
    "min_response_length": 150,
    "min_sources": 2
  },
  "test_cases": [
    {
      "id": "CC-001",
      "priority": "high",
      "question": "What are the main causes of climate change?",
      "markers": ["climate", "basic", "comprehensive"],
      "validation": {
        "expect_sources": true,
        "min_sources": 2,
        "expected_keywords": ["greenhouse", "emissions", "fossil"]
      }
    },
    {
      "id": "CC-002",
      "priority": "high",
      "question": "How is climate change affecting Europe?",
      "markers": ["climate", "europe", "impacts"],
      "validation": {
        "expect_sources": true,
        "min_sources": 3,
        "min_response_length": 200
      }
    }
  ]
}
```

---

## Using Markers

### Built-in Markers

```python
@pytest.mark.basic         # Basic functionality tests
@pytest.mark.ui            # UI-only tests (no LLM verification)
@pytest.mark.sources       # Source/citation tests
@pytest.mark.feedback      # Feedback feature tests
@pytest.mark.quality_check # Halloumi fact-check tests
@pytest.mark.llm           # LLM response tests
@pytest.mark.slow          # Long-running tests
@pytest.mark.always        # Tests that run regardless of data fixtures
```

### Priority Markers

```python
@pytest.mark.high          # High priority
@pytest.mark.medium        # Medium priority
@pytest.mark.low           # Low priority
```

### Topic Markers

Define markers based on your fixture topics:

```python
@pytest.mark.climate       # Climate change
@pytest.mark.air_quality   # Air quality
@pytest.mark.biodiversity  # Biodiversity
@pytest.mark.water         # Water quality
```

### Applying Markers

```python
@pytest.mark.basic
@pytest.mark.high
class TestBasicFunctionality:
    def test_loads(self, chatbot_page):
        pass


# Multiple markers on a method
@pytest.mark.basic
class TestSomething:

    @pytest.mark.slow
    def test_slow_operation(self, chatbot_page):
        pass
```

### Running with Markers

```bash
# Run only basic tests
chatbot-tests run -m basic

# Run high priority tests
chatbot-tests run -m high

# Run multiple markers (OR logic)
chatbot-tests run -m "basic,sources"

# Run with marker expression (AND logic)
chatbot-tests run -m "basic and high"

# Exclude markers
chatbot-tests run -m "not slow"
```

---

## Step-Based Logging

### Using the `step()` Context Manager

Every logical test action should be wrapped in a `step()`:

```python
from chatbot_tests.step import step

def test_example(self, chatbot_page):
    with step("Navigate to chatbot"):
        chatbot_page.goto(settings.chatbot_url)

    with step("Verify page loaded"):
        expect(chatbot_page.chat_window).to_be_visible()

    with step("Send message"):
        response = chatbot_page.send_message_and_wait("Hello")

    with step("Verify response"):
        assert response is not None
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
  "step_name": "Send message"
}
```

### Continue on Failure

For non-critical checks, use `continue_on_failure`:

```python
def test_with_optional_checks(self, chatbot_page):
    with step("Send message"):
        response = chatbot_page.send_message_and_wait("Test")

    # Critical check - test fails if this fails
    with step("Verify response exists"):
        assert response is not None

    # Optional check - logged but doesn't fail the test
    with step("Check for related questions", continue_on_failure=True):
        rq = chatbot_page.related_question_buttons
        assert rq.count() > 0, "No related questions found"
```

---

## Page Object Reference

### ChatbotPage Properties (Locators)

```python
chatbot_page.chat_window              # Main container
chatbot_page.messages                 # Message area
chatbot_page.conversation             # Scrollable message list
chatbot_page.empty_state              # Landing page
chatbot_page.textarea                 # Message input
chatbot_page.submit_button            # Send button
chatbot_page.clear_chat_button        # Clear chat button
chatbot_page.user_message             # User message locator
chatbot_page.assistant_message        # Assistant message locator
chatbot_page.answer_loader            # Loading indicator
chatbot_page.sources_container        # Sources section
chatbot_page.source_items             # Individual sources
chatbot_page.show_all_sources_button  # "See all sources" button
chatbot_page.related_questions_container
chatbot_page.related_question_buttons
chatbot_page.like_button
chatbot_page.dislike_button
chatbot_page.feedback_modal
chatbot_page.feedback_textarea
chatbot_page.fact_check_button
chatbot_page.halloumi_message
chatbot_page.starter_messages
```

### ChatbotPage Methods

```python
# Navigation
chatbot_page.goto(url)                          # Navigate to URL
chatbot_page.wait_for_chatbot_ready(timeout)    # Wait for UI ready
chatbot_page.wait_for_network_idle(timeout)     # Wait for network

# Messaging
chatbot_page.send_message(message, timeout)     # Send, return response immediately
chatbot_page.send_message_and_wait(message, timeout)  # Send and wait for completion

# Response handling
chatbot_page.get_response(assistant_message_id) # Get stored response
chatbot_page.responses                          # List of all responses
```

### StreamedResponse Methods

```python
response.get_message()           # Full message text
response.get_reasoning()         # Reasoning data (dict by ind)
response.get_search_tools()      # Search tool results
response.get_final_documents()   # Source documents
response.get_by_type(type, ind)  # Filter chunks by type

# Properties
response.user_message_id         # User message ID
response.assistant_message_id    # Assistant message ID
response.grouped_chunks          # Chunks grouped by ind
response.chunks                  # Raw chunk list
```

---

## LLM-Based Verification

### Setup

Enable LLM analysis in your `.env`:

```bash
ENABLE_LLM_ANALYSIS=true
LLM_MODEL=Inhouse-LLM/gpt-oss-120b
LLM_URL=https://llmgw.eea.europa.eu
LLM_API_KEY=your_api_key
```

### Using LLMAnalyzer in Tests

```python
from chatbot_tests.config import settings
from chatbot_tests.llm_analysis import create_analyzer_from_settings

@pytest.mark.llm
class TestWithLLMVerification:

    def test_response_quality(self, chatbot_page: ChatbotPage):
        question = "What is the EU doing about plastic pollution?"

        with step("Send question"):
            response = chatbot_page.send_message_and_wait(question)

        # Only run LLM checks if enabled
        if not settings.enable_llm_analysis:
            return

        analyzer = create_analyzer_from_settings(settings)
        if not analyzer:
            return

        message = response.get_message()
        documents = response.get_final_documents()

        with step("Verify response has citations (LLM)"):
            passes, explanation = analyzer.verify_has_citations(
                message, documents
            )
            assert passes, f"Missing citations: {explanation}"

        with step("Verify response is not vague (LLM)"):
            passes, explanation = analyzer.verify_not_vague(
                question, message
            )
            assert passes, f"Response too vague: {explanation}"

        with step("Verify response answers question (LLM)"):
            passes, explanation = analyzer.verify_answers_question(
                question, message
            )
            assert passes, f"Doesn't answer question: {explanation}"
```

### Available LLM Verification Methods

```python
# Check if response properly cites sources
passes, explanation = analyzer.verify_has_citations(response_text, documents)

# Check if response is specific, not vague
passes, explanation = analyzer.verify_not_vague(question, response_text)

# Check if response actually answers the question
passes, explanation = analyzer.verify_answers_question(question, response_text)

# Full structured analysis
analysis = analyzer.analyze_response(question, response_text, documents)
# Returns: ResponseAnalysis with accuracy, completeness, relevance scores
```

---

## Running and Analyzing Tests

### Running Tests

```bash
# Run all tests
chatbot-tests run

# Run with specific markers
chatbot-tests run -m basic
chatbot-tests run -m "sources and high"

# Run in headed mode (see browser)
chatbot-tests run --headed

# Custom output file
chatbot-tests run -o ./results.jsonl

# Force color output
chatbot-tests run --color

# Limit number of test cases
chatbot-tests run --limit 10
```

### Analyzing Results

```bash
# Summary
chatbot-tests analyze ./results.jsonl

# Show failures
chatbot-tests analyze ./results.jsonl --failures

# Show all steps
chatbot-tests analyze ./results.jsonl --steps

# Group by marker
chatbot-tests analyze ./results.jsonl --by-marker

# Show performance metrics (durations, slowest tests)
chatbot-tests analyze ./results.jsonl --performance

# Show auto-generated insights (health, recommendations)
chatbot-tests analyze ./results.jsonl --insights

# Show all analysis sections at once
chatbot-tests analyze ./results.jsonl --all

# JSON output
chatbot-tests analyze ./results.jsonl -f json

# Include LLM analysis
chatbot-tests analyze ./results.jsonl --llm
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
| `--llm` | LLM-powered analysis with risk assessment and root cause analysis |

### Comparing Runs

```bash
# Compare multiple runs
chatbot-tests compare ./run1.jsonl ./run2.jsonl ./run3.jsonl

# With LLM comparison
chatbot-tests compare ./run1.jsonl ./run2.jsonl --llm
```

---

## Complete Test Example

Here's a complete test file demonstrating all concepts:

```python
# tests/test_air_quality.py
"""End-to-end tests for air quality related questions."""

import pytest
from playwright.sync_api import expect
from chatbot_tests.step import step
from chatbot_tests.page_objects import ChatbotPage
from chatbot_tests.config import settings
from chatbot_tests.llm_analysis import create_analyzer_from_settings


@pytest.mark.air_quality
@pytest.mark.ui
class TestAirQualityUI:
    """UI tests for air quality questions."""

    @pytest.mark.always
    def test_chatbot_ready(self, chatbot_page: ChatbotPage):
        """Verify chatbot is ready to receive questions."""
        with step("Verify chat window visible"):
            expect(chatbot_page.chat_window).to_be_visible()

        with step("Verify input ready"):
            expect(chatbot_page.textarea).to_be_visible()
            expect(chatbot_page.submit_button).to_be_visible()


@pytest.mark.air_quality
@pytest.mark.sources
class TestAirQualitySources:
    """Tests verifying air quality responses include sources."""

    def test_sources_present(self, chatbot_page: ChatbotPage, data: dict):
        """Verify air quality responses include relevant sources."""

        validation = data.get("validation", {})
        if not validation.get("expect_sources"):
            pytest.skip("Test case doesn't require sources")

        with step(f"Send question ({data['id']})"):
            response = chatbot_page.send_message_and_wait(data["question"])

        with step("Verify sources in response"):
            documents = response.get_final_documents()
            min_sources = validation.get("min_sources", 1)
            assert len(documents) >= min_sources, \
                f"Expected {min_sources}+ sources, got {len(documents)}"

        with step("Verify sources displayed in UI"):
            expect(chatbot_page.sources_container).to_be_visible()


@pytest.mark.air_quality
@pytest.mark.llm
class TestAirQualityResponseQuality:
    """Tests verifying air quality response quality using LLM analysis."""

    def test_response_quality(self, chatbot_page: ChatbotPage, data: dict):
        """Verify response quality for air quality questions."""

        with step(f"Send question ({data['id']})"):
            response = chatbot_page.send_message_and_wait(data["question"])

        message = response.get_message()
        validation = data.get("validation", {})

        with step("Verify response length"):
            min_length = validation.get("min_response_length", 100)
            assert len(message) >= min_length, \
                f"Response too short: {len(message)} < {min_length}"

        # Optional LLM verification
        if not settings.enable_llm_analysis:
            return

        analyzer = create_analyzer_from_settings(settings)
        if not analyzer:
            return

        documents = response.get_final_documents()

        with step("LLM: Verify answers question"):
            passes, explanation = analyzer.verify_answers_question(
                data["question"], message
            )
            assert passes, f"Doesn't answer question: {explanation}"

        with step("LLM: Verify has citations", continue_on_failure=True):
            passes, explanation = analyzer.verify_has_citations(
                message, documents
            )
            assert passes, f"Missing citations: {explanation}"


@pytest.mark.air_quality
@pytest.mark.feedback
class TestAirQualityFeedback:
    """Tests for feedback on air quality responses."""

    def test_can_provide_feedback(self, chatbot_page: ChatbotPage, data: dict):
        """Verify user can provide feedback on air quality responses."""

        with step(f"Send question ({data['id']})"):
            chatbot_page.send_message_and_wait(data["question"])

        with step("Verify feedback buttons visible"):
            expect(chatbot_page.like_button).to_be_visible()
            expect(chatbot_page.dislike_button).to_be_visible()

        with step("Click like button"):
            chatbot_page.like_button.click()

        with step("Verify feedback recorded"):
            # Toast or visual confirmation
            pass
```

---

## Best Practices

1. **Always use steps**: Wrap every logical action in a `step()` for clear logging.

2. **Check validation requirements**: Skip tests when the data fixture doesn't require that feature.

3. **Use appropriate markers**: Apply markers that match the test's purpose and the fixture's markers.

4. **Handle LLM analysis conditionally**: Always check `settings.enable_llm_analysis` before using LLM verification.

5. **Use `continue_on_failure` for optional checks**: Non-critical validations shouldn't fail the entire test.

6. **Keep tests focused**: Each test should verify one specific behavior.

7. **Use descriptive step names**: Step names appear in reports and should be self-explanatory.

8. **Organize fixtures by topic**: Group related questions in topic-specific fixture files.

9. **Set appropriate timeouts**: LLM responses can be slow; use appropriate timeouts in `send_message_and_wait()`.

10. **Clean up state**: Use `clear_chat_button` when tests need a fresh conversation.

---

## Test Files Overview

The test suite contains two main test files:

### `test_basic.py` - Always-Run UI Tests

Contains UI/integration tests marked with `@pytest.mark.always`. These tests:
- Verify core chatbot functionality
- Test UI elements and interactions
- Run regardless of fixture data
- Are organized by feature (UI, feedback, halloumi, sources, etc.)

### `test_questions.py` - Data-Driven Question Tests

Contains comprehensive question validation tests. For each question in `fixtures/*.json`, this file:
1. Sends the question and validates response
2. Checks response content and length
3. Validates source citations
4. Triggers and validates Halloumi quality check (if enabled)
5. Validates related questions (if qgen enabled)
6. Validates feedback functionality (if enabled)
7. Runs LLM-based quality checks (if enabled)

**Running question tests:**
```bash
# Run all question tests
chatbot-tests run -m questions

# Run with limit
chatbot-tests run -m questions --limit 3

# Run high priority only
chatbot-tests run -m "questions and high"

# Run specific topic
chatbot-tests run -m "questions and air_quality"
```

---

## Fixture Validation Schema (v2.0)

Each test case in `fixtures/*.json` uses this schema:

```json
{
  "id": "GD-001",
  "priority": "high",
  "question": "What is the current state of air quality in Europe?",
  "markers": ["comprehensive", "air_quality"],
  "validation": {
    "response": {
      "min_length": 100,
      "expected_keywords": ["air", "pollution", "Europe"]
    },
    "sources": {
      "min_count": 2
    },
    "quality_check": {
      "min_score": 70
    },
    "related_questions": {
      "min_count": 2
    },
    "feedback": {
      "enabled": true
    },
    "llm": {
      "verify_answers_question": true,
      "verify_not_vague": true,
      "verify_citations": true
    }
  }
}
```

### Validation Sections

| Section | Options | Description |
|---------|---------|-------------|
| `response` | `min_length`, `expected_keywords` | Content validation |
| `sources` | `min_count` | Citation requirements |
| `quality_check` | `min_score` | Halloumi fact-check threshold (0-100) |
| `related_questions` | `min_count` | Minimum generated follow-up questions |
| `feedback` | `enabled` | Test feedback buttons visibility |
| `llm` | `verify_answers_question`, `verify_not_vague`, `verify_citations` | LLM-based quality checks |

### Block Config Behavior

Features skip gracefully when not enabled in block config:

| Feature | Block Config | Behavior When Disabled |
|---------|--------------|------------------------|
| Quality Check | `qualityCheck: "disabled"` | Logs info, skips validation |
| Related Questions | `enableQgen: false` | Logs info, skips validation |
| Feedback | `enableFeedback: false` | Logs info, skips validation |
| LLM Verification | `ENABLE_LLM_ANALYSIS=false` | Logs info, skips validation |

### Adding New Questions

1. Add test case to `fixtures/golden_dataset.json` or create a new topic file
2. Include appropriate `markers` for filtering
3. Set `priority` (high/medium/low) for selective runs
4. Configure `validation` sections with desired thresholds
5. Run tests: `chatbot-tests run -m questions`
