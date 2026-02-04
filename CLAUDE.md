# Chatbot Tests Framework

Playwright-based test automation framework for volto-eea-chatbot validation with step-based logging, JSONL output, and result analysis.

## Project Structure

```
chatbot_tests/
├── console.py       # Terminal output with ANSI colors
├── plugin.py        # Pytest plugin for event collection
├── step.py          # Context manager for test steps
├── output.py        # JSONL file writer
├── runner.py        # Test orchestration
├── main.py          # CLI entry point
├── analyze.py       # Result analysis and comparison
├── config.py        # Settings via pydantic-settings
├── llm_analysis.py  # LLM-based response analysis
├── page_objects/    # Playwright page object pattern
│   ├── __init__.py
│   ├── chatbot_page.py
│   └── response.py  # StreamedResponse model
├── tests/
│   ├── conftest.py  # Pytest fixtures and hooks
│   └── test_basic.py
├── fixtures/        # Test data JSON files
└── reports/         # Generated JSONL output files
```

## CLI Usage

By default, all commands load `config.json` from the current directory. Use `-c` to specify a different config file.

```bash
# Run tests (uses config.json by default)
chatbot-tests run                             # Run all tests
chatbot-tests run -m basic                    # Run tests with 'basic' marker
chatbot-tests run --headed                    # Show browser (overrides config headless)
chatbot-tests run --limit 5                   # Limit to first 5 questions
chatbot-tests run -o ./out.jsonl              # Custom output path
chatbot-tests run -c other.json               # Use different config file

# Analyze results
chatbot-tests analyze ./results.jsonl                 # Summary
chatbot-tests analyze ./results.jsonl --failures      # Show failures
chatbot-tests analyze ./results.jsonl --steps         # Show all steps
chatbot-tests analyze ./results.jsonl --by-marker     # Group by marker
chatbot-tests analyze ./results.jsonl -f json         # JSON output
chatbot-tests analyze ./results.jsonl --llm           # LLM analysis (uses config.json)

# Compare multiple runs
chatbot-tests compare ./run1.jsonl ./run2.jsonl ./run3.jsonl
chatbot-tests compare ./run1.jsonl ./run2.jsonl --llm  # LLM comparison
```

## Architecture

### Event Flow

```
CLI (main.py)
    │
    ▼
run_tests_sync() in runner.py
    │
    ├─── JSONLWriter opens output file
    │
    ▼
pytest.main() with ResultCollectorPlugin
    │
    ├─── Pytest hooks fire → TestEvent objects created
    │         │
    │         ├─── on_event callback:
    │         │         ├─── writer.write_event() → JSONL file
    │         │         └─── _print_event() → console output
    │         │
    │         └─── step() context manager → log_step() → plugin._emit()
    │
    ▼
TestRunSummary returned with pass/fail counts
```

### Module Responsibilities

| Module | Purpose |
|--------|---------|
| `console.py` | ANSI color output, bypasses pytest capture with `os.write(1, ...)` |
| `plugin.py` | Pytest plugin hooks, emits TestEvent via callbacks |
| `step.py` | `step()` context manager with timing, `info()` for untimed logging |
| `output.py` | JSONLWriter for real-time streaming to file |
| `runner.py` | TestRunner class, builds pytest args, coordinates execution |
| `main.py` | argparse CLI with run/analyze/compare subcommands |
| `analyze.py` | Parses JSONL, calculates stats, detects regressions |
| `config.py` | Pydantic settings from environment/.env |
| `llm_analysis.py` | LLM-based response quality analysis via LiteLLM |

## Writing Tests

### Using Steps

Steps have types that indicate their timing expectations:

| Type | Description | Timing Expected |
|------|-------------|-----------------|
| `action` | User interaction, browser action, API call | Yes |
| `info` | Logging results/status, informational message | No |
| `wait` | Waiting for async operations | Yes |

```python
from chatbot_tests.step import step, info
from playwright.sync_api import expect

@pytest.mark.basic
class TestChatbotBasicFunctionality:
    def test_chatbot_loads(self, chatbot_page: ChatbotPage):
        # Timed action steps (step_type="action")
        with step("Verify chat window is visible"):
            expect(chatbot_page.chat_window).to_be_visible()

        with step("Verify textarea is visible"):
            expect(chatbot_page.textarea).to_be_visible()

        # Informational logging (step_type="info", no timing)
        info("INFO: Feature X is enabled in block config")

        # Log LLM analysis results (no timing expected)
        info(f"LLM analysis: answer on-topic: {explanation}")
```

**`step()` context manager** (for timed actions):
- Measures execution time
- Logs pass/fail with duration to JSONL
- Sets `step_type="action"`
- Re-raises exceptions on failure

**`info()` function** (for informational messages):
- No timing measurement
- Logs message to JSONL with `step_type="info"`
- Use for status messages, configuration info, LLM analysis results
- LLM analyzer understands these don't need timing

### Markers

Apply pytest markers to organize tests:

```python
@pytest.mark.basic      # Basic functionality
@pytest.mark.feedback   # Feedback features
@pytest.mark.high       # High priority
```

Run specific markers: `chatbot-tests run -m basic,high`

### Fixtures

Available fixtures in `conftest.py`:

- `chatbot_page` - ChatbotPage instance, already navigated to chatbot URL
- `settings` - Settings instance with configuration
- `data` - Parametrized test case data from JSON fixtures

## JSONL Format

Each line is a JSON object with `event_type`:

```jsonl
{"event_type": "session_start", "timestamp": "2026-01-27T18:45:00.106678"}
{"event_type": "test_start", "nodeid": "tests/test_basic.py::TestChatbotBasicFunctionality::test_chatbot_loads[chromium]", "name": "test_chatbot_loads[chromium]", "markers": ["basic"]}
{"event_type": "step", "nodeid": "...", "name": "test_chatbot_loads[chromium]", "outcome": "passed", "duration_ms": 6, "step_name": "Verify chat window is visible", "step_type": "action"}
{"event_type": "step", "nodeid": "...", "name": "test_chatbot_loads[chromium]", "outcome": "passed", "step_name": "INFO: Feature enabled", "step_type": "info"}
{"event_type": "test_end", "nodeid": "...", "name": "test_chatbot_loads[chromium]", "outcome": "passed", "duration_seconds": 4.804}
{"event_type": "session_end", "timestamp": "2026-01-27T18:45:04.951433", "outcome": "passed"}
```

Step events include a `step_type` field:
- `"action"`: Timed browser/API actions (duration_ms is measured)
- `"info"`: Informational logging (duration_ms is null - this is expected)
- `"wait"`: Async waits (duration_ms is measured)

## Configuration

Settings can be loaded from:
1. **JSON config file** (via `--config` / `-c` CLI option)
2. **Environment variables**
3. **`.env` file**

Priority: Environment variables > .env file > JSON config file defaults

### JSON Config File

Create a JSON file with snake_case keys matching the settings:

```json
{
  "chatbot_base_url": "https://www.eea.europa.eu",
  "chatbot_path": "/chatbot",
  "headless": true,
  "browser": "chromium",
  "timeout": 120000,
  "reports_dir": "./reports",
  "fixtures_dir": "./fixtures/eea_site",
  "enable_llm_analysis": false
}
```

See `config.example.json` for a complete template.

### Settings Reference

| Setting | Default | Description |
|---------|---------|-------------|
| `CHATBOT_BASE_URL` | `http://localhost:3000` | Volto frontend URL |
| `CHATBOT_PATH` | `/chatbot` | Path to chatbot page |
| `HEADLESS` | `true` | Run browser headless |
| `BROWSER` | `chromium` | Browser: chromium/firefox/webkit |
| `TIMEOUT` | `120000` | Default timeout (ms) |
| `REPORTS_DIR` | `./chatbot_tests/reports` | Output directory |
| `FIXTURES_DIR` | `./fixtures` (relative to package) | Test fixtures directory |
| `ENABLE_LLM_ANALYSIS` | `false` | Enable LLM-based analysis |
| `LLM_MODEL` | `Inhouse-LLM/gpt-oss-120b` | LLM model for analysis |
| `LLM_URL` | `https://llmgw.eea.europa.eu` | LLM API endpoint |
| `LLM_API_KEY` | `` | API key for authentication |

## LLM-Based Analysis

The framework supports optional LLM-based analysis for:
1. **Test verification** - Verify response quality during tests
2. **Report analysis** - Analyze test reports with insights
3. **Run comparison** - Compare multiple test runs with recommendations

### Setup

1. Install litellm (in the external LLM service, not in this test framework)

2. Configure in `.env`:
```bash
ENABLE_LLM_ANALYSIS=true
LLM_MODEL=Inhouse-LLM/gpt-oss-120b       # Required - LLM model for analysis
LLM_URL=https://llmgw.eea.europa.eu      # Required - external litellm endpoint
LLM_API_KEY=your_api_key_here            # Required - API key for authentication
```

**Note:** Since litellm runs outside of this test framework, `LLM_MODEL`, `LLM_URL` and `LLM_API_KEY` are required when `ENABLE_LLM_ANALYSIS=true`. The framework will validate these settings on startup and raise an error if they're missing:

```
ValueError: LLM_MODEL must be set when ENABLE_LLM_ANALYSIS is true
ValueError: LLM_URL must be set when ENABLE_LLM_ANALYSIS is true
ValueError: LLM_API_KEY must be set when ENABLE_LLM_ANALYSIS is true
```

### Usage in Tests

Verify response quality during test execution:

```python
from chatbot_tests.config import settings
from chatbot_tests.llm_analysis import create_analyzer_from_settings

def test_response_quality(self, chatbot_page: ChatbotPage):
    response = chatbot_page.send_message_and_wait("What is air quality?")

    # Optional LLM verification (only runs if enabled)
    if settings.enable_llm_analysis:
        analyzer = create_analyzer_from_settings(settings)

        if analyzer:
            # Single LLM call for all verifications (returns JSON)
            with step("Verify response quality (LLM)"):
                verification = analyzer.verify_all(
                    "What is air quality?",
                    response.get_message(),
                    response.get_final_documents()
                )

                errors = []
                if not verification.answers_question:
                    errors.append(f"Doesn't answer: {verification.answers_question_explanation}")
                if not verification.not_vague:
                    errors.append(f"Too vague: {verification.not_vague_explanation}")
                if not verification.has_citations:
                    errors.append(f"Missing citations: {verification.has_citations_explanation}")

                assert not errors, f"LLM verification failed: {'; '.join(errors)}"
```

### Report Analysis

Analyze test reports with LLM insights:

```bash
# Analyze a single report with LLM
chatbot-tests analyze ./results.jsonl --llm

# Compare multiple runs with LLM
chatbot-tests compare ./run1.jsonl ./run2.jsonl --llm
```

The LLM will provide:
- **For analyze**: Overall quality assessment, insights, and recommendations
- **For compare**: Trend analysis, regression explanations, and improvement suggestions

### Available Verification Methods

- `verify_all()` - Single LLM call for all verifications, returns JSON with verdicts and explanations
- `analyze_test_report()` - Analyze test report with comprehensive insights
- `analyze_test_comparison()` - Analyze comparison of multiple test runs

## Key Implementation Details

### Bypassing Pytest Capture

Pytest's capture plugin intercepts stdout during test execution. To show real-time output:

1. Disable capture: `-p no:capture` in pytest args
2. Disable terminal plugin: `-p no:terminal` to suppress pytest's own output
3. Use `os.write(1, text.encode())` for direct file descriptor writes

This is handled in `runner.py`:
```python
pytest_args = [
    "-o", "addopts=-p no:terminal -p no:capture --tb=no",
    str(tests_dir),
]
```

### Step Context Tracking

The plugin tracks current test context via module-level state:

```python
# plugin.py
_current_plugin: Optional[ResultCollectorPlugin] = None

def log_step(name, outcome, message=None, duration_ms=None):
    if _current_plugin:
        _current_plugin._emit(TestEvent(
            "step",
            nodeid=_current_plugin._current_nodeid,
            name=_current_plugin._current_test_name,
            ...
        ))
```

The runner sets/clears this before/after pytest execution:
```python
set_current_plugin(plugin)
pytest.main(...)
set_current_plugin(None)
```

### Analysis Features

`analyze.py` provides:

- **Summary stats**: Total/passed/failed counts, pass rates
- **Step details**: Full step breakdown for each test
- **Marker grouping**: Tests organized by pytest marker
- **Multi-run comparison**:
  - Pass rate trends across runs
  - Regression detection (pass→fail)
  - Fix detection (fail→pass)
  - Flaky test detection (inconsistent results)

## Development

### Adding New Tests

1. Create test file in `tests/` directory
2. Use `step()` for logical test sections
3. Apply markers for organization
4. Use `chatbot_page` fixture for browser interaction

### Adding New Page Objects

1. Create class in `page_objects/`
2. Define locators as properties
3. Export from `page_objects/__init__.py`

### Extending Analysis

Add new analysis functions in `analyze.py`:
1. Define data extraction logic
2. Add print function for text output
3. Include in `to_dict()` for JSON output
4. Wire up in `main.py` CLI
