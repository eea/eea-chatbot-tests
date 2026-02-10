# Chatbot Tests Framework

Playwright-based test automation framework for volto-eea-chatbot validation with step-based logging, JSONL output, LLM-powered analysis, and PDF report generation.

## Project Structure

All source files live at the project root. The `chatbot_tests.` import prefix is configured via `pyproject.toml` package-dir mapping.

```
eea-chatbot-tests/
├── __init__.py
├── main.py              # CLI entry point (run/analyze/compare)
├── runner.py            # Test orchestration, pytest invocation
├── config.py            # Settings via pydantic-settings
├── plugin.py            # Pytest plugin for event collection
├── step.py              # step(), info(), llm_verdict() helpers
├── output.py            # JSONL file writer
├── console.py           # ANSI color output, bypasses pytest capture
├── analyze.py           # Result analysis, comparison, metrics
├── llm_analysis.py      # LLM-based quality verification and reports
├── utils.py             # Marker helpers, quality score stages
├── styles.css           # PDF styling
├── eea-logo.svg         # EEA logo for PDF headers
├── page_objects/
│   ├── __init__.py
│   ├── chatbot_page.py  # ChatbotPage with 40+ locators/methods
│   └── response.py      # StreamedResponse model
├── tests/
│   ├── conftest.py      # Fixtures, hooks, parametrization
│   ├── test_basic.py    # UI/integration tests (@pytest.mark.always)
│   └── test_questions.py # Data-driven question validation
├── fixtures/
│   └── golden_dataset.json  # Test questions dataset
├── fonts/               # Custom fonts for PDF generation
├── reports/             # Generated JSONL, JSON, and PDF files
├── pyproject.toml
├── pytest.ini
├── requirements.txt
├── config.json          # Active config (not committed)
├── config.example.json  # Config template
├── Makefile
├── CLAUDE.md            # This file (developer reference)
├── GUIDE.md             # Test writing guide
└── README.md            # User-facing documentation
```

## CLI Usage

The CLI entry point is `chatbot_tests` (defined in `pyproject.toml`). All commands load `config.json` from the current directory by default. Use `-c` to specify a different config file.

```bash
# Run tests
chatbot_tests run                             # Run all tests
chatbot_tests run -m basic                    # Run tests with 'basic' marker
chatbot_tests run -m "basic,high"             # Multiple markers (OR logic)
chatbot_tests run --headed                    # Show browser (overrides config headless)
chatbot_tests run --limit 5                   # Limit to first 5 fixture questions
chatbot_tests run -o ./out.jsonl              # Custom output path
chatbot_tests run -c other.json               # Use different config file
chatbot_tests run --color                     # Force color output

# Analyze results
chatbot_tests analyze ./results.jsonl                 # Summary
chatbot_tests analyze ./results.jsonl --failures      # Show failure details
chatbot_tests analyze ./results.jsonl --steps         # Show all steps
chatbot_tests analyze ./results.jsonl --by-marker     # Group by marker
chatbot_tests analyze ./results.jsonl --performance   # Duration metrics
chatbot_tests analyze ./results.jsonl --insights      # Auto-generated insights
chatbot_tests analyze ./results.jsonl --all           # All sections at once
chatbot_tests analyze ./results.jsonl --llm           # LLM analysis + PDF

# Compare multiple runs
chatbot_tests compare ./run1.jsonl ./run2.jsonl ./run3.jsonl
chatbot_tests compare ./run1.jsonl ./run2.jsonl --llm  # LLM comparison + PDF
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
| `main.py` | argparse CLI with run/analyze/compare subcommands, JSON/PDF output |
| `runner.py` | TestRunner class, builds pytest args, coordinates execution |
| `config.py` | Pydantic settings from JSON config / environment / .env |
| `plugin.py` | Pytest plugin hooks, emits TestEvent via callbacks |
| `step.py` | `step()` context manager with timing, `info()` for untimed logging, `llm_verdict()` for LLM quality assessments |
| `output.py` | JSONLWriter for real-time streaming to file |
| `console.py` | ANSI color output, bypasses pytest capture with `os.write(1, ...)` |
| `analyze.py` | Parses JSONL, calculates stats, detects regressions, generates insights |
| `llm_analysis.py` | LLM-based response quality verification and report analysis via LiteLLM |
| `utils.py` | Marker helpers, quality score stage definitions |

## Writing Tests

### Using Steps

Steps have types that indicate their timing expectations:

| Type | Description | Timing Expected |
|------|-------------|-----------------|
| `action` | User interaction, browser action, API call | Yes |
| `info` | Logging results/status, informational message | No |
| `wait` | Waiting for async operations | Yes |
| `llm_verdict` | LLM quality assessment of chatbot response | No |

```python
from chatbot_tests.step import step, info, llm_verdict
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

        # Log LLM quality verdicts (step_type="llm_verdict", no timing)
        llm_verdict(f"LLM analysis: answer on-topic - {explanation}")
```

**`step()` context manager** (for timed actions):
- Measures execution time
- Logs pass/fail with duration to JSONL
- Sets `step_type="action"`
- Supports `continue_on_failure=True` for non-critical checks
- Re-raises exceptions on failure

**`info()` function** (for informational messages):
- No timing measurement
- Logs message to JSONL with `step_type="info"`
- Use for status messages, configuration info

**`llm_verdict()` function** (for LLM quality assessments):
- No timing measurement
- Logs message to JSONL with `step_type="llm_verdict"`
- Use for external LLM quality evaluations of chatbot responses
- Tracked separately in analysis for verdict pass rates

### Markers

Defined in `pytest.ini`:

```
always      # Tests that always run regardless of marker filter
basic       # Basic chatbot functionality
halloumi    # Halloumi fact-check tests
feedback    # Feedback functionality
follow_up   # Follow-up query tests
high        # High priority (auto-added from fixture priority)
medium      # Medium priority
low         # Low priority
```

Additional markers are dynamically applied from fixture `markers` arrays (e.g., `satellite`, `security`, `copernicus`, `ai`, `policy`).

Run specific markers: `chatbot_tests run -m basic,high`

### Fixtures

Available fixtures in `conftest.py`:

- `chatbot_page` - ChatbotPage instance, navigated to chatbot URL with `?playwright=yes` param
- `settings` - Settings instance with configuration (session-scoped)
- `data` - Parametrized test case data from JSON fixtures
- `context` - Browser context with clipboard permissions and 1650x950 viewport
- `page` - Playwright page with configured timeout

### Test Files

| File | Purpose |
|------|---------|
| `test_basic.py` | UI/integration tests marked `@pytest.mark.always` — run regardless of fixture data |
| `test_questions.py` | Data-driven question validation — parametrized from `fixtures/*.json` |

## Fixture Format (v2.0)

```json
{
  "version": "2.0.0",
  "validation_thresholds": { ... },
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
      "priority": "medium",
      "question": "How does SatCen use AI...",
      "markers": ["satellite", "security"],
      "validation": { }
    }
  ]
}
```

Each test case inherits `default_validation` and can override specific fields via deep merge.

## JSONL Format

Each line is a JSON object with `event_type`:

```jsonl
{"event_type": "session_start", "timestamp": "..."}
{"event_type": "test_start", "nodeid": "...", "name": "...", "markers": [...]}
{"event_type": "step", "step_name": "...", "outcome": "passed", "duration_ms": 123, "step_type": "action"}
{"event_type": "step", "step_name": "INFO: ...", "outcome": "passed", "step_type": "info"}
{"event_type": "step", "step_name": "LLM analysis: ...", "outcome": "passed", "step_type": "llm_verdict"}
{"event_type": "test_end", "nodeid": "...", "outcome": "passed", "duration_seconds": 45.2}
{"event_type": "session_end", "outcome": "passed"}
```

Step events include a `step_type` field:
- `"action"`: Timed browser/API actions (duration_ms is measured)
- `"info"`: Informational logging (duration_ms is null — this is expected)
- `"wait"`: Async waits (duration_ms is measured)
- `"llm_verdict"`: LLM quality assessments (duration_ms is null — this is expected)

## Configuration

Settings can be loaded from:
1. **JSON config file** (via `--config` / `-c` CLI option, defaults to `config.json`)
2. **Environment variables**
3. **`.env` file**

Priority: Environment variables > .env file > JSON config file defaults

### JSON Config File

```json
{
  "chatbot_base_url": "https://www.eea.europa.eu",
  "chatbot_path": "/en/chatbot",
  "headless": true,
  "browser": "chromium",
  "timeout": 240000,
  "expect_timeout": 30000,
  "reports_dir": "./reports",
  "fixtures_dir": "./fixtures",
  "enable_llm_analysis": false,
  "llm_model": "Inhouse-LLM/gpt-oss-120b",
  "llm_url": "https://llmgw.eea.europa.eu",
  "llm_api_key": ""
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
| `TIMEOUT` | `120000` | Default timeout for operations (ms) |
| `EXPECT_TIMEOUT` | `30000` | Default timeout for expect assertions (ms) |
| `REPORTS_DIR` | `./chatbot_tests/reports` | Output directory for reports |
| `FIXTURES_DIR` | `./fixtures` (relative to package) | Test fixtures directory |
| `PDF_FONT` | `null` | Path to TTF font for PDF generation (Unicode/emoji support) |
| `ENABLE_LLM_ANALYSIS` | `false` | Enable LLM-based analysis |
| `LLM_MODEL` | `Inhouse-LLM/gpt-oss-120b` | LLM model for analysis |
| `LLM_URL` | `https://llmgw.eea.europa.eu` | LLM API endpoint |
| `LLM_API_KEY` | `` | API key for authentication |

## LLM-Based Analysis

The framework supports optional LLM-based analysis for:
1. **In-test verification** — Verify response quality during test execution
2. **Report analysis** — Analyze test reports with insights and PDF generation
3. **Run comparison** — Compare multiple test runs with recommendations and PDF

### Setup

Configure in `config.json` or `.env`:

```json
{
  "enable_llm_analysis": true,
  "llm_model": "Inhouse-LLM/gpt-oss-120b",
  "llm_url": "https://llmgw.eea.europa.eu",
  "llm_api_key": "your_api_key_here"
}
```

All three LLM settings are required when `enable_llm_analysis` is `true`. The framework validates on startup:

```
ValueError: LLM_MODEL must be set when ENABLE_LLM_ANALYSIS is true
ValueError: LLM_URL must be set when ENABLE_LLM_ANALYSIS is true
ValueError: LLM_API_KEY must be set when ENABLE_LLM_ANALYSIS is true
```

### In-Test Verification

The `LLMAnalyzer.verify_answer()` method performs a single LLM call that evaluates all dimensions at once, returning a `ResponseVerification` object:

```python
from chatbot_tests.config import get_settings
from chatbot_tests.llm_analysis import create_analyzer_from_settings

settings = get_settings()
if settings.enable_llm_analysis:
    analyzer = create_analyzer_from_settings(settings)
    if analyzer:
        verification = analyzer.verify_answer(
            question, response.get_message(), response.get_final_documents()
        )
        # verification.lack_information (bool + explanation)
        # verification.answers_question (bool + explanation)
        # verification.not_vague (bool + explanation)
        # verification.has_citations (bool + explanation)
```

### LLM Verdict Dimensions

| Dimension | Pass | Fail |
|-----------|------|------|
| Relevance | `answer on-topic` | `answer off-topic` |
| Specificity | `answer not vague` | `answer too vague` |
| Citations | `answer has citations` | `answer missing citations` |
| Information | `answer has information` | `answer lacks information` |

### Report Analysis

```bash
# Single report — generates JSON + PDF with LLM insights
chatbot_tests analyze ./results.jsonl --llm

# Multi-run comparison — generates JSON + PDF with trend analysis
chatbot_tests compare ./run1.jsonl ./run2.jsonl --llm
```

PDF reports include EEA logo header, run metadata, executive summary, risk assessment, and actionable recommendations.

## Analysis Features

`analyze.py` provides:

- **Summary stats**: Total/passed/failed counts, pass rates, health status
- **Performance metrics**: Avg test/step duration, slowest/fastest tests (excludes info and llm_verdict steps)
- **Failure categorization**: Timeout, UI Visibility, LLM Error, etc.
- **Step details**: Full step breakdown for each test
- **Marker grouping**: Tests organized by pytest marker
- **LLM verdict extraction**: Per-dimension pass rates from both llm_verdict and info step types (backward compat)
- **Auto-generated insights**: Health assessment, patterns, recommendations
- **Warnings detection**: Tests that pass overall but have failed steps

`compare_runs()` provides:

- **Executive summary table**: First vs last run with colored deltas
- **Pass rate trends** across runs
- **Regression detection** (pass→fail)
- **Fix detection** (fail→pass)
- **Flaky test detection** (inconsistent results)
- **LLM verdict trends**: Per-dimension pass rates across runs
- **Stability score** (1-10)

## Key Implementation Details

### Bypassing Pytest Capture

Pytest's capture plugin intercepts stdout during test execution. To show real-time output:

1. Disable capture: `-p no:capture` in pytest args (via `pytest.ini`)
2. Disable terminal plugin: `-p no:terminal` to suppress pytest's own output
3. Use `os.write(1, text.encode())` for direct file descriptor writes

### Step Context Tracking

The plugin tracks current test context via module-level state:

```python
# plugin.py
_current_plugin: Optional[ResultCollectorPlugin] = None

def log_step(name, outcome, message=None, duration_ms=None):
    if _current_plugin:
        _current_plugin._emit(TestEvent(...))
```

The runner sets/clears this before/after pytest execution:
```python
set_current_plugin(plugin)
pytest.main(...)
set_current_plugin(None)
```

### Chatbot Page Setup

The `chatbot_page` fixture:
1. Adds `?playwright=yes` query param to chatbot URL
2. Intercepts the `/_da/persona/` response to get `Assistant` info
3. Reads `window.__EEA_CHATBOT_TEST_CONFIG__` for block config / feature detection
4. Verifies assistant ID matches block config

### Dynamic Marker Application

In `conftest.py::pytest_collection_modifyitems`:
- Tests with `@pytest.mark.always` get the selected marker applied so they always run
- Fixture-driven tests get markers from the test case's `markers` array
- Priority markers (`high`, `medium`, `low`) are auto-added from fixture `priority` field

## Development

### Adding New Tests

1. Create test file in `tests/` directory
2. Use `step()` for logical test sections
3. Apply markers for organization
4. Use `chatbot_page` fixture for browser interaction
5. See `GUIDE.md` for detailed examples

### Adding New Questions

1. Add test case to `fixtures/golden_dataset.json` or create a new fixture file
2. Include appropriate `markers` for filtering
3. Set `priority` (high/medium/low) for selective runs
4. Configure `validation` sections (inherits from `default_validation`)
5. Run: `chatbot_tests run -m questions`

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
