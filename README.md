# EEA Chatbot Tests

Playwright-based end-to-end test framework for [volto-eea-chatbot](https://github.com/eea/volto-eea-chatbot) with step-based logging, JSONL output, LLM-powered quality analysis, and PDF report generation.

## Features

- **Browser automation** with Playwright (Chromium, Firefox, WebKit)
- **Step-based test logging** with timed actions and JSONL output
- **Data-driven testing** via parametrized fixtures with validation rules
- **LLM quality verification** of chatbot responses (relevance, specificity, citations, information)
- **Halloumi fact-check** integration and scoring
- **Result analysis** with performance metrics, failure categorization, and health assessment
- **Multi-run comparison** with regression/fix/flaky detection and stability scoring
- **PDF report generation** with EEA branding and executive summaries

## Quick Start

### Prerequisites

- Python 3.10+
- A running volto-eea-chatbot instance

### Installation

```bash
pip install -e .
playwright install chromium
```

Or using Make:

```bash
make install
```

### Configuration

Copy the example config and edit it:

```bash
cp config.example.json config.json
```

Set the chatbot URL and desired options:

```json
{
  "chatbot_base_url": "https://www.eea.europa.eu",
  "chatbot_path": "/en/chatbot",
  "headless": true,
  "browser": "chromium",
  "timeout": 240000,
  "expect_timeout": 30000,
  "reports_dir": "./reports",
  "fixtures_dir": "./fixtures"
}
```

### Run Tests

```bash
# Run all tests
chatbot_tests run

# Run specific markers
chatbot_tests run -m basic

# Run with visible browser
chatbot_tests run --headed

# Limit fixture questions
chatbot_tests run --limit 5
```

### Analyze Results

```bash
# Summary with all sections
chatbot_tests analyze ./reports/test_run_*.jsonl --all

# Compare multiple runs
chatbot_tests compare ./reports/run1.jsonl ./reports/run2.jsonl
```

## CLI Reference

### `chatbot_tests run`

Run Playwright tests against a chatbot instance.

| Option | Description |
|--------|-------------|
| `-c, --config` | Path to JSON config file (default: `config.json`) |
| `-m, --marker` | Comma-separated test markers to filter |
| `--headed` | Show browser window (overrides config `headless`) |
| `--limit N` | Limit to first N fixture questions |
| `-o, --output` | Custom output file or directory |
| `--color` | Force ANSI color output |

### `chatbot_tests analyze`

Analyze a test run JSONL file.

| Option | Description |
|--------|-------------|
| `input` | JSONL file to analyze |
| `-c, --config` | Path to JSON config file |
| `--failures` | Show failure details grouped by category |
| `--steps` | Show step-by-step breakdown |
| `--by-marker` | Group results by pytest marker |
| `--performance` | Show duration metrics |
| `--insights` | Show auto-generated health insights |
| `--all` | Show all analysis sections |
| `--llm` | Run LLM analysis and generate PDF report |

### `chatbot_tests compare`

Compare multiple test runs for trends and regressions.

| Option | Description |
|--------|-------------|
| `files` | Two or more JSONL files to compare |
| `-c, --config` | Path to JSON config file |
| `--llm` | Run LLM comparison and generate PDF report |

## Configuration

Settings are loaded from (in priority order):

1. **Environment variables** (e.g., `CHATBOT_BASE_URL`)
2. **`.env` file**
3. **JSON config file** (`config.json` by default)

### Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `chatbot_base_url` | `http://localhost:3000` | Volto frontend URL |
| `chatbot_path` | `/chatbot` | Path to chatbot page |
| `headless` | `true` | Run browser headless |
| `browser` | `chromium` | Browser engine |
| `timeout` | `120000` | Default timeout (ms) |
| `expect_timeout` | `30000` | Assertion timeout (ms) |
| `reports_dir` | `./chatbot_tests/reports` | Output directory |
| `fixtures_dir` | `./fixtures` | Test fixtures directory |
| `pdf_font` | `null` | TTF font path for PDF Unicode support |
| `enable_llm_analysis` | `false` | Enable LLM quality analysis |
| `llm_model` | `Inhouse-LLM/gpt-oss-120b` | LLM model identifier |
| `llm_url` | `https://llmgw.eea.europa.eu` | LLM API endpoint |
| `llm_api_key` | `` | LLM API key |

## Test Structure

### Test Files

| File | Description |
|------|-------------|
| `tests/test_basic.py` | UI and integration tests (`@pytest.mark.always`) — always run |
| `tests/test_questions.py` | Data-driven question validation from `fixtures/*.json` |

### Fixtures (v2.0)

Test questions are defined in `fixtures/*.json` with a minimal format:

```json
{
  "version": "2.0.0",
  "default_validation": {
    "response": { "min_length": 100 },
    "sources": { "min_count": 1 },
    "llm": {
      "verify_answers_question": true,
      "verify_not_vague": true,
      "verify_citations": true,
      "verify_lack_information": true
    }
  },
  "default_feedback": true,
  "test_cases": [
    {
      "id": "Q-001",
      "priority": "high",
      "question": "What is the current state of air quality in Europe?",
      "markers": ["air_quality"]
    }
  ]
}
```

Each test case inherits `default_validation` and can override specific fields.

### Markers

| Marker | Description |
|--------|-------------|
| `always` | Tests that run regardless of `-m` filter |
| `basic` | Basic chatbot functionality |
| `halloumi` | Halloumi fact-check tests |
| `feedback` | Feedback functionality |
| `follow_up` | Follow-up query tests |
| `high` / `medium` / `low` | Priority (auto-added from fixture `priority`) |

Topic markers (e.g., `satellite`, `copernicus`, `ai`) are dynamically applied from fixture `markers` arrays.

## LLM Quality Analysis

When enabled, the framework uses an external LLM to evaluate chatbot responses across four dimensions:

| Dimension | Pass | Fail |
|-----------|------|------|
| Information | Has sufficient information | Lacks information |
| Relevance | On-topic | Off-topic |
| Specificity | Not vague | Too vague |
| Citations | Has citations | Missing citations |

### Setup

Add LLM settings to `config.json`:

```json
{
  "enable_llm_analysis": true,
  "llm_model": "Inhouse-LLM/gpt-oss-120b",
  "llm_url": "https://llmgw.eea.europa.eu",
  "llm_api_key": "your_api_key"
}
```

### Report Generation

```bash
# Single run analysis with PDF
chatbot_tests analyze ./results.jsonl --llm

# Multi-run comparison with PDF
chatbot_tests compare ./run1.jsonl ./run2.jsonl --llm
```

Generated reports include executive summaries, risk assessments, trend analysis, and actionable recommendations.

## Output Files

Test runs produce files in the `reports/` directory:

| File | Description |
|------|-------------|
| `test_run_<timestamp>.jsonl` | Raw test execution events |
| `analysis_<timestamp>.json` | Analyzed test results |
| `analysis_<timestamp>.pdf` | LLM-generated analysis report |
| `comparison_<timestamp>.json` | Multi-run comparison data |
| `comparison_<timestamp>.pdf` | LLM-generated comparison report |

## Make Targets

```
make install    Install dependencies and Playwright browser
make run        Run all chatbot tests
make headed     Run tests with visible browser
make analyze    Analyze a test report (FILE=path)
make compare    Compare test runs (FILES="path1 path2")
make help       Show available targets
```

## Documentation

- **[GUIDE.md](GUIDE.md)** — Detailed guide for writing tests, page object reference, fixture format, LLM verification patterns
- **[CLAUDE.md](CLAUDE.md)** — Developer reference with architecture, implementation details, and module documentation

## Copyright and license

The Initial Owner of the Original Code is European Environment Agency (EEA).
All Rights Reserved.

See [LICENSE.md](https://github.com/eea/eea-website-backend/blob/master/LICENSE.md) for details.

## Funding

[European Environment Agency (EU)](http://eea.europa.eu)