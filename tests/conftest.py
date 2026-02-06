import json

import pytest
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from playwright.sync_api import Page, expect

from chatbot_tests.config import get_settings
from chatbot_tests.page_objects import (
    Assistant,
    ChatbotPageSelectors,
    ChatbotPage,
)
from chatbot_tests.step import step

tests_order = {
    "basic": 1
}


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_all_fixtures():
    """Load all JSON files from fixtures directory.

    Supports two fixture formats:
    1. Full validation per test case (legacy)
    2. Minimal format with default_validation at file level

    For minimal format, each test case inherits from default_validation
    and can override specific fields.
    """
    settings = get_settings()
    all_test_cases = []

    for json_file in settings.fixtures_path.glob("*.json"):
        with open(json_file) as f:
            data = json.load(f)

        # Get this file's thresholds and default validation
        file_thresholds = data.get("validation_thresholds", {})
        default_validation = data.get("default_validation", {})
        default_feedback = data.get("default_feedback", False)

        # Attach thresholds and merge validation to each test case
        for tc in data.get("test_cases", []):
            tc["_source"] = json_file.name
            tc["_thresholds"] = file_thresholds

            # Merge default_validation with test case specific validation
            tc_validation = tc.get("validation", {})
            tc_feedback = tc.get("feedback")
            tc["validation"] = deep_merge(default_validation, tc_validation)
            tc["feedback"] = tc_feedback or default_feedback

            all_test_cases.append(tc)

    return all_test_cases


@pytest.fixture
def context(browser):
    """Create a new browser context with clipboard permissions."""
    context = browser.new_context(
        permissions=["clipboard-read", "clipboard-write"],
        viewport={ 'width': 1650, 'height': 950 }
    )
    yield context
    context.close()


@pytest.fixture
def data(request):
    """Fixture that provides each test case data to tests."""
    return request.param


@pytest.fixture(scope="session", autouse=True)
def settings():
    """Get test settings."""
    return get_settings()


@pytest.fixture(scope="session", autouse=True)
def configure_expect_timeout(settings):
    """Set global expect timeout from settings."""
    expect.set_options(timeout=settings.expect_timeout)


@pytest.fixture
def page(context, settings):
    page = context.new_page()
    page.set_default_timeout(settings.timeout)  # For actions/responses
    yield page
    page.close()


@pytest.fixture
def chatbot_page(page: Page, settings) -> ChatbotPage:
    """Create a ChatbotPage instance and navigate to chatbot."""
    assistant = None
    url = urlparse(settings.chatbot_url)
    url = url._replace(query=urlencode({
        **parse_qs(url.query),
        "playwright": "yes",
    }))
    page.goto(urlunparse(url))

    with page.expect_response(
        lambda r: "/_da/persona/" in r.url
    ) as response_info:
        response = response_info.value
        response.finished()
        assistant = Assistant.model_validate(response.json())

    page.locator(ChatbotPageSelectors.CHAT_WINDOW).wait_for(state="visible")

    block_config = page.evaluate("""
        () => window.__EEA_CHATBOT_TEST_CONFIG__
    """)
    chatbot_page = ChatbotPage(page, block_config, assistant)

    with step("Verify chat window container renders on page"):
        expect(chatbot_page.chat_window).to_be_visible()

    with step("Verify assistant is loaded"):
        assert assistant is not None, "Assistant information is missing, chatbot block may be misconfigured"
        assert block_config.get("assistant") is not None, "Assistant ID is missing in chatbot block config"
        assert int(block_config.get("assistant")) == assistant.id, "Assistant ID mismatch with chatbot block config"

    return chatbot_page


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--limit",
        action="store",
        default=None,
        type=int,
        help="Limit to first N questions from fixtures (does not affect other tests)",
    )


def pytest_generate_tests(metafunc):
    """Dynamically generate test cases with limit applied at collection time."""
    if "data" in metafunc.fixturenames:
        all_cases = load_all_fixtures()
        limit = metafunc.config.getoption("--limit")

        if limit is not None:
            all_cases = all_cases[:limit]

        metafunc.parametrize("data", all_cases, ids=lambda q: q["id"])


def pytest_collection_modifyitems(config, items):
    """Apply markers from fixture data to test items.

    Tests marked with 'always' get the selected marker applied.
    Tests with the 'data' fixture get markers from the test case data.
    """
    selected_marker = config.getoption("-m")

    for item in items:
        # Tests with 'always' marker run regardless of marker filter
        if selected_marker and "always" in item.keywords:
            item.add_marker(selected_marker)
            continue

        # Only process tests that use the 'data' fixture
        if not (hasattr(item, "callspec") and "data" in item.callspec.params):
            continue

        data = item.callspec.params["data"]

        # Apply markers from the question's markers list
        for marker_name in data.get("markers", []):
            item.add_marker(getattr(pytest.mark, marker_name))

        # Auto-add priority as a marker (high, medium, low)
        priority = data.get("priority")
        if priority:
            item.add_marker(getattr(pytest.mark, priority))
