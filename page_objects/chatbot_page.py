"""Page object for chatbot Playwright interactions."""
import random

from dataclasses import dataclass
from typing import Optional
from pydantic import BaseModel
from contextlib import contextmanager

from playwright.sync_api import Page, Locator, Response, expect

from chatbot_tests.step import step

from .response import StreamedResponse


# @dataclass
# class ChatMessage:
#     """Represents a chat message."""

#     type: str  # 'user' or 'assistant'
#     content: str
#     sources: list[dict] = field(default_factory=list)
#     related_questions: list[str] = field(default_factory=list)
#     message_id: Optional[int] = None


# @dataclass
# class Source:
#     """Represents a source citation."""

#     title: str
#     url: str
#     source_type: str = "web"

class LLMException(Exception):
    """Exception raised when LLM fails to respond."""


class StarterMessage(BaseModel):
    """Represents a starter message."""
    name: str
    message: str


class Assistant(BaseModel):
    """Represents an assistant."""
    id: int
    name: str
    description: str
    starter_messages: list[StarterMessage]


class Messages(BaseModel):
    user: list[str]
    assistant: list[str]


class ChatbotPageSelectors:
    """CSS selectors for chatbot elements."""

    # Main containers
    CHAT_WINDOW = ".chat-window"
    MESSAGES_CONTAINER = ".messages"
    CONVERSATION_CONTAINER = ".conversation"
    EMPTY_STATE = ".empty-state"
    STARTER_CONTAINER = ".starter-messages-container"

    # Input elements
    CHAT_FORM = ".chat-form"
    CHAT_CONTROLS = ".chat-controls"
    TEXTAREA = "form .textarea-wrapper textarea"
    SUBMIT_BUTTON = "form .chat-right-actions [aria-label='Send']"
    CLEAR_CHAT_BUTTON = "[aria-label='Clear chat']"

    # Message elements
    USER_MESSAGES = ".comment:has(.circle.user)"
    ASSISTANT_MESSAGES = ".comment:has(.circle.assistant:not(.placeholder))"
    ASSISTANT_PLACEHOLDER = ".comment:has(.circle.assistant.placeholder)"

    # Multi tool (processing steps display)
    MULTI_TOOL = ".multi-tool-renderer"
    MULTI_TOOL_STREAMING = ".multi-tool-renderer.streaming"
    MULTI_TOOL_COMPLETE = ".multi-tool-renderer.complete"
    MULTI_TOOL_HEADER = ".tools-collapsed-header, .tools-summary-header"
    MULTI_TOOL_COUNT = ".tools-count"
    MULTI_TOOL_COUNT_VALUE = ".tools-count-value"
    MULTI_TOOL_COUNT_LABEL = ".tools-count-label"
    MULTI_TOOL_ITEM = ".tool-item-collapsed, .tool-item-expanded"
    MULTI_TOOL_ITEM_ACTIVE = ".tool-item-collapsed.active"
    MULTI_TOOL_ITEM_COMPLETED = ".tool-item-collapsed.completed"
    MULTI_TOOL_DONE_NODE = ".tool-done-node"
    MULTI_TOOL_EXPAND_CHEVRON = ".expand-chevron"

    # Tabs
    TABS = ".comment-tabs"
    TAB_MENU = ".ui.menu"
    TAB_MENU_ITEM_ACTIVE = ".item.active"
    TAB_MENU_ITEM_ANSWER = ".item.answer-tab"
    TAB_MENU_ITEM_SOURCES = ".item.sources-tab"
    TAB_CONTENT = ".ui.active.tab"
    ANSWER_CONTENT = ".answer-tab .answer-content"

    # Sources
    SOURCES_CONTAINER = ".answer-tab .sources"
    SOURCE_ITEMS = ".source"
    SHOW_ALL_SOURCES_BUTTON = ".show-all-sources-btn"
    SOURCES_SIDEBAR = ".sources-sidebar"

    # Related questions
    RELATED_QUESTIONS = ".chat-related-questions"
    RELATED_QUESTION_BUTTONS = ".relatedQuestionButton"

    # Feedback
    LIKE_BUTTON = ".message-actions [aria-label='Like']"
    DISLIKE_BUTTON = ".message-actions [aria-label='Dislike']"
    FEEDBACK_TOAST = ".message-actions .feedback-toast"
    FEEDBACK_MODAL = ".ui.modal, .feedback-modal"
    FEEDBACK_TEXTAREA = "textarea"
    FEEDBACK_NEGATIVE_REASON = ".reason-buttons"
    FEEDBACK_SUBMIT = ".ui.primary.button:has-text('Submit Feedback')"
    FEEDBACK_CANCEL = ".ui.button:has-text('Cancel')"

    # Halloumi (fact-check)
    FACT_CHECK_BUTTON = ".halloumi-feedback-button .claims-btn"
    VERIFY_CLAIMS_LOADING = ".verify-claims"
    HALLOUMI_MESSAGE = ".claim-message .content"
    CLAIMS = "span.claim"
    RETRY_BUTTON = "button:has-text('Retry Fact-check AI answer')"

    # Loading states
    ANSWER_LOADER = ".loader-container .loader"
    RELATED_QUESTIONS_LOADER = ".related-questions-loader"

    # Starter prompts
    STARTER_MESSAGES = ".starter-message"
    STARTER_MESSAGE_TITLE = ".starter-message-title"

    # Toggle controls
    FACT_CHECK_TOGGLE = ".quality-check-toggle"
    DEEP_RESEARCH_TOGGLE = ".deep-research-toggle"

    # Copy button
    COPY_BUTTON = "[aria-label='Copy']"

    # Error
    MESSAGE_ERROR = ".message-error .ui.error-message .error-content"


class ChatbotPage:
    """Page object for interacting with the chatbot."""
    DEFAULT_MESSAGE = "What are the main sources of air pollution?"
    assistant: Assistant
    block_id: int
    block_config: dict

    def __init__(self, page: Page, block_config: dict, assistant: Assistant):
        self.page = page
        self.block_config = block_config
        self.block_id = block_config.get('block_id')
        self.assistant = assistant
        self.selectors = ChatbotPageSelectors
        self.responses: list[StreamedResponse] = []
        self.related_questions: list[StreamedResponse] = []

        if not self.block_config:
            raise ValueError("Chatbot block configuration is missing")

        if not self.block_id:
            raise ValueError("Chatbot block ID is missing")

    # === Chatbot block configuration ===
    @property
    def show_starter_messages(self) -> bool:
        return (
            self.block_config.get("showAssistantPrompts") or
            self.block_config.get("enableStarterPrompts")
        )

    # === Containers ===
    @property
    def chat_window(self) -> Locator:
        data_attribute = f"[data-playwright-block-id=\"{self.block_id}\"]"
        return self.page.locator(self.selectors.CHAT_WINDOW + data_attribute)

    @property
    def messages(self) -> Locator:
        return self.chat_window.locator(self.selectors.MESSAGES_CONTAINER)

    @property
    def chat_form(self) -> Locator:
        return self.chat_window.locator(self.selectors.CHAT_FORM)

    @property
    def chat_controls(self) -> Locator:
        return self.chat_window.locator(self.selectors.CHAT_CONTROLS)

    @property
    def conversation(self) -> Locator:
        return self.messages.locator(self.selectors.CONVERSATION_CONTAINER)

    @property
    def empty_state(self) -> Locator:
        position = self.block_config.get('starterPromptsPosition')
        container = self.messages if position == "top" else self.chat_window
        return container.locator(self.selectors.EMPTY_STATE)

    @property
    def starter_container(self) -> Locator:
        return self.empty_state.locator(self.selectors.STARTER_CONTAINER)

    # === Input elements ===
    @property
    def textarea(self) -> Locator:
        return self.chat_form.locator(self.selectors.TEXTAREA)

    @property
    def submit_button(self) -> Locator:
        return self.chat_form.locator(self.selectors.SUBMIT_BUTTON)

    @property
    def clear_chat_button(self) -> Locator:
        return self.messages.locator(self.selectors.CLEAR_CHAT_BUTTON)

    # === Messages ===
    @property
    def user_messages(self) -> Locator:
        return self.conversation.locator(self.selectors.USER_MESSAGES)

    @property
    def assistant_messages(self) -> Locator:
        return self.conversation.locator(self.selectors.ASSISTANT_MESSAGES)

    @property
    def assistant_placeholder(self) -> Locator:
        return self.conversation.locator(self.selectors.ASSISTANT_PLACEHOLDER).last

    @property
    def starter_messages(self) -> Locator:
        return self.starter_container.locator(self.selectors.STARTER_MESSAGES)

    # === Multi tool (processing steps) ===
    @property
    def multi_tool(self) -> Locator:
        return self.assistant_messages.last.locator(self.selectors.MULTI_TOOL)

    @property
    def multi_tool_streaming(self) -> Locator:
        return self.assistant_messages.last.locator(self.selectors.MULTI_TOOL_STREAMING)

    @property
    def multi_tool_complete(self) -> Locator:
        return self.assistant_messages.last.locator(self.selectors.MULTI_TOOL_COMPLETE)

    @property
    def multi_tool_header(self) -> Locator:
        return self.multi_tool.locator(self.selectors.MULTI_TOOL_HEADER)

    @property
    def multi_tool_count(self) -> Locator:
        return self.multi_tool.locator(self.selectors.MULTI_TOOL_COUNT)

    @property
    def multi_tool_count_value(self) -> Locator:
        return self.multi_tool.locator(self.selectors.MULTI_TOOL_COUNT_VALUE)

    @property
    def multi_tool_items(self) -> Locator:
        return self.multi_tool.locator(self.selectors.MULTI_TOOL_ITEM)

    @property
    def multi_tool_active_item(self) -> Locator:
        return self.multi_tool.locator(self.selectors.MULTI_TOOL_ITEM_ACTIVE)

    @property
    def multi_tool_done_node(self) -> Locator:
        return self.multi_tool.locator(self.selectors.MULTI_TOOL_DONE_NODE)

    # === Tabs ===
    @property
    def tabs(self) -> Locator:
        return self.assistant_messages.last.locator(self.selectors.TABS)

    @property
    def tab_menu(self) -> Locator:
        return self.tabs.locator(self.selectors.TAB_MENU)

    @property
    def tab_menu_item_active(self) -> Locator:
        return self.tab_menu.locator(self.selectors.TAB_MENU_ITEM_ACTIVE)

    @property
    def tab_menu_item_answer(self) -> Locator:
        return self.tab_menu.locator(self.selectors.TAB_MENU_ITEM_ANSWER)

    @property
    def tab_menu_item_sources(self) -> Locator:
        return self.tab_menu.locator(self.selectors.TAB_MENU_ITEM_SOURCES)

    @property
    def tab_content(self) -> Locator:
        return self.tabs.locator(self.selectors.TAB_CONTENT)

    @property
    def answer_content(self) -> Locator:
        return self.assistant_messages.last.locator(self.selectors.ANSWER_CONTENT)

    # === Sources ===
    @property
    def sources_container(self) -> Locator:
        return self.assistant_messages.last.locator(self.selectors.SOURCES_CONTAINER)

    @property
    def source_items(self) -> Locator:
        return self.sources_container.locator(self.selectors.SOURCE_ITEMS)

    @property
    def show_all_sources_button(self) -> Locator:
        return self.sources_container.locator(self.selectors.SHOW_ALL_SOURCES_BUTTON)

    @property
    def sources_sidebar(self) -> Locator:
        return self.assistant_messages.last.locator(self.selectors.SOURCES_SIDEBAR)

    @property
    def sources_sidebar_items(self) -> Locator:
        return self.sources_sidebar.locator(self.selectors.SOURCE_ITEMS)

    # === Related questions ===
    @property
    def related_questions_container(self) -> Locator:
        return self.assistant_messages.last.locator(self.selectors.RELATED_QUESTIONS)

    @property
    def related_question_buttons(self) -> Locator:
        return self.related_questions_container.locator(self.selectors.RELATED_QUESTION_BUTTONS)

    # === Feedback ===
    @property
    def like_button(self) -> Locator:
        return self.assistant_messages.last.locator(self.selectors.LIKE_BUTTON)

    @property
    def dislike_button(self) -> Locator:
        return self.assistant_messages.last.locator(self.selectors.DISLIKE_BUTTON)

    @property
    def feedback_toast(self) -> Locator:
        return self.assistant_messages.last.locator(self.selectors.FEEDBACK_TOAST)

    @property
    def feedback_modal(self) -> Locator:
        return self.page.locator(self.selectors.FEEDBACK_MODAL)

    @property
    def feedback_textarea(self) -> Locator:
        return self.feedback_modal.locator(self.selectors.FEEDBACK_TEXTAREA)

    @property
    def feedback_negative_reason(self) -> Locator:
        return self.feedback_modal.locator(self.selectors.FEEDBACK_NEGATIVE_REASON)

    @property
    def feedback_submit_button(self) -> Locator:
        return self.feedback_modal.locator(self.selectors.FEEDBACK_SUBMIT)

    @property
    def feedback_cancel_button(self) -> Locator:
        return self.feedback_modal.locator(self.selectors.FEEDBACK_CANCEL)

    # === Halloumi (Quality Check) ===
    @property
    def fact_check_button(self) -> Locator:
        return self.assistant_messages.last.locator(self.selectors.FACT_CHECK_BUTTON)

    @property
    def verify_claims_loading(self) -> Locator:
        return self.assistant_messages.last.locator(self.selectors.VERIFY_CLAIMS_LOADING)

    @property
    def halloumi_message(self) -> Locator:
        return self.assistant_messages.last.locator(self.selectors.HALLOUMI_MESSAGE)

    @property
    def halloumi_claims(self) -> Locator:
        return self.assistant_messages.last.locator(self.selectors.CLAIMS)

    # @property
    # def halloumi_feedback(self) -> Locator:
    #     return self.assistant_messages.last.locator(self.selectors.HALLOUMI_FEEDBACK)

    @property
    def retry_button(self) -> Locator:
        return self.assistant_messages.last.locator(self.selectors.RETRY_BUTTON)

    # === Loading states ===
    @property
    def answer_loader(self) -> Locator:
        return self.assistant_placeholder.locator(self.selectors.ANSWER_LOADER)

    @property
    def related_questions_loader(self) -> Locator:
        return self.assistant_messages.last.locator(self.selectors.RELATED_QUESTIONS_LOADER)

    # === Toggle controls ===
    @property
    def fact_check_toggle(self) -> Locator:
        return self.chat_controls.locator(self.selectors.FACT_CHECK_TOGGLE)

    @property
    def deep_research_toggle(self) -> Locator:
        return self.chat_controls.locator(self.selectors.DEEP_RESEARCH_TOGGLE)

    # === Copy button ===
    @property
    def copy_button(self) -> Locator:
        return self.assistant_messages.last.locator(self.selectors.COPY_BUTTON)

    # === Error ===
    @property
    def message_error(self) -> Locator:
        return self.conversation.locator(self.selectors.MESSAGE_ERROR)

    # @property
    # def loader(self) -> Locator:
    #     return self.chat_window.locator(self.selectors.LOADER)

    # @property
    # def like_button(self) -> Locator:
    #     return self.chat_window.locator(self.selectors.LIKE_BUTTON)

    # @property
    # def dislike_button(self) -> Locator:
    #     return self.chat_window.locator(self.selectors.DISLIKE_BUTTON)

    # @property
    # def fact_check_button(self) -> Locator:
    #     return self.chat_window.locator(self.selectors.FACT_CHECK_BUTTON)

    # @property
    # def related_question_buttons(self) -> Locator:
    #     return self.chat_window.locator(
    #         self.selectors.RELATED_QUESTION_BUTTONS
    #     )

    # === Getters ===
    def get_response(self, assistant_message_id: int) -> Optional[StreamedResponse]:
        """Retrieve a stored response by assistant message ID."""
        for response in self.responses:
            if response.assistant_message_id == assistant_message_id:
                return response
        return None

    def get_predefined_message(self, index: int = 0, use_random: bool = False) -> tuple[str, int]:
        """Get predefined message without UI validation."""
        starter_messages = self._get_starter_messages()

        if not starter_messages:
            return self.DEFAULT_MESSAGE, -1

        selected_index = random.randint(0, len(starter_messages) - 1) if use_random else index

        if selected_index >= len(starter_messages):
            raise IndexError(f"Index {selected_index} out of range for {len(starter_messages)} messages")

        message = starter_messages[selected_index].message or self.DEFAULT_MESSAGE
        return message, selected_index

    def _get_starter_messages(self) -> list[StarterMessage]:
        """Internal: determine which message source to use."""
        if self.block_config.get("showAssistantPrompts"):
            return self.assistant.starter_messages or []
        elif self.block_config.get("enableStarterPrompts"):
            custom_starter_messages = self.block_config.get("starterPrompts", [])
            return [StarterMessage.model_validate(msg) for msg in custom_starter_messages]
        return []

    # === Validators ===
    def validate_predefined_message_in_ui(self, index: int, message: str | None = None) -> None:
        """Verify UI displays the correct predefined message."""
        expected_message = message or self.get_predefined_message(index)[0]
        message_locator = self.starter_messages.nth(index)

        expect(message_locator).to_be_visible()
        if self.block_config.get("showAssistantPrompts"):
            actual_message = message_locator.text_content()
            assert actual_message == expected_message, \
                f"UI mismatch: expected '{expected_message}', got '{actual_message}'"
        # We don't validate message_locator if enableStarterPrompts is True
        # because in that case the message is not displayed
        # only the title and the description of the custom message are displayed

    def validate_predefined_messages_in_ui(self) -> None:
        """Verify UI displays the correct predefined messages."""
        starter_messages = self._get_starter_messages()
        ui_starters_no = self.starter_messages.count()
        assert len(starter_messages) == ui_starters_no, \
            f"Number of messages mismatch: expected {len(starter_messages)}, got {ui_starters_no}"
        for index, starter_msg in enumerate(starter_messages):
            self.validate_predefined_message_in_ui(index, starter_msg.message)

    # === UI Validators ===
    def verify_empty_conversation(self):
        if self.show_starter_messages:
            expect(self.empty_state).to_be_visible()
        else:
            expect(self.empty_state).to_be_hidden()
        expect(self.user_messages.last).to_have_count(0)
        expect(self.assistant_messages.last).to_have_count(0)

    def verify_interactions_disabled(self):
        expect(self.clear_chat_button).to_be_visible()
        expect(self.submit_button).to_be_visible()
        expect(self.textarea).to_be_visible()
        expect(self.clear_chat_button, "Clear chat button should be disabled while response is streaming").to_be_disabled()
        expect(self.submit_button, "Submit button should be disabled while response is streaming").to_be_disabled()
        expect(self.textarea, "Textarea should be disabled while response is streaming").to_be_disabled()

    def verify_answer(self, response: StreamedResponse | None, description: str = ""):
        with step(description or "Verify assistant response"):
            if not response:
                raise Exception("No response received")
            assert response.assistant_message_id is not None, "Missing assistant message ID"
            assert response.get_message() != "", "Response should not be empty"
            expect(self.assistant_messages.last).to_be_visible()
            if response.error:
                expect(self.message_error).to_be_visible()
                expect(self.message_error).to_have_text(response.error)
                raise LLMException(response.error)
            assert len(response.chunks) > 0, "Response should have at least one chunk"
            assert response.stopped, "Response stream should have stopped"

    # === Processors ===
    def parse_response(self, response: Response) -> StreamedResponse:
        response.finished()  # Wait for body to complete

        body = response.body().decode('utf-8')

        if response.status != 200:
            raise Exception(f"Response status code: {response.status} | {body}")

        streamed_response = StreamedResponse.from_jsonl_body(body)

        self.responses.append(streamed_response)

        return streamed_response

    # === Actions ===
    @contextmanager
    def send_message(self, message: str = None, enter: bool = False):
        if message:
            self.textarea.fill(message)

        assert self.textarea.input_value(), "Textarea value should not be empty"

        class ResponseHolder:
            response_info = None
            value = None

        holder = ResponseHolder()

        with self.page.expect_response(
            lambda r: "/_da/chat/send-message" in r.url
        ) as response_info:
            if enter:
                self.textarea.press("Enter")
            else:
                self.submit_button.click()
            holder.response_info = response_info
            yield holder

        holder.value = self.parse_response(response_info.value)

    # === Actions ===
    @contextmanager
    def send_predefined_message(self, index: int, message: str | None = None):
        class ResponseHolder:
            response_info = None
            value = None

        holder = ResponseHolder()

        if index < 0 and message:
            with self.send_message(message) as response:
                yield holder
            holder.response_info = response.response_info
            holder.value = response.value
            return
        elif index < 0:
            raise ValueError("Predefined message index identifier must be non-negative")

        self.validate_predefined_message_in_ui(index, message)

        with self.page.expect_response(
            lambda r: "/_da/chat/send-message" in r.url
        ) as response_info:
            self.starter_messages.nth(index).click()
            holder.response_info = response_info
            yield holder

        holder.value = self.parse_response(response_info.value)

    @contextmanager
    def send_related_question(self, index: int):
        class ResponseHolder:
            response_info = None
            value = None

        holder = ResponseHolder()

        with self.page.expect_response(
            lambda r: "/_da/chat/send-message" in r.url
        ) as response_info:
            self.related_question_buttons.nth(index).click()
            holder.response_info = response_info
            yield holder

        holder.value = self.parse_response(response_info.value)

    @contextmanager
    def send_feedback(self):
        class ResponseHolder:
            response_info = None
            value = None

        holder = ResponseHolder()

        with self.page.expect_response(
            lambda r: "/_da/chat/create-chat-message-feedback" in r.url
        ) as response_info:
            self.feedback_submit_button.click()
            holder.response_info = response_info
            yield holder

    @contextmanager
    def wait_related_questions(self):
        class ResponseHolder:
            response_info = None
            value = None

        holder = ResponseHolder()

        with self.page.expect_response(
            lambda r: "/_rq/chat/send-message" in r.url
        ) as response_info:
            holder.response_info = response_info
            yield holder

        holder.value = self.parse_response(response_info.value)
