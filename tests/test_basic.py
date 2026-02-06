"""End-to-end tests for EEA Chatbot functionality."""

import pytest
import json
from playwright.sync_api import expect

from chatbot_tests.page_objects import ChatbotPage
from chatbot_tests.page_objects.chatbot_page import ChatbotPageSelectors
from chatbot_tests.step import step, info
from chatbot_tests.utils import quality_check_stages


@pytest.mark.basic
@pytest.mark.ui
class TestChatbotUI:
    """UI tests that verify the chatbot interface renders correctly."""

    def test_initial_ui_state_and_configuration(self, chatbot_page: ChatbotPage):
        """Verify the chatbot UI loads correctly with all initial elements.

        Tests: chat window, input, send button, empty state, assistant, starter messages.
        """

        quality_check = chatbot_page.block_config.get("qualityCheck")
        deep_research = chatbot_page.block_config.get("deepResearch")

        # === Core UI Elements ===
        with step("Verify message input textarea is ready for input"):
            expect(chatbot_page.textarea).to_be_visible()

        with step("Verify send button is present in chat interface"):
            expect(chatbot_page.submit_button).to_be_visible()

        # === Empty State ===
        if chatbot_page.show_starter_messages:
            with step("Verify empty state placeholder displays before conversation"):
                expect(chatbot_page.empty_state).to_be_visible()

            with step("Verify predefined starter messages section renders in empty state"):
                expect(chatbot_page.starter_container).to_be_visible()
                chatbot_page.validate_predefined_messages_in_ui()
        else:
            info("INFO: Chatbot block is configured to not show starter messages")

        # === Assistant ===
        with step("Verify assistant title matches configured name"):
            title = chatbot_page.messages.locator("h2")
            expect(title).to_have_text(chatbot_page.assistant.name)

        with step("Verify assistant description matches configured text"):
            description = chatbot_page.messages.locator("p")
            expect(description).to_have_text(chatbot_page.assistant.description)

        # === Fact-Check Toggle (optional - may not be configured) ===
        if quality_check == "ondemand_toggle":
            info("INFO: Chatbot block is configured to show a Halloumi quality fact-check toggle button, ON by default")
            with step("Verify Halloumi quality fact-check toggle changes state on click"):
                expect(chatbot_page.fact_check_toggle).to_be_visible()
                checkbox = chatbot_page.fact_check_toggle.locator(".ui.checkbox")
                input = chatbot_page.fact_check_toggle.locator("#quality-check-toggle")
                initial_fc_checked = input.is_checked()
                assert initial_fc_checked, "Fact-check toggle should be checked by default"
                checkbox.click()
                new_fc_checked = input.is_checked()
                assert new_fc_checked != initial_fc_checked, f"Fact-check toggle unchanged after click (was: {initial_fc_checked}, now: {new_fc_checked})"
                checkbox.click()  # Reset to original
        elif quality_check == "ondemand":
            info("INFO: Chatbot block is configured to show a Halloumi quality fact-check trigger button, after assistant response")
        elif quality_check == "enabled":
            info("INFO: Chatbot block is configured to always do Halloumi quality fact-check after assistant response")
        else:
            info("INFO: Chatbot block is configured to never do Halloumi quality fact-check after assistant response")

        # === Deep Research Toggle (optional - may not be configured) ===
        if deep_research in ["user_on", "user_off"]:
            on_or_off = "ON" if deep_research == "user_on" else "OFF"
            info(f"INFO: Chatbot block is configured to show a deep research toggle button, {on_or_off} by default")
            with step("Verify deep research toggle changes state on click"):
                expect(chatbot_page.deep_research_toggle).to_be_visible()
                checkbox = chatbot_page.deep_research_toggle.locator(".ui.checkbox")
                input = chatbot_page.deep_research_toggle.locator("#deep-research-toggle")
                initial_dr_checked = input.is_checked()
                if on_or_off == "ON":
                    assert initial_dr_checked, "Deep research toggle should be checked by default"
                else:
                    assert not initial_dr_checked, "Deep research toggle should be unchecked by default"
                checkbox.click()
                new_dr_checked = input.is_checked()
                assert new_dr_checked != initial_dr_checked, f"Deep research toggle unchanged after click (was: {initial_dr_checked}, now: {new_dr_checked})"
                checkbox.click()  # Reset to original
        elif deep_research == "always_on":
            info("INFO: Chatbot block is configured to always go into deep research")
            with step("Verify deep research is always enabled"):
                expect(chatbot_page.deep_research_toggle).to_be_visible()
                expect(chatbot_page.deep_research_toggle).to_have_text("Deep research on")
        else:
            info("INFO: Chatbot block is configured to never go into deep research")

    def test_user_input_interactions(self, chatbot_page: ChatbotPage):
        """Verify user can interact with input elements.

        Tests: typing in textarea, empty/whitespace blocked, clicking starter prompt, message appears.
        """

        # === Empty Message Validation ===
        with step("Verify send button is disabled when textarea is empty"):
            expect(chatbot_page.textarea).to_have_value("")
            is_disabled = chatbot_page.submit_button.is_disabled()
            assert is_disabled, "Send button should be disabled when textarea is empty"

        with step("Verify send button remains disabled for whitespace-only input"):
            chatbot_page.textarea.fill("   ")
            is_disabled = chatbot_page.submit_button.is_disabled()
            assert is_disabled, "Send button should be disabled for whitespace-only input"
            chatbot_page.textarea.fill("")

        # === Textarea Input ===
        test_message = "What is air quality?"
        first_line = "First line"
        second_line = "Second line"

        with step("Type a message in the textarea"):
            chatbot_page.textarea.fill(test_message)

        with step("Verify textarea contains the typed message"):
            expect(chatbot_page.textarea).to_have_value(test_message)

        with step("Verify send button is enabled when textarea has content"):
            is_disabled = chatbot_page.submit_button.is_disabled()
            assert not is_disabled, "Send button should be enabled when textarea has content"

        with step("Verify clear chat button is hidden"):
            expect(chatbot_page.clear_chat_button).to_be_hidden()

        with step(f"Send typed message by submit button: {test_message}"):
            with chatbot_page.send_message(test_message) as response:
                chatbot_page.verify_interactions_disabled()
                assert chatbot_page.textarea.input_value() == "", "Textarea should be cleared after message is sent"

        response = response.value
        chatbot_page.verify_answer(response)

        with step("Verify user message appears with correct text"):
            expect(chatbot_page.user_messages.last).to_be_visible()
            expect(chatbot_page.user_messages.last).to_contain_text(test_message)

        with step("Verify clear chat button is enabled"):
            expect(chatbot_page.clear_chat_button).to_be_visible()
            is_disabled = chatbot_page.clear_chat_button.is_disabled()
            assert not is_disabled, "Clear chat button should be enabled after assistant response completes"

        with step("Click clear chat button"):
            chatbot_page.clear_chat_button.click()

        with step("Verify conversation is cleared"):
            chatbot_page.verify_empty_conversation()

        # === Multiline Input ===

        with step("Verifying multiline input, type first line"):
            chatbot_page.textarea.fill(first_line)

        with step("Press Shift+Enter to create newline"):
            chatbot_page.textarea.press("Shift+Enter")

        with step("Type second line of message"):
            chatbot_page.textarea.type(second_line)

        with step("Verify textarea contains multiline text"):
            expected_text = f"{first_line}\n{second_line}"
            actual_value = chatbot_page.textarea.input_value()
            assert expected_text == actual_value, "Textarea should contain multiline text"
            assert first_line in actual_value, "First line should be present"
            assert second_line in actual_value, "Second line should be present"

        with step("Verify message was not sent"):
            assert chatbot_page.user_messages.count() == 0, "Message should not be sent on Shift+Enter"

        with step("Send multiline message via Enter key"):
            with chatbot_page.send_message(enter=True) as response:
                chatbot_page.verify_interactions_disabled()

        response = response.value
        chatbot_page.verify_answer(response)

        with step("Verify multiline user message is displayed"):
            expect(chatbot_page.user_messages.last).to_be_visible()
            user_msg_text = chatbot_page.user_messages.last.text_content()
            assert first_line in user_msg_text, "First line should be in sent message"

        with step("Click clear chat button"):
            chatbot_page.clear_chat_button.click()

        with step("Verify conversation is cleared"):
            chatbot_page.verify_empty_conversation()

        # === Starter messages Click ===
        if chatbot_page.show_starter_messages:
            with step("Capture first starter prompt text"):
                starter = chatbot_page.starter_messages.first
                prompt_text = starter.text_content()

            with step(f"Send message by clicking predefined starter prompt: {prompt_text}"):
                with chatbot_page.send_predefined_message(0) as response:
                    chatbot_page.verify_interactions_disabled()

            response = response.value
            chatbot_page.verify_answer(response)

            with step("Verify empty state disappears after interaction"):
                expect(chatbot_page.empty_state).to_be_hidden()

            with step("Verify user message appears with correct text"):
                expect(chatbot_page.user_messages.last).to_be_visible()
                expect(chatbot_page.user_messages.last).to_contain_text(prompt_text)

            with step("Verify clear chat button is enabled"):
                expect(chatbot_page.clear_chat_button).to_be_visible()
                is_disabled = chatbot_page.clear_chat_button.is_disabled()
                assert not is_disabled, "Clear chat button should be enabled after assistant response completes"

            with step("Click clear chat button"):
                chatbot_page.clear_chat_button.click()

            with step("Verify conversation is cleared"):
                chatbot_page.verify_empty_conversation()

        with step("Type a message in the textarea"):
            chatbot_page.textarea.fill(test_message)

        with step(f"Send typed message by pressing ENTER key: {test_message}"):
            with chatbot_page.send_message(test_message, enter=True) as response:
                chatbot_page.verify_interactions_disabled()

        response = response.value
        chatbot_page.verify_answer(response)

        with step("Verify textarea is cleared after Enter key send"):
            expect(chatbot_page.textarea).to_have_value("")

        with step("Wait for user message to appear with correct text"):
            expect(chatbot_page.user_messages.last).to_be_visible()
            expect(chatbot_page.user_messages.last).to_contain_text(test_message)

    def test_message_actions(self, chatbot_page: ChatbotPage):
        """Verify user can interact with message actions.

        Tests: copy answer and like/dislike feedback.
        """

        message, index = chatbot_page.get_predefined_message(use_random=True)

        with step(f"Send a{' predefined' if index >= 0 else ''} message: {message}"):
            with chatbot_page.send_predefined_message(index, message) as response:
                chatbot_page.verify_interactions_disabled()

        response = response.value
        chatbot_page.verify_answer(response)

        # === Copy ===
        with step("Verify copy button is visible"):
            expect(chatbot_page.copy_button).to_be_visible()

        with step("Click copy button", step_type="info"):
            chatbot_page.copy_button.click()

        with step("Verify copy button is disabled"):
            expect(chatbot_page.copy_button).to_be_disabled()

        with step("Wait for copy button to be enabled after a short period"):
            expect(chatbot_page.copy_button).to_be_enabled()

        with step("Verify clipboard contains the assistant response text"):
            clipboard_text = chatbot_page.page.evaluate("navigator.clipboard.readText()")
            assert clipboard_text == response.get_message(), "Clipboard text does not match assistant response"

        if not chatbot_page.block_config.get("enableFeedback"):
            info("INFO: Chatbot block is not configured to allow sending feedback")
            return

        # === Like ===
        with step("Verify like button is visible"):
            expect(chatbot_page.like_button).to_be_visible()

        with step("Click like button"):
            chatbot_page.like_button.click()

        with step("Verify feedback modal opens", continue_on_failure=True):
            expect(chatbot_page.feedback_modal).to_be_visible()
            expect(chatbot_page.feedback_textarea).to_be_visible()

        with step("Close feedback modal"):
            chatbot_page.feedback_cancel_button.click()

        with step("Verify feedback modal closes"):
            expect(chatbot_page.feedback_modal).to_be_hidden()

        with step("Re-open feedback modal via like button"):
            chatbot_page.like_button.click()

        with step("Enter feedback text in textarea"):
            feedback_text = "This is a test feedback"
            chatbot_page.feedback_textarea.fill(feedback_text)

        with step("Submit positive feedback and capture API response"):
            with chatbot_page.send_feedback() as response:
                request = response.response_info.value.request
                data = json.loads(request.post_data) if request.post_data else None
                assert data, "No post data found"
                assert data.get("feedback_text") == feedback_text, f"Requested feedback text doesn't match, expected: {feedback_text}, requested: {data.get('feedback_text')}"
                assert data.get("is_positive") is True, "Requested feedback is negative, expected positive feedback"

        response = response.response_info.value

        with step("Verify feedback succeeded and verify toast notification"):
            expect(chatbot_page.feedback_modal).to_be_hidden()
            expect(chatbot_page.feedback_toast).to_be_visible()
            expect(chatbot_page.feedback_toast).to_have_text("Thanks for your feedback!")
            assert response.status == 200, f"Expected status 200, got {response.status}"

        # === Dislike ===
        with step("Verify dislike button is visible"):
            expect(chatbot_page.dislike_button).to_be_visible()

        with step("Click dislike button"):
            chatbot_page.dislike_button.click()

        with step("Verify feedback modal opens", continue_on_failure=True):
            expect(chatbot_page.feedback_modal).to_be_visible()
            expect(chatbot_page.feedback_negative_reason).to_be_visible()
            expect(chatbot_page.feedback_textarea).to_be_visible()

        with step("Close feedback modal"):
            chatbot_page.feedback_cancel_button.click()

        with step("Verify feedback modal closes"):
            expect(chatbot_page.feedback_modal).to_be_hidden()

        with step("Re-open feedback modal via dislike button"):
            chatbot_page.dislike_button.click()

        with step("Enter feedback text in textarea"):
            feedback_text = "This is a test feedback"
            chatbot_page.feedback_textarea.fill(feedback_text)

        with step("Select first predefined negative feedback reason"):
            reason = chatbot_page.feedback_negative_reason.locator("button").nth(0)
            reason_text = reason.text_content()
            reason.click()
            classes = reason.get_attribute("class") or ""
            assert "inverted" not in classes.split()

        with step("Submit negative feedback and capture API response"):
            with chatbot_page.send_feedback() as response:
                request = response.response_info.value.request
                data = json.loads(request.post_data) if request.post_data else None
                assert data, "No post data found"
                assert data.get("feedback_text") == feedback_text, f"Requested feedback text doesn't match, expected: {feedback_text}, requested: {data.get('feedback_text')}"
                assert data.get("is_positive") is False, "Requested feedback is positive, expected negative feedback"
                assert data.get("predefined_feedback") == reason_text, f"Requested feedback doesn't match predefined feedback, expected: {reason_text}, requested: {data.get('predefined_feedback')}"

        response = response.response_info.value

        with step("Verify feedback succeeded and verify toast notification"):
            expect(chatbot_page.feedback_modal).to_be_hidden()
            expect(chatbot_page.feedback_toast).to_be_visible()
            expect(chatbot_page.feedback_toast).to_have_text("Thanks for your feedback!")
            assert response.status == 200, f"Expected status 200, got {response.status}"

        with step("Re-open feedback modal to verify button still works"):
            chatbot_page.dislike_button.click()

    def test_loading_states(self, chatbot_page: ChatbotPage):
        """Verify all loading states during a single response flow.

        Tests: answer_loader, multi-tool streaming/complete states,
        step count, expandability, active step indicator, and done node.
        """

        message, index = chatbot_page.get_predefined_message(use_random=True)

        info(f"Send a{' predefined' if index >= 0 else ''} message: {message}")
        with chatbot_page.send_message(message) as response:
            with step("Verify answer loader appears"):
                expect(chatbot_page.answer_loader).to_be_visible()

            if len(chatbot_page.block_config.get("showTools")) <= 0:
                info("INFO: Chatbot block is configured to never show multi-tool steps like reasoning or internal serach tool")
                return

            with step("Verify multi-tool renderer appears", step_type="info"):
                expect(chatbot_page.multi_tool).to_be_visible()

            with step("Verify multi-tool shows streaming state"):
                expect(chatbot_page.multi_tool_streaming).to_be_visible()

            with step("Verify multi-tool header is visible"):
                expect(chatbot_page.multi_tool_header).to_be_visible()

            with step("Verify multi-tool header is clickable (button role)"):
                expect(chatbot_page.multi_tool_header).to_have_attribute("role", "button")

            # === Step Count ===
            with step("Verify multi-tool step count is displayed"):
                expect(chatbot_page.multi_tool_count).to_be_visible()
                count_value = chatbot_page.multi_tool_count_value.text_content()
                assert count_value is not None
                step_count = int(count_value)
                assert step_count >= 1, "Should have at least 1 processing step"

            with step("Verify step count label grammar (step/steps)"):
                count_label = chatbot_page.multi_tool.locator(
                    ChatbotPageSelectors.MULTI_TOOL_COUNT_LABEL
                ).text_content()
                if step_count == 1:
                    assert count_label == "step", f"Expected 'step', got '{count_label}'"
                else:
                    assert count_label == "steps", f"Expected 'steps', got '{count_label}'"

            # === Expandability (during streaming) ===
            with step("Expand multi-tool by clicking header"):
                chatbot_page.multi_tool_header.click()

            with step("Verify multi-tool expanded state"):
                expect(chatbot_page.multi_tool_header).to_have_attribute(
                    "aria-expanded", "true"
                )

            with step("Verify active step indicator exists"):
                expect(chatbot_page.multi_tool_active_item).to_be_visible()

            with step("Collapse multi-tool by clicking header"):
                chatbot_page.multi_tool_header.click()

            with step("Verify multi-tool collapsed state"):
                expect(chatbot_page.multi_tool_header).to_have_attribute(
                    "aria-expanded", "false"
                )

        # === Wait for Completion ===
        response = response.value
        chatbot_page.verify_answer(response)

        # === Completion State ===
        with step("Verify multi-tool shows complete state"):
            expect(chatbot_page.multi_tool_complete).to_be_visible()

        with step("Verify done node appears"):
            expect(chatbot_page.multi_tool_done_node).to_be_visible()

        with step("Verify answer loader disappears"):
            expect(chatbot_page.answer_loader).to_be_hidden()

        with step("Verify assistant message is visible"):
            expect(chatbot_page.assistant_messages.first).to_be_visible()

    def test_message_sending_and_conversation_flow(self, chatbot_page: ChatbotPage):
        """Verify message sending methods and conversation flow work correctly.

        Tests: send button click, enter key, textarea clears, message ordering, auto-scroll.
        """

        first_message = "What is climate change?"
        second_message = "How does it affect Europe?"

        # === Send Button Click ===
        with step(f"Type first message in textarea: {first_message}"):
            chatbot_page.textarea.fill(first_message)

        info("Click send button and wait for API response")
        with chatbot_page.send_message() as response:
            with step("Verify interactions are disabled"):
                chatbot_page.verify_interactions_disabled()

            with step("Verify textarea is cleared after sending"):
                expect(chatbot_page.textarea).to_have_value("")

            with step("Verify first user message appears in conversation"):
                assert chatbot_page.user_messages.count() == 1, "Should have 1 user message"
                expect(chatbot_page.user_messages.last).to_be_visible()
                expect(chatbot_page.user_messages.last).to_contain_text(first_message)

        response = response.value
        chatbot_page.verify_answer(response)

        with step("Verify first assistant response is visible"):
            assert chatbot_page.assistant_messages.count() == 1, "Should have 1 assistant message"
            expect(chatbot_page.assistant_messages.last).to_be_visible()

        # === Enter Key Sends Message ===
        with step("Type second message"):
            chatbot_page.textarea.fill(second_message)

        info("Press Enter key to send message")
        with chatbot_page.send_message(enter=True) as response:
            with step("Verify interactions are disabled"):
                chatbot_page.verify_interactions_disabled()

            with step("Verify textarea is cleared after Enter key send"):
                expect(chatbot_page.textarea).to_have_value("")

        response = response.value
        chatbot_page.verify_answer(response)

        # === Conversation Flow Verification ===
        with step("Verify both user messages are visible"):
            assert chatbot_page.user_messages.count() == 2, "Should have 2 user messages"

        with step("Verify both assistant responses are visible"):
            assert chatbot_page.assistant_messages.count() == 2, "Should have 2 assistant messages"

        with step("Verify messages are in correct order"):
            first_user_msg = chatbot_page.user_messages.nth(0).text_content()
            second_user_msg = chatbot_page.user_messages.nth(1).text_content()
            assert first_message in first_user_msg, "First message should be first in list"
            assert second_message in second_user_msg, "Second message should be second in list"

        with step("Verify conversation scrolled to show latest message"):
            last_assistant = chatbot_page.assistant_messages.last
            expect(last_assistant).to_be_visible()

    def test_input_edge_cases(self, chatbot_page: ChatbotPage):
        """Verify edge cases in message input are handled correctly.

        Tests: special characters with unicode (1 LLM request), long messages (1 LLM request).
        """

        special_message = "What about CO₂ émissions & climate \"change\" in Zürich? (including <brackets>, 'quotes' and 🌍)"

        # === Special Characters & Unicode (combined into one request) ===
        with step("Type message with special characters and unicode"):
            chatbot_page.textarea.fill(special_message)

        with step("Verify textarea accepts special characters and unicode"):
            expect(chatbot_page.textarea).to_have_value(special_message)

        info(f"Send message with special characters and unicode: {special_message}")
        with chatbot_page.send_message() as response:
            with step("Verify user message displays special characters correctly", True):
                expect(chatbot_page.user_messages.last).to_be_visible()
                expect(chatbot_page.user_messages.last).not_to_have_text("")
                user_msg_text = chatbot_page.user_messages.last.text_content()
                # Check key special characters are preserved
                assert "CO₂" in user_msg_text or "CO2" in user_msg_text, f"Subscript or CO2 should be in message"
                assert "émissions" in user_msg_text or "emissions" in user_msg_text.lower(), "Accented characters should be preserved"

        response = response.value
        chatbot_page.verify_answer(response, f"Verify assistant response to special characters and unicode (message ID: {response.assistant_message_id})")

        with step("Clear chat for next test"):
            chatbot_page.clear_chat_button.click()
            chatbot_page.verify_empty_conversation()

        # === Long Message ===
        long_message = "Please explain in detail " + "the environmental impact of industrial activities " * 10

        with step("Type a long message"):
            chatbot_page.textarea.fill(long_message)

        with step("Verify textarea accepts long message"):
            actual_value = chatbot_page.textarea.input_value()
            assert len(actual_value) > 200, "Textarea should accept long messages"

        info("Send long message")
        with chatbot_page.send_message() as response:
            with step("Verify long user message is displayed"):
                expect(chatbot_page.user_messages.last).to_be_visible()

        response = response.value
        chatbot_page.verify_answer(response, f"Verify assistant response to long message (message ID: {response.assistant_message_id})")


@pytest.mark.basic
@pytest.mark.sources
class TestResponseSources:
    """Tests for source document handling in responses."""

    def test_sources_in_response_and_ui(self, chatbot_page: ChatbotPage):
        """Verify sources are present in response data and displayed in UI."""

        message, index = chatbot_page.get_predefined_message(use_random=True)

        with step(f"Send a{' predefined' if index >= 0 else ''} message: {message}"):
            with chatbot_page.send_predefined_message(index, message) as response:
                chatbot_page.verify_interactions_disabled()

        response = response.value
        chatbot_page.verify_answer(response)

        with step("Verify assistant response contains source documents"):
            documents = response.get_final_documents()
            assert len(documents) > 0, "No sources found in response"

        with step("Verify assistant response has inline citations"):
            citations = response.get_citations()
            assert len(citations) > 0, "No citations found in response"

        if len(citations) > 3:
            info("More than 3 citations found in response")
            with step("Verify show all sources button is visible", step_type="info"):
                expect(chatbot_page.show_all_sources_button).to_be_visible()
            with step("Click show all sources button"):
                chatbot_page.show_all_sources_button.click()
            with step("Verify sources sidebar is visible"):
                expect(chatbot_page.sources_sidebar).to_be_visible()
                ui_sources_count = chatbot_page.sources_sidebar_items.count()
            with step("Close sources sidebar"):
                chatbot_page.show_all_sources_button.click()
                expect(chatbot_page.sources_sidebar).to_be_hidden()
        else:
            info("Less than 3 citations found in response")
            with step("Verify show all sources button is hidden"):
                expect(chatbot_page.show_all_sources_button).to_be_hidden()
            with step("Verify sources sidebar is hidden"):
                expect(chatbot_page.sources_sidebar).to_be_hidden()
            with step("Verify sources are displayed in UI"):
                expect(chatbot_page.source_items.first).to_be_visible()
                ui_sources_count = chatbot_page.source_items.count()

        with step("Verify sources count"):
            assert ui_sources_count == len(citations), "Sources count doesn't match citations count"

        with step("Verify sources tab menu item is visible"):
            expect(chatbot_page.tab_menu_item_sources).to_be_visible()
            classes = chatbot_page.tab_menu_item_sources.get_attribute("class") or ""
            assert "active" not in classes.split(), "Sources tab menu should not be active"

        with step("Click sources tab menu item"):
            chatbot_page.tab_menu_item_sources.click()

        with step("Verify sources tab content is visible"):
            classes = chatbot_page.tab_menu_item_sources.get_attribute("class") or ""
            assert "active" in classes.split(), "Sources tab menu should be active"
            expect(chatbot_page.tab_content).to_be_visible()
            source_items = chatbot_page.tab_content.locator(chatbot_page.selectors.SOURCE_ITEMS)
            expect(source_items.first).to_be_visible()
            assert source_items.count() == len(citations), "Sources count doesn't match citations count"

        with step("Click answer tab menu item"):
            chatbot_page.tab_menu_item_answer.click()

        with step("Verify answer tab content is visible"):
            classes = chatbot_page.tab_menu_item_answer.get_attribute("class") or ""
            assert "active" in classes.split(), "Answer tab menu should be active"
            expect(chatbot_page.tab_content).to_be_visible()


@pytest.mark.basic
@pytest.mark.related_questions
class TestRelatedQuestions:
    """Tests for related questions functionality."""

    def test_related_questions_displayed(self, chatbot_page: ChatbotPage):
        """Verify related questions are displayed after response."""

        if not chatbot_page.block_config.get("enableQgen"):
            info("INFO: Chatbot block configuration does not have related questions generation enabled")
            return

        if not chatbot_page.block_config.get("qgenAsistantId"):
            info("INFO: Chatbot block configuration does not have a proper assistant ID set for related questions generation")
            return

        message, index = chatbot_page.get_predefined_message(use_random=True)

        with step(f"Send a{' predefined' if index >= 0 else ''} message: {message}"):
            with chatbot_page.send_predefined_message(index, message) as response:
                chatbot_page.verify_interactions_disabled()

        with chatbot_page.wait_related_questions() as response:
            with step("Verify related questions loader is visible", step_type="info"):
                expect(chatbot_page.related_questions_loader).to_be_visible()

        response = response.value
        chatbot_page.verify_answer(response)
        related_questions = response.get_related_questions()

        with step("Verify related questions loader disappears"):
            expect(chatbot_page.related_questions_loader).to_be_hidden()

        with step("Verify related questions container is visible"):
            expect(chatbot_page.related_questions_container).to_be_visible()

        with step("Verify related questions are displayed"):
            assert len(related_questions) > 0, "No related questions found"
            assert chatbot_page.related_question_buttons.count() == len(related_questions), "Incorrect number of related questions displayed"
            for i, question in enumerate(related_questions):
                assert chatbot_page.related_question_buttons.nth(i).text_content() == question, f"Question {i} is incorrectly displayed"

        info(f"Click the first related question: {chatbot_page.related_question_buttons.nth(0).text_content()}")
        with chatbot_page.send_related_question(0) as response:
            with step("Verify answer loader appears"):
                expect(chatbot_page.answer_loader).to_be_visible()

        response = response.value
        chatbot_page.verify_answer(response)

        with step("Verify answer loader disappears"):
            expect(chatbot_page.answer_loader).to_be_hidden()


@pytest.mark.basic
@pytest.mark.halloumi
class TestHalloumiFactCheck:
    """Tests for Halloumi fact-check functionality."""

    def test_halloumi_quality_fact_check(self, chatbot_page: ChatbotPage):
        """Verify fact-check button appears and executes successfully.

        Tests: button visibility, click action, result appears.
        """

        quality_check = chatbot_page.block_config.get("qualityCheck")
        message, index = chatbot_page.get_predefined_message(use_random=True)

        with step("Verify Halloumi quality fact-check is enabled"):
            assert quality_check and quality_check != "disabled", "Halloumi quality fact-check is disabled"

        if quality_check == "ondemand_toggle":
            with step("Verify Halloumi quality fact-check toggle is ON"):
                expect(chatbot_page.fact_check_toggle).to_be_visible()
                checkbox = chatbot_page.fact_check_toggle.locator(".ui.checkbox")
                input = chatbot_page.fact_check_toggle.locator("#quality-check-toggle")
                checked = input.is_checked()
                if not checked:
                    checkbox.click()
                    checked = input.is_checked()
                    assert checked, "Fact-check toggle should be checked by default"

        with step(f"Send a{' predefined' if index >= 0 else ''} message: {message}"):
            with chatbot_page.send_predefined_message(index, message) as response:
                chatbot_page.verify_interactions_disabled()

        response = response.value
        chatbot_page.verify_answer(response)

        with step("Verify Halloumi quality fact-check loading state"):
            with chatbot_page.page.expect_response(
                lambda r: "/_ha/generate" in r.url
            ) as response_info:
                if quality_check == "ondemand":
                    expect(chatbot_page.fact_check_button).to_be_visible()
                    chatbot_page.fact_check_button.click()
                expect(chatbot_page.verify_claims_loading).to_be_visible()

        with step("Wait for Halloumi quality fact-check to be fetched"):
            response = response_info.value
            response.finished()
            assert response.status == 200, f"Expected status 200, got {response.status}"
            expect(chatbot_page.verify_claims_loading).to_be_hidden()
            expect(chatbot_page.halloumi_message).to_be_visible()

        with step("Verify Halloumi quality fact-check loader disappears"):
            expect(chatbot_page.verify_claims_loading).to_be_hidden()
        with step("Verify Halloumi quality fact-check message is displayed"):
            expect(chatbot_page.halloumi_message).to_be_visible()
            assert chatbot_page.halloumi_message.text_content(), "Halloumi message is empty"
        with step("Verify Halloumi quality fact-check claims are displayed"):
            assert chatbot_page.halloumi_claims.count() > 0, "Claims are not displayed"

        # === Score Validation ===
        try:
            halloumi_text = chatbot_page.halloumi_message.text_content()
            score = int(halloumi_text.split(" ")[1].replace("%", ""))
            stage = quality_check_stages(score)
            info(f"Halloumi quality fact-check score: {score}%. {stage}")

            with step("Verify Halloumi quality fact-check score is in valid range"):
                # Validate score is in valid range
                assert 0 <= score <= 100, f"Score should be between 0-100, got {score}"

            with step("Verify Halloumi quality fact-check stage message is appropriate for the score"):
                # Verify stage message makes sense for the score
                if score < 20:
                    assert "not supported" in stage.lower(), f"Low score ({score}%) should indicate 'not supported'"
                elif score >= 95:
                    assert "fully supported" in stage.lower() or "safe" in stage.lower(), f"High score ({score}%) should indicate 'fully supported' or 'safe'"
        except (IndexError, ValueError) as e:
            info(f"Halloumi quality fact-check score cannot be determined: {str(e)}", "failed")

        # TODO: check claim modal

        # === Retry Functionality (if available) ===
        if chatbot_page.retry_button.is_visible():
            with step("Retry button is visible, testing retry functionality", True):
                with chatbot_page.page.expect_response(
                    lambda r: "/_ha/generate" in r.url
                ) as retry_response_info:
                    chatbot_page.retry_button.click()
                    expect(chatbot_page.verify_claims_loading).to_be_visible()
                retry_response = retry_response_info.value
                retry_response.finished()

                assert retry_response.status == 200, f"Retry failed with status {retry_response.status}"
                expect(chatbot_page.verify_claims_loading).to_be_hidden()
                expect(chatbot_page.halloumi_message).to_be_visible()
        else:
            info("INFO: Retry button not visible (fact-check may have succeeded)")


@pytest.mark.basic
@pytest.mark.error_handling
class TestErrorHandling:
    """Tests for error handling and display."""

    def test_network_error_displays_gracefully(self, chatbot_page: ChatbotPage):
        """Verify network errors are displayed with appropriate error UI."""

        with step("Set up network interception to simulate error"):
            chatbot_page.page.route(
                "**/chat/send-message",
                lambda route: route.abort("failed")
            )

        test_message = "Test network error handling"

        with step(f"Send message that will fail: {test_message}"):
            chatbot_page.textarea.fill(test_message)
            chatbot_page.submit_button.click()

        with step("Wait for user message to appear"):
            expect(chatbot_page.user_messages.last).to_be_visible()

        with step("Verify error UI is displayed"):
            expect(chatbot_page.message_error).to_be_visible(timeout=15000)

        with step("Verify error message content"):
            error_text = chatbot_page.message_error.text_content()
            assert error_text, "Error message should have content"

        info(f"Error message displayed: {error_text}")

        with step("Clean up route"):
            chatbot_page.page.unroute("**/chat/send-message")

    def test_api_error_response_handled(self, chatbot_page: ChatbotPage):
        """Verify API error responses (5xx) display appropriate error message."""

        error_message = "Internal server error: Model unavailable"

        with step("Set up route to return 500 error"):
            chatbot_page.page.route(
                "**/chat/send-message",
                lambda route: route.fulfill(
                    status=500,
                    content_type="application/json",
                    body=json.dumps({"message": error_message, "detail": error_message})
                )
            )

        test_message = "Test API error handling"

        with step(f"Send message: {test_message}"):
            chatbot_page.textarea.fill(test_message)
            chatbot_page.submit_button.click()

        with step("Wait for user message to appear"):
            expect(chatbot_page.user_messages.last).to_be_visible()

        with step("Verify error UI is displayed"):
            expect(chatbot_page.message_error).to_be_visible(timeout=15000)

        with step("Verify error message content"):
            error_text = chatbot_page.message_error.text_content()
            assert error_text, "Error message should have content"

        info(f"Error displayed: {error_text}")

        with step("Clean up route"):
            chatbot_page.page.unroute("**/chat/send-message")

    def test_malformed_response_handled(self, chatbot_page: ChatbotPage):
        """Verify malformed responses are handled gracefully without crashing."""

        with step("Set up route to return malformed response"):
            chatbot_page.page.route(
                "**/chat/send-message",
                lambda route: route.fulfill(
                    status=200,
                    content_type="application/json",
                    body="invalid json {"
                )
            )

        test_message = "Test malformed response"

        with step(f"Send message: {test_message}"):
            chatbot_page.textarea.fill(test_message)
            chatbot_page.submit_button.click()

        with step("Wait for user message to appear"):
            expect(chatbot_page.user_messages.last).to_be_visible()

        with step("Verify graceful handling (error or stable UI)", step_type="info"):
            # Either error message appears or UI remains stable
            try:
                expect(chatbot_page.message_error).to_be_visible(timeout=10000)
                graceful = True
            except Exception:
                # If no error, verify the UI doesn't crash
                expect(chatbot_page.chat_window).to_be_visible()
                graceful = False
        if graceful:
            info("Error message displayed for malformed response")
        else:
            info("UI remained stable despite malformed response")

        with step("Clean up route"):
            chatbot_page.page.unroute("**/chat/send-message")

    def test_empty_response_handled(self, chatbot_page: ChatbotPage):
        """Verify empty responses are handled gracefully."""

        with step("Set up route to return empty response"):
            chatbot_page.page.route(
                "**/chat/send-message",
                lambda route: route.fulfill(
                    status=200,
                    content_type="application/json",
                    body=""
                )
            )

        test_message = "Test empty response"

        with step(f"Send message: {test_message}"):
            chatbot_page.textarea.fill(test_message)
            chatbot_page.submit_button.click()

        with step("Wait for user message to appear"):
            expect(chatbot_page.user_messages.last).to_be_visible()

        with step("Verify graceful handling"):
            # Verify UI doesn't crash
            expect(chatbot_page.chat_window).to_be_visible()

        info("UI handled empty response gracefully")

        with step("Clean up route"):
            chatbot_page.page.unroute("**/chat/send-message")

    def test_error_recovery_allows_new_message(self, chatbot_page: ChatbotPage):
        """Verify user can send a new message after an error."""

        with step("Set up route to simulate error"):
            chatbot_page.page.route(
                "**/chat/send-message",
                lambda route: route.abort("failed")
            )

        with step("Send message that will fail"):
            chatbot_page.textarea.fill("Will fail")
            chatbot_page.submit_button.click()

        with step("Wait for error to appear"):
            expect(chatbot_page.message_error).to_be_visible(timeout=15000)

        with step("Remove error route"):
            chatbot_page.page.unroute("**/chat/send-message")

        with step("Verify textarea is enabled for new input"):
            expect(chatbot_page.textarea).to_be_enabled()

        message = "Will pass"

        with step(f"Send a new message: {message}"):
            with chatbot_page.send_message(message) as response:
                pass

        response = response.value
        chatbot_page.verify_answer(response)


@pytest.mark.basic
@pytest.mark.deep_research
class TestDeepResearch:
    """Tests for deep research functionality."""

    def test_deep_research_toggle_affects_request(self, chatbot_page: ChatbotPage):
        """Verify deep research toggle sends correct API parameters."""

        deep_research = chatbot_page.block_config.get("deepResearch")

        # Skip if deep research is completely disabled
        if deep_research == "disabled" or deep_research is None:
            info("INFO: Deep research is disabled in block config")
            return

        # Ensure toggle is ON
        if deep_research in ["user_on", "user_off"]:
            with step("Ensure deep research toggle is ON"):
                expect(chatbot_page.deep_research_toggle).to_be_visible()
                checkbox = chatbot_page.deep_research_toggle.locator(".ui.checkbox")
                input_el = chatbot_page.deep_research_toggle.locator("#deep-research-toggle")
                if not input_el.is_checked():
                    checkbox.click()
                expect(input_el).to_be_checked()

        message, index = chatbot_page.get_predefined_message(use_random=True)

        info("Send a message")
        with chatbot_page.send_predefined_message(index, message) as response:
            with step("Verify request contains use_agentic_search parameter", step_type="info"):
                request = response.response_info.value.request
                body = json.loads(request.post_data)
                assert body.get("use_agentic_search") is True, f"Request should have use_agentic_search=true, got: {body.get('use_agentic_search')}"

            with step("Verify multi-tool renderer appears", step_type="info"):
                expect(chatbot_page.multi_tool).to_be_visible()

            with step("Expand multi-tool header"):
                chatbot_page.multi_tool_header.click()
                expect(chatbot_page.multi_tool_header).to_have_attribute("aria-expanded", "true")

            # Look for "Thinking" status which indicates reasoning
            with step("Verify reasoning step is visible"):
                thinking_item = chatbot_page.multi_tool_items.locator(".tool-collapsed-status:has-text('Thinking')").last
                expect(thinking_item).to_be_visible(timeout=15000)

        response = response.value
        chatbot_page.verify_answer(response)

        with step("Verify response contains reasoning data"):
            reasoning = response.get_reasoning()
            assert len(reasoning) > 0, "Expected reasoning data"

    def test_deep_research_toggle_off(self, chatbot_page: ChatbotPage):
        """Verify no reasoning data when deep research is OFF."""

        deep_research = chatbot_page.block_config.get("deepResearch")

        if deep_research not in ["user_on", "user_off"]:
            info("INFO: Deep research toggle not available")
            return

        with step("Ensure deep research toggle is OFF"):
            expect(chatbot_page.deep_research_toggle).to_be_visible()
            checkbox = chatbot_page.deep_research_toggle.locator(".ui.checkbox")
            input_el = chatbot_page.deep_research_toggle.locator("#deep-research-toggle")
            if input_el.is_checked():
                checkbox.click()
            expect(input_el).not_to_be_checked()

        message, index = chatbot_page.get_predefined_message(use_random=True)

        with step("Send message and capture request"):
            with chatbot_page.send_predefined_message(index, message) as response:
                request = response.response_info.value.request

        response = response.value
        chatbot_page.verify_answer(response)

        with step("Verify request has use_agentic_search=false"):
            assert request is not None, "Request was not captured"
            body = json.loads(request.post_data)
            assert body.get("use_agentic_search") is False, f"Request should have use_agentic_search=false, got: {body.get('use_agentic_search')}"
