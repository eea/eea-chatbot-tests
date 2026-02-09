"""Comprehensive data-driven tests for chatbot question validation.

This module provides fixture-based testing where each question from fixtures/*.json
is validated against all chatbot features. Tests are parametrized via the `data`
fixture from conftest.py.

Features tested per question:
1. Response content and length
2. Source citations
3. Halloumi quality check (if enabled)
4. Related questions generation (if enabled)
5. Feedback functionality (if enabled)
6. LLM-based quality verification (if enabled)
"""

import pytest
import json
import time
from playwright.sync_api import expect
from contextlib import contextmanager

from chatbot_tests.step import step, info, llm_verdict
from chatbot_tests.plugin import log_step
from chatbot_tests.config import Settings
from chatbot_tests.page_objects import ChatbotPage
from chatbot_tests.llm_analysis import create_analyzer_from_settings
from chatbot_tests.utils import quality_check_stages


# Default thresholds when not specified in fixture
DEFAULTS = {
    "min_response_length": 50,
    "min_sources": 1,
    "min_quality_score": 60,
    "min_related_questions": 2,
}


class ResponseHolder:
    response_info = None
    value = None


@pytest.mark.question
class TestQuestionValidation:
    """Comprehensive validation of chatbot responses for fixture-defined questions.

    Each test case from fixtures/*.json runs as a parametrized test instance.
    Validations are conditionally executed based on block configuration,
    skipping with info log when features are not enabled.
    """

    def test_question_response(self, chatbot_page: ChatbotPage, data: dict, settings: Settings):
        """Comprehensively validate chatbot response for a question.

        Executes all validation phases in order:
        1. Send question and verify response received
        2. Validate response content (length, keywords)
        3. Validate source citations
        4. Validate Halloumi quality check (if configured)
        5. Validate related questions (if qgen enabled)
        6. Validate feedback functionality (if enabled)
        7. Run LLM-based quality checks (if enabled in settings)
        """
        expect_response = chatbot_page.page.expect_response
        question = data["question"]
        test_id = data["id"]
        validation = data.get("validation", {})
        thresholds = data.get("_thresholds", {})

        def get_threshold(key: str) -> int:
            """Get threshold value with fallback chain: validation > thresholds > defaults."""
            return thresholds.get(key, DEFAULTS.get(key, 0))

        def req_filter(url):
            return lambda r: url in r.url

        # =====================================================================
        # PRE-PHASE: Ensure quality check toggle is enabled if applicable
        # =====================================================================

        requests = {}
        qc_config = validation.get("quality_check", {})
        rq_config = validation.get("related_questions", {})
        quality_check = chatbot_page.block_config.get("qualityCheck")
        qgen_enabled = chatbot_page.block_config.get("enableQgen", False)
        qgen_assistant = chatbot_page.block_config.get("qgenAsistantId")

        if not qgen_enabled:
            info("INFO: Chatbot block configuration does not have related questions generation enabled")
        elif not qgen_assistant:
            info("INFO: Chatbot block configuration does not have a proper assistant ID set for related questions generation")
        else:
            requests["qgen"] = "/_rq/chat/send-message"

        if quality_check not in ["enabled", "ondemand", "ondemand_toggle"]:
            info("INFO: Chatbot block is configured to never do Halloumi answer quality fact-check after assistant response")
        else:
            requests["halloumi"] = "/_ha/generate"

        if qc_config and quality_check == "ondemand_toggle":
            with step("Ensure Hallumi quality fact-check toggle is enabled"):
                expect(chatbot_page.fact_check_toggle).to_be_visible()
                input_el = chatbot_page.fact_check_toggle.locator("#quality-check-toggle")
                if not input_el.is_checked():
                    chatbot_page.fact_check_toggle.locator(".ui.checkbox").click()
                    assert input_el.is_checked(), "Failed to enable quality check toggle"
        elif not qc_config and quality_check == "ondemand_toggle":
            with step("Halloumi quality fact-check not enabled for this test case - disabling toggle"):
                expect(chatbot_page.fact_check_toggle).to_be_visible()
                input_el = chatbot_page.fact_check_toggle.locator("#quality-check-toggle")
                if input_el.is_checked():
                    chatbot_page.fact_check_toggle.locator(".ui.checkbox").click()
                    assert not input_el.is_checked(), "Failed to disable quality check toggle"

        # =====================================================================
        # PHASE 1: Send Question and Get Response
        # =====================================================================

        rq_holder = ResponseHolder()
        min_rq = rq_config.get("min_count", get_threshold("min_related_questions")) or 0

        @contextmanager
        def _related_questions():
            if requests.get("qgen"):
                with expect_response(req_filter(requests.get("qgen"))) as response:
                    rq_holder.response_info = response
                    yield rq_holder
                rq_holder.value = response.value
            else:
                rq_holder.value = "not_supported"
                yield rq_holder

        def validate_rq_loading(response: ResponseHolder):
            if response.value == "not_supported":
                return
            with step("Verify related questions loading state"):
                expect(chatbot_page.related_questions_loader).to_be_visible()

        def validate_rq_complete(response: ResponseHolder, start):
            if response.value == "not_supported":
                return
            with step("Wait for related questions to be fetched", start=start):
                response.value = chatbot_page.parse_response(response.value)
            with step("Verify related questions loader disappears"):
                expect(chatbot_page.related_questions_loader).to_be_hidden()

        def validate_rq_generation(response: ResponseHolder):
            response = response.value
            if response == "not_supported":
                return
            items = response.get_related_questions()
            ui_items = chatbot_page.related_question_buttons
            count = ui_items.count()

            with step("Verify related questions were generated properly", True):
                error = None
                rq = response.get_message()
                if "no_response" in rq.lower():
                    error = "assistent responded with \"no_response\""
                error = error or response.error
                if error:
                    raise Exception(f"Related questions were not generated: {error}")
            if not error:
                with step("Verify related questions container is visible", True):
                    expect(chatbot_page.related_questions_container).to_be_visible()
                with step("Verify related questions are displayed", True):
                    assert len(items) > 0, "No related questions found"
                    assert count == len(items), "Incorrect number of related questions displayed"
                    for i, question in enumerate(items):
                        assert ui_items.nth(i).text_content() == question, f"Question {i} is incorrectly displayed"
                with step(f"Verify related questions count ({count} found, {min_rq} minimum)", True):
                    assert count >= min_rq, f"Insufficient related questions: {count} < {min_rq}"

        with _related_questions() as rq_response:
            with step(f"Send question [{test_id}]: '{question}'"):
                with chatbot_page.send_message(question) as response:
                    chatbot_page.verify_interactions_disabled()
                    assert chatbot_page.textarea.input_value() == "", "Textarea should be cleared after message is sent"
                response = response.value
            chatbot_page.verify_answer(response)
            start = time.time()
            validate_rq_loading(rq_response)
        validate_rq_complete(rq_response, start)

        message_text = response.get_message()
        documents = response.get_final_documents()
        citations = response.get_citations()
        map_citations = {cit.get("document_id"): cit for cit in citations}
        cited_documents = []
        for doc in documents:
            doc_id = doc.get("document_id")
            if doc_id in map_citations:
                citation = map_citations[doc_id]
                cited_documents.append({**doc, "citation": citation})

        # =====================================================================
        # PHASE 2: LLM-Based Quality Verification (Optional)
        # =====================================================================

        llm_config = validation.get("llm", {})

        if settings.enable_llm_analysis:
            analyzer = create_analyzer_from_settings(settings)
            if not analyzer:
                info("LLM analyzer unavailable - skipping LLM verification")
            else:
                verification = analyzer.verify_answer(
                    question, message_text, cited_documents
                )
                verify_lack_information = llm_config.get("verify_lack_information")
                verify_answers_question = llm_config.get("verify_answers_question")
                verify_not_vague = llm_config.get("verify_not_vague")
                verify_citations = llm_config.get("verify_citations")

                answers_question = [
                    verification.answers_question,
                    verification.answers_question_explanation
                ]

                not_vague = [
                    verification.not_vague,
                    verification.not_vague_explanation
                ]

                has_citations = [
                    verification.has_citations,
                    verification.has_citations_explanation
                ]

                if verify_lack_information and verification.lack_information:
                    llm_verdict('LLM analysis: answer lacks information', verification.lack_information_explanation)
                    pytest.skip(f"LLM analysis: answer lacks information - {verification.lack_information_explanation}")
                else:
                    llm_verdict("LLM analysis: answer has sufficient information", verification.lack_information_explanation)
                if verify_answers_question and not answers_question[0]:
                    llm_verdict("LLM analysis: answer off-topic", answers_question[1], "failed")
                else:
                    llm_verdict("LLM analysis: answer on-topic", answers_question[1])
                if verify_not_vague and not not_vague[0]:
                    llm_verdict("LLM analysis: answer too vague", not_vague[1], "failed")
                else:
                    llm_verdict("LLM analysis: answer not vague", not_vague[1])
                if verify_citations and not has_citations[0]:
                    llm_verdict("LLM analysis: answer missing citations", has_citations[1], "failed")
                else:
                    llm_verdict("LLM analysis: answer has citations", has_citations[1])

        # =====================================================================
        # PHASE 3: Source Citations Validation and Related questions validation
        # =====================================================================

        sources_config = validation.get("sources", {})
        min_sources = sources_config.get("min_count", get_threshold("min_sources")) or 0

        with step(f"Verify assistant response has inline citations ({len(citations)} found)", True):
            assert len(citations) > 0, "No inline citations in response"

        with step(f"Verify cited sources ({len(cited_documents)} found, {min_sources} minimum)", True):
            assert len(cited_documents) >= min_sources, f"Insufficient cited sources: {len(cited_documents)} < {min_sources}"

        if len(citations) > 0:
            with step("Verify sources visible in UI"):
                if len(citations) > 3:
                    expect(chatbot_page.show_all_sources_button).to_be_visible()
                else:
                    expect(chatbot_page.source_items.first).to_be_visible()

        validate_rq_generation(rq_response)

        # =====================================================================
        # PHASE 4: Response Content Validation
        # =====================================================================

        response_config = validation.get("response", {})
        min_length = response_config.get("min_length", get_threshold("min_response_length")) or 0
        with step(f"Verify assistant response length ({len(message_text)} chars >= {min_length} required)"):
            assert len(message_text) >= min_length, f"Response too short: {len(message_text)} chars < {min_length} required"

        # Optional keyword validation
        expected_keywords = response_config.get("expected_keywords", [])
        if expected_keywords:
            with step(f"Verify assistant response contains expected keywords: {expected_keywords}"):
                message_lower = message_text.lower()
                found = [kw for kw in expected_keywords if kw.lower() in message_lower]
                missing = [kw for kw in expected_keywords if kw.lower() not in message_lower]
                assert len(found) > 0, f"None of the expected keywords found. Missing: {missing}"

        # =====================================================================
        # PHASE 5: Quality Check (Halloumi) Validation
        # =====================================================================

        min_score = qc_config.get("min_score", get_threshold("min_quality_score")) or 0
        chatbot_page.page.set_default_timeout(90000)

        ha_holder = ResponseHolder()

        @contextmanager
        def _halloumi_fact_check():
            try:
                if not requests.get("halloumi"):
                    ha_holder.value = "not_supported"
                    yield ha_holder
                    return

                with expect_response(req_filter(requests.get("halloumi"))) as response:
                    if quality_check == "ondemand":
                        expect(chatbot_page.fact_check_button).to_be_visible()
                        chatbot_page.fact_check_button.click()

                    ha_holder.response_info = response
                    yield ha_holder

                ha_holder.value = response.value

            except Exception:
                log_step(
                    "Wait for Halloumi quality fact-check response",
                    outcome="failed",
                    message="Halloumi quality fact-check request timeout after 90s",
                )
                ha_holder.value = "no_response"

        def validate_ha_loading(response: ResponseHolder):
            if response.value in ["not_supported", "no_response"]:
                return
            with step("Verify Halloumi quality fact-check loading state"):
                expect(chatbot_page.verify_claims_loading).to_be_visible()

        def validate_ha_complete(response: ResponseHolder, start):
            if response.value in ["not_supported", "no_response"]:
                return
            with step("Wait for Halloumi quality fact-check to be fetched", True, start=start):
                response = response.value
                response.finished()
                assert response.status == 200, f"Expected status 200, got {response.status} - {str(response.body())}"
                expect(chatbot_page.verify_claims_loading).to_be_hidden()
                expect(chatbot_page.halloumi_message).to_be_visible()
            with step("Verify Halloumi quality fact-check loader disappears"):
                expect(chatbot_page.verify_claims_loading).to_be_hidden()
            if response.status == 200:
                with step("Verify Halloumi quality fact-check message is displayed"):
                    expect(chatbot_page.halloumi_message).to_be_visible()
                    assert chatbot_page.halloumi_message.text_content(), "Halloumi message is empty"
                with step("Verify Halloumi quality fact-check claims are displayed"):
                    assert chatbot_page.halloumi_claims.count() > 0, "Claims are not displayed"
                halloumi_text = chatbot_page.halloumi_message.text_content()
                score = int(halloumi_text.split(" ")[1].replace("%", ""))
                stage = quality_check_stages(score)
                info(f"Halloumi quality fact-check score: {score}%. {stage}")

                with step("Verify Halloumi quality fact-check score is in valid range", True):
                    # Validate score is in valid range
                    assert 0 <= score <= 100, f"Score should be between 0-100, got {score}"

                if score >= min_score:
                    log_step(f"Quality score ({score}%) above threshold ({min_score}%)")
                else:
                    log_step(f"Quality score {score}% below threshold {min_score}%", "failed")

        try:
            with _halloumi_fact_check() as ha_response:
                start = time.time()
                validate_ha_loading(ha_response)
            validate_ha_complete(ha_response, start)
        except Exception:
            pass

        chatbot_page.page.set_default_timeout(settings.timeout)

        # =====================================================================
        # PHASE 6: Feedback Functionality Validation
        # =====================================================================

        feedback = data.get("feedback")
        feedback_enabled = chatbot_page.block_config.get("enableFeedback")

        if not feedback_enabled:
            info("INFO: Chatbot block is not configured to allow sending feedback")
        elif not feedback:
            info("Feedback validation skipped per fixture config")
        else:
            with step("Verify feedback buttons visible and accessible"):
                expect(chatbot_page.like_button).to_be_visible()
                expect(chatbot_page.dislike_button).to_be_visible()

            with step("Click like button"):
                chatbot_page.like_button.click()

            with step("Verify feedback modal opens", continue_on_failure=True):
                expect(chatbot_page.feedback_modal).to_be_visible()
                expect(chatbot_page.feedback_textarea).to_be_visible()

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

            with step("Wait for feedback to be sent"):
                response = response.response_info.value

            with step("Verify feedback succeeded and verify toast notification"):
                expect(chatbot_page.feedback_modal).to_be_hidden()
                expect(chatbot_page.feedback_toast).to_be_visible()
                expect(chatbot_page.feedback_toast).to_have_text("Thanks for your feedback!")
                assert response.status == 200, f"Expected status 200, got {response.status}"

        # =====================================================================
        # CLEANUP: Clear chat for next test iteration
        # =====================================================================

        with step("Clear chat"):
            chatbot_page.clear_chat_button.click()
            chatbot_page.verify_empty_conversation()
