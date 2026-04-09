"""Tests for bonus/answer_shaping.py — intent-based prompt templates."""

from __future__ import annotations

from bonus.answer_shaping import AnswerShape, detect_answer_shape, get_system_prompt


# ---------------------------------------------------------------------------
# detect_answer_shape
# ---------------------------------------------------------------------------


class TestDetectAnswerShape:
    # SUMMARY (highest priority)
    def test_summarize(self):
        assert detect_answer_shape("Summarize the paper") == AnswerShape.SUMMARY

    def test_overview(self):
        assert detect_answer_shape("Give me a brief overview") == AnswerShape.SUMMARY

    def test_main_findings(self):
        assert detect_answer_shape("What are the main findings?") == AnswerShape.SUMMARY

    def test_key_takeaways(self):
        assert detect_answer_shape("Key takeaways from the document") == AnswerShape.SUMMARY

    def test_tldr(self):
        assert detect_answer_shape("TL;DR") == AnswerShape.SUMMARY

    def test_briefly_explain(self):
        assert detect_answer_shape("Can you briefly explain the results?") == AnswerShape.SUMMARY

    # COMPARISON
    def test_compare(self):
        assert detect_answer_shape("Compare method A and method B") == AnswerShape.COMPARISON

    def test_difference(self):
        assert detect_answer_shape("What is the difference between X and Y?") == AnswerShape.COMPARISON

    def test_pros_and_cons(self):
        assert detect_answer_shape("Pros and cons of this approach") == AnswerShape.COMPARISON

    def test_vs(self):
        assert detect_answer_shape("GPT-4 vs Claude") == AnswerShape.COMPARISON

    def test_tradeoffs(self):
        assert detect_answer_shape("Discuss the trade-offs of each method") == AnswerShape.COMPARISON

    def test_differ(self):
        assert detect_answer_shape("How does method A differ from B?") == AnswerShape.COMPARISON

    # LIST
    def test_list(self):
        assert detect_answer_shape("List the main methods") == AnswerShape.LIST

    def test_enumerate(self):
        assert detect_answer_shape("Enumerate the steps") == AnswerShape.LIST

    def test_what_are_the(self):
        assert detect_answer_shape("What are the different approaches?") == AnswerShape.LIST

    def test_name_all(self):
        assert detect_answer_shape("Name all the techniques") == AnswerShape.LIST

    def test_which_methods(self):
        assert detect_answer_shape("Which methods are discussed?") == AnswerShape.LIST

    # DEFINITION
    def test_what_is(self):
        assert detect_answer_shape("What is in-context learning?") == AnswerShape.DEFINITION

    def test_define(self):
        assert detect_answer_shape("Define retrieval augmented generation") == AnswerShape.DEFINITION

    def test_what_does_mean(self):
        assert detect_answer_shape("What does ICL mean?") == AnswerShape.DEFINITION

    def test_meaning_of(self):
        assert detect_answer_shape("What is the meaning of RAG?") == AnswerShape.DEFINITION

    # GENERAL fallback
    def test_general(self):
        assert detect_answer_shape("How was this implemented?") == AnswerShape.GENERAL

    def test_general_tell_me(self):
        assert detect_answer_shape("Tell me about the results section") == AnswerShape.GENERAL

    def test_general_empty(self):
        assert detect_answer_shape("") == AnswerShape.GENERAL


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------


class TestPriorityOrdering:
    def test_summary_beats_list(self):
        """'main findings' matches both summary and list patterns — summary wins."""
        assert detect_answer_shape("What are the main findings?") == AnswerShape.SUMMARY

    def test_comparison_beats_list(self):
        """'difference' should match comparison, not list."""
        assert detect_answer_shape("What is the difference between A and B?") == AnswerShape.COMPARISON


# ---------------------------------------------------------------------------
# get_system_prompt
# ---------------------------------------------------------------------------


class TestGetSystemPrompt:
    def test_all_shapes_have_prompts(self):
        for shape in AnswerShape:
            prompt = get_system_prompt(shape)
            assert isinstance(prompt, str)
            assert len(prompt) > 50

    def test_all_prompts_contain_grounding(self):
        for shape in AnswerShape:
            prompt = get_system_prompt(shape)
            assert "ONLY" in prompt

    def test_list_prompt_mentions_list(self):
        prompt = get_system_prompt(AnswerShape.LIST)
        assert "list" in prompt.lower()

    def test_comparison_prompt_mentions_comparison(self):
        prompt = get_system_prompt(AnswerShape.COMPARISON)
        assert "comparison" in prompt.lower() or "table" in prompt.lower()

    def test_summary_prompt_mentions_concise(self):
        prompt = get_system_prompt(AnswerShape.SUMMARY)
        assert "concise" in prompt.lower() or "summary" in prompt.lower()

    def test_definition_prompt_mentions_definition(self):
        prompt = get_system_prompt(AnswerShape.DEFINITION)
        assert "definition" in prompt.lower()
