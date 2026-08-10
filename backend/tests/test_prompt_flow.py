import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app import build_contextual_answer_prompt, should_ask_follow_up


def test_should_ask_follow_up_when_context_is_sparse():
    assert should_ask_follow_up(
        "Can you explain the benefits of this product?",
        "",
        "Brand"
    ) is True


def test_should_not_ask_follow_up_when_question_is_specific_and_context_is_present():
    assert should_ask_follow_up(
        "What is the recommended dosage for maize?",
        "The product is recommended for maize at 2 liters per hectare.",
        "Brand"
    ) is False


def test_prompt_includes_context_and_clarification_guidance():
    prompt = build_contextual_answer_prompt(
        message="How does it work?",
        relevant_context="The product improves soil microbes and nutrient uptake.",
        personal_info={"name": "AgriGrow"},
        source_summary="SOURCE SUMMARY:\n- Uploaded Document: test.pdf\n"
    )

    assert "AgriGrow" in prompt
    assert "How does it work?" in prompt
    assert "RETRIEVED_KNOWLEDGE" in prompt
    assert "SOURCE SUMMARY" in prompt
    assert "If the user asks a broad or ambiguous question" in prompt
