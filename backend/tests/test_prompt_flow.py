import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app import (
    build_contextual_answer_prompt, build_precise_attribution,
    should_ask_follow_up, synthesize_local_answer,
)


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


def test_local_fallback_enumerates_microorganisms_from_context():
    context = """[Source: Brand Knowledge Base] Relevant Information 1:
Microbial Mechanism & Functions:
- Azospirillum - Nitrogen fixation
- Azotobacter - Nitrogen enrichment
- PSB (Phosphate Solubilizing Bacteria) - Unlocks phosphorus
"""

    answer = synthesize_local_answer(
        "enumerate every named microorganism and its function",
        context,
        {"name": "Navyakosh"},
    )

    assert "Azospirillum: Nitrogen fixation" in answer
    assert "Azotobacter: Nitrogen enrichment" in answer
    assert "PSB (Phosphate Solubilizing Bacteria): Unlocks phosphorus" in answer


def test_precise_attribution_names_brand_record_and_uploaded_document():
    attribution = build_precise_attribution([
        {
            "label": "Brand Knowledge Base", "source_type": "brand_kb",
            "record_type": "mechanism",
        },
        {
            "label": "Uploaded Document: report.pdf", "source_type": "file_upload",
            "record_type": "uploaded_document", "chunk_index": 2, "document_id": 7,
        },
    ])

    assert "Microbial Mechanism & Functions record" in attribution
    assert "Uploaded document: report.pdf" in attribution
    assert "chunk" not in attribution
    assert "document ID" not in attribution
