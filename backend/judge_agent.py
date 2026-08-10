"""LLM-as-a-judge evaluation for chatbot responses."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from llm_client import LLMClient


class LLMJudgeAgent:
    """Evaluate each chatbot answer without interrupting answer delivery."""

    def __init__(self, client: LLMClient | None = None, enabled: bool | None = None):
        self.client = client or LLMClient()
        self.enabled = (
            enabled
            if enabled is not None
            else os.getenv("LLM_JUDGE_ENABLED", "true").lower() in ("1", "true", "yes", "on")
        )
        self.logger = logging.getLogger(__name__)

    def judge(
        self,
        *,
        question: str,
        response: str,
        context: str = "",
        agent_name: str = "chatbot",
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"status": "disabled", "verdict": "not_judged", "score": None}

        prompt = self._build_prompt(question, response, context, agent_name)
        try:
            if hasattr(self.client, "generate_json"):
                raw_result = self.client.generate_json(prompt, self._response_schema())
            else:
                raw_result = self.client.generate(prompt)
            result = self._parse_result(raw_result)
            result["status"] = "completed"
            result["agent"] = agent_name
            return result
        except (json.JSONDecodeError, ValueError) as exc:
            self.logger.warning("LLM judge returned invalid structured output for %s: %s", agent_name, exc)
            return self._fallback_evaluation(question, response, context, agent_name, "invalid LLM output")
        except Exception as exc:
            self.logger.warning("LLM judge failed for %s: %s", agent_name, exc)
            return self._fallback_evaluation(question, response, context, agent_name, "LLM unavailable")

    def judge_without_llm(
        self,
        *,
        question: str,
        response: str,
        context: str = "",
        agent_name: str = "chatbot",
    ) -> dict[str, Any]:
        """Return an immediate scored evaluation without another model call."""
        return self._fallback_evaluation(
            question, response, context, agent_name, "Fast deterministic review"
        )

    @staticmethod
    def _fallback_evaluation(question: str, response: str, context: str, agent_name: str, reason: str) -> dict[str, Any]:
        """Return a useful conservative critique when the LLM judge cannot run."""
        answer = (response or "").strip()
        words = re.findall(r"\b\w+\b", answer)
        issues = []
        missing = []
        relevance = 72
        clarity = 75
        groundedness = 68 if context else 55
        correctness = 65  # Cannot be fully verified without the model judge.
        safety = 85

        if len(words) < 45:
            issues.append("The answer is brief and may not fully cover every part of the request.")
            missing.append("More specific explanation, examples, or actionable detail.")
            clarity = 68
        if context and not re.search(r"\b(source|document|according|based on|evidence)\b", answer, re.IGNORECASE):
            issues.append("The answer does not clearly connect its claims to the supplied knowledge sources.")
            missing.append("Explicit evidence or source-to-claim attribution from the retrieved context.")
            groundedness = 58
        if not re.search(r"\b(however|depends|may|might|caveat|limitation|verify|validate)\b", answer, re.IGNORECASE):
            missing.append("Relevant limitations, assumptions, or facts that should be validated.")
        if not issues:
            issues.append("Automated fallback review cannot verify every factual claim against the full context.")

        dimensions = {
            "relevance": relevance,
            "correctness": correctness,
            "groundedness": groundedness,
            "clarity": clarity,
            "safety": safety,
        }
        score = min(79, round(sum(dimensions.values()) / len(dimensions)))
        return {
            "status": "fallback",
            "verdict": "warning" if score >= 60 else "fail",
            "score": score,
            "dimensions": dimensions,
            "missing_information": missing[:5],
            "issues": issues[:5],
            "feedback": f"A conservative fallback review was used because of {reason}. Address the missing items above and verify factual claims against the retrieved sources.",
            "agent": agent_name,
            "evaluation_method": "deterministic_fallback",
        }

    @staticmethod
    def _response_schema() -> dict[str, Any]:
        score = {"type": "integer", "minimum": 0, "maximum": 100}
        return {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "enum": ["pass", "warning", "fail"]},
                "score": score,
                "dimensions": {
                    "type": "object",
                    "properties": {name: score for name in ("relevance", "correctness", "groundedness", "clarity", "safety")},
                    "required": ["relevance", "correctness", "groundedness", "clarity", "safety"],
                },
                "missing_information": {"type": "array", "items": {"type": "string"}},
                "issues": {"type": "array", "items": {"type": "string"}},
                "feedback": {"type": "string"},
            },
            "required": ["verdict", "score", "dimensions", "missing_information", "issues", "feedback"],
        }

    @staticmethod
    def _build_prompt(question: str, response: str, context: str, agent_name: str) -> str:
        question = (question or "")[:1000]
        response = (response or "")[:4000]
        context = (context or "No reference context was supplied.").strip()
        if len(context) > 3500:
            context = context[:3500]
        try:
            from db_utils import get_agent_prompt
            saved = get_agent_prompt("llm_judge")
        except Exception:
            saved = None
        if saved and saved.get("prompt_text"):
            template = saved["prompt_text"]
        else:
            template = (Path(__file__).resolve().parent / "prompts" / "llm_judge_prompt.txt").read_text(encoding="utf-8")
        replacements = {
            "{agent_name}": agent_name,
            "{user_question}": question,
            "{reference_context}": context,
            "{chatbot_response}": response,
        }
        for placeholder, value in replacements.items():
            template = template.replace(placeholder, value)
        return template

    @staticmethod
    def _parse_result(raw_result: str) -> dict[str, Any]:
        cleaned = (raw_result or "").strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            cleaned = match.group(0)
        payload = json.loads(cleaned)
        if not isinstance(payload, dict):
            raise ValueError("Judge response must be a JSON object")

        dimensions = payload.get("dimensions")
        if not isinstance(dimensions, dict):
            raise ValueError("Judge response is missing dimension scores")
        dimension_names = ("relevance", "correctness", "groundedness", "clarity", "safety")
        normalized_dimensions = {
            name: LLMJudgeAgent._bounded_score(dimensions.get(name)) for name in dimension_names
        }
        issues = payload.get("issues", [])
        if not isinstance(issues, list):
            issues = [str(issues)]
        missing_information = payload.get("missing_information", [])
        if not isinstance(missing_information, list):
            missing_information = [str(missing_information)]
        missing_information = [str(item).strip() for item in missing_information[:5] if str(item).strip()]

        # Derive the overall result from dimension scores instead of trusting a
        # model-provided headline score. A response with an identified omission
        # is incomplete by definition and therefore cannot receive a pass.
        score = round(sum(normalized_dimensions.values()) / len(normalized_dimensions))
        if missing_information:
            score = min(score, 79)
        expected_verdict = "pass" if score >= 80 else "warning" if score >= 60 else "fail"

        return {
            "verdict": expected_verdict,
            "score": score,
            "dimensions": normalized_dimensions,
            "issues": [str(issue) for issue in issues[:5]],
            "missing_information": missing_information,
            "feedback": str(payload.get("feedback", "")).strip(),
        }

    @staticmethod
    def _bounded_score(value: Any) -> int:
        try:
            return max(0, min(100, int(round(float(value)))))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid judge score: {value}") from exc
