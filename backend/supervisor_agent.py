"""Supervisor agent for routing general-chat requests to specialist agents."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from llm_client import LLMClient


AGENTS = {
    "general": "General Knowledge Agent",
    "marketing": "Marketing Strategist",
    "competitive_intelligence": "Competitive Intelligence Strategist",
    "tracker": "Implementation Tracker Agent",
}


class SupervisorAgent:
    def __init__(self, client: LLMClient | None = None):
        self.client = client or LLMClient()
        self.logger = logging.getLogger(__name__)

    def route(self, message: str, *, is_admin: bool = False) -> dict[str, Any]:
        available = ["general", "marketing", "competitive_intelligence"] + (["tracker"] if is_admin else [])
        prompt = self._build_prompt(message, available)
        try:
            if hasattr(self.client, "generate_json"):
                raw = self.client.generate_json(prompt, self._response_schema(available))
            else:
                raw = self.client.generate(prompt)
            decision = self._parse(raw, available)
        except Exception as exc:
            self.logger.warning("Supervisor routing failed; using deterministic fallback: %s", exc)
            decision = self._fallback(message, available)
            decision["routing_method"] = "fallback"
        decision["display_name"] = AGENTS[decision["agent"]]
        return decision

    @staticmethod
    def _response_schema(available: list[str]) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "agent": {"type": "string", "enum": available},
                "reason": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["agent", "reason", "confidence"],
        }

    @staticmethod
    def _build_prompt(message: str, available: list[str]) -> str:
        routes = "\n".join(
            {
                "general": "- general: product questions, uploaded documents, brand facts, agriculture Q&A, and ordinary conversation",
                "marketing": "- marketing: campaigns, growth plans, positioning, audiences, channels, content, sales/dealer activation, go-to-market, SEO, and promotion",
                "competitive_intelligence": "- competitive_intelligence: competitor research, competitive comparisons, market threats, differentiation, competitive advantages, and strategies to outperform rivals",
                "tracker": "- tracker: saved marketing plans, owners, implementation status, progress, and execution tracking",
            }[agent]
            for agent in available
        )
        try:
            from db_utils import get_agent_prompt
            saved = get_agent_prompt("supervisor")
        except Exception:
            saved = None
        if saved and saved.get("prompt_text"):
            template = saved["prompt_text"]
        else:
            template = (Path(__file__).resolve().parent / "prompts" / "supervisor_prompt.txt").read_text(encoding="utf-8")
        return template.replace("{available_routes}", routes).replace("{user_request}", message)

    @staticmethod
    def _parse(raw: str, available: list[str]) -> dict[str, Any]:
        match = re.search(r"\{.*\}", raw or "", flags=re.DOTALL)
        payload = json.loads(match.group(0) if match else raw)
        agent = str(payload.get("agent", "")).strip().lower()
        if agent not in available:
            raise ValueError(f"Supervisor selected unavailable agent: {agent}")
        try:
            confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.5))))
        except (TypeError, ValueError):
            confidence = 0.5
        return {
            "agent": agent,
            "reason": str(payload.get("reason", "Request matched this specialist.")).strip(),
            "confidence": round(confidence, 2),
            "routing_method": "llm",
        }

    @staticmethod
    def _fallback(message: str, available: list[str]) -> dict[str, Any]:
        text = (message or "").lower()
        tracker_terms = ("tracker", "implementation status", "plan status", "plan owner", "progress of")
        marketing_terms = (
            "marketing", "campaign", "go-to-market", "growth plan", "promotion", "promote",
            "brand awareness", "target audience", "content strategy", "social media", "seo",
            "dealer activation", "lead generation", "advertising", "market strategy",
        )
        competitor_terms = (
            "competitor", "competitors", "competition", "competitive analysis",
            "competitive landscape", "market rival", "market rivals", "rival",
            "rivals", "outperform", "get ahead of", "beat the competition",
            "compare us", "compare our company", "market positioning against",
        )
        if "tracker" in available and any(term in text for term in tracker_terms):
            agent, reason = "tracker", "The request concerns implementation tracker data."
        elif any(term in text for term in competitor_terms):
            agent, reason = "competitive_intelligence", "The request asks for competitor assessment or competitive strategy."
        elif any(term in text for term in marketing_terms):
            agent, reason = "marketing", "The request asks for marketing expertise."
        else:
            agent, reason = "general", "The request is best handled by general knowledge retrieval."
        return {"agent": agent, "reason": reason, "confidence": 0.7}
