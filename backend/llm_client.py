"""Provider-independent chat completion client.

The application defaults to Ollama, but this module keeps the application code
independent of any one model host.  Providers are selected entirely through
environment variables; API keys are only required by remote providers.
"""
from dataclasses import dataclass
import os
from typing import Optional

import requests
import logging
import json
from pathlib import Path


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    base_url: str
    api_key: Optional[str] = None
    timeout: int = 120
    temperature: float = 0.2
    max_tokens: int = 1024

    @classmethod
    def from_env(cls) -> "LLMConfig":
        # Allow a repository-local JSON config file to override environment vars.
        cfg_path = Path(__file__).parent / "llm_config.json"
        file_cfg = {}
        if cfg_path.exists():
            try:
                file_cfg = json.loads(cfg_path.read_text(encoding="utf-8")) or {}
            except Exception:
                file_cfg = {}

        def _get(name, default=None):
            # keys in JSON are lowercase names matching the env var names
            return file_cfg.get(name) if name in file_cfg else os.getenv(name.upper(), default)

        provider = str(_get("llm_provider", "ollama")).strip().lower()
        if provider == "ollama":
            return cls(
                provider=provider,
                model=str(_get("ollama_model", os.getenv("OLLAMA_MODEL", "llama3.2:3b"))),
                base_url=str(_get("ollama_base_url", os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))),
                timeout=int(_get("llm_timeout", os.getenv("LLM_TIMEOUT", "120"))),
                temperature=float(_get("llm_temperature", os.getenv("LLM_TEMPERATURE", "0.2"))),
                max_tokens=int(_get("llm_max_tokens", os.getenv("LLM_MAX_TOKENS", "1024"))),
            )

        # OpenAI-compatible covers OpenAI, DeepSeek, Groq, Together, vLLM, etc.
        # Resolve model/base_url with mapping for known providers (e.g., DeepSeek)
        model_val = str(_get("llm_model", os.getenv("LLM_MODEL", os.getenv("DEEPSEEK_MODEL", "gpt-4o-mini"))))
        base_url_val = str(_get("llm_base_url", os.getenv("LLM_BASE_URL", os.getenv("DEEPSEEK_API_URL", "https://api.openai.com/v1"))))
        api_key_val = str(_get("llm_api_key", os.getenv("LLM_API_KEY", os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"))))

        # Automatic compatibility mapping: DeepSeek requires explicit model names
        try:
            host = base_url_val.lower()
            if "deepseek" in host:
                # Map common generic model aliases to DeepSeek's model names
                if model_val in ("default", "gpt-4o-mini", "gpt-4o", "gpt-4o-mini-dev"):
                    model_val = "deepseek-v4-pro"
        except Exception:
            pass

        return cls(
            provider="openai_compatible",
            model=model_val,
            base_url=base_url_val,
            api_key=api_key_val,
            timeout=int(_get("llm_timeout", os.getenv("LLM_TIMEOUT", "120"))),
            temperature=float(_get("llm_temperature", os.getenv("LLM_TEMPERATURE", "0.2"))),
            max_tokens=int(_get("llm_max_tokens", os.getenv("LLM_MAX_TOKENS", "1024"))),
        )


class LLMClient:
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig.from_env()
        self.logger = logging.getLogger(__name__)

    @property
    def is_configured(self) -> bool:
        return self.config.provider == "ollama" or bool(self.config.api_key)

    def generate(self, prompt: str) -> str:
        if self.config.provider == "ollama":
            return self._generate_ollama(prompt)
        return self._generate_openai_compatible(prompt)

    def _generate_ollama(self, prompt: str) -> str:
        endpoint = f"{self.config.base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": self.config.temperature, "num_predict": self.config.max_tokens},
        }
        try:
            response = requests.post(endpoint, json=payload, timeout=self.config.timeout)
            response.raise_for_status()
            content = (response.json().get("message") or {}).get("content", "")
        except requests.RequestException as exc:
            raise RuntimeError(
                f"Ollama request failed. Ensure Ollama is running and `{self.config.model}` is installed "
                f"(`ollama pull {self.config.model}`). Details: {exc}"
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Ollama returned an empty or invalid chat response.")
        return content.strip()

    def _generate_openai_compatible(self, prompt: str) -> str:
        if not self.config.api_key:
            raise RuntimeError("LLM_API_KEY is required when LLM_PROVIDER is openai_compatible.")
        base = self.config.base_url.rstrip('/')
        endpoint = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        try:
            response = requests.post(
                endpoint,
                json=payload,
                headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
                timeout=self.config.timeout,
            )
            # If we got a non-2xx, try a couple of compatibility fallbacks before failing
            if not response.ok:
                body = response.text
                self.logger.warning("LLM provider returned HTTP %s: %s", response.status_code, body)
                # Some OpenAI-compatible providers expect a single `input` field instead
                # of a `messages` array. Try that form once before failing.
                if response.status_code == 400:
                    alt_payload = {
                        "model": self.config.model,
                        "input": prompt,
                        "temperature": self.config.temperature,
                        "max_tokens": self.config.max_tokens,
                    }
                    try:
                        alt_resp = requests.post(
                            endpoint,
                            json=alt_payload,
                            headers={"Authorization": f"Bearer {self.config.api_key}", "Content-Type": "application/json"},
                            timeout=self.config.timeout,
                        )
                        if alt_resp.ok:
                            response = alt_resp
                        else:
                            self.logger.warning("Alternate payload also failed (HTTP %s): %s", alt_resp.status_code, alt_resp.text)
                    except requests.RequestException as e:
                        self.logger.exception("Alternate LLM request failed: %s", e)
                # Finally, raise to be handled below with richer context
            response.raise_for_status()
            # Try several common response shapes for OpenAI-compatible APIs
            j = response.json()
            content = ""
            # Chat-style: choices[].message.content
            try:
                content = ((j.get("choices") or [{}])[0].get("message") or {}).get("content", "")
            except Exception:
                content = ""
            # Text-style: choices[].text
            if not content:
                try:
                    content = (j.get("choices") or [{}])[0].get("text", "")
                except Exception:
                    content = ""
            # Some providers use top-level output fields
            if not content:
                for key in ("output", "generated_text", "result"):
                    if key in j and isinstance(j[key], str):
                        content = j[key]
                        break
        except requests.RequestException as exc:
            # If a response object exists on the exception include its body
            resp = getattr(exc, 'response', None)
            extra = ''
            if resp is not None:
                try:
                    extra = f" Response body: {resp.text}"
                except Exception:
                    extra = ''
            raise RuntimeError(f"OpenAI-compatible LLM request failed: {exc}.{extra}") from exc
        if not isinstance(content, str) or not content.strip():
            # Do not expose provider payloads (which can include internal reasoning,
            # usage metadata, and credentials-adjacent identifiers) to the UI.
            raise RuntimeError("The selected model did not return a final answer. Please try again.")
        return content.strip()
