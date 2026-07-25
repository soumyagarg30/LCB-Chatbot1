from llm_client import LLMConfig


def test_ollama_llama_3_2_is_the_default(monkeypatch):
    for key in ("LLM_PROVIDER", "OLLAMA_MODEL", "OLLAMA_BASE_URL"):
        monkeypatch.delenv(key, raising=False)

    config = LLMConfig.from_env()

    assert config.provider == "ollama"
    assert config.model == "llama3.2:3b"
    assert config.base_url == "http://localhost:11434"


def test_openai_compatible_provider_is_configurable(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LLM_MODEL", "custom-model")
    monkeypatch.setenv("LLM_API_KEY", "test-key")

    config = LLMConfig.from_env()

    assert config.provider == "openai_compatible"
    assert config.base_url == "https://example.test/v1"
    assert config.model == "custom-model"
    assert config.api_key == "test-key"
