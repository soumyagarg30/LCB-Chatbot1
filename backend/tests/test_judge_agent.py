from judge_agent import LLMJudgeAgent


class StubClient:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    def generate(self, prompt):
        if self.error:
            raise self.error
        return self.result


def test_judge_returns_structured_normalized_result():
    client = StubClient('''```json
    {"verdict":"pass","score":92,"dimensions":{"relevance":95,"correctness":90,
    "groundedness":88,"clarity":94,"safety":100},"missing_information":["A concrete example"],
    "issues":["Too abstract"],"feedback":"Add an example and supporting detail."}
    ```''')
    result = LLMJudgeAgent(client=client, enabled=True).judge(
        question="What is it?", response="An answer", context="Reference text"
    )

    assert result["status"] == "completed"
    assert result["verdict"] == "warning"
    assert result["score"] == 79
    assert result["dimensions"]["safety"] == 100
    assert result["missing_information"] == ["A concrete example"]


def test_judge_uses_score_to_enforce_verdict_band():
    client = StubClient('''{"verdict":"pass","score":65,"dimensions":{"relevance":65,
    "correctness":65,"groundedness":65,"clarity":65,"safety":65},"issues":[],"feedback":""}''')
    result = LLMJudgeAgent(client=client, enabled=True).judge(question="Q", response="A")
    assert result["verdict"] == "warning"


def test_judge_failure_does_not_fail_chatbot_response():
    result = LLMJudgeAgent(
        client=StubClient(error=RuntimeError("judge offline")), enabled=True
    ).judge(question="Q", response="A")

    assert result["status"] == "fallback"
    assert result["verdict"] in ("warning", "fail")
    assert isinstance(result["score"], int)


def test_malformed_json_uses_scored_fallback_not_unavailable():
    result = LLMJudgeAgent(client=StubClient('{"score": 70,,}'), enabled=True).judge(
        question="Q", response="A"
    )
    assert result["status"] == "fallback"
    assert result["score"] is not None
    assert result["evaluation_method"] == "deterministic_fallback"
