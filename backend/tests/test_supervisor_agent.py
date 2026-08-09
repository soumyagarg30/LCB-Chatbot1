from supervisor_agent import SupervisorAgent


class StubClient:
    def __init__(self, result=None, error=None):
        self.result, self.error = result, error

    def generate(self, prompt):
        if self.error:
            raise self.error
        return self.result


def test_routes_marketing_request_from_llm_decision():
    supervisor = SupervisorAgent(StubClient('{"agent":"marketing","reason":"Campaign request","confidence":0.94}'))
    result = supervisor.route("Create a social campaign")
    assert result["agent"] == "marketing"
    assert result["display_name"] == "Marketing Strategist"


def test_tracker_is_unavailable_to_non_admin():
    supervisor = SupervisorAgent(StubClient('{"agent":"tracker","reason":"Status request","confidence":0.9}'))
    result = supervisor.route("Show tracker status", is_admin=False)
    assert result["agent"] == "general"
    assert result["routing_method"] == "fallback"


def test_fallback_routes_marketing_when_supervisor_llm_is_offline():
    supervisor = SupervisorAgent(StubClient(error=RuntimeError("offline")))
    result = supervisor.route("Build a dealer activation strategy")
    assert result["agent"] == "marketing"
    assert result["routing_method"] == "fallback"


def test_routes_competitor_request_from_llm_decision():
    supervisor = SupervisorAgent(StubClient('{"agent":"competitive_intelligence","reason":"Competitor assessment request","confidence":0.96}'))
    result = supervisor.route("Assess our competitors and tell us how to get ahead")
    assert result["agent"] == "competitive_intelligence"
    assert result["display_name"] == "Competitive Intelligence Strategist"


def test_fallback_routes_competitor_request_when_supervisor_llm_is_offline():
    supervisor = SupervisorAgent(StubClient(error=RuntimeError("offline")))
    result = supervisor.route("Compare us with our competitors and recommend how to outperform them")
    assert result["agent"] == "competitive_intelligence"
    assert result["routing_method"] == "fallback"
