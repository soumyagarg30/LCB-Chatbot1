from app import build_tracker_answer


def test_build_tracker_answer_summarizes_plan_data():
    plans = [
        {
            "title": "Dealer activation",
            "strategy": "Run field demos and distributor incentives.",
            "owner": "Ravi",
            "status": "in_progress",
            "created_by": "admin",
        },
        {
            "title": "Farmer education",
            "strategy": "Use WhatsApp campaigns and demo plots.",
            "owner": "Meera",
            "status": "not_started",
            "created_by": "admin",
        },
    ]

    answer = build_tracker_answer("What strategies are in the tracker?", plans)

    assert "Dealer activation" in answer
    assert "Farmer education" in answer
    assert "field demos" in answer.lower()
    assert "WhatsApp" in answer
