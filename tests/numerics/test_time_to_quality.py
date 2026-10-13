from scripts.benchmark_time_to_quality import crossing


def test_crossing_requires_two_adjacent_successes() -> None:
    rows = [
        {"seconds": 1.0, "passed": False},
        {"seconds": 2.0, "passed": True},
        {"seconds": 3.0, "passed": False},
        {"seconds": 4.0, "passed": True},
        {"seconds": 5.0, "passed": True},
    ]
    assert crossing(rows) == {
        "status": "observed",
        "lower_seconds": 3.0,
        "upper_seconds": 4.0,
        "confirmed_seconds": 5.0,
        "left_censored": False,
    }
    assert crossing(rows[:4])["status"] == "not_confirmed"


def test_crossing_marks_unknown_time_before_first_snapshot() -> None:
    result = crossing([{"seconds": 8.0, "passed": True}, {"seconds": 9.0, "passed": True}])
    assert result["left_censored"] is True
    assert result["lower_seconds"] is None
