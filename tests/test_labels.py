from triage.labels import FailureCategory, VALID_CATEGORIES


def test_categories_match_project_definition():
    assert VALID_CATEGORIES == {
        "configuration",
        "timeout",
        "communication",
        "resource_exhaustion",
    }


def test_category_values_are_unique():
    values = [
        category.value
        for category in FailureCategory.__members__.values()
    ]

    assert len(values) == len(set(values))


def test_unknown_category_is_not_accepted():
    assert "unknown_failure" not in VALID_CATEGORIES