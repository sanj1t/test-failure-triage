"""Failure-category definitions used throughout the project."""

from enum import Enum


class FailureCategory(str, Enum):
    CONFIGURATION = "configuration"
    TIMEOUT = "timeout"
    COMMUNICATION = "communication"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


VALID_CATEGORIES = frozenset(
    category.value for category in FailureCategory
)