"""Инварианты контрактов: то, на что этапы вправе полагаться."""

import pytest
from pydantic import ValidationError

from reqcheck.models import (
    DefectType,
    FindingSource,
    Location,
    ModelFinding,
    RawFinding,
    Scope,
)
from tests.factories import make_finding, make_report


def test_location_rejects_reversed_lines() -> None:
    with pytest.raises(ValidationError):
        Location(line_start=5, line_end=4)


def test_finding_needs_a_quote() -> None:
    with pytest.raises(ValidationError):
        ModelFinding(defect_type=DefectType.AMBIGUOUS, quotes=(), explanation="x")


def test_finding_rejects_short_quote() -> None:
    with pytest.raises(ValidationError):
        ModelFinding(defect_type=DefectType.AMBIGUOUS, quotes=("быстро",), explanation="x")


def test_contradiction_needs_both_sides() -> None:
    with pytest.raises(ValidationError):
        ModelFinding(
            defect_type=DefectType.CONTRADICTION,
            quotes=("лимит задаётся в одной валюте",),
            explanation="x",
        )


def test_rule_finding_needs_rule_id() -> None:
    with pytest.raises(ValidationError):
        RawFinding(
            defect_type=DefectType.AMBIGUOUS,
            quotes=("работать быстро",),
            explanation="x",
            scope=Scope.REQUIREMENT,
            source=FindingSource.RULE,
        )


def test_contracts_are_frozen() -> None:
    report = make_report()
    with pytest.raises(ValidationError):
        report.verdict = "go"


def test_contracts_reject_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Location(line_start=1, line_end=1, column=3)


def test_report_rejects_requirement_missing_from_registry() -> None:
    with pytest.raises(ValidationError):
        make_report(findings=(make_finding(requirement_ids=("R-999",)),))


def test_report_rejects_unknown_fix_first() -> None:
    with pytest.raises(ValidationError):
        make_report(fix_first=("D-404",))
