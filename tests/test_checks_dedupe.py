"""Слияние дублей FR-6: из замечаний с одинаковыми цитатами остаётся самое серьёзное."""

from reqcheck.llm import FindingDraft, SetCheckResponse
from reqcheck.models import DefectType, Impact, RequirementKind
from reqcheck.stages import checks
from reqcheck.taxonomy import SEVERITY_ORDER, severity_for
from tests.test_checks import FakeLLM, document, req

QUOTES = ("принимает поставку номер 1", "принимает поставку номер 2")


def draft(defect: DefectType, impact: Impact, explanation: str) -> FindingDraft:
    return FindingDraft(
        defect_type=defect,
        requirement_ids=("R-001", "R-002"),
        quotes=QUOTES,
        explanation=explanation,
        suggestion="Было: «номер 2» → Стало: «номер 1»",
        impact=impact,
    )


def test_more_severe_duplicate_wins_regardless_of_order() -> None:
    reqs = (req(1), req(2, RequirementKind.FUNCTIONAL))
    weak = draft(DefectType.DUPLICATE, Impact.LOW, "Два требования говорят одно и то же.")
    strong = draft(DefectType.CONTRADICTION, Impact.HIGH, "A требует X, B требует не-X.")
    rank = SEVERITY_ORDER.index
    assert rank(severity_for(strong.defect_type, strong.impact)) < rank(
        severity_for(weak.defect_type, weak.impact)
    )
    llm = FakeLLM([SetCheckResponse(findings=(weak,)), SetCheckResponse(findings=(strong,))])
    result, _ = checks.check_set(document(reqs), reqs, llm)
    assert [f.defect_type for f in result] == [DefectType.CONTRADICTION]


def test_equal_severity_keeps_longer_explanation() -> None:
    reqs = (req(1), req(2))
    short = draft(DefectType.CONTRADICTION, Impact.HIGH, "Конфликт.")
    long = draft(DefectType.CONTRADICTION, Impact.HIGH, "A требует приёмку за час, B — за два.")
    llm = FakeLLM([SetCheckResponse(findings=(short, long))])
    result, _ = checks.check_set(document(reqs), reqs, llm)
    assert [f.explanation for f in result] == [long.explanation]
