"""Фабрики минимальных валидных объектов для тестов."""

from datetime import UTC, datetime

from reqcheck.models import (
    Criterion,
    DefectType,
    Document,
    DocumentFormat,
    Finding,
    FindingClass,
    FindingSource,
    Fragment,
    FragmentKind,
    Location,
    Passport,
    Quote,
    Report,
    Requirement,
    RunMode,
    Scope,
    Severity,
    Summary,
    Verdict,
)

SHA = "0" * 64
LINE = Location(line_start=3, line_end=3)
TEXT = "Система должна работать быстро."


def make_document() -> Document:
    fragment = Fragment(id="F-0001", kind=FragmentKind.PARAGRAPH, text=TEXT, location=LINE)
    return Document(
        name="doc.md", sha256=SHA, format=DocumentFormat.MARKDOWN, fragments=(fragment,)
    )


def make_requirement(req_id: str = "R-001") -> Requirement:
    return Requirement(
        id=req_id,
        summary="Система работает быстро",
        quote=TEXT,
        location=LINE,
        fragment_id="F-0001",
    )


def make_finding(
    finding_id: str = "D-001", requirement_ids: tuple[str, ...] = ("R-001",)
) -> Finding:
    return Finding(
        id=finding_id,
        scope=Scope.REQUIREMENT,
        defect_type=DefectType.UNVERIFIABLE,
        criterion=Criterion.REQ_VERIFIABLE,
        finding_class=FindingClass.MISSING_INFO,
        severity=Severity.CRITICAL,
        standard_ref="ISO/IEC/IEEE 29148:2018, 5.2.5",
        requirement_ids=requirement_ids,
        quotes=(Quote(text="работать быстро", location=LINE, requirement_id="R-001"),),
        explanation="Нет измеримого порога.",
        source=FindingSource.MODEL,
    )


def make_report(verdict: Verdict = Verdict.NO_GO, **overrides: object) -> Report:
    fields: dict[str, object] = {
        "passport": Passport(
            document_name="doc.md",
            document_sha256=SHA,
            run_at=datetime(2026, 1, 1, tzinfo=UTC),
            agent_version="test",
            dirty=False,
            mode=RunMode.RULES_ONLY,
        ),
        "requirements": (make_requirement(),),
        "findings": (make_finding(),),
        "set_check_done": False,
        "summary": Summary(requirements_total=1, findings_total=1),
        "verdict": verdict,
    }
    fields.update(overrides)
    return Report(**fields)
