"""Сборка отчёта: паспорт, сводка, итог (FR-10, FR-11)."""

from datetime import UTC, datetime

from reqcheck.llm import LLMUsage
from reqcheck.models import (
    Document,
    Finding,
    GroundingResult,
    Passport,
    Report,
    Requirement,
    RunMode,
    Severity,
    Summary,
    TraceReport,
    Verdict,
)
from reqcheck.version import agent_version

MAX_FIX_FIRST = 10
CRITICAL_SEVERITIES = (Severity.BLOCKER, Severity.CRITICAL)


def decide_verdict(findings: tuple[Finding, ...], *, complete: bool) -> Verdict:
    """Итог считает код (README: итог ставит код, не модель).

    - есть замечание BLOCKER — NO_GO;
    - иначе есть CRITICAL — CONDITIONAL_GO;
    - иначе GO.

    Проверка неполная (прогон без модели или есть непрочитанные страницы) —
    итог не выше CONDITIONAL_GO: «чисто» нельзя утверждать о том, чего не видели.
    """
    severities = {f.severity for f in findings}
    if Severity.BLOCKER in severities:
        return Verdict.NO_GO
    if Severity.CRITICAL in severities or not complete:
        return Verdict.CONDITIONAL_GO
    return Verdict.GO


def _fix_first(findings: tuple[Finding, ...]) -> tuple[str, ...]:
    """Все BLOCKER, затем CRITICAL; не больше MAX_FIX_FIRST; внутри уровня — по id."""
    ordered: list[str] = []
    for severity in CRITICAL_SEVERITIES:
        ordered += sorted(f.id for f in findings if f.severity is severity)
    return tuple(ordered[:MAX_FIX_FIRST])


def _summary(requirements: tuple[Requirement, ...], findings: tuple[Finding, ...]) -> Summary:
    by_severity = {s: sum(1 for f in findings if f.severity is s) for s in Severity}
    critical = sorted(f.id for f in findings if f.severity in CRITICAL_SEVERITIES)
    return Summary(
        requirements_total=len(requirements),
        findings_total=len(findings),
        by_severity=by_severity,
        critical_finding_ids=tuple(critical),
    )


def _passport(
    document: Document,
    grounded: GroundingResult,
    usage: LLMUsage | None,
    duration_s: float,
    model_error: str | None,
) -> Passport:
    version, dirty = agent_version()
    counters = usage if usage is not None else LLMUsage()
    return Passport(
        document_name=document.name,
        document_sha256=document.sha256,
        run_at=datetime.now(UTC),
        agent_version=version,
        dirty=dirty,
        mode=RunMode.FULL if usage is not None and model_error is None else RunMode.RULES_ONLY,
        model_error=model_error,
        model=counters.model,
        llm_requests=counters.requests,
        cache_hits=counters.cache_hits,
        tokens_in=counters.tokens_in,
        tokens_out=counters.tokens_out,
        quotes_checked=grounded.quotes_checked,
        quotes_confirmed=grounded.quotes_confirmed,
        findings_dropped=grounded.findings_dropped,
        duration_s=max(duration_s, 0.0),
        unread_pages=document.unread_pages,
    )


def build_report(
    document: Document,
    requirements: tuple[Requirement, ...],
    grounded: GroundingResult,
    trace: TraceReport | None,
    usage: LLMUsage | None,
    duration_s: float,
    model_error: str | None = None,
) -> Report:
    """Собрать Report. Итог GO / CONDITIONAL GO / NO-GO считает код, не модель.

    usage=None — прогон без модели: mode=RULES_ONLY, set_check_done=False.
    model_error — модель упала посреди прогона: отчёт выпускается, mode=RULES_ONLY,
    set_check_done=False, итог не выше CONDITIONAL_GO; счётчики usage остаются в паспорте.
    Правило итога — в `decide_verdict`.
    """
    findings = grounded.findings
    complete = usage is not None and model_error is None and not document.unread_pages
    return Report(
        passport=_passport(document, grounded, usage, duration_s, model_error),
        requirements=requirements,
        findings=findings,
        trace=trace,
        set_check_done=usage is not None and model_error is None and trace is not None,
        summary=_summary(requirements, findings),
        verdict=decide_verdict(findings, complete=complete),
        fix_first=_fix_first(findings),
    )
