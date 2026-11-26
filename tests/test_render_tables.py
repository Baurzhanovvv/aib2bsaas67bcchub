"""Сводные таблицы отчёта: реестр, критерии приёмки, вопросы заказчику, покрытие."""

from reqcheck.models import (
    AcceptanceLink,
    Finding,
    Location,
    Quote,
    RequirementKind,
    RunMode,
    Severity,
    Summary,
    TraceReport,
    Verdict,
)
from reqcheck.stages.render import render_markdown
from tests.factories import make_finding, make_report, make_requirement


def req(rid: str, kind: RequirementKind = RequirementKind.FUNCTIONAL, section=(), summary="Кратко"):
    return make_requirement(rid).model_copy(
        update={"kind": kind, "section": section, "summary": summary}
    )


def fnd(fid: str, rids: tuple[str, ...] = ("R-001",), **update: object) -> Finding:
    return make_finding(fid, rids).model_copy(update=update)


def report(requirements, findings, **overrides):
    crit = tuple(f.id for f in findings if f.severity in (Severity.BLOCKER, Severity.CRITICAL))
    fields = {
        "requirements": tuple(requirements),
        "findings": tuple(findings),
        "summary": Summary(
            requirements_total=len(requirements),
            findings_total=len(findings),
            by_severity={f.severity: 1 for f in findings},
            critical_finding_ids=crit,
        ),
    }
    fields.update(overrides)
    return make_report(**fields)


def test_registry_lists_all_requirements_with_findings_column_and_short_summary() -> None:
    long_summary = "слово " * 30
    rep = report([req("R-001", summary=long_summary), req("R-002")], [fnd("D-001")])
    text = render_markdown(rep)
    assert "| ID в отчёте | ID в документе | Тип | Кратко | Место | Замечания |" in text
    row1 = next(x for x in text.splitlines() if x.startswith("| R-001 |"))
    row2 = next(x for x in text.splitlines() if x.startswith("| R-002 |"))
    assert row1.endswith("| D-001 |")
    assert row2.endswith("| — |")
    assert "…" in row1 and len(row1) < 200


def test_registry_finding_found_via_quote_requirement_id() -> None:
    quote = Quote(
        text="работать быстро", location=Location(line_start=3, line_end=3), requirement_id="R-002"
    )
    f = fnd("D-001", ("R-001",), quotes=(quote,))
    text = render_markdown(report([req("R-001"), req("R-002")], [f]))
    row2 = next(x for x in text.splitlines() if x.startswith("| R-002 |"))
    assert row2.endswith("| D-001 |")


def test_acceptance_table_statuses() -> None:
    acc = RequirementKind.ACCEPTANCE
    rep = report(
        [req("R-001"), req("R-002", acc), req("R-003", acc)],
        [fnd("D-001", ("R-003",))],
        passport=make_report().passport.model_copy(update={"mode": RunMode.FULL, "model": "m"}),
        set_check_done=True,
        trace=TraceReport(links=(AcceptanceLink(acceptance_id="R-002", covers=("R-001",)),)),
    )
    text = render_markdown(rep)
    assert "### Таблица критериев приёмки" in text
    assert "| R-002 | Кратко | R-001 | ✓ |" in text
    assert "| R-003 | Кратко | — | ✗ D-001 |" in text
    assert text.index("Таблица критериев приёмки") < text.index("Трассировка критериев приёмки")


def test_acceptance_table_absent_criteria_line() -> None:
    text = render_markdown(report([req("R-001")], []))
    assert "Критериев приёмки в документе не найдено." in text


def test_customer_questions_listed_before_fix_first() -> None:
    f1 = fnd("D-002", suggestion="ВОПРОС ЗАКАЗЧИКУ: какой срок хранения?")
    f2 = fnd("D-001", suggestion="Вопрос заказчику — кто утверждает?")
    f3 = fnd("D-003", suggestion="Задать порог.")
    text = render_markdown(report([req("R-001")], [f1, f2, f3], fix_first=("D-001",)))
    assert "### Вопросы заказчику" in text
    assert "1. Кто утверждает? (D-001; R-001)" in text
    assert "2. Какой срок хранения? (D-002; R-001)" in text
    assert "Задать порог" not in text.split("### Вопросы заказчику")[1].split("Исправить")[0]
    assert text.index("### Вопросы заказчику") < text.index("Исправить в первую очередь")


def test_no_customer_questions_no_subsection() -> None:
    assert "Вопросы заказчику" not in render_markdown(report([req("R-001")], [fnd("D-001")]))


def test_coverage_by_sections() -> None:
    rep = report(
        [
            req("R-001", section=("Доставка", "Сроки")),
            req("R-002", section=("Доставка",)),
            req("R-003"),
        ],
        [
            fnd("D-001", ("R-001", "R-002"), severity=Severity.BLOCKER),
            fnd("D-002", ("R-002",), severity=Severity.MINOR),
        ],
    )
    text = render_markdown(rep)
    assert "| Доставка | 2 | 2 | 1 |" in text
    assert "| вне разделов | 1 | 0 | 0 |" in text


def test_severity_table_has_marks() -> None:
    text = render_markdown(report([req("R-001")], [fnd("D-001")]))
    assert "| 🟠 критичное | 1 |" in text
    assert "| 🔴 блокирующее | 0 |" in text


def test_verdict_unchanged_marker() -> None:
    assert report([req("R-001")], [], verdict=Verdict.GO).verdict is Verdict.GO
