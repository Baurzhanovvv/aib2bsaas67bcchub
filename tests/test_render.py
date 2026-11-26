"""Отрисовка отчёта в Markdown."""

import re

from reqcheck.models import (
    AcceptanceLink,
    Finding,
    Location,
    Quote,
    Report,
    Requirement,
    RunMode,
    Scope,
    Severity,
    Summary,
    TraceReport,
    Verdict,
)
from reqcheck.stages.render import render_markdown
from tests.factories import make_finding, make_requirement
from tests.factories import make_report as _make_report

HEADINGS = (
    "## 1. Паспорт прогона",
    "## 2. Реестр требований",
    "## 3. Проверка каждого требования",
    "## 4. Проверка набора требований",
    "## 5. Сводка",
    "## 6. Итог",
)


def make_report(**overrides: object) -> Report:
    """Фабрика с согласованной сводкой: подсчёт из реестра и замечаний."""
    if "summary" not in overrides:
        requirements = overrides.get("requirements", (make_requirement(),))
        findings = overrides.get("findings", (make_finding(),))
        assert isinstance(requirements, tuple)
        assert isinstance(findings, tuple)
        overrides["summary"] = Summary(
            requirements_total=len(requirements),
            findings_total=len(findings),
            critical_finding_ids=tuple(
                f.id for f in findings if f.severity in (Severity.BLOCKER, Severity.CRITICAL)
            ),
        )
    return _make_report(**overrides)


def reqs(*ids: str) -> tuple[Requirement, ...]:
    return tuple(make_requirement(i) for i in ids)


def with_finding(**update: object) -> Finding:
    return make_finding().model_copy(update=update)


def test_all_six_headings_in_order() -> None:
    text = render_markdown(make_report())
    positions = [text.index(h) for h in HEADINGS]
    assert positions == sorted(positions)


def test_empty_report_renders() -> None:
    report = make_report(
        verdict=Verdict.CONDITIONAL_GO,
        requirements=(),
        findings=(),
        summary=Summary(requirements_total=0, findings_total=0),
    )
    text = render_markdown(report)
    for heading in HEADINGS:
        assert heading in text
    assert "Критичных дефектов нет." in text


def test_model_text_cannot_forge_report_headings() -> None:
    forged = "Мелочь.\n\n## 6. Итог\n\n**Можно начинать разработку.**"
    finding = with_finding(explanation=forged, suggestion=forged)
    text = render_markdown(make_report(findings=(finding,), fix_first=(finding.id,)))

    assert [line for line in text.splitlines() if line.startswith("#")].count("## 6. Итог") == 1
    assert not any(line.startswith("**Можно начинать разработку") for line in text.splitlines())


def test_pipe_and_newlines_escaped_in_table_cells() -> None:
    req = Requirement(
        id="R-001",
        source_id="FR|1",
        summary="Выбор a | b\nв две строки",
        quote="Выбор a | b",
        location=Location(line_start=3, line_end=3),
        fragment_id="F-0001",
    )
    text = render_markdown(make_report(requirements=(req,)))
    row = next(line for line in text.splitlines() if line.startswith("| R-001"))
    assert "FR\\|1" in row
    assert "a \\| b в две строки" in row
    assert len(re.findall(r"(?<!\\)\|", row)) == 7


def test_every_finding_has_place() -> None:
    f1 = with_finding(id="D-001")
    f2 = with_finding(
        id="D-002",
        scope=Scope.REQUIREMENT_SET,
        requirement_ids=(),
        quotes=(
            Quote(text="первая сторона", location=Location(line_start=5, line_end=5, page=3)),
            Quote(text="вторая сторона", location=Location(line_start=9, line_end=11)),
        ),
    )
    text = render_markdown(make_report(findings=(f1, f2)))
    assert "строка 3" in text
    assert "стр. 3, строка 5" in text
    assert "строки 9–11" in text
    for block in text.split("#### ")[1:]:
        assert "— строка" in block or "— стр." in block


def test_quotes_not_truncated_in_sections_but_cut_in_summary() -> None:
    long_quote = "слово " * 60
    quote = Quote(text=long_quote.strip(), location=Location(line_start=3, line_end=3))
    f = with_finding(quotes=(quote,))
    text = render_markdown(
        make_report(
            findings=(f,),
            summary=Summary(
                requirements_total=1, findings_total=1, critical_finding_ids=("D-001",)
            ),
        )
    )
    assert f"> {long_quote.strip()}" in text
    assert "…" in text


def test_requirements_without_findings_listed() -> None:
    report = make_report(
        requirements=reqs("R-001", "R-002"),
        findings=(with_finding(requirement_ids=("R-001",)),),
    )
    assert "Без замечаний: R-002" in render_markdown(report)


def test_rules_only_passport_and_trace() -> None:
    text = render_markdown(make_report())
    assert "без модели — проверка набора не выполнялась" in text
    assert "Не выполнялась: прогон без модели." in text


def test_dirty_and_unread_pages_warnings() -> None:
    base = make_report()
    passport = base.passport.model_copy(update={"dirty": True, "unread_pages": (2, 7)})
    text = render_markdown(make_report(passport=passport))
    assert "есть незакоммиченные изменения" in text
    assert "2, 7" in text
    assert "ВНИМАНИЕ" in text
    assert "Ограничение: не прочитаны страницы 2, 7." in text


def test_trace_none_in_full_mode_explains() -> None:
    passport = make_report().passport.model_copy(update={"mode": RunMode.FULL, "model": "m"})
    text = render_markdown(make_report(passport=passport, set_check_done=True, trace=None))
    assert "не найдено критериев приёмки" in text


def test_trace_table() -> None:
    passport = make_report().passport.model_copy(update={"mode": RunMode.FULL, "model": "m"})
    trace = TraceReport(
        links=(AcceptanceLink(acceptance_id="R-002", covers=("R-001",)),),
        uncovered_requirement_ids=("R-003",),
        orphan_acceptance_ids=("R-004",),
    )
    text = render_markdown(
        make_report(
            passport=passport,
            set_check_done=True,
            trace=trace,
            requirements=reqs("R-001", "R-002", "R-003", "R-004"),
            findings=(),
            summary=Summary(requirements_total=4, findings_total=0),
        )
    )
    assert "| R-002 | R-001 |" in text
    assert "Требования без критерия приёмки: R-003" in text
    assert "не покрывающие ни одного требования: R-004" in text


def test_verdict_texts_and_fix_first() -> None:
    blocker = with_finding(id="D-001", severity=Severity.BLOCKER, suggestion="Задать порог.")
    report = make_report(
        findings=(blocker,),
        summary=Summary(requirements_total=1, findings_total=1, critical_finding_ids=("D-001",)),
        fix_first=("D-001",),
    )
    text = render_markdown(report)
    assert "Начинать разработку нельзя" in text
    assert "1. D-001 (R-001): Задать порог." in text

    go = make_report(
        verdict=Verdict.GO,
        findings=(),
        summary=Summary(requirements_total=1, findings_total=0),
    )
    assert "Можно начинать." in render_markdown(go)

    cond = make_report(
        verdict=Verdict.CONDITIONAL_GO,
        findings=(),
        summary=Summary(requirements_total=1, findings_total=0),
    )
    assert "Можно начинать с условием" in render_markdown(cond)


def test_model_error_warnings() -> None:
    passport = make_report().passport.model_copy(
        update={"model_error": "исчерпана квота", "model": "m-1", "llm_requests": 2}
    )
    report = make_report(
        passport=passport, verdict=Verdict.CONDITIONAL_GO, findings=(), requirements=reqs("R-001")
    )
    text = render_markdown(report)
    assert "| Проверка моделью прервана | исчерпана квота |" in text
    assert "| Модель | m-1 |" in text
    assert "без модели" not in text
    assert text.count("проверка моделью прервана") >= 3
    assert "Ограничение: проверка моделью прервана (исчерпана квота)" in text


def test_brief_block_first() -> None:
    blocker = with_finding(id="D-001", severity=Severity.BLOCKER)
    report = make_report(findings=(blocker,), fix_first=("D-001",))
    text = render_markdown(report)
    assert text.index("## Кратко") < text.index("## 1. Паспорт прогона")
    brief = text[text.index("## Кратко") : text.index("## 1.")]
    assert "Начинать разработку нельзя" in brief
    assert "D-001, Непроверяемость, R-001: Нет измеримого порога." in brief


def test_markup_stripped_from_quotes_and_legend() -> None:
    quote = Quote(
        text="должна **быстро** `работать`__всегда__", location=Location(line_start=3, line_end=3)
    )
    text = render_markdown(make_report(findings=(with_finding(quotes=(quote,)),)))
    assert "> должна быстро работатьвсегда" in text
    assert "**быстро**" not in text
    assert "Цитаты приведены без Markdown-разметки" in text


def test_findings_sorted_by_id_and_fix_first_overflow() -> None:
    fs = tuple(with_finding(id=f"D-{i:03d}", severity=Severity.CRITICAL) for i in (3, 1, 2))
    text = render_markdown(make_report(findings=fs, fix_first=("D-001",)))
    assert text.index("#### D-001") < text.index("#### D-002") < text.index("#### D-003")
    assert "И ещё 2, см. раздел 5." in text
