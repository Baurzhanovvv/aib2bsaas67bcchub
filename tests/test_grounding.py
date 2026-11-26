"""Проверка цитат, справочника и дедупликации (FR-7, FR-9, FR-16, FR-18)."""

import pytest

from reqcheck import taxonomy
from reqcheck.models import (
    DefectType,
    Document,
    DocumentFormat,
    FindingSource,
    Fragment,
    FragmentKind,
    Impact,
    Location,
    RawFinding,
    Requirement,
    RequirementKind,
    Scope,
    Severity,
)
from reqcheck.stages.grounding import verify_and_classify

SHA = "0" * 64


def frag(n: int, text: str, line: int, page: int | None = None) -> Fragment:
    lines = text.count("\n")
    return Fragment(
        id=f"F-{n:04d}",
        kind=FragmentKind.PARAGRAPH,
        text=text,
        location=Location(line_start=line, line_end=line + lines, page=page),
    )


def req(
    n: int, fragment: Fragment, kind: RequirementKind = RequirementKind.FUNCTIONAL
) -> Requirement:
    return Requirement(
        id=f"R-{n:03d}",
        kind=kind,
        summary="s",
        quote=fragment.text,
        location=fragment.location,
        fragment_id=fragment.id,
    )


def doc(*fragments: Fragment) -> Document:
    return Document(name="d.md", sha256=SHA, format=DocumentFormat.MARKDOWN, fragments=fragments)


def raw(
    quotes: tuple[str, ...],
    ids: tuple[str, ...] = ("R-001",),
    defect: DefectType = DefectType.AMBIGUOUS,
    scope: Scope = Scope.REQUIREMENT,
    source: FindingSource = FindingSource.MODEL,
    suggestion: str | None = None,
    explanation: str = "Пояснение.",
    impact: Impact = Impact.MEDIUM,
) -> RawFinding:
    return RawFinding(
        defect_type=defect,
        requirement_ids=ids,
        quotes=quotes,
        explanation=explanation,
        impact=impact,
        suggestion=suggestion,
        scope=scope,
        source=source,
        rule_id="r1" if source is FindingSource.RULE else None,
    )


F1 = frag(1, "Склад должен «оперативно»  выдавать\nгрузы курьерам по заявке.", 10, page=2)
F2 = frag(2, "Курьер обязан подтверждать получение груза подписью.", 20)
DOC = doc(F1, F2)
REQS = (req(1, F1), req(2, F2))


def run(raws: list[RawFinding], document: Document = DOC, reqs: tuple[Requirement, ...] = REQS):
    return verify_and_classify(document, reqs, raws)


def test_quote_with_other_spaces_quotes_and_newline_returned_original() -> None:
    result = run([raw(('склад должен "оперативно" выдавать грузы'.replace("с", "С", 1),))])
    quote = result.findings[0].quotes[0]
    assert quote.text == "Склад должен «оперативно»  выдавать\nгрузы"
    assert (quote.location.line_start, quote.location.line_end) == (10, 11)
    assert quote.location.page == 2
    assert quote.requirement_id == "R-001"
    assert result.quotes_checked == result.quotes_confirmed == 1


def test_single_line_quote_line_numbers() -> None:
    multi = frag(1, "Первая строка текста.\nВторая строка с нужной цитатой.", 5)
    result = run([raw(("нужной цитатой",), ids=())], doc(multi), (req(1, multi),))
    loc = result.findings[0].quotes[0].location
    assert (loc.line_start, loc.line_end) == (6, 6)
    assert result.findings[0].requirement_ids == ("R-001",)


def test_pdf_hyphenated_word_wrap() -> None:
    f = frag(1, "Система должна выполнять требова-\nние по отчётности.", 3, page=1)
    result = run([raw(("выполнять требование по отчётности",))], doc(f), (req(1, f),))
    quote = result.findings[0].quotes[0]
    assert quote.text == "выполнять требова-\nние по отчётности"
    assert (quote.location.line_start, quote.location.line_end) == (3, 4)


def test_dashes_and_nbsp_and_soft_hyphen() -> None:
    f = frag(1, "Срок хранения — 5 лет, ука­зан в договоре.", 1)
    result = run([raw(("Срок хранения - 5 лет, указан",))], doc(f), (req(1, f),))
    assert result.findings


def test_case_is_not_normalized() -> None:
    assert not run([raw(("склад должен «оперативно»",))]).findings


def test_missing_quote_drops_finding() -> None:
    result = run([raw(("такой цитаты в документе нет",))])
    assert result.findings == ()
    assert (result.quotes_checked, result.quotes_confirmed, result.findings_dropped) == (1, 0, 1)


def test_contradiction_with_one_missing_quote_dropped() -> None:
    finding = raw(
        ("Курьер обязан подтверждать", "выдуманная вторая сторона"),
        ids=("R-002",),
        defect=DefectType.CONTRADICTION,
        scope=Scope.REQUIREMENT_SET,
    )
    result = run([finding])
    assert result.findings == ()
    assert (result.quotes_checked, result.quotes_confirmed, result.findings_dropped) == (2, 1, 1)


def test_contradiction_is_blocker_with_both_quotes() -> None:
    finding = raw(
        ("Курьер обязан подтверждать", "Склад должен «оперативно»"),
        ids=("R-001", "R-002"),
        defect=DefectType.CONTRADICTION,
        scope=Scope.REQUIREMENT_SET,
        impact=Impact.HIGH,
    )
    f = run([finding]).findings[0]
    assert f.severity is Severity.BLOCKER
    assert [q.requirement_id for q in f.quotes] == ["R-002", "R-001"]


def test_prefers_place_in_listed_requirement_then_first() -> None:
    a = frag(1, "Общий срок ответа указан здесь.", 1)
    b = frag(2, "Общий срок ответа указан здесь же.", 5)
    document, reqs = doc(a, b), (req(1, a), req(2, b))
    in_second = run([raw(("Общий срок ответа",), ids=("R-002",))], document, reqs)
    assert in_second.findings[0].quotes[0].location.line_start == 5
    unlisted = run([raw(("Общий срок ответа",), ids=())], document, reqs)
    assert unlisted.findings[0].quotes[0].location.line_start == 1


def test_unknown_requirement_ids_are_removed() -> None:
    f = run([raw(("Курьер обязан подтверждать",), ids=("R-999",))]).findings[0]
    assert f.requirement_ids == ("R-002",)


def test_taxonomy_covers_every_defect_type() -> None:
    assert set(taxonomy.TAXONOMY) == set(DefectType)
    assert all(m.standard_ref for m in taxonomy.TAXONOMY.values())


@pytest.mark.parametrize("defect", list(DefectType))
def test_severity_table_complete_and_monotonic(defect: DefectType) -> None:
    ranks = [taxonomy.SEVERITY_ORDER.index(taxonomy.severity_for(defect, i)) for i in Impact]
    assert ranks == sorted(ranks)


def test_same_type_different_impact_different_severity() -> None:
    sev = {i: taxonomy.severity_for(DefectType.CONTRADICTION, i) for i in Impact}
    assert sev == {
        Impact.HIGH: Severity.BLOCKER,
        Impact.MEDIUM: Severity.CRITICAL,
        Impact.LOW: Severity.MAJOR,
    }
    quote = ("Курьер обязан подтверждать",)
    got = [run([raw(quote, ids=("R-002",), impact=i)]).findings[0].severity for i in Impact]
    assert got == [Severity.MAJOR, Severity.MAJOR, Severity.MINOR]


def test_markup_applied() -> None:
    f = run([raw(("Курьер обязан подтверждать",), ids=("R-002",))]).findings[0]
    assert f.standard_ref.startswith("ISO/IEC/IEEE 29148:2018, 5.2.5")


def test_noop_suggestion_dropped() -> None:
    same_as_quote = raw(
        ("Курьер обязан подтверждать",), ids=("R-002",), suggestion="курьер обязан подтверждать."
    )
    same_as_req = raw(
        ("Курьер обязан",), ids=("R-002",), suggestion=F2.text, defect=DefectType.UNVERIFIABLE
    )
    changed = raw(
        ("получение груза",), ids=("R-002",), suggestion="Подтверждение — электронной подписью."
    )
    result = run([same_as_quote, same_as_req, changed])
    assert len(result.findings) == 1
    assert result.findings_dropped == 2


def test_rule_and_model_duplicates_merge_preferring_model() -> None:
    rule = raw(("оперативно",), source=FindingSource.RULE, explanation="Слово-маркер.")
    model = raw(("«оперативно»  выдавать",), explanation="Нет порога времени выдачи.")
    result = run([rule, model])
    assert len(result.findings) == 1
    assert result.findings[0].source is FindingSource.MODEL
    assert result.findings[0].explanation == "Нет порога времени выдачи."


def test_overlapping_group_types_merge_to_more_severe() -> None:
    amb = raw(("должен «оперативно»",), defect=DefectType.AMBIGUOUS)
    unv = raw(("«оперативно»  выдавать",), defect=DefectType.UNVERIFIABLE)
    result = run([amb, unv])
    assert [f.defect_type for f in result.findings] == [DefectType.UNVERIFIABLE]


def test_non_overlapping_group_types_stay_separate() -> None:
    amb = raw(("Склад должен",), defect=DefectType.AMBIGUOUS)
    unv = raw(("курьерам по заявке",), defect=DefectType.UNVERIFIABLE)
    assert len(run([amb, unv]).findings) == 2


def test_set_scope_dedup_by_quotes() -> None:
    quotes = ("Курьер обязан подтверждать", "Склад должен «оперативно»")
    a = raw(
        quotes, defect=DefectType.CONTRADICTION, scope=Scope.REQUIREMENT_SET, ids=("R-001", "R-002")
    )
    b = raw(
        quotes[::-1], defect=DefectType.CONTRADICTION, scope=Scope.REQUIREMENT_SET, ids=("R-002",)
    )
    assert len(run([a, b]).findings) == 1


def test_ids_and_order_deterministic() -> None:
    minor = raw(("Курьер обязан подтверждать",), ids=("R-002",), defect=DefectType.NOT_SINGULAR)
    major_late = raw(("получение груза",), ids=("R-002",), defect=DefectType.INCOMPLETE)
    major_early = raw(("Склад должен",), defect=DefectType.AMBIGUOUS)
    crit = raw(("по заявке",), defect=DefectType.UNVERIFIABLE, impact=Impact.HIGH)
    first = run([minor, major_late, major_early, crit])
    second = run([crit, major_early, major_late, minor])
    assert [f.id for f in first.findings] == ["D-001", "D-002", "D-003", "D-004"]
    assert [f.defect_type for f in first.findings] == [
        DefectType.UNVERIFIABLE,
        DefectType.AMBIGUOUS,
        DefectType.INCOMPLETE,
        DefectType.NOT_SINGULAR,
    ]
    assert first == second


def test_acceptance_promotion_option(monkeypatch: pytest.MonkeyPatch) -> None:
    acc = frag(1, "Приёмка: заказ считается выданным после проверки.", 1)
    reqs = (req(1, acc, RequirementKind.ACCEPTANCE),)
    finding = raw(("после проверки",), defect=DefectType.INCOMPLETE)
    assert run([finding], doc(acc), reqs).findings[0].severity is Severity.MAJOR
    monkeypatch.setattr(taxonomy, "PROMOTE_ACCEPTANCE_TO_BLOCKER", True)
    assert run([finding], doc(acc), reqs).findings[0].severity is Severity.BLOCKER


def test_contradiction_with_same_place_quotes_dropped() -> None:
    finding = raw(
        ("Курьер обязан подтверждать", "курьер обязан".capitalize() + " подтверждать получение"),
        ids=("R-002",),
        defect=DefectType.CONTRADICTION,
        scope=Scope.REQUIREMENT_SET,
    )
    result = run([finding])
    assert result.findings == ()
    assert result.findings_dropped == 1


def test_markdown_emphasis_ignored_and_original_text_returned() -> None:
    f = frag(1, "**ТР-1.** Система **должна** хранить `журнал` и __архив__ в snake_case.", 7)
    document, reqs = doc(f), (req(1, f),)
    result = run([raw(("Система должна хранить журнал и архив в snake_case",))], document, reqs)
    quote = result.findings[0].quotes[0]
    assert quote.text == "Система **должна** хранить `журнал` и __архив__ в snake_case"
    assert quote.location.line_start == 7
    assert run([raw(("Система должна хранить",))], document, reqs).findings
    assert not run([raw(("snakecase и всё прочее",))], document, reqs).findings


def test_cross_scope_merge_keeps_more_quotes_and_max_severity() -> None:
    pair = ("Склад должен «оперативно»", "Курьер обязан подтверждать")
    single = raw(
        ("«оперативно»  выдавать",),
        defect=DefectType.INCONSISTENT_TERMS,
        impact=Impact.HIGH,
    )
    wide = raw(
        pair,
        ids=("R-001", "R-002"),
        defect=DefectType.CONTRADICTION,
        scope=Scope.REQUIREMENT_SET,
        impact=Impact.LOW,
    )
    result = run([single, wide])
    assert len(result.findings) == 1
    f = result.findings[0]
    assert f.defect_type is DefectType.CONTRADICTION
    assert f.scope is Scope.REQUIREMENT_SET
    assert f.severity is Severity.MAJOR
    assert len(f.quotes) == 2


def test_cross_group_types_not_merged() -> None:
    a = raw(("должен «оперативно»",), defect=DefectType.AMBIGUOUS)
    b = raw(("«оперативно»  выдавать",), defect=DefectType.INCOMPLETE)
    assert len(run([a, b]).findings) == 2


def test_blocker_needs_functional_or_acceptance_quote() -> None:
    quotes = ("Склад должен «оперативно»", "Курьер обязан подтверждать")

    def severity(kind_a: RequirementKind, kind_b: RequirementKind) -> Severity:
        reqs = (req(1, F1, kind_a), req(2, F2, kind_b))
        finding = raw(
            quotes,
            ids=("R-001", "R-002"),
            defect=DefectType.CONTRADICTION,
            scope=Scope.REQUIREMENT_SET,
            impact=Impact.HIGH,
        )
        return run([finding], DOC, reqs).findings[0].severity

    interface, functional = RequirementKind.INTERFACE, RequirementKind.FUNCTIONAL
    assert severity(interface, interface) is Severity.CRITICAL
    assert severity(interface, functional) is Severity.BLOCKER


def test_heading_marks_stripped_from_quote() -> None:
    head = Fragment(
        id="F-0001",
        kind=FragmentKind.HEADING,
        text="Раздел про доставку",
        location=Location(line_start=1, line_end=1),
    )
    result = run([raw(("## Раздел про доставку",), ids=())], doc(head), (req(1, head),))
    assert result.findings[0].quotes[0].text == "Раздел про доставку"
