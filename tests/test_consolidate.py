"""Смысловая склейка на фейковом клиенте без сети."""

from dataclasses import dataclass, field

from pydantic import BaseModel

from reqcheck.errors import ModelError
from reqcheck.llm import LLMResult, LLMUsage
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
    GroundingResult,
    Location,
    Quote,
    Scope,
    Severity,
)
from reqcheck.stages.consolidate import DuplicateGroups, GroupDraft, consolidate

DOC = Document(name="d.md", sha256="b" * 64, format=DocumentFormat.MARKDOWN, fragments=())


@dataclass
class FakeLLM:
    reply: BaseModel | None = None
    error: Exception | None = None
    calls: list[tuple[str, str]] = field(default_factory=list)

    def complete(self, *, system: str, user: str, response_model: type) -> LLMResult:
        self.calls.append((system, user))
        if self.error:
            raise self.error
        assert isinstance(self.reply, response_model)
        return LLMResult(data=self.reply, model="fake")

    def usage(self) -> LLMUsage:
        return LLMUsage()


def finding(
    num: int,
    line: int,
    defect: DefectType = DefectType.AMBIGUOUS,
    severity: Severity = Severity.MAJOR,
    reqs: tuple[str, ...] = ("R-001",),
    text: str = "",
    source: FindingSource = FindingSource.MODEL,
) -> Finding:
    return Finding(
        id=f"D-{num:03d}",
        scope=Scope.REQUIREMENT,
        defect_type=defect,
        criterion=Criterion.REQ_UNAMBIGUOUS,
        finding_class=FindingClass.RECOMMENDATION,
        severity=severity,
        standard_ref="ref",
        requirement_ids=reqs,
        quotes=(
            Quote(
                text=text or f"цитата номер {line}",
                location=Location(line_start=line, line_end=line),
            ),
        ),
        explanation=f"Пояснение {num}.",
        source=source,
        rule_id="r1" if source is FindingSource.RULE else None,
    )


def grounded(*findings: Finding) -> GroundingResult:
    return GroundingResult(findings=findings, quotes_checked=5, quotes_confirmed=5)


def groups(*ids: tuple[str, ...]) -> DuplicateGroups:
    return DuplicateGroups(groups=tuple(GroupDraft(ids=i, reason="одна проблема") for i in ids))


def run(g: GroundingResult, llm: FakeLLM | None) -> GroundingResult:
    return consolidate(DOC, (), g, llm)


def test_no_llm_or_single_finding_returns_as_is() -> None:
    g = grounded(finding(1, 1), finding(2, 2))
    assert run(g, None) is g
    one = grounded(finding(1, 1))
    llm = FakeLLM(groups(("D-001",)))
    assert run(one, llm) is one
    assert llm.calls == []


def test_group_merged_and_renumbered() -> None:
    a = finding(1, 5)
    b = finding(2, 9, DefectType.UNVERIFIABLE, Severity.CRITICAL, reqs=("R-001", "R-002"))
    c = finding(3, 2, DefectType.NOT_SINGULAR, Severity.MINOR)
    result = run(grounded(a, b, c), FakeLLM(groups(("D-001", "D-002"))))
    assert [(f.id, f.defect_type) for f in result.findings] == [
        ("D-001", DefectType.UNVERIFIABLE),
        ("D-002", DefectType.NOT_SINGULAR),
    ]
    merged = result.findings[0]
    assert merged.severity is Severity.CRITICAL
    assert merged.requirement_ids == ("R-001", "R-002")
    assert [q.text for q in merged.quotes] == ["цитата номер 9", "цитата номер 5"]
    assert result.quotes_checked == 5


def test_duplicate_quotes_not_repeated() -> None:
    a, b = finding(1, 5, text="одна и та же цитата"), finding(2, 5, text="одна и та же цитата")
    result = run(grounded(a, b), FakeLLM(groups(("D-001", "D-002"))))
    assert len(result.findings) == 1
    assert len(result.findings[0].quotes) == 1


def test_unknown_and_repeated_ids_dropped() -> None:
    g = grounded(finding(1, 1), finding(2, 2))
    unchanged = run(g, FakeLLM(groups(("D-001", "D-099"), ("D-002", "D-002"))))
    assert unchanged is g


def test_overlapping_groups_first_wins() -> None:
    g = grounded(finding(1, 1), finding(2, 2), finding(3, 3))
    result = run(g, FakeLLM(groups(("D-001", "D-002"), ("D-002", "D-003"))))
    assert len(result.findings) == 2
    assert {len(f.quotes) for f in result.findings} == {2, 1}


def test_model_error_returns_unchanged() -> None:
    g = grounded(finding(1, 1), finding(2, 2))
    assert run(g, FakeLLM(error=ModelError("квота"))) is g


def test_findings_are_wrapped_as_data() -> None:
    llm = FakeLLM(groups())
    run(grounded(finding(1, 1), finding(2, 2)), llm)
    system, user = llm.calls[0]
    assert "<<<DOC-" in system and user.startswith("<<<DOC-")


def test_explanation_keeps_both_types_without_service_notes() -> None:
    a = finding(1, 5)
    b = finding(2, 9, DefectType.UNVERIFIABLE, Severity.CRITICAL)
    merged = run(grounded(a, b), FakeLLM(groups(("D-001", "D-002")))).findings[0]
    assert merged.explanation == "Пояснение 2. Пояснение 1."


def test_oversized_group_skipped() -> None:
    g = grounded(*(finding(n, n) for n in range(1, 6)))
    ids = tuple(f"D-00{n}" for n in range(1, 6))
    assert run(g, FakeLLM(groups(ids))) is g
    assert len(run(g, FakeLLM(groups(ids[:4]))).findings) == 2


def test_group_without_common_requirement_or_section_skipped() -> None:
    g = grounded(finding(1, 1, reqs=("R-001",)), finding(2, 9, reqs=("R-002",)))
    assert run(g, FakeLLM(groups(("D-001", "D-002")))) is g


def test_common_requirement_via_quote_requirement_id() -> None:
    a = finding(1, 1, reqs=("R-001",))
    b = finding(2, 9, reqs=("R-002",))
    b = b.model_copy(
        update={"quotes": (b.quotes[0].model_copy(update={"requirement_id": "R-001"}),)}
    )
    assert len(run(grounded(a, b), FakeLLM(groups(("D-001", "D-002")))).findings) == 1


def section_doc(*sections: tuple[str, ...]) -> Document:
    fragments = tuple(
        Fragment(
            id=f"F-{n:04d}",
            kind=FragmentKind.PARAGRAPH,
            text="текст фрагмента",
            location=Location(line_start=n * 10, line_end=n * 10 + 9),
            section=section,
        )
        for n, section in enumerate(sections, start=1)
    )
    return DOC.model_copy(update={"fragments": fragments})


def test_overlapping_quote_lines_in_one_section_allowed() -> None:
    g = grounded(finding(1, 12, reqs=("R-001",)), finding(2, 12, reqs=("R-002",)))
    llm = FakeLLM(groups(("D-001", "D-002")))
    assert len(consolidate(section_doc(("A",), ("A",)), (), g, llm).findings) == 1
    llm = FakeLLM(groups(("D-001", "D-002")))
    other = g.model_copy(
        update={"findings": (finding(1, 12, reqs=("R-001",)), finding(2, 22, reqs=("R-002",)))}
    )
    assert consolidate(section_doc(("A",), ("A",), ("B",)), (), other, llm) is other


def test_rule_merged_into_model_finding() -> None:
    """Правило и модель об одном: текст модели, серьёзность — самая высокая в группе."""
    rule = finding(1, 5, source=FindingSource.RULE, severity=Severity.CRITICAL)
    model = finding(2, 5, severity=Severity.MAJOR)
    result = run(grounded(rule, model), FakeLLM(groups(("D-001", "D-002"))))
    assert len(result.findings) == 1
    merged = result.findings[0]
    assert merged.source is FindingSource.MODEL and merged.rule_id is None
    assert merged.explanation.startswith("Пояснение 2.")
    assert merged.severity is Severity.CRITICAL


def test_two_rule_findings_never_merged() -> None:
    first = finding(1, 5, source=FindingSource.RULE)
    second = finding(2, 5, source=FindingSource.RULE)
    model = finding(3, 9, reqs=("R-002",))
    g = grounded(first, second, model)
    assert run(g, FakeLLM(groups(("D-001", "D-002")))) is g
    only_rules = grounded(first, second)
    llm = FakeLLM(groups(("D-001", "D-002")))
    assert run(only_rules, llm) is only_rules
    assert not llm.calls


def test_shared_line_links_findings_across_sections() -> None:
    """Одна и та же строка процитирована в двух замечаниях без общего требования."""
    wide = finding(1, 29, reqs=())
    extra = tuple(
        Quote(text=f"цитата номер {n}", location=Location(line_start=n, line_end=n))
        for n in (52, 65)
    )
    wide = wide.model_copy(update={"quotes": wide.quotes + extra})
    narrow = finding(2, 52, reqs=("R-005",))
    result = run(grounded(wide, narrow), FakeLLM(groups(("D-001", "D-002"))))
    assert len(result.findings) == 1
    assert len(result.findings[0].quotes) == 3


def test_merged_findings_of_other_type_keep_their_gist() -> None:
    """Склеены два замечания модели разного типа: суть второго дописывается, а не теряется."""
    first = finding(1, 5, severity=Severity.CRITICAL)
    second = finding(2, 5, severity=Severity.MAJOR).model_copy(
        update={"defect_type": DefectType.INTERFACE_MISUSE}
    )
    assert first.defect_type is not second.defect_type
    result = run(grounded(first, second), FakeLLM(groups(("D-001", "D-002"))))
    assert len(result.findings) == 1
    assert result.findings[0].explanation == "Пояснение 1. Пояснение 2."
