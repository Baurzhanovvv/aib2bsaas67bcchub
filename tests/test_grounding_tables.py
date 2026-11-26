"""Таблицы, составные цитаты и склейка по месту (FR-7, FR-16): отрицательные случаи."""

from reqcheck.models import (
    DefectType,
    Document,
    Fragment,
    FragmentKind,
    Impact,
    Location,
    Requirement,
    Scope,
    Severity,
)
from tests.test_grounding import doc, frag, raw, req, run

ROW = "| Роль | Пользователь | Лимит 1000000 | Администратор без лимита |"


def row(n: int, text: str, line: int) -> Fragment:
    return Fragment(
        id=f"F-{n:04d}",
        kind=FragmentKind.TABLE_ROW,
        text=text,
        location=Location(line_start=line, line_end=line),
    )


def table_doc() -> tuple[Document, tuple[Requirement, ...]]:
    rows = [row(1, ROW, 4), row(2, "| volume | decimal | Объём груза в литрах |", 5)]
    return doc(*rows), (req(1, rows[0]), req(2, rows[1]))


def grounded(quote: str) -> list[str]:
    document, reqs = table_doc()
    result = run([raw((quote,), ids=())], document, reqs)
    return [q.text for f in result.findings for q in f.quotes]


def test_adjacent_cells_accepted_with_verbatim_text() -> None:
    assert grounded("Роль: Пользователь") == ["Роль | Пользователь"]
    assert grounded("Лимит 1000000: Администратор") == ["Лимит 1000000 | Администратор без лимита"]


def test_non_adjacent_cells_rejected() -> None:
    assert grounded("Роль: Администратор") == []


def test_number_prefix_not_on_word_boundary_rejected() -> None:
    assert grounded("Пользователь: Лимит 10") == []
    assert grounded("Пол: Лимит 10") == []


def test_segments_from_different_rows_rejected() -> None:
    assert grounded("Роль: Объём груза в литрах") == []


def test_composite_duplicate_parts_not_confirmed() -> None:
    a = frag(1, "Система должна отправлять уведомление.", 1)
    assert not run(
        [raw(("Система должна\nСистема должна",), ids=())], doc(a), (req(1, a),)
    ).findings


def test_composite_prefers_fragments_near_requirement() -> None:
    frags = [frag(i, f"Общая фраза номер {i}.", i) for i in range(1, 9)]
    frags[0] = frag(1, "Общая фраза для всех требований.", 1)
    frags[7] = frag(8, "Общая фраза для всех требований.", 8)
    frags[6] = frag(7, "Срок хранения данных составляет год.", 7)
    quote = "Общая фраза для всех требований.\nСрок хранения данных составляет год."
    result = run([raw((quote,), ids=("R-008",))], doc(*frags), (req(8, frags[7]),))
    assert [q.location.line_start for q in result.findings[0].quotes] == [8, 7]


PAIR = frag(1, "Система должна отправлять уведомление пользователю сразу.", 1)
PAIR_REQS = (req(1, PAIR), req(2, frag(2, "Другой текст требования номер два.", 2)))


def merged(a_ids: tuple[str, ...], b_ids: tuple[str, ...], b_quote: str) -> int:
    first = raw(("должна отправлять уведомление",), ids=a_ids, defect=DefectType.AMBIGUOUS)
    second = raw((b_quote,), ids=b_ids, defect=DefectType.UNVERIFIABLE)
    return len(run([first, second], doc(PAIR), PAIR_REQS).findings)


def test_merge_requires_shared_requirement_ids() -> None:
    quote = "отправлять уведомление пользователю"
    assert merged(("R-001",), ("R-001",), quote) == 1
    assert merged(("R-001",), ("R-002",), quote) == 2


def test_merge_requires_half_overlap_of_shorter_quote() -> None:
    assert merged(("R-001",), ("R-001",), "Система должна отправлять") == 1
    assert merged(("R-001",), ("R-001",), "уведомление пользователю сразу") == 2
    assert merged(("R-001",), ("R-001",), "Система должна") == 2


def test_merged_type_and_severity_come_from_most_serious() -> None:
    wide = raw(
        ("Система должна отправлять уведомление", "Система должна отправлять"),
        scope=Scope.REQUIREMENT_SET,
        defect=DefectType.AMBIGUOUS,
    )
    serious = raw(
        ("должна отправлять уведомление",), defect=DefectType.UNVERIFIABLE, impact=Impact.HIGH
    )
    f = run([wide, serious], doc(PAIR), PAIR_REQS).findings[0]
    assert (f.defect_type, f.severity) == (DefectType.UNVERIFIABLE, Severity.CRITICAL)
    only_ambiguous = run([wide], doc(PAIR), PAIR_REQS).findings[0]
    assert only_ambiguous.severity is Severity.MAJOR
