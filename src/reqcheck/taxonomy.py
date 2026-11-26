"""Справочник разметки замечаний (FR-9).

Модель называет только тип дефекта; критерий, класс, серьёзность и пункт
стандарта берутся отсюда. Таблица — данные: чтобы поправить серьёзность или
пункт, достаточно изменить строку, код трогать не нужно.
"""

from typing import NamedTuple

from reqcheck.models import Criterion, DefectType, FindingClass, Impact, RequirementKind, Severity

_ISO = "ISO/IEC/IEEE 29148:2018"
_IEEE = "IEEE 830-1998"


class Markup(NamedTuple):
    criterion: Criterion
    finding_class: FindingClass
    standard_ref: str


TAXONOMY: dict[DefectType, Markup] = {
    DefectType.CONTRADICTION: Markup(
        Criterion.SET_CONSISTENT,
        FindingClass.CONFLICT,
        f"{_ISO}, 5.2.6 (Consistent)",
    ),
    DefectType.MISSING_REQUIREMENT: Markup(
        Criterion.SET_COMPLETE,
        FindingClass.MISSING_INFO,
        f"{_ISO}, 5.2.6 (Complete)",
    ),
    DefectType.INTERFACE_MISUSE: Markup(
        Criterion.REQ_CORRECT,
        FindingClass.CONFLICT,
        f"{_ISO}, 5.2.5 (Correct)",
    ),
    DefectType.UNVERIFIABLE: Markup(
        Criterion.REQ_VERIFIABLE,
        FindingClass.MISSING_INFO,
        f"{_ISO}, 5.2.5 (Verifiable)",
    ),
    DefectType.INCOMPLETE: Markup(
        Criterion.REQ_COMPLETE,
        FindingClass.MISSING_INFO,
        f"{_ISO}, 5.2.5 (Complete)",
    ),
    DefectType.AMBIGUOUS: Markup(
        Criterion.REQ_UNAMBIGUOUS,
        FindingClass.RECOMMENDATION,
        f"{_ISO}, 5.2.5 (Unambiguous)",
    ),
    DefectType.UNDEFINED_REFERENCE: Markup(
        Criterion.REQ_COMPLETE,
        FindingClass.MISSING_INFO,
        f"{_IEEE}, 4.3.3",
    ),
    DefectType.INCONSISTENT_TERMS: Markup(
        Criterion.SET_CONSISTENT,
        FindingClass.CONFLICT,
        f"{_ISO}, 5.2.6 (Consistent)",
    ),
    DefectType.INFEASIBLE: Markup(
        Criterion.REQ_FEASIBLE,
        FindingClass.RECOMMENDATION,
        f"{_ISO}, 5.2.5 (Feasible)",
    ),
    DefectType.NOT_SINGULAR: Markup(
        Criterion.REQ_SINGULAR,
        FindingClass.RECOMMENDATION,
        f"{_ISO}, 5.2.5 (Singular)",
    ),
    DefectType.DESIGN_CONSTRAINT: Markup(
        Criterion.REQ_APPROPRIATE,
        FindingClass.RECOMMENDATION,
        f"{_ISO}, 5.2.5 (Appropriate)",
    ),
    DefectType.DUPLICATE: Markup(
        Criterion.SRS_MODIFIABLE,
        FindingClass.RECOMMENDATION,
        f"{_IEEE}, 4.3.7 (Modifiable)",
    ),
    DefectType.UNTRACEABLE: Markup(
        Criterion.SRS_TRACEABLE,
        FindingClass.RECOMMENDATION,
        f"{_IEEE}, 4.3.8 (Traceable)",
    ),
}

_B, _C, _M, _N = Severity.BLOCKER, Severity.CRITICAL, Severity.MAJOR, Severity.MINOR

# Серьёзность = функция (тип дефекта, последствие). Столбцы: high, medium, low.
# Тип задаёт потолок, последствие (ставит модель по определению Impact) опускает его:
SEVERITY: dict[DefectType, tuple[Severity, Severity, Severity]] = {
    # противоречие: BLOCKER только когда сторон не выполнить разом в ключевом поведении
    DefectType.CONTRADICTION: (_B, _C, _M),
    # пробел, непроверяемость, неверный интерфейс: потолок CRITICAL
    DefectType.MISSING_REQUIREMENT: (_C, _M, _N),
    DefectType.INTERFACE_MISUSE: (_C, _M, _N),
    DefectType.UNVERIFIABLE: (_C, _M, _N),
    # читается, но реализуется по-разному или с догадками: потолок MAJOR
    DefectType.INCOMPLETE: (_M, _M, _N),
    DefectType.AMBIGUOUS: (_M, _M, _N),
    DefectType.UNDEFINED_REFERENCE: (_M, _M, _N),
    DefectType.INCONSISTENT_TERMS: (_M, _M, _N),
    DefectType.INFEASIBLE: (_M, _M, _N),
    # форма, а не содержание: потолок MAJOR только при высоком последствии
    DefectType.NOT_SINGULAR: (_M, _N, _N),
    DefectType.DESIGN_CONSTRAINT: (_M, _N, _N),
    DefectType.DUPLICATE: (_M, _N, _N),
    DefectType.UNTRACEABLE: (_M, _N, _N),
}
_IMPACT_COLUMN = {Impact.HIGH: 0, Impact.MEDIUM: 1, Impact.LOW: 2}


def severity_for(defect_type: DefectType, impact: Impact) -> Severity:
    return SEVERITY[defect_type][_IMPACT_COLUMN[impact]]


SEVERITY_ORDER: tuple[Severity, ...] = (
    Severity.BLOCKER,
    Severity.CRITICAL,
    Severity.MAJOR,
    Severity.MINOR,
)

# Повышение до BLOCKER по правилу (опция, по умолчанию выключена): если цитата
# замечания взята из критерия приёмки, то пробел или отсутствие требования
# означает, что критерий проверить нельзя.
PROMOTE_ACCEPTANCE_TO_BLOCKER = False
ACCEPTANCE_PROMOTED_TYPES: frozenset[DefectType] = frozenset(
    {DefectType.INCOMPLETE, DefectType.MISSING_REQUIREMENT}
)

# Группы типов, которые на одном и том же месте документа считаются одним дефектом (FR-16).
MERGEABLE_GROUPS: tuple[frozenset[DefectType], ...] = (
    frozenset(
        {DefectType.CONTRADICTION, DefectType.INCONSISTENT_TERMS, DefectType.INTERFACE_MISUSE}
    ),
    frozenset({DefectType.AMBIGUOUS, DefectType.UNVERIFIABLE}),
    frozenset({DefectType.INCOMPLETE, DefectType.MISSING_REQUIREMENT}),
)

# BLOCKER возможен, только если хотя бы одна цитата из требования такого вида:
# интерфейсное расхождение критично, но разработку не останавливает.
BLOCKER_REQUIREMENT_KINDS: frozenset[RequirementKind] = frozenset(
    {RequirementKind.FUNCTIONAL, RequirementKind.ACCEPTANCE}
)
