"""Справочник разметки замечаний (FR-9).

Модель называет только тип дефекта; критерий, класс, серьёзность и пункт
стандарта берутся отсюда. Таблица — данные: чтобы поправить серьёзность или
пункт, достаточно изменить строку, код трогать не нужно.
"""

from typing import NamedTuple

from reqcheck.models import Criterion, DefectType, FindingClass, Severity

_ISO = "ISO/IEC/IEEE 29148:2018"
_IEEE = "IEEE 830-1998"


class Markup(NamedTuple):
    criterion: Criterion
    finding_class: FindingClass
    severity: Severity
    standard_ref: str


# Серьёзность зависит от типа и области, а не от мнения модели:
#   BLOCKER  - противоречие в наборе: две стороны не выполнить разом.
#   CRITICAL - нет требования, нельзя проверить, интерфейс описан неверно.
#   MAJOR    - читается, но реализуется по-разному: пробел, неоднозначность.
#   MINOR    - форма, а не содержание: не атомарно, дубль, дизайн, трассировка.
TAXONOMY: dict[DefectType, Markup] = {
    DefectType.CONTRADICTION: Markup(
        Criterion.SET_CONSISTENT,
        FindingClass.CONFLICT,
        Severity.BLOCKER,
        f"{_ISO}, 5.2.6 (Consistent)",
    ),
    DefectType.MISSING_REQUIREMENT: Markup(
        Criterion.SET_COMPLETE,
        FindingClass.MISSING_INFO,
        Severity.CRITICAL,
        f"{_ISO}, 5.2.6 (Complete)",
    ),
    DefectType.INTERFACE_MISUSE: Markup(
        Criterion.REQ_CORRECT,
        FindingClass.CONFLICT,
        Severity.CRITICAL,
        f"{_ISO}, 5.2.5 (Correct)",
    ),
    DefectType.UNVERIFIABLE: Markup(
        Criterion.REQ_VERIFIABLE,
        FindingClass.MISSING_INFO,
        Severity.CRITICAL,
        f"{_ISO}, 5.2.5 (Verifiable)",
    ),
    DefectType.INCOMPLETE: Markup(
        Criterion.REQ_COMPLETE,
        FindingClass.MISSING_INFO,
        Severity.MAJOR,
        f"{_ISO}, 5.2.5 (Complete)",
    ),
    DefectType.AMBIGUOUS: Markup(
        Criterion.REQ_UNAMBIGUOUS,
        FindingClass.RECOMMENDATION,
        Severity.MAJOR,
        f"{_ISO}, 5.2.5 (Unambiguous)",
    ),
    DefectType.UNDEFINED_REFERENCE: Markup(
        Criterion.REQ_COMPLETE,
        FindingClass.MISSING_INFO,
        Severity.MAJOR,
        f"{_IEEE}, 4.3.3",
    ),
    DefectType.INCONSISTENT_TERMS: Markup(
        Criterion.SET_CONSISTENT,
        FindingClass.CONFLICT,
        Severity.MAJOR,
        f"{_ISO}, 5.2.6 (Consistent)",
    ),
    DefectType.INFEASIBLE: Markup(
        Criterion.REQ_FEASIBLE,
        FindingClass.RECOMMENDATION,
        Severity.MAJOR,
        f"{_ISO}, 5.2.5 (Feasible)",
    ),
    DefectType.NOT_SINGULAR: Markup(
        Criterion.REQ_SINGULAR,
        FindingClass.RECOMMENDATION,
        Severity.MINOR,
        f"{_ISO}, 5.2.5 (Singular)",
    ),
    DefectType.DESIGN_CONSTRAINT: Markup(
        Criterion.REQ_APPROPRIATE,
        FindingClass.RECOMMENDATION,
        Severity.MINOR,
        f"{_ISO}, 5.2.5 (Appropriate)",
    ),
    DefectType.DUPLICATE: Markup(
        Criterion.SRS_MODIFIABLE,
        FindingClass.RECOMMENDATION,
        Severity.MINOR,
        f"{_IEEE}, 4.3.7 (Modifiable)",
    ),
    DefectType.UNTRACEABLE: Markup(
        Criterion.SRS_TRACEABLE,
        FindingClass.RECOMMENDATION,
        Severity.MINOR,
        f"{_IEEE}, 4.3.8 (Traceable)",
    ),
}

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

# Группы типов, которые на одной и той же цитате считаются одним дефектом (FR-16).
MERGEABLE_GROUPS: tuple[frozenset[DefectType], ...] = (
    frozenset({DefectType.AMBIGUOUS, DefectType.UNVERIFIABLE}),
)
