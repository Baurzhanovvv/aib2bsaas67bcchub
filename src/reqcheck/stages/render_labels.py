"""Русские названия перечислений для отчёта."""

from reqcheck.models import Criterion, DefectType, RequirementKind, Severity

DEFECT_RU: dict[DefectType, str] = {
    DefectType.AMBIGUOUS: "Неоднозначность",
    DefectType.UNVERIFIABLE: "Непроверяемость",
    DefectType.INCOMPLETE: "Неполнота",
    DefectType.NOT_SINGULAR: "Неатомарность",
    DefectType.INFEASIBLE: "Невыполнимость",
    DefectType.DESIGN_CONSTRAINT: "Подмена требования решением",
    DefectType.UNDEFINED_REFERENCE: "Неопределённая ссылка",
    DefectType.INTERFACE_MISUSE: "Некорректное описание интерфейса",
    DefectType.CONTRADICTION: "Противоречие",
    DefectType.MISSING_REQUIREMENT: "Пропущенное требование",
    DefectType.DUPLICATE: "Дублирование",
    DefectType.INCONSISTENT_TERMS: "Несогласованные термины",
    DefectType.UNTRACEABLE: "Отсутствие трассировки",
}

SEVERITY_RU: dict[Severity, str] = {
    Severity.BLOCKER: "блокирующее",
    Severity.CRITICAL: "критичное",
    Severity.MAJOR: "существенное",
    Severity.MINOR: "незначительное",
}

CRITERION_RU: dict[Criterion, str] = {
    Criterion.REQ_NECESSARY: "необходимость",
    Criterion.REQ_APPROPRIATE: "уместность",
    Criterion.REQ_UNAMBIGUOUS: "однозначность",
    Criterion.REQ_COMPLETE: "полнота",
    Criterion.REQ_SINGULAR: "атомарность",
    Criterion.REQ_FEASIBLE: "выполнимость",
    Criterion.REQ_VERIFIABLE: "проверяемость",
    Criterion.REQ_CORRECT: "корректность",
    Criterion.REQ_CONFORMING: "соответствие шаблону",
    Criterion.SET_COMPLETE: "полнота набора",
    Criterion.SET_CONSISTENT: "согласованность набора",
    Criterion.SET_FEASIBLE: "выполнимость набора",
    Criterion.SET_COMPREHENSIBLE: "понятность набора",
    Criterion.SET_VALIDATABLE: "валидируемость набора",
    Criterion.SRS_TRACEABLE: "трассируемость",
    Criterion.SRS_RANKED: "ранжирование по важности",
    Criterion.SRS_MODIFIABLE: "изменяемость",
}

KIND_RU: dict[RequirementKind, str] = {
    RequirementKind.FUNCTIONAL: "функциональное",
    RequirementKind.NON_FUNCTIONAL: "нефункциональное",
    RequirementKind.INTERFACE: "интерфейсное",
    RequirementKind.ACCEPTANCE: "критерий приёмки",
    RequirementKind.CONSTRAINT: "ограничение",
    RequirementKind.OTHER: "прочее",
}
