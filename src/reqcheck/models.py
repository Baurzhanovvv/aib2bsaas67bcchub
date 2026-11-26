"""Контракты данных между этапами конвейера.

Каждый этап принимает и возвращает только эти модели. Менять поля — только
вместе с тестами в tests/test_models.py и с оглядкой на соседние этапы.

Правило разделения труда: модель называет тип дефекта из закрытого списка и
приводит дословные цитаты; серьёзность, класс и пункт стандарта ставит код
(справочник, FR-9), номера строк считает код (FR-7).
"""

from enum import StrEnum
from typing import Annotated

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

REQUIREMENT_ID_PATTERN = r"^R-\d{3,}$"
FINDING_ID_PATTERN = r"^D-\d{3,}$"
FRAGMENT_ID_PATTERN = r"^F-\d{4,}$"
MIN_QUOTE_LENGTH = 8

RequirementId = Annotated[str, Field(pattern=REQUIREMENT_ID_PATTERN)]


class Contract(BaseModel):
    """База всех контрактов: лишние поля запрещены, объекты неизменяемы."""

    model_config = ConfigDict(extra="forbid", frozen=True)


# --- Документ -----------------------------------------------------------------


class DocumentFormat(StrEnum):
    MARKDOWN = "markdown"
    PDF = "pdf"
    DOCX = "docx"


class FragmentKind(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE_ROW = "table_row"
    CODE = "code"


class Location(Contract):
    """Место в исходном документе. Строки считаются с 1.

    Для Markdown — строки файла. Для PDF — номер страницы и строки на ней.
    """

    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    page: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _ordered(self) -> "Location":
        if self.line_end < self.line_start:
            raise ValueError("line_end меньше line_start")
        return self


class Fragment(Contract):
    """Минимальная единица текста: абзац, пункт списка, строка таблицы."""

    id: str = Field(pattern=FRAGMENT_ID_PATTERN)
    kind: FragmentKind
    text: str = Field(min_length=1)
    location: Location
    section: tuple[str, ...] = ()
    """Путь заголовков от корня до фрагмента."""


class Document(Contract):
    """Результат чтения (FR-2, FR-3, FR-20)."""

    name: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    format: DocumentFormat
    fragments: tuple[Fragment, ...]
    unread_pages: tuple[int, ...] = ()
    """Страницы, которые не удалось прочитать (скан, диаграмма)."""

    @property
    def degraded(self) -> bool:
        return bool(self.unread_pages)


# --- Реестр требований ----------------------------------------------------------


class RequirementKind(StrEnum):
    FUNCTIONAL = "functional"
    NON_FUNCTIONAL = "non_functional"
    INTERFACE = "interface"
    ACCEPTANCE = "acceptance"
    CONSTRAINT = "constraint"
    OTHER = "other"


class Requirement(Contract):
    """Строка реестра (FR-4). Нарезает код, модель уточняет kind и summary."""

    id: str = Field(pattern=REQUIREMENT_ID_PATTERN)
    source_id: str | None = None
    """Идентификатор из документа, если он там был."""
    kind: RequirementKind = RequirementKind.OTHER
    summary: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    """Дословный текст требования из документа."""
    location: Location
    fragment_id: str = Field(pattern=FRAGMENT_ID_PATTERN)
    section: tuple[str, ...] = ()
    """Путь заголовков фрагмента — контекст для модели."""


# --- Замечания ------------------------------------------------------------------


class DefectType(StrEnum):
    """Закрытый список типов дефекта. Его выбирает модель или правило."""

    AMBIGUOUS = "ambiguous"
    UNVERIFIABLE = "unverifiable"
    INCOMPLETE = "incomplete"
    NOT_SINGULAR = "not_singular"
    INFEASIBLE = "infeasible"
    DESIGN_CONSTRAINT = "design_constraint"
    UNDEFINED_REFERENCE = "undefined_reference"
    INTERFACE_MISUSE = "interface_misuse"
    CONTRADICTION = "contradiction"
    MISSING_REQUIREMENT = "missing_requirement"
    DUPLICATE = "duplicate"
    INCONSISTENT_TERMS = "inconsistent_terms"
    UNTRACEABLE = "untraceable"


class Criterion(StrEnum):
    """Свойства качества: ISO/IEC/IEEE 29148 §5.2.5, §5.2.6 и IEEE 830 §4.3."""

    REQ_NECESSARY = "req_necessary"
    REQ_APPROPRIATE = "req_appropriate"
    REQ_UNAMBIGUOUS = "req_unambiguous"
    REQ_COMPLETE = "req_complete"
    REQ_SINGULAR = "req_singular"
    REQ_FEASIBLE = "req_feasible"
    REQ_VERIFIABLE = "req_verifiable"
    REQ_CORRECT = "req_correct"
    REQ_CONFORMING = "req_conforming"
    SET_COMPLETE = "set_complete"
    SET_CONSISTENT = "set_consistent"
    SET_FEASIBLE = "set_feasible"
    SET_COMPREHENSIBLE = "set_comprehensible"
    SET_VALIDATABLE = "set_validatable"
    SRS_TRACEABLE = "srs_traceable"
    SRS_RANKED = "srs_ranked"
    SRS_MODIFIABLE = "srs_modifiable"


class Scope(StrEnum):
    REQUIREMENT = "requirement"
    REQUIREMENT_SET = "requirement_set"


class FindingClass(StrEnum):
    CONFLICT = "conflict"
    MISSING_INFO = "missing_info"
    RECOMMENDATION = "recommendation"


class Severity(StrEnum):
    BLOCKER = "blocker"
    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


class FindingSource(StrEnum):
    RULE = "rule"
    MODEL = "model"


class ModelFinding(Contract):
    """Замечание в том виде, в каком его возвращает модель.

    Цитата обязательна всегда, в том числе для пропуска (MISSING_REQUIREMENT,
    UNTRACEABLE): тогда цитируется место, из которого пропуск следует, —
    требование, которое без недостающего нельзя реализовать или проверить.
    Задание требует, чтобы каждое нарушение ссылалось на место в документе.

    requirement_ids может содержать id, которых нет в реестру: этап, который
    создаёт RawFinding, обязан их отбросить, иначе отчёт не соберётся.
    """

    defect_type: DefectType
    requirement_ids: tuple[RequirementId, ...] = ()
    quotes: tuple[str, ...] = Field(min_length=1)
    """Дословные цитаты из документа. Для противоречия — обе стороны."""
    explanation: str = Field(min_length=1)
    suggestion: str | None = None

    @model_validator(mode="after")
    def _check(self) -> "ModelFinding":
        for quote in self.quotes:
            if len(quote.strip()) < MIN_QUOTE_LENGTH:
                raise ValueError(f"цитата короче {MIN_QUOTE_LENGTH} символов")
        if self.defect_type is DefectType.CONTRADICTION and len(set(self.quotes)) < 2:
            raise ValueError("для противоречия нужны две разные цитаты — обе стороны")
        return self


class RawFinding(ModelFinding):
    """Замечание до проверки цитат: от модели или от правила."""

    scope: Scope
    source: FindingSource
    rule_id: str | None = None

    @model_validator(mode="after")
    def _rule_has_id(self) -> "RawFinding":
        if self.source is FindingSource.RULE and not self.rule_id:
            raise ValueError("у замечания от правила должен быть rule_id")
        if self.source is FindingSource.MODEL and self.rule_id:
            raise ValueError("у замечания от модели не бывает rule_id")
        return self


class Quote(Contract):
    """Цитата, найденная в документе кодом."""

    text: str = Field(min_length=MIN_QUOTE_LENGTH)
    """Исходный текст цитаты из документа (не нормализованный)."""
    location: Location
    """Точные строки цитаты, а не всего фрагмента."""
    requirement_id: RequirementId | None = None


class Finding(Contract):
    """Замечание, попадающее в отчёт: цитаты подтверждены, разметка проставлена."""

    id: str = Field(pattern=FINDING_ID_PATTERN)
    scope: Scope
    defect_type: DefectType
    criterion: Criterion
    finding_class: FindingClass
    severity: Severity
    standard_ref: str = Field(min_length=1)
    """Пункт стандарта, например «ISO/IEC/IEEE 29148:2018, 5.2.5»."""
    requirement_ids: tuple[RequirementId, ...] = ()
    quotes: tuple[Quote, ...] = Field(min_length=1)
    explanation: str = Field(min_length=1)
    suggestion: str | None = None
    source: FindingSource
    rule_id: str | None = None


class GroundingResult(Contract):
    """Итог проверки цитат (FR-7) и разметки по справочнику (FR-9)."""

    findings: tuple[Finding, ...] = ()
    quotes_checked: int = Field(default=0, ge=0)
    quotes_confirmed: int = Field(default=0, ge=0)
    findings_dropped: int = Field(default=0, ge=0)


# --- Трассировка критериев приёмки (FR-6) --------------------------------------


class AcceptanceLink(Contract):
    acceptance_id: RequirementId
    covers: tuple[RequirementId, ...] = ()


class TraceReport(Contract):
    links: tuple[AcceptanceLink, ...] = ()
    uncovered_requirement_ids: tuple[RequirementId, ...] = ()
    """Требования, на которые не ссылается ни один критерий приёмки."""
    orphan_acceptance_ids: tuple[RequirementId, ...] = ()
    """Критерии приёмки, не покрывающие ни одного требования."""


# --- Отчёт ----------------------------------------------------------------------


class RunMode(StrEnum):
    FULL = "full"
    RULES_ONLY = "rules_only"


class Verdict(StrEnum):
    GO = "go"
    CONDITIONAL_GO = "conditional_go"
    NO_GO = "no_go"


class Passport(Contract):
    """Паспорт прогона (FR-10). Все значения ставит код."""

    document_name: str
    document_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    run_at: AwareDatetime
    agent_version: str
    """Тег или короткий хеш коммита."""
    dirty: bool
    """Есть незакоммиченные изменения."""
    mode: RunMode
    model: str | None = None
    """Имя модели из ответа API, а не из настроек."""
    model_error: str | None = None
    """Почему проверка моделью прервалась (квота, сеть, ответ не по схеме).

    Прогон при этом не падает: правила отработали, отчёт выпускается в режиме
    RULES_ONLY, а то, что модель успела найти до сбоя, остаётся в отчёте.
    """
    llm_requests: int = Field(default=0, ge=0)
    cache_hits: int = Field(default=0, ge=0)
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    quotes_checked: int = Field(default=0, ge=0)
    quotes_confirmed: int = Field(default=0, ge=0)
    findings_dropped: int = Field(default=0, ge=0)
    """Замечания, отброшенные кодом: цитата не найдена в документе или правка «Было = Стало»."""
    duration_s: float = Field(default=0, ge=0)
    unread_pages: tuple[int, ...] = ()


class Summary(Contract):
    requirements_total: int = Field(ge=0)
    findings_total: int = Field(ge=0)
    by_severity: dict[Severity, int] = Field(default_factory=dict)
    critical_finding_ids: tuple[str, ...] = ()
    """Замечания уровня blocker и critical."""


class Report(Contract):
    """Полный результат прогона; из него рендерятся все форматы отчёта."""

    passport: Passport
    requirements: tuple[Requirement, ...]
    findings: tuple[Finding, ...]
    trace: TraceReport | None = None
    set_check_done: bool
    """False в режиме без модели: набор целиком не проверялся (FR-15)."""
    summary: Summary
    verdict: Verdict
    fix_first: tuple[str, ...] = ()
    """Идентификаторы замечаний, которые исправлять в первую очередь."""

    @model_validator(mode="after")
    def _ids_exist(self) -> "Report":
        known = _unique_ids("требований", [r.id for r in self.requirements])
        finding_ids = _unique_ids("замечаний", [f.id for f in self.findings])
        for finding in self.findings:
            referenced = set(finding.requirement_ids)
            referenced |= {q.requirement_id for q in finding.quotes if q.requirement_id}
            _require_subset(f"{finding.id}: нет в реестре", referenced, known)
        if self.trace is not None:
            _require_subset("трассировка: нет в реестре", _trace_ids(self.trace), known)
        _require_subset("fix_first: нет таких замечаний", set(self.fix_first), finding_ids)
        _require_subset(
            "сводка: нет таких замечаний", set(self.summary.critical_finding_ids), finding_ids
        )
        return self

    @model_validator(mode="after")
    def _summary_matches(self) -> "Report":
        if self.summary.requirements_total != len(self.requirements):
            raise ValueError("сводка: число требований не совпадает с реестром")
        if self.summary.findings_total != len(self.findings):
            raise ValueError("сводка: число замечаний не совпадает со списком")
        if self.passport.mode is RunMode.RULES_ONLY and self.set_check_done:
            raise ValueError("в режиме без модели набор не может быть проверен")
        return self


def _unique_ids(what: str, ids: list[str]) -> set[str]:
    unique = set(ids)
    if len(unique) != len(ids):
        raise ValueError(f"повторяющиеся идентификаторы {what}")
    return unique


def _require_subset(message: str, ids: set[str], known: set[str]) -> None:
    unknown = ids - known
    if unknown:
        raise ValueError(f"{message} {sorted(unknown)}")


def _trace_ids(trace: TraceReport) -> set[str]:
    ids = set(trace.uncovered_requirement_ids) | set(trace.orphan_acceptance_ids)
    for link in trace.links:
        ids.add(link.acceptance_id)
        ids.update(link.covers)
    return ids
