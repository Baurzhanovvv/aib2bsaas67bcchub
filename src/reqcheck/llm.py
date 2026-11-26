"""Контракт вызова модели: что спрашиваем и что принимаем в ответ.

Реализация клиента (схема, повтор, кэш, фолбэк — FR-8, NFR-3, NFR-4) живёт
отдельно и обязана соответствовать протоколу LLMClient. Этапы проверок
зависят только от этого файла, поэтому тестируются с мок-клиентом.
"""

from typing import Generic, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from reqcheck.models import Contract, DefectType, Impact, RequirementKind

T = TypeVar("T", bound=BaseModel)


class LLMResult(BaseModel, Generic[T]):
    """Разобранный ответ модели вместе с учётом для паспорта прогона."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    data: T
    model: str
    """Имя модели из ответа API."""
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    cached: bool = False


class LLMUsage(Contract):
    """Накопленный учёт клиента за прогон — уходит в паспорт отчёта."""

    model: str | None = None
    """Имя модели из последнего ответа API; None, если вызовов не было."""
    requests: int = Field(default=0, ge=0)
    cache_hits: int = Field(default=0, ge=0)
    tokens_in: int = Field(default=0, ge=0)
    tokens_out: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0)
    """Расчётная стоимость запросов прогона в долларах; 0 для бесплатных моделей."""


class LLMClient(Protocol):
    def complete(self, *, system: str, user: str, response_model: type[T]) -> LLMResult[T]:
        """Вернуть ответ, прошедший схему response_model.

        Не прошёл схему — один повтор с текстом ошибки; второй провал или
        недоступность всех провайдеров — reqcheck.errors.ModelError.
        """
        ...

    def usage(self) -> LLMUsage:
        """Учёт всех вызовов этого клиента с начала прогона."""
        ...


# --- Схемы ответов модели ---------------------------------------------------------
#
# Схемы ответов намеренно мягкие: без шаблонов id, минимальных длин и
# валидаторов. Модель о них не знает (в JSON-схеме валидаторов нет), и одно
# кривое замечание не должно ронять весь ответ — а с ним и все проверки
# моделью. Строгая проверка — поштучно у потребителя: черновик превращается в
# RawFinding / Requirement внутри try, битый элемент отбрасывается.
# Перечисления (тип дефекта, вид требования) оставлены: Gemini соблюдает enum
# в схеме при структурированном выводе.


class Draft(BaseModel):
    """База черновиков ответа модели: лишние поля молча игнорируются."""

    model_config = ConfigDict(extra="ignore", frozen=True)


class RequirementRefinement(Draft):
    id: str
    kind: RequirementKind
    summary: str = ""


class RegistryRefinement(Draft):
    """Уточнение реестра (FR-4): тип и краткая формулировка, состав не меняется."""

    items: tuple[RequirementRefinement, ...] = ()


class FindingDraft(Draft):
    """Замечание модели до проверки. В RawFinding — только через валидацию."""

    defect_type: DefectType
    requirement_ids: tuple[str, ...] = ()
    quotes: tuple[str, ...] = ()
    explanation: str = ""
    suggestion: str | None = None
    impact: Impact = Impact.MEDIUM


class LinkDraft(Draft):
    acceptance_id: str
    covers: tuple[str, ...] = ()


class RequirementCheckResponse(Draft):
    """Проверка требований по одному (FR-5)."""

    findings: tuple[FindingDraft, ...] = ()


class SetCheckResponse(Draft):
    """Проверка набора и трассировка критериев приёмки (FR-6)."""

    findings: tuple[FindingDraft, ...] = ()
    links: tuple[LinkDraft, ...] = ()
