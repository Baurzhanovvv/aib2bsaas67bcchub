"""Профили домена (FR-19): дополнительные проверки под флагом `--profile`.

Профиль не меняет общий агент: без флага ни его правила, ни его текст в промпт не
попадают. Профиль — отдельный файл с перечнем того, что домен требует задать явно,
и с правилами без модели.
"""

from collections.abc import Callable
from dataclasses import dataclass

from reqcheck.errors import InputError
from reqcheck.models import Document, RawFinding, Requirement

Rule = Callable[[Document, tuple[Requirement, ...]], list[RawFinding]]


@dataclass(frozen=True)
class Profile:
    name: str
    title: str
    checklist: str
    """Текст для свободного прохода модели: что домен требует задать явно."""
    rules: tuple[Rule, ...] = ()


def names() -> tuple[str, ...]:
    return ("bank",)


def load(name: str | None) -> Profile | None:
    if not name:
        return None
    if name == "bank":
        from reqcheck.profiles import bank

        return bank.PROFILE
    raise InputError(f"Неизвестный профиль «{name}». Доступны: {', '.join(names())}.")
