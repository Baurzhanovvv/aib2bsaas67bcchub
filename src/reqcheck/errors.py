"""Коды выхода и ошибки, которые видит пользователь (FR-11, NFR-6)."""

from enum import IntEnum


class ExitCode(IntEnum):
    OK = 0
    """Отчёт создан, итог GO или CONDITIONAL GO."""
    NO_GO = 1
    """Отчёт создан, итог NO-GO."""
    FAILED = 2
    """Отчёт о проверке не создан: плохой вход, сбой модели, этап не готов."""


class ReqcheckError(Exception):
    """Ошибка с текстом для пользователя на русском, без traceback."""

    exit_code: ExitCode = ExitCode.FAILED

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InputError(ReqcheckError):
    """Входной файл не годится: нет файла, пустой, не тот формат."""


class ModelError(ReqcheckError):
    """Модель недоступна или дважды вернула ответ не по схеме (FR-8)."""


class StageNotReady(ReqcheckError):
    """Этап конвейера ещё не реализован."""

    def __init__(self, stage: str) -> None:
        super().__init__(f"Этап «{stage}» ещё не реализован.")
