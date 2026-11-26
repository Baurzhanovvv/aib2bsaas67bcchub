"""Конвейер прогона: порядок этапов и больше ничего.

Логика этапов живёт в reqcheck.stages; здесь только их вызов по порядку.
Сбой модели не обрушивает прогон: правила уже отработали, отчёт выпускается
с причиной сбоя в паспорте (режим RULES_ONLY).
"""

import os
import tempfile
import time
from pathlib import Path

from reqcheck.errors import ExitCode, InputError, ModelError
from reqcheck.llm import LLMClient
from reqcheck.models import RawFinding, Report, Requirement, TraceReport, Verdict
from reqcheck.stages import (
    checks,
    consolidate,
    grounding,
    reader,
    registry,
    render,
    report,
    rules,
)


def _model_checks(
    document, requirements: tuple[Requirement, ...], llm: LLMClient
) -> tuple[list[RawFinding], TraceReport | None, str | None]:
    """Проверки моделью. При сбое — то, что успели, и текст причины."""
    found: list[RawFinding] = []
    try:
        found += checks.check_requirements(requirements, llm, document=document)
        set_findings, trace = checks.check_set(document, requirements, llm)
    except ModelError as error:
        return found, None, error.message
    return found + set_findings, trace, None


def build(input_path: Path, llm: LLMClient | None) -> Report:
    """Прочитать документ и собрать отчёт. llm=None — режим без модели (FR-15)."""
    started = time.monotonic()
    document = reader.read_document(input_path)

    model_error: str | None = None
    try:
        requirements = registry.build_registry(document, llm)
    except ModelError as error:
        model_error = error.message
        requirements = registry.build_registry(document, None)

    raw: list[RawFinding] = rules.apply_rules(document, requirements)
    trace: TraceReport | None = None
    if llm is not None and model_error is None:
        model_found, trace, model_error = _model_checks(document, requirements, llm)
        raw += model_found

    grounded = grounding.verify_and_classify(document, requirements, raw)
    merger = llm if model_error is None else None
    grounded = consolidate.consolidate(document, requirements, grounded, merger)
    usage = llm.usage() if llm is not None else None
    duration_s = time.monotonic() - started
    return report.build_report(
        document, requirements, grounded, trace, usage, duration_s, model_error=model_error
    )


def _write_atomically(path: Path, text: str) -> None:
    """Через временный файл с непредсказуемым именем: отчёт не бывает записан наполовину."""
    tmp: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except OSError as error:
        if tmp is not None and os.path.exists(tmp):
            os.unlink(tmp)
        raise InputError(f"Не удалось записать отчёт в {path}: {error.strerror}") from error


def run(input_path: Path, output_path: Path, llm: LLMClient | None) -> ExitCode:
    """Полный прогон: отчёт записан в output_path, возвращён код выхода."""
    result = build(input_path, llm)
    _write_atomically(output_path, render.render_markdown(result))
    return ExitCode.NO_GO if result.verdict is Verdict.NO_GO else ExitCode.OK
