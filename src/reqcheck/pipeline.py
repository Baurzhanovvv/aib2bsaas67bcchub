"""Конвейер прогона: порядок этапов и больше ничего.

Логика этапов живёт в reqcheck.stages; здесь только их вызов по порядку.
Сбой модели не обрушивает прогон: правила уже отработали, отчёт выпускается
с причиной сбоя в паспорте (режим RULES_ONLY).
"""

import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from reqcheck.errors import ExitCode, InputError, ModelError
from reqcheck.llm import LLMClient
from reqcheck.models import RawFinding, Report, Requirement, TraceReport, Verdict
from reqcheck.profiles import Profile
from reqcheck.stages import (
    checks,
    consolidate,
    grounding,
    reader,
    registry,
    render,
    render_html,
    report,
    rules,
)

Progress = Callable[[str], None]
"""Куда сообщать о начале этапа (FR-23); в отчёт это не попадает."""
STEPS = 6


def _silent(_: str) -> None:
    return None


def _model_checks(
    document, requirements: tuple[Requirement, ...], llm: LLMClient, profile: Profile | None
) -> tuple[list[RawFinding], TraceReport | None, str | None]:
    """Проверки моделью. При сбое — то, что успели, и текст причины."""
    found: list[RawFinding] = []
    try:
        found += checks.check_requirements(requirements, llm, document=document)
        extra = {"profile": profile.checklist} if profile else {}
        set_findings, trace = checks.check_set(document, requirements, llm, **extra)
    except ModelError as error:
        return found, None, error.message
    return found + set_findings, trace, None


def build(
    input_path: Path,
    llm: LLMClient | None,
    progress: Progress = _silent,
    profile: Profile | None = None,
) -> Report:
    """Прочитать документ и собрать отчёт. llm=None — режим без модели (FR-15)."""
    started = time.monotonic()
    progress(f"[1/{STEPS}] Чтение документа")
    document = reader.read_document(input_path)
    progress(f"[2/{STEPS}] Реестр требований")

    model_error: str | None = None
    try:
        requirements = registry.build_registry(document, llm)
    except ModelError as error:
        model_error = error.message
        requirements = registry.build_registry(document, None)

    progress(f"[3/{STEPS}] Правила без модели: требований {len(requirements)}")
    raw: list[RawFinding] = rules.apply_rules(document, requirements)
    for rule in profile.rules if profile else ():
        raw += rule(document, requirements)
    trace: TraceReport | None = None
    if llm is not None and model_error is None:
        progress(f"[4/{STEPS}] Проверки моделью (несколько запросов, обычно 1–3 минуты)")
        model_found, trace, model_error = _model_checks(document, requirements, llm, profile)
        raw += model_found

    progress(f"[5/{STEPS}] Сверка цитат и склейка дублей: замечаний до сверки {len(raw)}")
    grounded = grounding.verify_and_classify(document, requirements, raw)
    merger = llm if model_error is None else None
    grounded = consolidate.consolidate(document, requirements, grounded, merger)
    usage = llm.usage() if llm is not None else None
    duration_s = time.monotonic() - started
    return report.build_report(
        document,
        requirements,
        grounded,
        trace,
        usage,
        duration_s,
        model_error=model_error,
        profile=profile.title if profile else None,
    )


def _write_atomically(path: Path, text: str) -> None:
    """Через временный файл с непредсказуемым именем: отчёт не бывает записан наполовину."""
    tmp: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        _publish(tmp, path)
        tmp = None
    except OSError as error:
        raise InputError(f"Не удалось записать отчёт в {path}: {error.strerror}") from error
    finally:
        _discard(tmp)


def _publish(tmp: str, path: Path) -> None:
    """Права как у обычного файла (mkstemp даёт 0600) и замена одним действием."""
    mask = os.umask(0)
    os.umask(mask)
    os.chmod(tmp, 0o666 & ~mask)
    os.replace(tmp, path)


def _discard(tmp: str | None) -> None:
    """Убрать временный файл при любом сбое, включая прерывание пользователем."""
    if tmp is not None and os.path.exists(tmp):
        os.unlink(tmp)


def _write_pdf(path: Path, result: Report) -> None:
    tmp: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        os.close(fd)
        render_html.write_pdf(result, Path(tmp))
        _publish(tmp, path)
        tmp = None
    except OSError as error:
        raise InputError(f"Не удалось записать отчёт в {path}: {error.strerror}") from error
    finally:
        _discard(tmp)


def run(
    input_path: Path,
    output_path: Path,
    llm: LLMClient | None,
    progress: Progress = _silent,
    profile: Profile | None = None,
) -> ExitCode:
    """Полный прогон: отчёт записан в output_path, возвращён код выхода.

    Формат отчёта — по расширению: .html, .pdf, иначе Markdown (FR-22).
    """
    result = build(input_path, llm, progress, profile)
    progress(f"[6/{STEPS}] Запись отчёта: {output_path}")
    suffix = output_path.suffix.lower()
    if suffix == ".pdf":
        _write_pdf(output_path, result)
    elif suffix in (".html", ".htm"):
        _write_atomically(output_path, render_html.render_html(result))
    else:
        _write_atomically(output_path, render.render_markdown(result))
    return ExitCode.NO_GO if result.verdict is Verdict.NO_GO else ExitCode.OK
