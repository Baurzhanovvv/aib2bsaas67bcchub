"""Отчёт в Markdown: шесть разделов задания (FR-10). PDF и HTML — FR-22."""

from collections import defaultdict
from datetime import datetime

from reqcheck.models import (
    Criterion,
    DefectType,
    Finding,
    Location,
    Report,
    RequirementKind,
    RunMode,
    Scope,
    Severity,
    Verdict,
)

TABLE_QUOTE_LIMIT = 100

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

_INCOMPLETE = "отчёт неполный, набор требований не проверен моделью."
_INTERRUPTED = "ВНИМАНИЕ: проверка моделью прервана, " + _INCOMPLETE

SEVERITY_ORDER = (Severity.BLOCKER, Severity.CRITICAL, Severity.MAJOR, Severity.MINOR)


# --- Мелкие помощники ---------------------------------------------------------------


def _cell(text: str) -> str:
    """Безопасное содержимое ячейки таблицы: без переноса строк и неэкранированных `|`."""
    flat = " ".join(text.split())
    return flat.replace("\\", "\\\\").replace("|", "\\|")


def _inline(text: str) -> str:
    """Текст от модели в одну строку: переносом нельзя начать свой заголовок или раздел."""
    return " ".join(text.split())


def _short(text: str, limit: int = TABLE_QUOTE_LIMIT) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _place(loc: Location) -> str:
    """«строка 42», «строки 12–14», «стр. 3, строка 12»."""
    if loc.line_start == loc.line_end:
        lines = f"строка {loc.line_start}"
    else:
        lines = f"строки {loc.line_start}–{loc.line_end}"
    return f"стр. {loc.page}, {lines}" if loc.page is not None else lines


def _table(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows]
    return lines


def _blockquote(text: str) -> list[str]:
    return ["> " + line if line.strip() else ">" for line in text.strip().splitlines()]


def _pages(pages: tuple[int, ...]) -> str:
    return ", ".join(str(p) for p in pages)


def _dt(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S %Z").strip()


# --- Разделы --------------------------------------------------------------------------


def _passport(report: Report) -> list[str]:
    p = report.passport
    model = p.model or "не указана"
    if p.mode is RunMode.RULES_ONLY and not p.model_error:
        model = "без модели — проверка набора не выполнялась"
    version = p.agent_version + (" (есть незакоммиченные изменения)" if p.dirty else "")
    if p.mode is RunMode.FULL:
        mode = "полный (правила и модель)"
    elif p.model_error:
        mode = "только правила (модель отказала в ходе прогона)"
    else:
        mode = "только правила, без модели"
    rows: list[tuple[str, str]] = [
        ("Документ", p.document_name),
        ("SHA-256", p.document_sha256[:12]),
        ("Дата и время", _dt(p.run_at)),
        ("Модель", model),
        ("Версия агента", version),
        ("Режим", mode),
        (
            "Запросы к модели",
            f"{p.llm_requests}, из кэша: {p.cache_hits}, токены: {p.tokens_in} вх. / "
            f"{p.tokens_out} исх.",
        ),
        (
            "Цитаты",
            f"проверено {p.quotes_checked}, подтверждено {p.quotes_confirmed}; отброшено "
            f"замечаний с неподтверждёнными цитатами: {p.findings_dropped}",
        ),
        ("Время прогона", f"{p.duration_s:.1f} с"),
    ]
    if p.model_error:
        rows.append(("Проверка моделью прервана", p.model_error))
    if p.unread_pages:
        rows.append(
            (
                "Непрочитанные страницы",
                f"{_pages(p.unread_pages)} — ВНИМАНИЕ: текст этих страниц не проверялся, "
                "проверка неполная",
            )
        )
    return ["## 1. Паспорт прогона", "", *_table(("Параметр", "Значение"), rows)]


def _registry(report: Report) -> list[str]:
    rows = [
        (r.id, r.source_id or "—", KIND_RU[r.kind], _short(r.summary), _place(r.location))
        for r in report.requirements
    ]
    head = "## 2. Реестр требований"
    note = "Идентификатор в документе указан, если он там был; «—» значит, что агент присвоил свой."
    if not rows:
        return [head, "", "Требования в документе не найдены."]
    header = ("ID в отчёте", "ID в документе", "Тип", "Кратко", "Место")
    return [head, "", note, "", *_table(header, rows)]


def _finding_block(f: Finding) -> list[str]:
    reqs = ", ".join(f.requirement_ids) or "—"
    lines = [
        f"#### {f.id} — {SEVERITY_RU[f.severity]}, {DEFECT_RU[f.defect_type]}",
        "",
        f"- Требования: {reqs}",
        f"- Свойство: {CRITERION_RU[f.criterion]}; пункт стандарта: {f.standard_ref}",
        "- Цитаты:",
        "",
    ]
    for q in f.quotes:
        lines += _blockquote(q.text)
        lines += [">", f"> — {_place(q.location)}", ""]
    lines.append(f"Почему это дефект: {_inline(f.explanation)}")
    if f.suggestion:
        lines += ["", f"Предложение правки: {_inline(f.suggestion)}"]
    return [*lines, ""]


def _per_requirement(report: Report) -> list[str]:
    out = ["## 3. Проверка каждого требования", ""]
    if report.passport.model_error:
        out += [_INTERRUPTED, ""]
    elif report.passport.mode is RunMode.RULES_ONLY:
        out += [
            "Прогон без модели: применены только правила, смысловая проверка не выполнялась.",
            "",
        ]
    by_req: dict[str, list[Finding]] = defaultdict(list)
    detached: list[Finding] = []
    for f in report.findings:
        if f.scope is not Scope.REQUIREMENT:
            continue
        if f.requirement_ids:
            by_req[f.requirement_ids[0]].append(f)
        else:
            detached.append(f)
    for req in report.requirements:
        if req.id not in by_req:
            continue
        out += [f"### {req.id}" + (f" ({req.source_id})" if req.source_id else ""), ""]
        out += [*_blockquote(req.quote), ">", f"> — {_place(req.location)}", ""]
        for f in by_req[req.id]:
            out += _finding_block(f)
    if detached:
        out += ["### Замечания без привязки к одному требованию", ""]
        for f in detached:
            out += _finding_block(f)
    flagged = {rid for f in report.findings for rid in f.requirement_ids}
    clean = [r.id for r in report.requirements if r.id not in flagged]
    if not by_req and not detached:
        out += ["Замечаний к отдельным требованиям нет.", ""]
    if clean:
        out.append("Без замечаний: " + ", ".join(clean))
    return out


def _trace(report: Report) -> list[str]:
    out = ["### Трассировка критериев приёмки", ""]
    trace = report.trace
    if not report.set_check_done:
        if report.passport.model_error:
            return [*out, "Не выполнялась: проверка моделью прервана."]
        if report.passport.mode is RunMode.FULL:
            return [*out, "Не построена: в документе не найдено критериев приёмки."]
        return [*out, "Не выполнялась: прогон без модели."]
    if trace is None or not (trace.links or trace.orphan_acceptance_ids):
        return [*out, "Не построена: в документе не найдено критериев приёмки."]
    if trace.links:
        rows = [(link.acceptance_id, ", ".join(link.covers) or "—") for link in trace.links]
        out += [*_table(("Критерий приёмки", "Покрываемые требования"), rows), ""]
    uncovered = ", ".join(trace.uncovered_requirement_ids) or "нет"
    orphans = ", ".join(trace.orphan_acceptance_ids) or "нет"
    out.append(f"Требования без критерия приёмки: {uncovered}")
    out += ["", f"Критерии приёмки, не покрывающие ни одного требования: {orphans}"]
    return out


def _set_check(report: Report) -> list[str]:
    out = ["## 4. Проверка набора требований", ""]
    if report.passport.model_error:
        out += [_INTERRUPTED, ""]
    if not report.set_check_done:
        out += ["Не выполнялась: противоречия и пробелы набора не искались.", ""]
    set_findings = [f for f in report.findings if f.scope is Scope.REQUIREMENT_SET]
    for f in set_findings:
        out += _finding_block(f)
    if report.set_check_done and not set_findings:
        out += ["Замечаний к набору в целом нет.", ""]
    return [*out, *_trace(report)]


def _finding_target(f: Finding) -> str:
    return ", ".join(f.requirement_ids) or "набор в целом"


def _summary(report: Report) -> list[str]:
    s = report.summary
    by_id = {f.id: f for f in report.findings}
    out = [
        "## 5. Сводка",
        "",
        f"- Требований: {s.requirements_total}",
        f"- Замечаний: {s.findings_total}",
    ]
    for sev in SEVERITY_ORDER:
        out.append(f"  - {SEVERITY_RU[sev]}: {s.by_severity.get(sev, 0)}")
    out += ["", "Критичные дефекты (блокирующие и критичные):", ""]
    rows = []
    for fid in s.critical_finding_ids:
        f = by_id[fid]
        rows.append(
            (
                fid,
                DEFECT_RU[f.defect_type],
                _finding_target(f),
                "«" + _short(f.quotes[0].text) + "»",
            )
        )
    if rows:
        out += _table(("Замечание", "Тип", "Требование", "Цитата"), rows)
    else:
        out.append("Критичных дефектов нет.")
    return out


def _verdict_text(report: Report) -> str:
    if report.verdict is Verdict.NO_GO:
        return "Начинать разработку нельзя: есть блокирующие дефекты."
    if report.verdict is Verdict.CONDITIONAL_GO:
        if report.summary.critical_finding_ids:
            return "Можно начинать с условием: сначала исправить критичные дефекты."
        return "Можно начинать с условием: проверка неполная, см. ограничения ниже."
    return "Можно начинать."


def _verdict(report: Report) -> list[str]:
    by_id = {f.id: f for f in report.findings}
    out = ["## 6. Итог", "", f"**{_verdict_text(report)}**", ""]
    p = report.passport
    if p.model_error:
        out += [f"Ограничение: проверка моделью прервана ({p.model_error}); {_INCOMPLETE}", ""]
    elif p.mode is RunMode.RULES_ONLY:
        out += ["Ограничение: прогон без модели, итог не выше «можно начинать с условием».", ""]
    if p.unread_pages:
        out += [f"Ограничение: не прочитаны страницы {_pages(p.unread_pages)}.", ""]
    crit = [by_id[i] for i in report.summary.critical_finding_ids]
    if crit:
        out += ["Основание:", ""]
        out += [
            f"- {f.id} ({SEVERITY_RU[f.severity]}, {DEFECT_RU[f.defect_type]}; "
            f"{_finding_target(f)}): «{_short(f.quotes[0].text)}»"
            for f in crit
        ]
        out.append("")
    out.append("Исправить в первую очередь:")
    out.append("")
    if not report.fix_first:
        out.append("Нечего исправлять в первую очередь: блокирующих и критичных дефектов нет.")
        return out
    for n, fid in enumerate(report.fix_first, start=1):
        f = by_id[fid]
        action = _inline(f.suggestion) if f.suggestion else _inline(f.explanation)
        out.append(f"{n}. {f.id} ({_finding_target(f)}): {action}")
    return out


def render_markdown(report: Report) -> str:
    """Отрисовать отчёт на русском: паспорт, реестр, проверка каждого
    требования, проверка набора, сводка, итог. Ни одной фразы без места в документе.
    """
    sections = [
        ["# Отчёт о проверке требований", ""],
        _passport(report),
        _registry(report),
        _per_requirement(report),
        _set_check(report),
        _summary(report),
        _verdict(report),
    ]
    return "\n\n".join("\n".join(s).rstrip() for s in sections) + "\n"
