"""Сводные таблицы отчёта и общие помощники отрисовки (только из данных Report)."""

import re
from collections import defaultdict

from reqcheck import taxonomy
from reqcheck.models import DefectType, Finding, Location, Report, RequirementKind, Severity
from reqcheck.stages.render_labels import DEFECT_RU, KIND_RU, SEVERITY_RU

TABLE_QUOTE_LIMIT = 100
REGISTRY_SUMMARY_LIMIT = 80
REGISTRY_QUOTE_LIMIT = 90
OUTSIDE_SECTIONS = "вне разделов"
QUESTION_PREFIX = "вопрос заказчику"
SEVERITY_MARK: dict[Severity, str] = {
    Severity.BLOCKER: "🔴",
    Severity.CRITICAL: "🟠",
    Severity.MAJOR: "🟡",
    Severity.MINOR: "⚪",
}
CRITICAL = (Severity.BLOCKER, Severity.CRITICAL)

_MARKUP = re.compile(r"\*\*|__|`")


def cell(text: str) -> str:
    """Безопасное содержимое ячейки таблицы: без переноса строк и неэкранированных `|`."""
    flat = " ".join(text.split())
    return flat.replace("\\", "\\\\").replace("|", "\\|")


def plain(text: str) -> str:
    """Убрать Markdown-выделение (**, __, обратные кавычки); слова остаются дословными."""
    return _MARKUP.sub("", text)


def short(text: str, limit: int = TABLE_QUOTE_LIMIT) -> str:
    flat = " ".join(plain(text).split())
    return flat if len(flat) <= limit else cut_at_word(flat, limit)


def cut_at_word(flat: str, limit: int) -> str:
    """Обрезать по границе слова, а не посреди него."""
    head = flat[: limit - 1]
    if flat[limit - 1 : limit].isalnum() and " " in head:
        head = head[: head.rindex(" ")]
    return head.rstrip(" ,;:—-«(") + "…"


def table(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(cell(c) for c in row) + " |" for row in rows]
    return lines


def findings_by_requirement(report: Report) -> dict[str, list[Finding]]:
    """Замечания, ссылающиеся на требование: по requirement_ids или Quote.requirement_id."""
    result: dict[str, list[Finding]] = defaultdict(list)
    for f in sorted(report.findings, key=lambda x: x.id):
        ids = set(f.requirement_ids) | {q.requirement_id for q in f.quotes if q.requirement_id}
        for rid in sorted(ids):
            result[rid].append(f)
    return result


def _ids(findings: list[Finding]) -> str:
    return ", ".join(f.id for f in findings) or "—"


def place(loc: Location) -> str:
    """«строка 42», «строки 12–14», «стр. 3, строка 12»."""
    if loc.line_start == loc.line_end:
        lines = f"строка {loc.line_start}"
    else:
        lines = f"строки {loc.line_start}–{loc.line_end}"
    return f"стр. {loc.page}, {lines}" if loc.page is not None else lines


def registry_table(report: Report) -> list[str]:
    by_req = findings_by_requirement(report)
    rows = [
        (
            r.id,
            r.source_id or "—",
            KIND_RU[r.kind],
            short(r.summary, REGISTRY_SUMMARY_LIMIT),
            "«" + short(r.quote, REGISTRY_QUOTE_LIMIT) + "»",
            place(r.location),
            _ids(by_req.get(r.id, [])),
        )
        for r in report.requirements
    ]
    header = ("ID в отчёте", "ID в документе", "Тип", "Кратко", "Цитата", "Место", "Замечания")
    return table(header, rows)


def verdict_table(report: Report, checked_by_model: bool) -> list[str]:
    """Результат по каждому требованию: худшая серьёзность и все замечания, где оно названо."""
    by_req = findings_by_requirement(report)
    clean = "замечаний нет" if checked_by_model else "правилами не найдено"
    rows = []
    for r in report.requirements:
        found = by_req.get(r.id, [])
        worst = (
            min(found, key=lambda f: taxonomy.SEVERITY_ORDER.index(f.severity)) if found else None
        )
        rows.append(
            (
                r.id,
                r.source_id or "—",
                SEVERITY_RU[worst.severity] if worst else clean,
                ", ".join(dict.fromkeys(DEFECT_RU[f.defect_type] for f in found)) or "—",
                _ids(found),
            )
        )
    header = ("ID в отчёте", "ID в документе", "Результат", "Типы дефектов", "Замечания")
    return table(header, rows)


def acceptance_table(report: Report) -> list[str]:
    """Критерии приёмки: что покрывают и есть ли к ним замечания."""
    out = ["### Таблица критериев приёмки", ""]
    items = [r for r in report.requirements if r.kind is RequirementKind.ACCEPTANCE]
    if not items:
        return [*out, "Критериев приёмки в документе не найдено."]
    covers = (
        {link.acceptance_id: link.covers for link in report.trace.links} if report.trace else {}
    )
    by_req = findings_by_requirement(report)
    rows = []
    for r in items:
        found = by_req.get(r.id, [])
        status = "✗ " + _ids(found) if found else "✓"
        rows.append(
            (
                r.id,
                short(r.summary, REGISTRY_SUMMARY_LIMIT),
                ", ".join(covers.get(r.id, ())) or "—",
                status,
            )
        )
    header = ("Критерий", "Кратко", "Покрывает требования", "Статус")
    return [*out, *table(header, rows), ""]


def coverage_table(report: Report) -> list[str]:
    """Покрытие по разделам верхнего уровня документа."""
    by_req = findings_by_requirement(report)
    sections: dict[str, list[str]] = {}
    for r in report.requirements:
        sections.setdefault(r.section[0] if r.section else OUTSIDE_SECTIONS, []).append(r.id)
    rows = []
    for name, ids in sections.items():
        found = {f.id: f for rid in ids for f in by_req.get(rid, [])}
        crit = sum(1 for f in found.values() if f.severity in CRITICAL)
        rows.append((name, str(len(ids)), str(len(found)), str(crit)))
    loose = [f for f in report.findings if not f.requirement_ids]
    if loose:
        crit = sum(1 for f in loose if f.severity in CRITICAL)
        rows.append(("Набор в целом (без привязки к требованию)", "—", str(len(loose)), str(crit)))
    header = ("Раздел документа", "Требований", "Замечаний", "🔴🟠 Блокирующих и критичных")
    return ["Покрытие по разделам документа:", "", *table(header, rows)]


def severity_table(report: Report) -> list[str]:
    counts = report.summary.by_severity
    rows = [(f"{SEVERITY_MARK[s]} {SEVERITY_RU[s]}", str(counts.get(s, 0))) for s in SEVERITY_MARK]
    return table(("Серьёзность", "Замечаний"), rows)


ROLE_PRODUCT = "Владельцу продукта (бизнес-правила)"
ROLE_ARCHITECT = "Архитектору и владельцу API (интерфейс, данные, нефункциональные)"
ROLE_QA = "Руководителю тестирования (проверяемость и приёмка)"
ROLES = (ROLE_PRODUCT, ROLE_ARCHITECT, ROLE_QA)
_TECH_TYPES = {DefectType.INTERFACE_MISUSE, DefectType.INCONSISTENT_TERMS}
_TECH_WORDS = re.compile(r"\b(?:API|HTTP|URL|Swagger|OpenAPI|JSON|эндпоинт\w*)\b")
_TECH_KINDS = {RequirementKind.INTERFACE, RequirementKind.NON_FUNCTIONAL}


def question_role(finding: Finding, kinds: dict[str, RequirementKind]) -> str:
    """Кому адресовать вопрос (FR-17): по типу дефекта и виду затронутых требований."""
    touched = {kinds[i] for i in finding.requirement_ids if i in kinds}
    technical = bool(finding.suggestion and _TECH_WORDS.search(finding.suggestion))
    if finding.defect_type in _TECH_TYPES or (touched and touched <= _TECH_KINDS) or technical:
        return ROLE_ARCHITECT
    if finding.defect_type is DefectType.UNVERIFIABLE or touched == {RequirementKind.ACCEPTANCE}:
        return ROLE_QA
    return ROLE_PRODUCT


def customer_questions(report: Report) -> list[str]:
    """Замечания, чья правка — вопрос заказчику; пусто, если таких нет."""
    items = [
        f
        for f in sorted(report.findings, key=lambda x: x.id)
        if f.suggestion and f.suggestion.strip().lower().startswith(QUESTION_PREFIX)
    ]
    if not items:
        return []
    kinds = {r.id: r.kind for r in report.requirements}
    by_role: dict[str, list[Finding]] = {role: [] for role in ROLES}
    for f in items:
        by_role[question_role(f, kinds)].append(f)
    out = ["### Вопросы заказчику", "", "Сгруппированы по тому, кто может ответить.", ""]
    number = 0
    for role, asked in by_role.items():
        if not asked:
            continue
        out += [f"**{role}**", ""]
        for f in asked:
            number += 1
            assert f.suggestion is not None
            text = " ".join(f.suggestion.split())[len(QUESTION_PREFIX) :].lstrip(" :—-–")
            reqs = ", ".join(f.requirement_ids) or "набор в целом"
            out.append(f"{number}. {text[:1].upper()}{text[1:]} ({f.id}; {reqs})")
        out.append("")
    return out
