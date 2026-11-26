"""Проверки с моделью: каждое требование (FR-5) и набор целиком (FR-6).

Текст документа подаётся модели как данные внутри маркеров (FR-21). Маркер
выводится из sha256 входа, а не случайно: одинаковый вход даёт одинаковый
промпт, и кэш клиента срабатывает. Ошибки клиента (ModelError) не глушатся.
"""

import hashlib
import math
import re
from collections.abc import Iterable

from pydantic import ValidationError

from reqcheck import prompts
from reqcheck.llm import (
    FindingDraft,
    LinkDraft,
    LLMClient,
    RequirementCheckResponse,
    SetCheckResponse,
)
from reqcheck.models import (
    MIN_QUOTE_LENGTH,
    AcceptanceLink,
    DefectType,
    Document,
    FindingSource,
    Fragment,
    FragmentKind,
    RawFinding,
    Requirement,
    RequirementKind,
    Scope,
    TraceReport,
)
from reqcheck.taxonomy import SEVERITY_ORDER, severity_for

BATCH_MAX = 25
"""Требований в одном вызове FR-5; 60 требований — 3 вызова по 20."""
SET_CHAR_BUDGET = 60_000
"""Символов текста документа в одном вызове FR-6; больше — делим по разделам."""

REQUIREMENT_TYPES = frozenset(
    {
        DefectType.AMBIGUOUS,
        DefectType.UNVERIFIABLE,
        DefectType.INCOMPLETE,
        DefectType.NOT_SINGULAR,
        DefectType.INFEASIBLE,
        DefectType.DESIGN_CONSTRAINT,
        DefectType.UNDEFINED_REFERENCE,
        DefectType.INTERFACE_MISUSE,
        DefectType.CONTRADICTION,
    }
)
"""Связи между требованиями (дубли, термины, пропуски) — только в FR-6."""
_D = DefectType
FOCUS_TYPES: dict[str, frozenset[DefectType]] = {
    "rules": frozenset({_D.CONTRADICTION, _D.MISSING_REQUIREMENT, _D.DUPLICATE, _D.INFEASIBLE}),
    "acceptance": frozenset({_D.CONTRADICTION, _D.UNDEFINED_REFERENCE, _D.MISSING_REQUIREMENT}),
    "interface": frozenset(
        {
            _D.INTERFACE_MISUSE,
            _D.INCONSISTENT_TERMS,
            _D.CONTRADICTION,
            _D.MISSING_REQUIREMENT,
            _D.UNDEFINED_REFERENCE,
            _D.INCOMPLETE,
            _D.DESIGN_CONSTRAINT,
        }
    ),
}
"""FR-6 — три узких вызова, у каждого свой список типов (prompts.SET_FOCUSES)."""
SET_TYPES = frozenset().union(*FOCUS_TYPES.values())
"""Свойства одного требования — в FR-5; incomplete — для служебных блоков вне реестра."""
TRACED_KINDS = frozenset({RequirementKind.FUNCTIONAL})
"""Покрытие приёмкой ждём только у бизнес-правил; строки API им не покрываются."""

_ID_TAG = re.compile(r"^\s*\[R-\d{3,}\]\s*")


# --- Обёртка данных (FR-21) -----------------------------------------------------------


def run_token(seed: str) -> str:
    """Токен маркера на прогон: детерминирован от содержимого входа."""
    return hashlib.sha256(f"reqcheck-marker:{seed}".encode()).hexdigest()[:16]


def markers(token: str) -> tuple[str, str]:
    return f"<<<DOC-{token}>>>", f"<<<END-DOC-{token}>>>"


def wrap(token: str, body: str) -> str:
    """Обернуть данные маркерами; токен внутри данных обезвреживается."""
    open_marker, close_marker = markers(token)
    safe = body.replace(token, "[маркер]")
    return f"{open_marker}\n{safe}\n{close_marker}"


# --- FR-5 -------------------------------------------------------------------------------


def check_requirements(
    requirements: tuple[Requirement, ...], llm: LLMClient, document: Document | None = None
) -> list[RawFinding]:
    """Проверить требования по одному по ISO/IEC/IEEE 29148 §5.2.5.

    С document модель видит текст документа как контекст: «не указано» про то, что описано
    в соседнем разделе, — главный источник ложных замечаний.
    """
    if not requirements:
        return []
    token = run_token(document.sha256 if document else _requirements_seed(requirements))
    system = prompts.requirement_system(*markers(token))
    findings: list[RawFinding] = []
    for batch in split_batches(requirements):
        body = "\n\n".join(format_requirement(req, req.section) for req in batch)
        context = batch_context(document, batch, requirements) if document else None
        user = prompts.requirement_user(
            wrap(token, body), wrap(token, context) if context else None
        )
        result = llm.complete(system=system, user=user, response_model=RequirementCheckResponse)
        known = {req.id for req in batch}
        for finding in result.data.findings:
            if finding.defect_type not in REQUIREMENT_TYPES:
                continue
            raw = to_raw(finding, known, Scope.REQUIREMENT)
            if raw is not None and raw.requirement_ids:
                findings.append(raw)
    return _dedupe(findings)


def split_batches(requirements: tuple[Requirement, ...]) -> list[tuple[Requirement, ...]]:
    """Ровные батчи не больше BATCH_MAX с сохранением порядка документа."""
    if not requirements:
        return []
    size = math.ceil(len(requirements) / math.ceil(len(requirements) / BATCH_MAX))
    return [requirements[i : i + size] for i in range(0, len(requirements), size)]


def format_requirement(req: Requirement, section: tuple[str, ...] = ()) -> str:
    where = " > ".join(section) or "—"
    return f"[{req.id}] вид: {req.kind.value}; раздел: {where}\n{req.quote}"


def batch_context(
    document: Document, batch: tuple[Requirement, ...], requirements: tuple[Requirement, ...]
) -> str:
    """Весь документ, если влезает в бюджет; иначе — разделы верхнего уровня пачки."""
    fragments = document.fragments
    if sum(len(f.text) for f in fragments) > SET_CHAR_BUDGET:
        sections = {f.id: f.section for f in fragments}
        tops = {_top(req.section or sections.get(req.fragment_id, ())) for req in batch}
        fragments = tuple(f for f in fragments if _top(f.section) in tops)
    return render_fragments(fragments, requirements)


def _top(section: tuple[str, ...]) -> str:
    return section[0] if section else ""


def _requirements_seed(requirements: Iterable[Requirement]) -> str:
    return "\n".join(f"{req.id}\t{req.quote}" for req in requirements)


# --- Разбор ответа модели ------------------------------------------------------------------


def to_raw(finding: FindingDraft, known: set[str], scope: Scope) -> RawFinding | None:
    """Черновик модели → RawFinding: чужие id отброшены, метки [R-NNN] срезаны с цитат.

    None — если после чистки замечание невалидно: одно битое замечание не роняет ответ.
    """
    if not finding.explanation.strip() or _mostly_latin(finding.explanation):
        return None
    suggestion = finding.suggestion
    if suggestion is not None and _mostly_latin(suggestion):
        suggestion = None
    ids = tuple(dict.fromkeys(i for i in finding.requirement_ids if i in known))
    cleaned = (_ID_TAG.sub("", quote).strip() for quote in finding.quotes)
    quotes = tuple(dict.fromkeys(q for q in cleaned if len(q) >= MIN_QUOTE_LENGTH))
    if not quotes:
        return None
    if finding.defect_type is DefectType.CONTRADICTION and len(quotes) < 2:
        return None
    try:
        return RawFinding(
            defect_type=finding.defect_type,
            requirement_ids=ids,
            quotes=quotes,
            explanation=finding.explanation.strip(),
            suggestion=suggestion,
            impact=finding.impact,
            scope=scope,
            source=FindingSource.MODEL,
        )
    except ValidationError:
        return None


def _mostly_latin(text: str) -> bool:
    """Больше половины букв — латиница: модель ответила не на русском (сбой, не перевод)."""
    letters = [ch for ch in text if ch.isalpha()]
    latin = sum(1 for ch in letters if ch.isascii())
    return bool(letters) and latin * 2 > len(letters)


def _dedupe(findings: list[RawFinding]) -> list[RawFinding]:
    """Одна проблема — одно замечание: те же цитаты под другим типом — дубль.

    Из дублей остаётся самое серьёзное (справочник FR-9), при равенстве — с более
    подробным объяснением; порядок — по первому появлению набора цитат.
    """
    best: dict[frozenset[str], RawFinding] = {}
    for finding in findings:
        key = frozenset(finding.quotes)
        kept = best.get(key)
        if kept is None or _rank(finding) < _rank(kept):
            best[key] = finding
    return list(best.values())


def _rank(finding: RawFinding) -> tuple[int, int]:
    severity = severity_for(finding.defect_type, finding.impact)
    return SEVERITY_ORDER.index(severity), -len(finding.explanation)


# --- FR-6 -------------------------------------------------------------------------------


def check_set(
    document: Document, requirements: tuple[Requirement, ...], llm: LLMClient
) -> tuple[list[RawFinding], TraceReport]:
    """Проверить набор по §5.2.6 и IEEE 830 §4.3 и связать критерии приёмки с требованиями.

    Три узких вызова с общим контекстом: совместимость правил, прогон критериев приёмки
    (и трассировка), интерфейс и данные. Без критериев приёмки второй не вызывается.
    """
    if not requirements:
        return [], TraceReport()
    has_acceptance = any(r.kind is RequirementKind.ACCEPTANCE for r in requirements)
    token = run_token(document.sha256)
    sections = {f.id: f.section for f in document.fragments}
    registry = wrap(token, render_registry(requirements, sections))
    chunks = split_document(document.fragments)
    texts = [wrap(token, render_fragments(chunk, requirements)) for chunk in chunks]
    findings: list[RawFinding] = []
    links: list[LinkDraft] = []
    for focus in prompts.SET_FOCUSES:
        if focus == "acceptance" and not has_acceptance:
            continue
        system = prompts.set_system(*markers(token), focus=focus)
        for number, text in enumerate(texts, start=1):
            part = f"{number} из {len(texts)}" if len(texts) > 1 else None
            user = prompts.set_user(registry, text, part=part)
            result = llm.complete(system=system, user=user, response_model=SetCheckResponse)
            findings += _set_findings(result.data, focus, requirements)
            if focus == "acceptance":
                links.extend(result.data.links)
    trace = build_trace(links, requirements) if has_acceptance else TraceReport()
    return _dedupe(findings), trace


def _set_findings(
    data: SetCheckResponse, focus: str, requirements: tuple[Requirement, ...]
) -> list[RawFinding]:
    allowed = FOCUS_TYPES[focus]
    known = {r.id for r in requirements}
    raws = (
        to_raw(finding, known, Scope.REQUIREMENT_SET)
        for finding in data.findings
        if finding.defect_type in allowed
    )
    return [raw for raw in raws if raw is not None]


def render_registry(
    requirements: tuple[Requirement, ...], sections: dict[str, tuple[str, ...]]
) -> str:
    lines = []
    for req in requirements:
        where = " > ".join(sections.get(req.fragment_id, ())) or "—"
        lines.append(f"[{req.id}] {req.kind.value} | {where} | {req.summary}")
    return "\n".join(lines)


def render_fragments(fragments: Iterable[Fragment], requirements: tuple[Requirement, ...]) -> str:
    """Текст документа по порядку; перед фрагментом-требованием — его id."""
    tags: dict[str, list[str]] = {}
    for req in requirements:
        tags.setdefault(req.fragment_id, []).append(f"[{req.id}]")
    lines = []
    for fragment in fragments:
        if fragment.kind is FragmentKind.HEADING:
            lines.append(f"\n# {fragment.text}")
            continue
        prefix = " ".join(tags.get(fragment.id, ()))
        lines.append(f"{prefix} {fragment.text}" if prefix else fragment.text)
    return "\n".join(lines).strip()


def split_document(fragments: tuple[Fragment, ...]) -> list[tuple[Fragment, ...]]:
    """Части по разделам верхнего уровня, каждая не больше SET_CHAR_BUDGET.

    Раздел, который сам больше бюджета, не режется: подаётся отдельной частью.
    """
    groups: list[list[Fragment]] = []
    current_top: str | None = None
    for fragment in fragments:
        top = fragment.section[0] if fragment.section else ""
        if not groups or top != current_top:
            groups.append([])
            current_top = top
        groups[-1].append(fragment)
    chunks: list[list[Fragment]] = []
    size = 0
    for group in groups:
        group_size = sum(len(f.text) for f in group)
        if chunks and size + group_size <= SET_CHAR_BUDGET:
            chunks[-1].extend(group)
            size += group_size
        else:
            chunks.append(list(group))
            size = group_size
    return [tuple(chunk) for chunk in chunks] or [()]


def build_trace(links: Iterable[LinkDraft], requirements: tuple[Requirement, ...]) -> TraceReport:
    """Связи только между существующими id (битые отброшены); покрытие считает код."""
    acceptance = [r.id for r in requirements if r.kind is RequirementKind.ACCEPTANCE]
    targets = {r.id for r in requirements if r.kind is not RequirementKind.ACCEPTANCE}
    covers: dict[str, dict[str, None]] = {aid: {} for aid in acceptance}
    for link in links:
        if link.acceptance_id not in covers:
            continue
        for rid in link.covers:
            if rid in targets:
                covers[link.acceptance_id][rid] = None
    covered = {rid for ids in covers.values() for rid in ids}
    return TraceReport(
        links=tuple(
            AcceptanceLink(acceptance_id=aid, covers=tuple(ids))
            for aid, ids in covers.items()
            if ids
        ),
        uncovered_requirement_ids=tuple(
            r.id for r in requirements if r.kind in TRACED_KINDS and r.id not in covered
        ),
        orphan_acceptance_ids=tuple(aid for aid, ids in covers.items() if not ids),
    )
