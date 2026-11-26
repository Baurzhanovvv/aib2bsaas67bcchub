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
SET_TYPES = frozenset(
    {
        DefectType.CONTRADICTION,
        DefectType.MISSING_REQUIREMENT,
        DefectType.DUPLICATE,
        DefectType.INCONSISTENT_TERMS,
        DefectType.INFEASIBLE,
        DefectType.INTERFACE_MISUSE,
        DefectType.UNDEFINED_REFERENCE,
        DefectType.DESIGN_CONSTRAINT,
    }
)
"""Свойства одного требования уже проверены в FR-5; покрытие считает код по links."""
TRACED_KINDS = frozenset({RequirementKind.FUNCTIONAL, RequirementKind.INTERFACE})

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


def check_requirements(requirements: tuple[Requirement, ...], llm: LLMClient) -> list[RawFinding]:
    """Проверить требования по одному по ISO/IEC/IEEE 29148 §5.2.5."""
    if not requirements:
        return []
    token = run_token(_requirements_seed(requirements))
    system = prompts.requirement_system(*markers(token))
    findings: list[RawFinding] = []
    for batch in split_batches(requirements):
        body = "\n\n".join(format_requirement(req, req.section) for req in batch)
        user = prompts.requirement_user(wrap(token, body))
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


def _requirements_seed(requirements: Iterable[Requirement]) -> str:
    return "\n".join(f"{req.id}\t{req.quote}" for req in requirements)


# --- Разбор ответа модели ------------------------------------------------------------------


def to_raw(finding: FindingDraft, known: set[str], scope: Scope) -> RawFinding | None:
    """Черновик модели → RawFinding: чужие id отброшены, метки [R-NNN] срезаны с цитат.

    None — если после чистки замечание невалидно: одно битое замечание не роняет ответ.
    """
    if not finding.explanation.strip():
        return None
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
            suggestion=finding.suggestion,
            scope=scope,
            source=FindingSource.MODEL,
        )
    except ValidationError:
        return None


def _dedupe(findings: list[RawFinding]) -> list[RawFinding]:
    seen: set[tuple[DefectType, frozenset[str]]] = set()
    unique: list[RawFinding] = []
    for finding in findings:
        key = (finding.defect_type, frozenset(finding.quotes))
        if key not in seen:
            seen.add(key)
            unique.append(finding)
    return unique


# --- FR-6 -------------------------------------------------------------------------------


def check_set(
    document: Document, requirements: tuple[Requirement, ...], llm: LLMClient
) -> tuple[list[RawFinding], TraceReport]:
    """Проверить набор по §5.2.6 и IEEE 830 §4.3 и связать критерии приёмки с требованиями."""
    if not requirements:
        return [], TraceReport()
    has_acceptance = any(r.kind is RequirementKind.ACCEPTANCE for r in requirements)
    token = run_token(document.sha256)
    system = prompts.set_system(*markers(token), has_acceptance=has_acceptance)
    sections = {f.id: f.section for f in document.fragments}
    registry = wrap(token, render_registry(requirements, sections))
    chunks = split_document(document.fragments)
    known = {r.id for r in requirements}
    findings: list[RawFinding] = []
    links: list[LinkDraft] = []
    for number, chunk in enumerate(chunks, start=1):
        part = f"{number} из {len(chunks)}" if len(chunks) > 1 else None
        text = wrap(token, render_fragments(chunk, requirements))
        user = prompts.set_user(registry, text, part=part)
        result = llm.complete(system=system, user=user, response_model=SetCheckResponse)
        for finding in result.data.findings:
            if finding.defect_type not in SET_TYPES:
                continue
            raw = to_raw(finding, known, Scope.REQUIREMENT_SET)
            if raw is not None:
                findings.append(raw)
        links.extend(result.data.links)
    trace = build_trace(links, requirements) if has_acceptance else TraceReport()
    return _dedupe(findings), trace


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
