"""Проверка цитат (FR-7), справочник разметки (FR-9), дедупликация (FR-16)."""

import unicodedata
from dataclasses import dataclass, replace

from reqcheck import taxonomy
from reqcheck.models import (
    MIN_QUOTE_LENGTH,
    DefectType,
    Document,
    Finding,
    FindingSource,
    Fragment,
    GroundingResult,
    Location,
    Quote,
    RawFinding,
    Requirement,
    RequirementKind,
    Scope,
    Severity,
)

_QUOTE_CHARS = "«»„“”‟‹›〝〞\"'‘’‚‛"
_DASH_CHARS = "‐‑‒–—―−⁃﹘﹣－-"
_INVISIBLE = "­​‌‍⁠﻿"
# Разметка выделения Markdown: `*` и обратные кавычки убираем всегда, `_` только серией `__`
# (одиночный `_` бывает внутри идентификаторов вроде snake_case).
_MARKUP = "*`"
_LINE_BREAKS = "\n\r\x0b\x0c  \x85"

# --- Нормализация с картой индексов ---------------------------------------------


@dataclass(frozen=True)
class _Norm:
    text: str
    starts: tuple[int, ...]
    ends: tuple[int, ...]


def _clusters(text: str) -> list[tuple[str, int, int]]:
    """Символ вместе с комбинирующими знаками, чтобы NFKC не ломал «й», «ё»."""
    result: list[tuple[str, int, int]] = []
    for index, char in enumerate(text):
        if result and unicodedata.combining(char):
            prev, start, _ = result[-1]
            result[-1] = (prev + char, start, index + 1)
        else:
            result.append((char, index, index + 1))
    return result


def _map_char(char: str) -> str:
    if char in _QUOTE_CHARS:
        return '"'
    if char in _DASH_CHARS:
        return "-"
    return char


def _double_underscores(text: str) -> set[int]:
    """Индексы символов `_` в сериях из двух и более подряд (маркеры выделения)."""
    skipped: set[int] = set()
    i = 0
    while i < len(text):
        j = i
        while j < len(text) and text[j] == "_":
            j += 1
        if j - i >= 2:
            skipped.update(range(i, j))
        i = max(j, i + 1)
    return skipped


def _tokens(text: str) -> list[list]:
    """[символ, начало, конец, был_перенос]; пробельные серии схлопнуты в один пробел."""
    tokens: list[list] = []
    skipped = _double_underscores(text)
    for cluster, start, end in _clusters(text):
        for char in unicodedata.normalize("NFKC", cluster):
            if char in _INVISIBLE or char in _MARKUP or start in skipped:
                continue
            if char.isspace():
                if tokens and tokens[-1][0] == " ":
                    tokens[-1][2] = end
                    tokens[-1][3] = tokens[-1][3] or char in _LINE_BREAKS
                else:
                    tokens.append([" ", start, end, char in _LINE_BREAKS])
            else:
                tokens.append([_map_char(char), start, end, False])
    return tokens


def _is_wrapped_hyphen(tokens: list[list], i: int) -> bool:
    """«требова-\\nние»: дефис между буквами на конце строки склеивает слово."""
    if tokens[i][0] != "-" or i == 0 or i + 2 >= len(tokens):
        return False
    gap, after = tokens[i + 1], tokens[i + 2]
    return tokens[i - 1][0].isalpha() and gap[0] == " " and gap[3] and after[0].islower()


def normalize(text: str) -> _Norm:
    tokens = _tokens(text)
    kept: list[list] = []
    i = 0
    while i < len(tokens):
        if _is_wrapped_hyphen(tokens, i):
            i += 2
            continue
        kept.append(tokens[i])
        i += 1
    return _Norm(
        "".join(t[0] for t in kept),
        tuple(t[1] for t in kept),
        tuple(t[2] for t in kept),
    )


def _norm_text(text: str) -> str:
    return normalize(text).text.strip()


# --- Поиск цитат ----------------------------------------------------------------


@dataclass(frozen=True)
class _Hit:
    text: str
    location: Location
    requirement_id: str | None
    fragment_index: int
    start: int
    end: int
    key: str


class _Index:
    """Нормализованный документ и поиск цитат по фрагментам."""

    def __init__(self, document: Document, requirements: tuple[Requirement, ...]) -> None:
        self._fragments = document.fragments
        self._norms = [normalize(f.text) for f in self._fragments]
        self._position = {f.id: i for i, f in enumerate(self._fragments)}
        self.by_id = {r.id: r for r in requirements}
        self._owners: dict[str, list[str]] = {}
        for req in requirements:
            self._owners.setdefault(req.fragment_id, []).append(req.id)

    def _order(self, ids: tuple[str, ...]) -> list[int]:
        first = [
            self._position[self.by_id[i].fragment_id]
            for i in ids
            if i in self.by_id and self.by_id[i].fragment_id in self._position
        ]
        first = list(dict.fromkeys(first))
        return first + [i for i in range(len(self._fragments)) if i not in first]

    def _owner(self, fragment: Fragment, ids: tuple[str, ...]) -> str | None:
        owners = self._owners.get(fragment.id, [])
        return next((o for o in owners if o in ids), owners[0] if owners else None)

    def find(self, quote: str, ids: tuple[str, ...]) -> _Hit | None:
        needle = _norm_text(quote)
        if not needle:
            return None
        for index in self._order(ids):
            norm = self._norms[index]
            pos = norm.text.find(needle)
            if pos >= 0:
                return self._hit(index, pos, pos + len(needle), needle, ids)
        return None

    def _hit(self, index: int, pos: int, stop: int, key: str, ids: tuple[str, ...]) -> _Hit | None:
        fragment, norm = self._fragments[index], self._norms[index]
        src_start, src_end = norm.starts[pos], norm.ends[stop - 1]
        text = fragment.text[src_start:src_end]
        if len(text.strip()) < MIN_QUOTE_LENGTH:
            return None
        base = fragment.location.line_start
        location = Location(
            line_start=base + fragment.text.count("\n", 0, src_start),
            line_end=base + fragment.text.count("\n", 0, src_end),
            page=fragment.location.page,
        )
        return _Hit(text, location, self._owner(fragment, ids), index, pos, stop, key)


# --- Кандидаты, дедупликация ----------------------------------------------------


@dataclass(frozen=True)
class _Cand:
    raw: RawFinding
    ids: tuple[str, ...]
    hits: tuple[_Hit, ...]
    severity: Severity
    suggestion: str | None

    @property
    def first_position(self) -> tuple[int, int]:
        return min((h.fragment_index, h.start) for h in self.hits)


def _rank(severity: Severity) -> int:
    return taxonomy.SEVERITY_ORDER.index(severity)


def _severity(raw: RawFinding, hits: tuple[_Hit, ...], index: _Index) -> Severity:
    severity = taxonomy.TAXONOMY[raw.defect_type].severity
    if (
        taxonomy.PROMOTE_ACCEPTANCE_TO_BLOCKER
        and raw.defect_type in taxonomy.ACCEPTANCE_PROMOTED_TYPES
    ):
        for hit in hits:
            req = index.by_id.get(hit.requirement_id or "")
            if req is not None and req.kind is RequirementKind.ACCEPTANCE:
                return Severity.BLOCKER
    return severity


def _is_noop_suggestion(cand: _Cand, index: _Index) -> bool:
    """FR-18: правка «Было = Стало» — предложение равно цитате или тексту требования."""
    if not cand.suggestion:
        return False
    canon = _canon(cand.suggestion)
    originals = [h.text for h in cand.hits]
    originals += [index.by_id[i].quote for i in cand.ids if i in index.by_id]
    return any(canon == _canon(text) for text in originals)


def _canon(text: str) -> str:
    return _norm_text(text).strip(" .;,:").casefold()


def _ground(raw: RawFinding, index: _Index) -> tuple[_Cand | None, int, int]:
    """Вернуть кандидата (или None) и счётчики (проверено, подтверждено)."""
    ids = tuple(sorted({i for i in raw.requirement_ids if i in index.by_id}))
    hits = [index.find(q, ids) for q in raw.quotes]
    confirmed = sum(h is not None for h in hits)
    if confirmed < len(hits):
        return None, len(hits), confirmed
    found = tuple(h for h in hits if h is not None)
    if raw.defect_type is DefectType.CONTRADICTION and _has_overlap(found):
        return None, len(hits), confirmed
    if raw.scope is Scope.REQUIREMENT and not ids:
        ids = tuple(sorted({h.requirement_id for h in found if h.requirement_id}))
    cand = _Cand(raw, ids, found, _severity(raw, found, index), raw.suggestion)
    return cand, len(hits), confirmed


def _dedup_key(cand: _Cand) -> tuple:
    if cand.raw.scope is Scope.REQUIREMENT and cand.ids:
        return ("requirement", cand.ids, cand.raw.defect_type)
    keys = tuple(sorted({h.key for h in cand.hits}))
    return ("set", cand.raw.defect_type, keys)


def _prefer(a: _Cand, b: _Cand) -> _Cand:
    """Победитель при слиянии: модель богаче объяснением, затем серьёзнее, затем длиннее текст."""

    def order(c: _Cand) -> tuple[int, int, int]:
        return (c.raw.source is FindingSource.RULE, _rank(c.severity), -len(c.raw.explanation))

    winner, loser = (a, b) if order(a) <= order(b) else (b, a)
    if winner.suggestion is None and loser.suggestion is not None:
        winner = replace(winner, suggestion=loser.suggestion)
    return winner


def _same_group(a: DefectType, b: DefectType) -> bool:
    return any(a in group and b in group for group in taxonomy.MERGEABLE_GROUPS)


def _overlap(a: _Cand, b: _Cand) -> bool:
    return any(
        x.fragment_index == y.fragment_index and x.start < y.end and y.start < x.end
        for x in a.hits
        for y in b.hits
    )


def _has_overlap(hits: tuple[_Hit, ...]) -> bool:
    """Две цитаты противоречия указывают на одно и то же место документа."""
    return any(
        x.fragment_index == y.fragment_index and x.start < y.end and y.start < x.end
        for i, x in enumerate(hits)
        for y in hits[i + 1 :]
    )


def _mergeable(a: _Cand, b: _Cand) -> bool:
    return (
        a.raw.scope is Scope.REQUIREMENT
        and b.raw.scope is Scope.REQUIREMENT
        and bool(a.ids)
        and a.ids == b.ids
        and _same_group(a.raw.defect_type, b.raw.defect_type)
        and _overlap(a, b)
    )


def _merge_overlapping(cands: list[_Cand]) -> list[_Cand]:
    """Перекрывающиеся цитаты, один набор id, типы одной группы: берём более серьёзный."""
    result = list(cands)
    changed = True
    while changed:
        changed = False
        for i, a in enumerate(result):
            j = next((k for k in range(i + 1, len(result)) if _mergeable(a, result[k])), None)
            if j is None:
                continue
            other = result.pop(j)
            more_severe = a if _rank(a.severity) <= _rank(other.severity) else other
            result[i] = _prefer(more_severe, other if more_severe is a else a)
            changed = True
            break
    return result


def _dedup(cands: list[_Cand]) -> list[_Cand]:
    unique: dict[tuple, _Cand] = {}
    for cand in cands:
        key = _dedup_key(cand)
        unique[key] = _prefer(unique[key], cand) if key in unique else cand
    return _merge_overlapping(list(unique.values()))


def _sort_key(cand: _Cand) -> tuple:
    return (
        _rank(cand.severity),
        cand.first_position,
        cand.raw.defect_type.value,
        cand.ids,
        cand.raw.explanation,
    )


def _build(number: int, cand: _Cand) -> Finding:
    markup = taxonomy.TAXONOMY[cand.raw.defect_type]
    return Finding(
        id=f"D-{number:03d}",
        scope=cand.raw.scope,
        defect_type=cand.raw.defect_type,
        criterion=markup.criterion,
        finding_class=markup.finding_class,
        severity=cand.severity,
        standard_ref=markup.standard_ref,
        requirement_ids=cand.ids,
        quotes=tuple(
            Quote(text=h.text, location=h.location, requirement_id=h.requirement_id)
            for h in cand.hits
        ),
        explanation=cand.raw.explanation,
        suggestion=cand.suggestion,
        source=cand.raw.source,
        rule_id=cand.raw.rule_id,
    )


def verify_and_classify(
    document: Document, requirements: tuple[Requirement, ...], raw: list[RawFinding]
) -> GroundingResult:
    """Оставить только замечания с цитатами, найденными в документе.

    Цитата нормализуется (пробелы, переносы, кавычки) и ищется точным
    вхождением; не найдена — замечание отбрасывается и считается в
    findings_dropped. Серьёзность, класс и пункт стандарта ставит справочник
    по defect_type. Идентификаторы D-001… присваиваются здесь.
    """
    index = _Index(document, requirements)
    checked = confirmed = dropped = 0
    cands: list[_Cand] = []
    for item in raw:
        cand, n_checked, n_confirmed = _ground(item, index)
        checked += n_checked
        confirmed += n_confirmed
        if cand is None or _is_noop_suggestion(cand, index):
            dropped += 1
            continue
        cands.append(cand)
    ordered = sorted(_dedup(cands), key=_sort_key)
    return GroundingResult(
        findings=tuple(_build(n, c) for n, c in enumerate(ordered, start=1)),
        quotes_checked=checked,
        quotes_confirmed=confirmed,
        findings_dropped=dropped,
    )
