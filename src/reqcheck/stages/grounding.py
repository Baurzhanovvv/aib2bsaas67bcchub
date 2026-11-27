"""Проверка цитат (FR-7), справочник разметки (FR-9), дедупликация (FR-16)."""

import re
from dataclasses import dataclass, replace

from reqcheck import taxonomy
from reqcheck.models import (
    MIN_QUOTE_LENGTH,
    DefectType,
    Document,
    Finding,
    FindingSource,
    Fragment,
    FragmentKind,
    GroundingResult,
    Impact,
    Location,
    Quote,
    RawFinding,
    Requirement,
    RequirementKind,
    Scope,
    Severity,
)
from reqcheck.text_norm import (
    match_adjacent_cells,
    norm_text,
    normalize,
    strip_heading_marks,
    table_segments,
)

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

    def _order(self, ids: tuple[str, ...], near: bool = False) -> list[int]:
        """Сначала фрагменты требований из ids, затем (near) соседние ±2, затем остальные."""
        first = [
            self._position[self.by_id[i].fragment_id]
            for i in ids
            if i in self.by_id and self.by_id[i].fragment_id in self._position
        ]
        order = list(dict.fromkeys(first))
        if near:
            around = sorted({j for i in order for j in range(i - 2, i + 3)})
            order += [j for j in around if 0 <= j < len(self._fragments) and j not in order]
        return order + [i for i in range(len(self._fragments)) if i not in order]

    def _owner(self, fragment: Fragment, ids: tuple[str, ...]) -> str | None:
        owners = self._owners.get(fragment.id, [])
        return next((o for o in owners if o in ids), owners[0] if owners else None)

    def find(self, quote: str, ids: tuple[str, ...]) -> list[_Hit]:
        """Найти цитату; пустой список — не найдена. Составная цитата даёт несколько мест."""
        quote = strip_heading_marks(quote)
        whole = self._find_exact(norm_text(quote), ids)
        if whole is not None:
            return [whole]
        return self._find_composite(quote, ids) or self._find_table_row(norm_text(quote), ids)

    def _find_exact(self, needle: str, ids: tuple[str, ...], near: bool = False) -> _Hit | None:
        if not needle:
            return None
        for index in self._order(ids, near):
            pos = self._norms[index].text.find(needle)
            if pos >= 0:
                return self._hit(index, pos, pos + len(needle), needle, ids)
        return None

    def _find_composite(self, quote: str, ids: tuple[str, ...]) -> list[_Hit]:
        """Два места, склеенные моделью через перевод строки: части ищем рядом с требованием."""
        parts = list(dict.fromkeys(norm_text(p) for p in quote.split("\n") if p.strip()))
        if len(parts) < 2 or any(len(p) < MIN_QUOTE_LENGTH for p in parts):
            return []
        hits = [self._find_exact(p, ids, near=True) for p in parts]
        if not all(hits):
            return []
        unique = {(h.fragment_index, h.start, h.end): h for h in hits if h}
        return list(unique.values())

    def _find_table_row(self, needle: str, ids: tuple[str, ...]) -> list[_Hit]:
        """«ключ: значение» вместо строки таблицы: сегменты в соседних ячейках одной строки."""
        segments = table_segments(needle)
        if not segments:
            return []
        for index in self._order(ids):
            if self._fragments[index].kind is not FragmentKind.TABLE_ROW:
                continue
            span = match_adjacent_cells(self._norms[index].text, segments)
            if span is not None:
                hit = self._hit(index, span[0], span[1], needle, ids)
                return [hit] if hit else []
        return []

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


MERGE_OVERLAP = 0.5

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


def _kinds(hits: tuple[_Hit, ...], index: _Index) -> set[RequirementKind]:
    reqs = (index.by_id.get(h.requirement_id or "") for h in hits)
    return {r.kind for r in reqs if r is not None}


def _severity(raw: RawFinding, hits: tuple[_Hit, ...], index: _Index) -> Severity:
    severity = taxonomy.severity_for(raw.defect_type, raw.impact)
    kinds = _kinds(hits, index)
    promote = taxonomy.PROMOTE_ACCEPTANCE_TO_BLOCKER
    if promote and raw.defect_type in taxonomy.ACCEPTANCE_PROMOTED_TYPES:
        if RequirementKind.ACCEPTANCE in kinds:
            return Severity.BLOCKER
    # Два бизнес-правила, которые нельзя выполнить разом, блокируют разработку при любой
    # оценке последствия моделью: реализовать можно только одно из них.
    owners = {h.requirement_id for h in hits if h.requirement_id}
    if (
        raw.defect_type is DefectType.CONTRADICTION
        and raw.impact is not Impact.LOW
        and len(owners) >= 2
        and kinds
        and kinds <= taxonomy.BLOCKER_REQUIREMENT_KINDS
    ):
        return Severity.BLOCKER
    # Блокирует разработку расхождение в бизнес-поведении, а не в оформлении протокола.
    if severity is Severity.BLOCKER and not kinds & taxonomy.BLOCKER_REQUIREMENT_KINDS:
        return Severity.CRITICAL
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
    return norm_text(text).strip(" .;,:").casefold()


def _ground(raw: RawFinding, index: _Index) -> tuple[_Cand | None, int, int]:
    """Вернуть кандидата (или None) и счётчики (проверено, подтверждено)."""
    ids = tuple(sorted({i for i in raw.requirement_ids if i in index.by_id}))
    per_quote = [index.find(q, ids) for q in raw.quotes]
    confirmed = sum(bool(h) for h in per_quote)
    if confirmed < len(per_quote):
        return None, len(per_quote), confirmed
    found = tuple(h for hits in per_quote for h in hits)
    if raw.defect_type is DefectType.CONTRADICTION and _has_overlap(found):
        return None, len(per_quote), confirmed
    if raw.scope is Scope.REQUIREMENT and not ids:
        ids = tuple(sorted({h.requirement_id for h in found if h.requirement_id}))
    cand = _Cand(raw, ids, found, _severity(raw, found, index), raw.suggestion)
    return cand, len(per_quote), confirmed


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
    top = min((winner.severity, loser.severity), key=_rank)
    winner = replace(winner, severity=top)
    if winner.suggestion is None and loser.suggestion is not None:
        winner = replace(winner, suggestion=loser.suggestion)
    return winner


def _same_group(a: DefectType, b: DefectType) -> bool:
    return any(a in group and b in group for group in taxonomy.MERGEABLE_GROUPS)


def _hits_overlap(x: _Hit, y: _Hit) -> bool:
    return x.fragment_index == y.fragment_index and x.start < y.end and y.start < x.end


def _has_overlap(hits: tuple[_Hit, ...]) -> bool:
    """Две цитаты противоречия указывают на одно и то же место документа."""
    return any(
        x.fragment_index == y.fragment_index and x.start < y.end and y.start < x.end
        for i, x in enumerate(hits)
        for y in hits[i + 1 :]
    )


def _overlap_ratio(x: _Hit, y: _Hit) -> float:
    """Доля перекрытия от длины более короткой из двух цитат."""
    if x.fragment_index != y.fragment_index:
        return 0.0
    common = min(x.end, y.end) - max(x.start, y.start)
    return max(common, 0) / min(x.end - x.start, y.end - y.start)


def _mergeable(a: _Cand, b: _Cand) -> bool:
    """Одна группа типов, общие id (или оба без id), цитаты перекрыты не меньше чем наполовину.

    Правило и модель на одном месте — одна находка, даже если id требований у них разные:
    правило уровня набора может не называть требований вовсе."""
    ids_ok = (not a.ids and not b.ids) or bool(set(a.ids) & set(b.ids))
    if a.raw.source is not b.raw.source:
        # Правило вливается в замечание модели того же типа либо в то, что процитировало
        # все его места: иначе цитаты правила остались бы в отчёте без объяснения.
        rule, model = (a, b) if a.raw.source is FindingSource.RULE else (b, a)
        ids_ok = rule.raw.defect_type is model.raw.defect_type or all(
            any(_overlap_ratio(x, y) >= MERGE_OVERLAP for y in model.hits) for x in rule.hits
        )
    return (
        ids_ok
        and _same_group(a.raw.defect_type, b.raw.defect_type)
        and any(_overlap_ratio(x, y) >= MERGE_OVERLAP for x in a.hits for y in b.hits)
    )


# Точка после цифры («204. Done») концом предложения не считается.
_SENTENCE_END = re.compile(r"(?<=[^\d\s][.!?])\s+(?=[A-ZА-ЯЁ])")


def with_rule_fact(text: str, rule_text: str) -> str:
    """Дописать к пояснению модели первое предложение пояснения правила."""
    fact = _SENTENCE_END.split(rule_text, maxsplit=1)[0]
    return text if fact in text else f"{text.rstrip()} {fact}"


def _combine(a: _Cand, b: _Cand) -> _Cand:
    """Тип и серьёзность берём у самого серьёзного (потолок типа не превышается)."""
    types = list(taxonomy.TAXONOMY)

    def order(c: _Cand) -> tuple:
        return (
            _rank(c.severity),
            -len(c.hits),
            c.raw.source is FindingSource.RULE,
            types.index(c.raw.defect_type),
            -len(c.raw.explanation),
        )

    win, lose = (a, b) if order(a) <= order(b) else (b, a)
    rule, model = (win, lose) if win.raw.source is FindingSource.RULE else (lose, win)
    if rule.raw.source is not model.raw.source:
        # Текст и вопрос заказчику — у модели; тип и серьёзность берём у победителя,
        # а факт правила другого типа дописываем, чтобы он не пропал при склейке.
        text = model.raw.explanation
        if rule.raw.defect_type is not model.raw.defect_type:
            text = with_rule_fact(text, rule.raw.explanation)
        kept = model.raw.model_copy(
            update={"defect_type": win.raw.defect_type, "explanation": text}
        )
        win, lose = replace(model, raw=kept, severity=win.severity), rule
    elif win.raw.defect_type is not lose.raw.defect_type:
        # Два замечания разного типа на одном месте — два дефекта: суть второго не теряем.
        text = with_rule_fact(win.raw.explanation, lose.raw.explanation)
        win = replace(win, raw=win.raw.model_copy(update={"explanation": text}))
    extra = tuple(h for h in lose.hits if not any(_hits_overlap(h, w) for w in win.hits))
    return replace(
        win,
        ids=tuple(sorted({*win.ids, *lose.ids})),
        hits=win.hits + tuple(dict.fromkeys(extra)),
        suggestion=win.suggestion or lose.suggestion,
    )


def _merge_overlapping(cands: list[_Cand]) -> list[_Cand]:
    """Склейка по месту: перекрывающиеся цитаты, типы одной группы."""
    result = list(cands)
    changed = True
    while changed:
        changed = False
        for i, a in enumerate(result):
            j = next((k for k in range(i + 1, len(result)) if _mergeable(a, result[k])), None)
            if j is None:
                continue
            result[i] = _combine(a, result.pop(j))
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
