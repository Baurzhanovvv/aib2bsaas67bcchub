"""Правила без модели (FR-14, профиль домена — FR-19)."""

import re
from collections.abc import Iterable

from reqcheck.models import (
    MIN_QUOTE_LENGTH,
    DefectType,
    Document,
    FindingSource,
    FragmentKind,
    Impact,
    RawFinding,
    Requirement,
    Scope,
)
from reqcheck.stages import rules_api
from reqcheck.stages import rules_lexicon as lex

_EDGE_MARKUP = " \t*_`|"
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ«\"(\d])|\n[ \t]*\n")
_REF_ID = re.compile(r"(?<![\w-])([A-ZА-ЯЁ]{1,6})-(\d+(?:\.\d+)*)(?![\w-])")
_HOMOGLYPHS = str.maketrans("АВЕКМНОРСТХ", "ABEKMHOPCTX")


# --- Цитаты -------------------------------------------------------------------


def _line_bounds(text: str, pos: int) -> tuple[int, int]:
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    return start, len(text) if end == -1 else end


def _extend_left(text: str, start: int, line_start: int) -> int:
    head = text[line_start:start].rstrip()
    if not head:
        return start
    cut = max(head.rfind(" "), head.rfind("\t")) + 1
    return line_start + cut


def _extend_right(text: str, end: int, line_end: int) -> int:
    tail = text[end:line_end]
    rest = tail.lstrip()
    if not rest:
        return end
    gap = len(tail) - len(rest)
    word = re.match(r"\S+", rest)
    return end + gap + (word.end() if word else 0)


def _grow(text: str, s: int, e: int, lo: int, hi: int) -> tuple[int, int]:
    while len(text[s:e].strip(_EDGE_MARKUP)) < MIN_QUOTE_LENGTH or text.count(text[s:e]) > 1:
        ns = _extend_left(text, s, lo)
        if ns != s:
            s = ns
            continue
        ne = _extend_right(text, e, hi)
        if ne == e:
            break
        e = ne
    return s, e


def make_quote(text: str, start: int, end: int) -> str:
    """Дословная подстрока вокруг [start, end): ≥ 8 символов и без повторов в тексте.

    Сначала расширяется в пределах строки; если не хватило (маркер в коротком
    хвосте строки), — через перевод строки.
    """
    line_start, line_end = _line_bounds(text, start)
    s, e = _grow(text, start, end, line_start, line_end)
    if len(text[s:e].strip(_EDGE_MARKUP)) < MIN_QUOTE_LENGTH or text.count(text[s:e]) > 1:
        s, e = _grow(text, s, e, 0, len(text))
    quote = text[s:e].strip(_EDGE_MARKUP)
    return quote if len(quote) >= MIN_QUOTE_LENGTH else text[s:e].strip()


def _quotes(text: str, spans: Iterable[tuple[int, int]]) -> tuple[str, ...]:
    result: list[str] = []
    for start, end in sorted(spans):
        quote = make_quote(text, start, end)
        if len(quote) >= MIN_QUOTE_LENGTH and quote not in result:
            result.append(quote)
    return tuple(result)


def _finding(
    req: Requirement,
    rule_id: str,
    defect: DefectType,
    quotes: tuple[str, ...],
    explanation: str,
    suggestion: str,
    impact: Impact,
    scope: Scope = Scope.REQUIREMENT,
) -> RawFinding | None:
    if not quotes:
        return None
    return RawFinding(
        defect_type=defect,
        requirement_ids=(req.id,),
        quotes=quotes,
        explanation=explanation,
        suggestion=suggestion,
        scope=scope,
        impact=impact,
        source=FindingSource.RULE,
        rule_id=rule_id,
    )


def _sentence_bounds(text: str, pos: int) -> tuple[int, int]:
    start = 0
    for sep in _SENTENCE_END.finditer(text):
        if sep.end() <= pos:
            start = sep.end()
        elif sep.start() >= pos:
            return start, sep.start()
    return start, len(text)


# --- RULE-VAGUE ---------------------------------------------------------------


def _vague_spans(text: str) -> tuple[list[tuple[int, int]], bool]:
    spans: list[tuple[int, int]] = []
    measurable = False
    hits = [(m, q) for rx, q in lex.VAGUE_PATTERNS for m in rx.finditer(text)]
    hits += [(m, True) for m in lex.VAGUE_WITHOUT_NUMBER.finditer(text)]
    for match, quality in hits:
        s, e = _sentence_bounds(text, match.start())
        sentence = text[s:e]
        if lex.NUMBER_WITH_UNIT.search(sentence):
            continue
        if match.re is lex.VAGUE_WITHOUT_NUMBER and lex.ANY_DIGIT.search(sentence):
            continue
        spans.append(match.span())
        measurable = measurable or quality
    return spans, measurable


def rule_vague(req: Requirement) -> RawFinding | None:
    spans, measurable = _vague_spans(req.quote)
    quotes = _quotes(req.quote, spans)
    words = ", ".join(f"«{req.quote[s:e]}»" for s, e in sorted(spans))
    defect = DefectType.UNVERIFIABLE if measurable else DefectType.AMBIGUOUS
    return _finding(
        req,
        "RULE-VAGUE",
        defect,
        quotes,
        f"Расплывчатые формулировки без измеримого значения: {words}. Такое требование "
        "понимают по-разному и не могут проверить тестом (ISO/IEC/IEEE 29148, 5.2.5: "
        "однозначность и верифицируемость).",
        "Заменить оценочные слова числом с единицей измерения или перечислить все случаи явно.",
        # Признак: оценочное слово про измеримое свойство (время, объём, частота) — MEDIUM.
        Impact.MEDIUM if measurable else Impact.LOW,
    )


# --- RULE-TBD -----------------------------------------------------------------


def _empty_cell_quote(req: Requirement, is_row: bool) -> str | None:
    text = req.quote.strip()
    if not (is_row or text.startswith("|")) or text.count("|") < 3:
        return None
    cells = [c.strip() for c in text.strip("|").split("|")]
    if len(cells) < 3 or all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
        return None
    if any(not c for c in cells[:-1]) and len(text) >= MIN_QUOTE_LENGTH:
        return text
    return None


def rule_tbd(req: Requirement, is_row: bool) -> RawFinding | None:
    spans = [m.span() for rx in lex.TBD_PATTERNS for m in rx.finditer(req.quote)]
    quotes = list(_quotes(req.quote, spans))
    empty = _empty_cell_quote(req, is_row)
    if empty and empty not in quotes:
        quotes.append(empty)
    return _finding(
        req,
        "RULE-TBD",
        DefectType.INCOMPLETE,
        tuple(quotes),
        "Требование содержит заглушку или пропуск вместо значения (TBD, «уточняется», "
        "«???», пустая ячейка). Требование неполно: его нельзя реализовать и проверить "
        "(ISO/IEC/IEEE 29148, 5.2.5: полнота).",
        "Указать конкретное значение или вынести вопрос в список открытых с ответственным.",
        Impact.MEDIUM,
    )


# --- RULE-HTTP-* --------------------------------------------------------------


def rule_http_safe_method(req: Requirement) -> RawFinding | None:
    text = req.quote
    method = lex.HTTP_SAFE_METHOD.search(text)
    if not method:
        return None
    if not (lex.HTTP_PATH_AFTER_METHOD.search(text) or lex.HTTP_CONTEXT.search(text)):
        return None
    verb = next(
        (
            m
            for rx in (lex.MUTATION_RU, lex.MUTATION_EN, lex.MUTATION_IDENT)
            for m in rx.finditer(text)
        ),
        None,
    )
    if verb is None:
        return None
    return _finding(
        req,
        "RULE-HTTP-SAFE-METHOD",
        DefectType.INTERFACE_MISUSE,
        _quotes(text, [method.span(), verb.span()]),
        f"Метод {method.group(1)} безопасен и идемпотентен по определению HTTP (RFC 9110, 9.2.1), "
        "но операция описана как изменяющая состояние. Описание интерфейса противоречит "
        "себе и вводит в заблуждение разработчика и кэши "
        "(ISO/IEC/IEEE 29148, 5.2.5: корректность).",
        "Использовать POST, PUT, PATCH или DELETE для изменения состояния, "
        "а GET оставить для чтения.",
        Impact.HIGH,
    )


def rule_http_error_200(req: Requirement) -> RawFinding | None:
    text = req.quote
    status = lex.STATUS_200.search(text)
    if not status or not lex.HTTP_STATUS_CONTEXT.search(text):
        return None
    if lex.OTHER_ERROR_STATUS.search(text):
        return None
    error = lex.ERROR_MARKER.search(text)
    if not error:
        return None
    return _finding(
        req,
        "RULE-HTTP-ERROR-200",
        DefectType.INTERFACE_MISUSE,
        _quotes(text, [status.span(), error.span()]),
        "Ошибка передаётся в ответе с кодом 200, а код 200 означает успех. Клиент, "
        "ориентирующийся на код, примет сбой за успех (ISO/IEC/IEEE 29148, 5.2.5: "
        "корректность и однозначность).",
        "Возвращать для ошибки код 4xx или 5xx, соответствующий её причине.",
        Impact.HIGH,
    )


# --- RULE-UNDEFINED-REF -------------------------------------------------------


def _norm_id(prefix: str, number: str) -> str:
    head = prefix.translate(_HOMOGLYPHS).upper()
    if re.fullmatch(r"\d+", number):
        number = str(int(number))
    return f"{head}-{number}"


def _known_ids(requirements: tuple[Requirement, ...]) -> tuple[set[str], set[str]]:
    ids: set[str] = set()
    prefixes: set[str] = set()
    for req in requirements:
        found = _REF_ID.search(req.source_id or "")
        if found:
            ids.add(_norm_id(*found.groups()))
            prefixes.add(found.group(1).translate(_HOMOGLYPHS).upper())
    return ids, prefixes


def _is_own_label(text: str, match: re.Match[str]) -> bool:
    before = text[: match.start()].strip(" *_|#>(-")
    after = text[match.end() : match.end() + 1]
    return not before and after in (".", ":", ")", "")


def rule_undefined_ref(req: Requirement, ids: set[str], prefixes: set[str]) -> RawFinding | None:
    spans: list[tuple[int, int]] = []
    names: list[str] = []
    for match in _REF_ID.finditer(req.quote):
        prefix, number = match.groups()
        if prefix.translate(_HOMOGLYPHS).upper() not in prefixes:
            continue
        if _norm_id(prefix, number) in ids or _is_own_label(req.quote, match):
            continue
        spans.append(match.span())
        names.append(match.group(0))
    return _finding(
        req,
        "RULE-UNDEFINED-REF",
        DefectType.UNDEFINED_REFERENCE,
        _quotes(req.quote, spans),
        f"Ссылка на требование, которого нет в документе: {', '.join(dict.fromkeys(names))}. "
        "Требование нельзя проследить до источника (ISO/IEC/IEEE 29148, 5.2.6: "
        "прослеживаемость и согласованность набора).",
        "Исправить идентификатор ссылки или добавить требование, на которое она указывает.",
        Impact.MEDIUM,
        scope=Scope.REQUIREMENT_SET,
    )


# --- Точка входа --------------------------------------------------------------


def apply_rules(document: Document, requirements: tuple[Requirement, ...]) -> list[RawFinding]:
    """Найти дефекты общими правилами, не обращаясь к модели.

    Каждое замечание: source=RULE, непустой rule_id, цитата — дословный кусок
    документа. Правила не содержат строк из какого-либо конкретного ТЗ (FR-12).
    """
    kinds = {f.id: f.kind for f in document.fragments}
    ids, prefixes = _known_ids(requirements)
    findings: list[RawFinding | None] = []
    for req in requirements:
        kind = kinds.get(req.fragment_id)
        if kind is not FragmentKind.CODE:
            findings.append(rule_vague(req))
            findings.append(rule_tbd(req, kind is FragmentKind.TABLE_ROW))
        findings.append(rule_http_safe_method(req))
        findings.append(rule_http_error_200(req))
        if prefixes:
            findings.append(rule_undefined_ref(req, ids, prefixes))
    findings += rules_api.apply_api_rules(document, requirements)
    return [f for f in findings if f is not None]
