"""Правила контракта API без модели (FR-14): сверка описания интерфейса между фрагментами.

Правила из rules.py смотрят на одно требование. Эти сверяют несколько фрагментов: метод
и назначение операции, код ответа и тело под ним, перечень статусов и текст, таблицы
запроса и ответа, текст и машиночитаемую спецификацию. Строк из конкретного ТЗ здесь нет
(FR-12). Ложное срабатывание дороже пропуска: правило молчит, если признак не однозначен.
"""

from collections import defaultdict
from collections.abc import Iterable, Iterator
from itertools import combinations

from reqcheck.models import (
    MIN_QUOTE_LENGTH,
    DefectType,
    Document,
    FindingSource,
    Fragment,
    FragmentKind,
    Impact,
    RawFinding,
    Requirement,
    Scope,
)
from reqcheck.stages import rules_lexicon as lex
from reqcheck.stages import rules_tables as tbl

MIN_STATUS_LIST = 3
MIN_SPEC_LINES = 5
MAX_ROW_QUOTES = 4


class _Ctx:
    def __init__(self, document: Document, requirements: tuple[Requirement, ...]) -> None:
        self.fragments = document.fragments
        self._req_of = {r.fragment_id: r.id for r in requirements}
        self.tables = tbl.tables(document.fragments)
        self.explained_statuses: set[str] = set()
        """Коды, уже объяснённые правилом о подписи: отдельного замечания «вне перечня» не нужно."""

    def ids(self, *fragments: Fragment) -> list[str]:
        return [self._req_of[f.id] for f in fragments if f.id in self._req_of]


def _line(text: str, pos: int) -> str:
    start = text.rfind("\n", 0, pos) + 1
    end = text.find("\n", pos)
    return text[start : len(text) if end == -1 else end].strip()


def _finding(
    rule_id: str,
    defect: DefectType,
    quotes: Iterable[str],
    explanation: str,
    suggestion: str,
    impact: Impact,
    ids: Iterable[str] = (),
) -> RawFinding | None:
    kept = tuple(dict.fromkeys(q for q in quotes if len(q.strip()) >= MIN_QUOTE_LENGTH))
    if not kept:
        return None
    return RawFinding(
        defect_type=defect,
        requirement_ids=tuple(sorted(set(ids))),
        quotes=kept,
        explanation=explanation,
        suggestion=suggestion,
        scope=Scope.REQUIREMENT_SET,
        impact=impact,
        source=FindingSource.RULE,
        rule_id=rule_id,
    )


# --- RULE-API-METHOD-MUTATES ----------------------------------------------------


def _mutation(fragments: list[Fragment], skip: Fragment) -> tuple[Fragment, int, str] | None:
    """Назначение операции начинается со слова-действия либо действие стоит в имени пути."""
    for rx in (lex.API_PURPOSE_MUTATION, lex.API_PATH_MUTATION):
        for fragment in fragments:
            if fragment is skip or fragment.kind is FragmentKind.CODE:
                continue
            match = rx.search(fragment.text)
            if match and not lex.NEGATION_BEFORE.search(fragment.text[: match.start(1)]):
                return fragment, match.start(1), match.group(1)
    return None


def rule_method_mutates(ctx: _Ctx) -> Iterator[RawFinding | None]:
    by_section: dict[tuple[str, ...], list[Fragment]] = defaultdict(list)
    for fragment in ctx.fragments:
        if fragment.kind is not FragmentKind.HEADING:
            by_section[fragment.section].append(fragment)
    for fragments in by_section.values():
        declared = next(
            (
                (f, m)
                for f in fragments
                if f.kind is not FragmentKind.CODE
                for m in [lex.API_METHOD_DECL.search(f.text)]
                if m
            ),
            None,
        )
        evidence = _mutation(fragments, declared[0]) if declared else None
        if declared is None or evidence is None:
            continue
        (where, method), (other, pos, word) = declared, evidence
        yield _finding(
            "RULE-API-METHOD-MUTATES",
            DefectType.INTERFACE_MISUSE,
            (_line(where.text, method.start()), _line(other.text, pos)),
            f"Метод объявлен как {method.group(1)}, а операция в том же разделе описана как "
            f"изменяющая состояние («{word}»). {method.group(1)} по определению HTTP "
            "безопасен (RFC 9110, 9.2.1): кэши, прокси и повторы запроса исходят из того, "
            "что он ничего не меняет (ISO/IEC/IEEE 29148, 5.2.5: корректность).",
            "Вопрос заказчику: каким методом выполняется операция? Варианты: 1) POST, PUT "
            "или PATCH с параметрами в теле; 2) операция только читает — тогда исправить "
            "описание и имя метода.",
            Impact.HIGH,
            ctx.ids(where, other),
        )


# --- RULE-API-ERROR-UNDER-SUCCESS -------------------------------------------------


def _success_title(title: str) -> bool:
    if lex.OTHER_ERROR_STATUS.search(title):
        return False
    if lex.SUCCESS_STATUS_HEAD.match(title):
        return True
    return bool(lex.SUCCESS_STATUS.search(title) and lex.HTTP_STRICT.search(title))


def rule_error_under_success(ctx: _Ctx) -> Iterator[RawFinding | None]:
    seen: set[tuple[str, ...]] = set()
    for fragment in ctx.fragments:
        if fragment.kind is not FragmentKind.CODE or not fragment.section:
            continue
        title = fragment.section[-1]
        body = lex.ERROR_BODY.search(fragment.text)
        if body is None or fragment.section in seen or not _success_title(title):
            continue
        seen.add(fragment.section)
        # Цитата — только тело: заголовок раздела занят другим дефектом (подпись кода).
        quote = _line(fragment.text, body.start())
        yield _finding(
            "RULE-API-ERROR-UNDER-SUCCESS",
            DefectType.INTERFACE_MISUSE,
            (quote if len(quote) >= MIN_QUOTE_LENGTH else title,),
            f"Пример ответа с ошибкой приведён в разделе успешного кода («{title}»). Клиент, "
            "который смотрит на код ответа, примет сбой за успех (ISO/IEC/IEEE 29148, 5.2.5: "
            "корректность и однозначность).",
            "Вопрос заказчику: каким кодом отвечать на ошибку? Варианты: 1) кодом 4xx по "
            "причине ошибки; 2) кодом успеха с признаком ошибки в теле — тогда описать это "
            "правило явно и убрать расхождение с перечнем статусов.",
            Impact.HIGH,
            ctx.ids(fragment),
        )


# --- RULE-API-STATUS-PHRASE -------------------------------------------------------


def rule_status_phrase(ctx: _Ctx) -> Iterator[RawFinding | None]:
    listed = _listed_statuses(ctx)
    for fragment in ctx.fragments:
        if fragment.kind is FragmentKind.CODE:
            continue
        context = " ".join(fragment.section)
        for match in lex.STATUS_WITH_PHRASE.finditer(fragment.text):
            code, phrase = int(match.group(1)), match.group(2)
            expected = lex.REASON_PHRASES[phrase.lower()]
            line = _line(fragment.text, match.start())
            if code == expected or not lex.HTTP_STRICT.search(f"{line} {context}"):
                continue
            # Тот же код вне перечня — та же путаница: одно замечание с обеими цитатами.
            also = _first_mention(ctx, str(expected)) if str(expected) not in listed else None
            ctx.explained_statuses.add(str(expected))
            yield _finding(
                "RULE-API-STATUS-PHRASE",
                DefectType.INTERFACE_MISUSE,
                (line, also) if also else (line,),
                f"Код ответа {code} подписан фразой «{phrase}», а это стандартная фраза кода "
                f"{expected} (RFC 9110). Неясно, какой код возвращает метод"
                + (
                    f"; код {expected} при этом упомянут в тексте, но в перечне статусов его нет"
                    if also
                    else ""
                )
                + " (ISO/IEC/IEEE 29148, 5.2.5: однозначность).",
                f"Вопрос заказчику: какой код возвращается? Варианты: 1) {code} — тогда "
                f"исправить подпись; 2) {expected} — тогда исправить код и описать его в "
                "перечне статусов.",
                Impact.MEDIUM,
                ctx.ids(fragment),
            )
            break


# --- RULE-API-STATUS-UNLISTED -----------------------------------------------------


def _listed_statuses(ctx: _Ctx) -> set[str]:
    """Коды из перечня: строки «код — описание», столбец кодов, ключи и строки HTTP в коде."""
    listed: set[str] = set()
    for fragment in ctx.fragments:
        if fragment.kind is FragmentKind.CODE:
            listed.update(lex.STATUS_SPEC_KEY.findall(fragment.text))
            listed.update(lex.STATUS_HTTP_LINE.findall(fragment.text))
            continue
        for line in fragment.text.splitlines():
            head = lex.STATUS_LIST_HEAD.match(line)
            if head:
                listed.add(head.group(1))
    for header, rows in ctx.tables:
        column = tbl.column(header, lex.STATUS_COLUMN)
        for row in rows if column is not None else ():
            cells = tbl.cells(row.text)
            if column < len(cells) and lex.STATUS_CELL.fullmatch(cells[column]):
                listed.add(cells[column])
    return listed


def _first_mention(ctx: _Ctx, code: str) -> str | None:
    for fragment in ctx.fragments:
        if fragment.kind not in (FragmentKind.PARAGRAPH, FragmentKind.LIST_ITEM):
            continue
        for line in fragment.text.splitlines():
            if code in _mentioned(line):
                return line.strip()
    return None


def _mentioned(line: str) -> Iterator[str]:
    for match in lex.STATUS_MENTION.finditer(line):
        yield match.group(1)
        following = lex.STATUS_NEXT.match(line, match.end())
        while following:
            yield following.group(1)
            following = lex.STATUS_NEXT.match(line, following.end())


def rule_status_unlisted(ctx: _Ctx) -> Iterator[RawFinding | None]:
    listed = _listed_statuses(ctx)
    if len(listed) < MIN_STATUS_LIST:
        return
    reported: set[str] = set(ctx.explained_statuses)
    for fragment in ctx.fragments:
        if fragment.kind not in (FragmentKind.PARAGRAPH, FragmentKind.LIST_ITEM):
            continue
        for line in fragment.text.splitlines():
            for code in _mentioned(line):
                if code in listed or code in reported:
                    continue
                reported.add(code)
                yield _finding(
                    "RULE-API-STATUS-UNLISTED",
                    DefectType.INTERFACE_MISUSE,
                    (line.strip(),),
                    f"Код ответа {code} упоминается в тексте, но в перечне статусов его нет "
                    f"(в перечне: {', '.join(sorted(listed))}). Условие, при котором он "
                    "возвращается, и его тело не описаны (ISO/IEC/IEEE 29148, 5.2.5: полнота "
                    "и согласованность).",
                    f"Вопрос заказчику: возвращает ли метод код {code}? Варианты: 1) да — "
                    "добавить его в перечень статусов с условием и примером ответа; 2) нет — "
                    "убрать упоминание.",
                    Impact.MEDIUM,
                    ctx.ids(fragment),
                )


# --- RULE-API-TYPE-MISMATCH -------------------------------------------------------


def _typed_fields(table: tbl.Table) -> dict[str, tuple[str, Fragment]]:
    header, rows = table
    name, kind = tbl.column(header, lex.NAME_COLUMN), tbl.column(header, lex.TYPE_COLUMN)
    fields: dict[str, tuple[str, Fragment]] = {}
    if name is None or kind is None:
        return fields
    for row in rows:
        cells = tbl.cells(row.text)
        word = lex.TYPE_WORD.match(cells[kind]) if max(name, kind) < len(cells) else None
        if word and cells[name]:
            declared = word.group(0).lower()
            fields.setdefault(cells[name], (lex.TYPE_SYNONYMS.get(declared, declared), row))
    return fields


def _request_and_response(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    """Таблицы лежат под общим разделом, одна в части запроса, другая в части ответа."""
    common = next((i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y), None)
    if not common:
        return False
    first, second = " ".join(a[common:]), " ".join(b[common:])

    def is_pair(request: str, response: str) -> bool:
        return bool(
            lex.REQUEST_PART.search(request)
            and lex.RESPONSE_PART.search(response)
            and not lex.RESPONSE_PART.search(request)
        )

    return is_pair(first, second) or is_pair(second, first)


def rule_type_mismatch(ctx: _Ctx) -> Iterator[RawFinding | None]:
    typed = [fields for fields in map(_typed_fields, ctx.tables) if fields]
    for first, second in combinations(typed, 2):
        a_row = next(iter(first.values()))[1]
        b_row = next(iter(second.values()))[1]
        if not _request_and_response(a_row.section, b_row.section):
            continue
        clashes = [n for n in first if n in second and first[n][0] != second[n][0]]
        if not clashes:
            continue
        rows = [row for n in clashes for row in (first[n][1], second[n][1])]
        listed = "; ".join(f"«{n}» — {first[n][0]} и {second[n][0]}" for n in clashes)
        yield _finding(
            "RULE-API-TYPE-MISMATCH",
            DefectType.INCONSISTENT_TERMS,
            [row.text for row in rows[:MAX_ROW_QUOTES]],
            f"Поле с одним именем описано в запросе и в ответе с разными типами: {listed}. "
            "Клиент и сервер разойдутся в формате значения (ISO/IEC/IEEE 29148, 5.2.6: "
            "согласованность).",
            "Вопрос заказчику: какой тип у поля? Варианты: 1) один тип в обеих таблицах; "
            "2) типы различаются намеренно — тогда описать правило преобразования.",
            Impact.MEDIUM,
            ctx.ids(*rows),
        )


# --- RULE-API-SPEC-* ----------------------------------------------------------------


def _first_quote(text: str) -> str:
    return next((ln.strip() for ln in text.splitlines() if len(ln.strip()) >= MIN_QUOTE_LENGTH), "")


def _bare(path: str) -> str:
    return lex.PATH_PARAM.sub("{}", path).rstrip("./")


def _in_spec(path: str, code: str, keys: set[str]) -> bool:
    """Путь есть в спецификации целиком либо его хвост — ключ `paths` (префикс в servers)."""
    return path in code or any(key and _bare(path).endswith(key) for key in keys)


def rule_spec(ctx: _Ctx) -> Iterator[RawFinding | None]:
    blocks = [f for f in ctx.fragments if f.kind is FragmentKind.CODE]
    specs = [f for f in blocks if lex.SPEC_MARK.search(f.text)]
    if not specs:
        return
    for block in specs:
        lines = [ln for ln in block.text.splitlines() if ln.strip()]
        flat = all(ln[:1] not in (" ", "\t") for ln in lines)
        if len(lines) >= MIN_SPEC_LINES and flat and lex.SPEC_PATHS_LINE.search(block.text):
            yield _finding(
                "RULE-API-SPEC-STRUCTURE",
                DefectType.INTERFACE_MISUSE,
                (_first_quote(block.text),),
                "В машиночитаемой спецификации нет отступов: вложенность ключей потеряна, "
                "документ не разбирается как описание API. По нему нельзя сгенерировать "
                "клиент или проверить контракт (ISO/IEC/IEEE 29148, 5.2.5: корректность).",
                "Привести спецификацию к валидному виду и проверить её линтером.",
                Impact.MEDIUM,
            )
    code = "\n".join(block.text for block in blocks)
    keys = {_bare(key) for key in lex.SPEC_PATH_KEY.findall(code)}
    reported: set[str] = set()
    for fragment in ctx.fragments:
        if fragment.kind is FragmentKind.CODE:
            continue
        for match in lex.PATH_DECL.finditer(fragment.text):
            path = match.group(1).rstrip("./")
            if len(path) < 2 or path in reported or _in_spec(path, code, keys):
                continue
            reported.add(path)
            yield _finding(
                "RULE-API-SPEC-PATH",
                DefectType.MISSING_REQUIREMENT,
                (_line(fragment.text, match.start()),),
                f"Путь {path} описан в тексте, но в машиночитаемой спецификации его нет. "
                "Клиент, собранный по спецификации, этого метода не получит "
                "(ISO/IEC/IEEE 29148, 5.2.6: полнота и согласованность набора).",
                "Добавить путь в спецификацию или убрать его описание из текста.",
                Impact.MEDIUM,
                ctx.ids(fragment),
            )


# --- Точка входа ------------------------------------------------------------------

_RULES = (
    rule_method_mutates,
    rule_error_under_success,
    rule_status_phrase,
    rule_status_unlisted,
    rule_type_mismatch,
    rule_spec,
)


def apply_api_rules(document: Document, requirements: tuple[Requirement, ...]) -> list[RawFinding]:
    """Найти расхождения в описании API без модели. Цитаты — дословные строки документа."""
    ctx = _Ctx(document, requirements)
    return [found for rule in _RULES for found in rule(ctx) if found is not None]
