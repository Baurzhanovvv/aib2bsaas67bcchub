"""Нормализация текста для поиска цитат с картой индексов в исходный текст (FR-7)."""

import re
import unicodedata
from dataclasses import dataclass

_QUOTE_CHARS = "«»„“”‟‹›〝〞\"'‘’‚‛"
_DASH_CHARS = "‐‑‒–—―−⁃﹘﹣－-"
_INVISIBLE = "­​‌‍⁠﻿"
# Форматные и управляющие символы (метки направления письма, остатки PDF): в тексте
# не видны, но ломают дословное совпадение цитаты и годятся для скрытой вставки.
_CONTROL = ("Cf", "Cc")
# Разметка выделения Markdown: `*` и обратные кавычки убираем всегда, `_` только серией `__`
# (одиночный `_` бывает внутри идентификаторов вроде snake_case).
_MARKUP = "*`"
_LINE_BREAKS = "\n\r\x0b\x0c  \x85"

# --- Нормализация с картой индексов ---------------------------------------------


@dataclass(frozen=True)
class Norm:
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
            if not char.isspace() and unicodedata.category(char) in _CONTROL:
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


def normalize(text: str) -> Norm:
    tokens = _tokens(text)
    kept: list[list] = []
    i = 0
    while i < len(tokens):
        if _is_wrapped_hyphen(tokens, i):
            i += 2
            continue
        kept.append(tokens[i])
        i += 1
    return Norm(
        "".join(t[0] for t in kept),
        tuple(t[1] for t in kept),
        tuple(t[2] for t in kept),
    )


def norm_text(text: str) -> str:
    return normalize(text).text.strip()


_HEADING_MARK = re.compile(r"(?m)^[ \t]*#{1,6}[ \t]+")
_SEGMENT_SPLIT = re.compile(r"\s*[:|;]\s*|\s+-\s+")
MIN_SEGMENT = 3
MIN_SEGMENTS_TOTAL = 8


def strip_heading_marks(quote: str) -> str:
    """Срезать `# ` в начале каждой строки цитаты: reader отдаёт заголовок без решёток."""
    return _HEADING_MARK.sub("", quote)


def table_segments(needle: str) -> list[str]:
    """Цитата «ключ: значение» как сегменты по `:` `|` `;` ` — `; пусто, если мало смысла."""
    parts = [p.strip() for p in _SEGMENT_SPLIT.split(needle)]
    segments = [p for p in parts if len(p) >= MIN_SEGMENT]
    if len(segments) < 2 or sum(len(p) for p in segments) < MIN_SEGMENTS_TOTAL:
        return []
    return segments


_CELL = re.compile(r"[^|;]+")


def _cells(text: str) -> list[tuple[int, int]]:
    """Границы непустых ячеек строки таблицы (разделители `|` и `;`) без крайних пробелов."""
    cells = []
    for m in _CELL.finditer(text):
        raw = m.group()
        start = m.start() + len(raw) - len(raw.lstrip())
        end = m.end() - (len(raw) - len(raw.rstrip()))
        if end > start:
            cells.append((start, end))
    return cells


def _matches_cell(cell: str, segment: str) -> bool:
    """Сегмент равен ячейке или её началу по границе слова («Лимит 10» не начало «Лимит 1000»)."""
    if cell == segment:
        return True
    return cell.startswith(segment) and not cell[len(segment)].isalnum()


def match_adjacent_cells(text: str, segments: list[str]) -> tuple[int, int] | None:
    """Сегменты в соседних ячейках по порядку: границы от начала первой до конца последней."""
    cells = _cells(text)
    for first in range(len(cells) - len(segments) + 1):
        window = cells[first : first + len(segments)]
        if all(_matches_cell(text[a:b], seg) for (a, b), seg in zip(window, segments, strict=True)):
            return window[0][0], window[-1][1]
    return None
