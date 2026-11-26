"""Разбор Markdown на фрагменты с номерами строк исходного файла (FR-2).

Решения:
- Текст абзаца и пункта списка — исходные строки через «\n» (концевые пробелы
  срезаны): номер строки цитаты = начало + число «\n» до неё.
- Каждая строка таблицы (заголовок, разделитель `|---|`, тело) — фрагмент
  TABLE_ROW с сырой исходной строкой.
- Заголовок входит в собственный `section`.
"""

import re
from collections.abc import Callable

from markdown_it import MarkdownIt
from markdown_it.token import Token

from reqcheck.models import Fragment, FragmentKind, Location

_WS = re.compile(r"\s+")
_MD = MarkdownIt("commonmark").enable("table")


def _collapse(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _location(token: Token) -> Location:
    start, end = token.map or (0, 1)
    return Location(line_start=start + 1, line_end=max(start + 1, end))


def _raw_lines(lines: list[str], token: Token) -> str:
    """Исходные строки блока через \n, концевые пробелы срезаны."""
    start, end = token.map or (0, 1)
    return "\n".join(line.rstrip() for line in lines[start:end]).strip()


def parse_markdown(source: str, next_id: Callable[[], str]) -> list[Fragment]:
    tokens = _MD.parse(source)
    lines = source.splitlines()
    fragments: list[Fragment] = []
    headings: list[tuple[int, str]] = []
    containers: list[str] = []

    def emit(kind: FragmentKind, text: str, token: Token) -> None:
        if text:
            section = tuple(title for _, title in headings)
            fragments.append(
                Fragment(
                    id=next_id(), kind=kind, text=text, location=_location(token), section=section
                )
            )

    index = 0
    while index < len(tokens):
        token = tokens[index]
        kind = token.type
        if kind == "list_item_open":
            containers.append("li")
        elif kind == "blockquote_open":
            containers.append("bq")
        elif kind in ("list_item_close", "blockquote_close"):
            containers.pop()
        elif kind == "heading_open":
            title = _collapse(tokens[index + 1].content)
            level = int(token.tag[1:])
            while headings and headings[-1][0] >= level:
                headings.pop()
            if title:
                headings.append((level, title))
            emit(FragmentKind.HEADING, title, token)
        elif kind == "paragraph_open":
            in_item = bool(containers) and containers[-1] == "li"
            para = FragmentKind.LIST_ITEM if in_item else FragmentKind.PARAGRAPH
            emit(para, _raw_lines(lines, token), token)
        elif kind in ("fence", "code_block"):
            emit(FragmentKind.CODE, token.content.strip("\n").rstrip(), token)
        elif kind == "table_open":
            start, end = token.map or (0, 0)
            for number in range(start, end):
                row = lines[number].strip()
                where = Token("tr_open", "tr", 1)
                where.map = [number, number + 1]
                emit(FragmentKind.TABLE_ROW, row, where)
            while tokens[index].type != "table_close":
                index += 1
        index += 1
    return fragments
