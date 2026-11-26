"""Разбор таблиц Markdown для правил: заголовок, строки данных, ячейки."""

from re import Pattern

from reqcheck.models import Fragment, FragmentKind
from reqcheck.stages import rules_lexicon as lex

_CELL_MARKUP = " \t*_`"

Table = tuple[list[str], list[Fragment]]
"""Ячейки заголовка и строки данных одной таблицы."""


def cells(row: str) -> list[str]:
    return [c.strip(_CELL_MARKUP) for c in lex.CELL_SPLIT.split(row.strip().strip("|"))]


def tables(fragments: tuple[Fragment, ...]) -> list[Table]:
    """Таблицы Markdown: заголовок — строка перед разделителем `|---|`, дальше строки подряд."""
    rows = [f for f in fragments if f.kind is FragmentKind.TABLE_ROW]
    found: list[Table] = []
    open_table = False
    for index, row in enumerate(rows):
        adjacent = index > 0 and (
            row.location.page == rows[index - 1].location.page
            and row.location.line_start == rows[index - 1].location.line_end + 1
        )
        if lex.TABLE_SEPARATOR.match(row.text.strip()):
            continue
        following = rows[index + 1].text.strip() if index + 1 < len(rows) else ""
        if lex.TABLE_SEPARATOR.match(following):
            found.append((cells(row.text), []))
            open_table = True
        elif open_table and adjacent:
            found[-1][1].append(row)
        else:
            open_table = False
    return found


def column(header: list[str], pattern: Pattern[str]) -> int | None:
    return next((i for i, cell in enumerate(header) if pattern.fullmatch(cell)), None)
