"""Разбор PDF с текстовым слоем на фрагменты (FR-2, FR-3, FR-20).

Решения:
- Фрагмент — абзац (подряд идущие строки одного блока и одного стиля), пункт
  списка (строка с маркером начинает новый пункт) или строка таблицы.
- Заголовок — строка крупнее основного текста или короткая жирная строка без
  точки в конце; уровень — по рангу размера шрифта. Не уверены — это абзац.
- `line_start/line_end` — номера строк на странице в порядке чтения; строка
  таблицы занимает одну строку.
- Строки таблиц берутся из `find_tables()` (только таблицы с рамками), включая
  заголовок: текст — ячейки через « | » с обрамляющими «|».
- Повторяющиеся колонтитулы и номера страниц у краёв страницы отбрасываются.
- Страница без текста попадает в `unread_pages`; если текста нет нигде — скан.
"""

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import pymupdf

from reqcheck.errors import InputError
from reqcheck.models import Fragment, FragmentKind, Location

_MARKER = re.compile(r"^(?:[•●▪◦‣\-–—*·]|\d{1,3}[.)]|[a-zA-Zа-яА-Я][.)])\s+\S")
_WS = re.compile(r"\s+")
_EDGE = 0.08
_HEADING_RATIO = 1.15
_MAX_BOLD_HEADING = 100
_MIN_REPEAT_PAGES = 3


@dataclass(frozen=True)
class _Line:
    text: str
    size: float
    bold: bool
    block: int
    y0: float
    edge: bool


@dataclass(frozen=True)
class _Piece:
    kind: FragmentKind
    text: str
    y0: float
    lines: int
    heading_key: tuple[int, bool] | None = None


def _norm(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _page_tables(page: pymupdf.Page) -> list[tuple[pymupdf.Rect, list[list[str]], list[float]]]:
    found = []
    try:
        tables = page.find_tables().tables
    except Exception:  # noqa: BLE001 — find_tables нестабилен; текст останется абзацами
        return []
    for table in tables:
        if table.row_count < 2 or table.col_count < 2:
            continue
        rows = [[_norm(cell or "") for cell in row] for row in table.extract()]
        keep = [
            i
            for i in range(max(map(len, rows), default=0))
            if any(i < len(r) and r[i] for r in rows)
        ]
        rows = [[r[i] if i < len(r) else "" for i in keep] for r in rows]
        ys = [row.bbox[1] for row in table.rows]
        if len(ys) == len(rows):
            found.append((pymupdf.Rect(table.bbox), rows, ys))
    return found


def _page_lines(page: pymupdf.Page, rects: list[pymupdf.Rect]) -> list[_Line]:
    height = page.rect.height
    lines: list[_Line] = []
    for block in page.get_text("dict", sort=True)["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            spans = [s for s in line["spans"] if s["text"].strip()]
            text = _norm("".join(s["text"] for s in spans))
            if not text:
                continue
            x0, y0, x1, y1 = line["bbox"]
            centre = pymupdf.Point((x0 + x1) / 2, (y0 + y1) / 2)
            if any(rect.contains(centre) for rect in rects):
                continue
            bold = all(s["flags"] & 16 or "bold" in s["font"].lower() for s in spans)
            edge = y0 < height * _EDGE or y1 > height * (1 - _EDGE)
            size = max(s["size"] for s in spans)
            lines.append(_Line(text, size, bold, block["number"], y0, edge))
    return lines


def _repeated_edges(pages: list[list[_Line]]) -> set[str]:
    counter: Counter[str] = Counter()
    for lines in pages:
        for key in {re.sub(r"\d+", "#", ln.text) for ln in lines if ln.edge}:
            counter[key] += 1
    return {key for key, count in counter.items() if count >= _MIN_REPEAT_PAGES}


def _drop_furniture(lines: list[_Line], repeated: set[str]) -> list[_Line]:
    return [
        ln
        for ln in lines
        if not (ln.edge and (ln.text.isdigit() or re.sub(r"\d+", "#", ln.text) in repeated))
    ]


def _body_size(pages: list[list[_Line]]) -> float:
    weight: Counter[float] = Counter()
    for lines in pages:
        for ln in lines:
            weight[round(ln.size, 1)] += len(ln.text)
    return weight.most_common(1)[0][0] if weight else 0.0


def _heading_key(line: _Line, body: float, body_bold: bool) -> tuple[int, bool] | None:
    if _MARKER.match(line.text):
        return None
    if line.size >= body * _HEADING_RATIO:
        return (round(line.size), line.bold)
    short = len(line.text) <= _MAX_BOLD_HEADING and not line.text.endswith((".", ";", ":", ","))
    if line.bold and not body_bold and short:
        return (round(body), True)
    return None


def _pieces(lines: list[_Line], body: float, body_bold: bool) -> list[_Piece]:
    pieces: list[_Piece] = []
    current: list[_Line] = []
    current_key: tuple[int, bool] | None = None
    current_kind = FragmentKind.PARAGRAPH

    def flush() -> None:
        if current:
            text = "\n".join(ln.text for ln in current)
            pieces.append(_Piece(current_kind, text, current[0].y0, len(current), current_key))
            current.clear()

    for line in lines:
        key = _heading_key(line, body, body_bold)
        marker = key is None and bool(_MARKER.match(line.text))
        same_block = bool(current) and current[-1].block == line.block
        if not (same_block and key == current_key and not marker):
            flush()
            current_key = key
            if key is not None:
                current_kind = FragmentKind.HEADING
            else:
                current_kind = FragmentKind.LIST_ITEM if marker else FragmentKind.PARAGRAPH
        current.append(line)
    flush()
    return pieces


def _table_pieces(tables: list[tuple[pymupdf.Rect, list[list[str]], list[float]]]) -> list[_Piece]:
    pieces: list[_Piece] = []
    for _, rows, ys in tables:
        for row, y0 in zip(rows, ys, strict=True):
            if any(row):
                pieces.append(_Piece(FragmentKind.TABLE_ROW, "| " + " | ".join(row) + " |", y0, 1))
    return sorted(pieces, key=lambda p: p.y0)


def _merge(text: list[_Piece], table: list[_Piece]) -> list[_Piece]:
    merged: list[_Piece] = []
    pending = list(table)
    for piece in text:
        while pending and pending[0].y0 <= piece.y0:
            merged.append(pending.pop(0))
        merged.append(piece)
    return merged + pending


def _open(path_bytes: bytes) -> pymupdf.Document:
    try:
        doc = pymupdf.open(stream=path_bytes, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 — pymupdf бросает разные типы на битых файлах
        raise InputError(
            "PDF повреждён или усечён и не открывается. Сохраните документ заново "
            "или передайте Markdown."
        ) from exc
    if doc.needs_pass:
        raise InputError(
            "PDF зашифрован (требуется пароль). Снимите защиту и запустите проверку снова."
        )
    if doc.page_count == 0:
        raise InputError("В PDF нет ни одной страницы: файл повреждён.")
    return doc


def _page_content(
    doc: pymupdf.Document,
) -> tuple[list[list[_Line]], list[list[_Piece]], list[int]]:
    page_lines: list[list[_Line]] = []
    page_tables: list[list[_Piece]] = []
    unread: list[int] = []
    for number, page in enumerate(doc, start=1):
        try:
            tables = _page_tables(page)
            lines = _page_lines(page, [rect for rect, _, _ in tables])
            table_pieces = _table_pieces(tables)
        except Exception:  # noqa: BLE001 — страница не читается: фиксируем, идём дальше
            lines, table_pieces = [], []
        if not lines and not table_pieces:
            unread.append(number)
        page_lines.append(lines)
        page_tables.append(table_pieces)
    return page_lines, page_tables, unread


def parse_pdf(data: bytes, next_id: Callable[[], str]) -> tuple[list[Fragment], tuple[int, ...]]:
    with _open(data) as doc:
        page_lines, page_tables, unread = _page_content(doc)
    if len(unread) == len(page_lines):
        raise InputError(
            "В PDF нет текстового слоя (похоже на скан). Нужен PDF с текстовым слоем или Markdown."
        )
    repeated = _repeated_edges(page_lines)
    page_lines = [_drop_furniture(lines, repeated) for lines in page_lines]
    body = _body_size(page_lines)
    all_lines = [ln for lines in page_lines for ln in lines]
    body_bold = bool(all_lines) and sum(ln.bold for ln in all_lines) > len(all_lines) / 2
    per_page = [
        _merge(_pieces(lines, body, body_bold), tables)
        for lines, tables in zip(page_lines, page_tables, strict=True)
    ]
    keys = sorted({p.heading_key for page in per_page for p in page if p.heading_key}, reverse=True)
    levels = {key: rank + 1 for rank, key in enumerate(keys)}
    return _fragments(per_page, levels, next_id), tuple(unread)


def _fragments(
    per_page: list[list[_Piece]], levels: dict[tuple[int, bool], int], next_id: Callable[[], str]
) -> list[Fragment]:
    fragments: list[Fragment] = []
    headings: list[tuple[int, str]] = []
    for page_number, pieces in enumerate(per_page, start=1):
        line = 1
        for piece in pieces:
            if piece.heading_key is not None:
                level = levels[piece.heading_key]
                while headings and headings[-1][0] >= level:
                    headings.pop()
                headings.append((level, piece.text))
            fragments.append(
                Fragment(
                    id=next_id(),
                    kind=piece.kind,
                    text=piece.text,
                    location=Location(
                        page=page_number, line_start=line, line_end=line + piece.lines - 1
                    ),
                    section=tuple(title for _, title in headings),
                )
            )
            line += piece.lines
    return fragments
