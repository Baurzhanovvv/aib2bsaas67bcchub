"""Чтение DOCX (FR-20): документ переводится в Markdown и идёт через тот же разбор.

Заголовки берутся по стилю (Heading N / Заголовок N), списки — по стилю списка,
таблицы — построчно. Читается текст правок в режиме рецензирования (вставки), полей
шаблона и вложенных таблиц; удалённый рецензентом текст и колонтитулы не читаются.
Номер строки в отчёте — номер строки текстового представления: у DOCX нет ни строк,
ни надёжных страниц, поэтому место ищут по цитате и разделу.
"""

import io
import re
import zipfile
from collections.abc import Iterator

from docx import Document as open_docx
from docx.opc.exceptions import PackageNotFoundError
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from reqcheck.errors import InputError

_HEADING = re.compile(r"(?:heading|заголовок)\s*(\d)", re.IGNORECASE)
_LIST = re.compile(r"list|список", re.IGNORECASE)
_WS = re.compile(r"\s+")
# Начало абзаца, которое Markdown принял бы за разметку: заголовок, список, цитата, черта.
_MD_LEAD = re.compile(r"^(?:#{1,6}(?:\s|$)|[-+*>](?:\s|$)|[-=_*]{3,}\s*$|`{3}|~{3})")
MAX_HEADING_LEVEL = 6
MAX_UNPACKED_BYTES = 40 * 1024 * 1024
_SPACE_TAGS = {qn("w:tab"), qn("w:br"), qn("w:cr")}
_P, _TBL, _SDT, _SDT_CONTENT, _T = (
    qn("w:p"),
    qn("w:tbl"),
    qn("w:sdt"),
    qn("w:sdtContent"),
    qn("w:t"),
)


def _flat(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _text(element) -> str:
    """Весь видимый текст элемента, включая вставки рецензента и поля шаблона."""
    parts: list[str] = []
    for node in element.iter():
        if node.tag == _T:
            parts.append(node.text or "")
        elif node.tag in _SPACE_TAGS:
            parts.append(" ")
    return _flat("".join(parts))


def _blocks(parent, owner) -> Iterator[Paragraph | Table]:
    """Абзацы и таблицы по порядку; блочные поля шаблона (w:sdt) раскрываются."""
    for child in parent.iterchildren():
        if child.tag == _P:
            yield Paragraph(child, owner)
        elif child.tag == _TBL:
            yield Table(child, owner)
        elif child.tag == _SDT:
            for content in child.iterchildren(_SDT_CONTENT):
                yield from _blocks(content, owner)


def _paragraph(item: Paragraph) -> list[str]:
    text = _text(item._p)
    if not text:
        return []
    style = item.style.name if item.style is not None and item.style.name else ""
    heading = _HEADING.search(style)
    if heading or style.lower() == "title":
        level = min(int(heading.group(1)), MAX_HEADING_LEVEL) if heading else 1
        return [f"{'#' * max(level, 1)} {text}", ""]
    if _LIST.search(style):
        return [f"- {text}"]
    return [f"\\{text}" if _MD_LEAD.match(text) else text, ""]


def _cell_text(cell) -> str:
    """Текст ячейки вместе с вложенными таблицами (их ячейки через «; »)."""
    parts: list[str] = []
    for block in _blocks(cell._tc, cell):
        if isinstance(block, Paragraph):
            parts.append(_text(block._p))
        else:
            parts += [_cell_text(inner) for row in block.rows for inner in row.cells]
    return "; ".join(p for p in parts if p).replace("|", "\\|")


def _table(item: Table) -> list[str]:
    rows = []
    for row in item.rows:
        cells: list[str] = []
        seen: set[int] = set()
        for cell in row.cells:
            # Объединённая ячейка приходит несколько раз одним и тем же объектом XML.
            if id(cell._tc) in seen:
                continue
            seen.add(id(cell._tc))
            cells.append(_cell_text(cell))
        if any(cells):
            rows.append(cells)
    if not rows:
        return []
    width = max(len(r) for r in rows)
    lines = ["| " + " | ".join(r + [""] * (width - len(r))) + " |" for r in rows]
    lines.insert(1, "|" + " --- |" * width)
    return ["", *lines, ""]


def _check_size(data: bytes, name: str) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            unpacked = sum(info.file_size for info in archive.infolist())
    except zipfile.BadZipFile:
        return  # причину назовёт открытие документа ниже
    if unpacked > MAX_UNPACKED_BYTES:
        raise InputError(
            f"Файл «{name}» слишком велик для проверки: {unpacked // (1024 * 1024)} МБ после "
            f"распаковки при пределе {MAX_UNPACKED_BYTES // (1024 * 1024)} МБ. "
            "Разделите документ на части."
        )


def docx_to_markdown(data: bytes, name: str) -> str:
    """Текст DOCX в Markdown. Битый, защищённый или слишком большой файл — InputError."""
    _check_size(data, name)
    try:
        document = open_docx(io.BytesIO(data))
    except (PackageNotFoundError, zipfile.BadZipFile, KeyError, ValueError) as exc:
        raise InputError(
            f"Файл «{name}» не читается как DOCX (повреждён или защищён паролем). "
            "Сохраните документ заново или передайте PDF либо Markdown."
        ) from exc
    lines: list[str] = []
    for item in _blocks(document.element.body, document):
        block = _paragraph(item) if isinstance(item, Paragraph) else _table(item)
        if not block:
            continue
        is_list = block[0].startswith("- ")
        was_list = bool(lines) and lines[-1].startswith("- ")
        if lines and lines[-1] != "" and is_list != was_list:
            lines.append("")
        lines += block
    return "\n".join(lines).strip() + "\n"
