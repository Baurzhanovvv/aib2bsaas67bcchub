"""Тесты этапа чтения документа: preflight, Markdown, PDF."""

import hashlib
from pathlib import Path

import pymupdf
import pytest

from reqcheck.errors import InputError
from reqcheck.models import DocumentFormat, FragmentKind
from reqcheck.stages.reader import read_document

MD = """# Склад

Поставщик передаёт накладную
на приёмку товара.

## Приёмка

- Кладовщик сканирует паллету.
- Система печатает этикетку.

| Поле | Значение |
|------|----------|
| Срок | 2 часа |
| Статус | Принято |

```
ship --all
```
"""


def write(tmp_path: Path, name: str, data: bytes | str) -> Path:
    path = tmp_path / name
    path.write_bytes(data if isinstance(data, bytes) else data.encode("utf-8"))
    return path


def make_pdf(pages: list[list[str]], path: Path) -> Path:
    doc = pymupdf.open()
    for lines in pages:
        page = doc.new_page()
        y = 80
        for text in lines:
            page.insert_text((60, y), text, fontsize=11, fontname="helv")
            y += 16
        if not lines:
            page.draw_rect(pymupdf.Rect(100, 100, 200, 200))
    doc.save(path)
    doc.close()
    return path


# --- preflight -------------------------------------------------------------------


def test_empty_file(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="пустой"):
        read_document(write(tmp_path, "a.md", b""))


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="не найден"):
        read_document(tmp_path / "нет.md")


@pytest.mark.parametrize(
    ("name", "data", "what"),
    [
        ("a.md", b"\x89PNG\r\n\x1a\n" + b"0" * 20, "PNG"),
        ("a.md", b"PK\x03\x04" + b"0" * 20, "ZIP"),
        ("a.md", b"%PDF-1.7\n" + b"0" * 20, "PDF"),
        ("a.pdf", b"\x89PNG\r\n\x1a\n" + b"0" * 20, "PNG"),
        ("a.pdf", b"# plain markdown, not a pdf", "не PDF"),
    ],
)
def test_signature_mismatch(tmp_path: Path, name: str, data: bytes, what: str) -> None:
    with pytest.raises(InputError, match=what):
        read_document(write(tmp_path, name, data))


def test_binary_garbage_under_md(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="бинарные"):
        read_document(write(tmp_path, "a.md", b"\x80\x81\x00\x00abc"))


def _docx(tmp_path: Path) -> Path:
    import docx

    document = docx.Document()
    document.add_heading("Склад", level=1)
    document.add_heading("Списание", level=2)
    document.add_paragraph("Система должна списывать позицию за 2 секунды.")
    document.add_paragraph("Первый пункт перечня", style="List Bullet")
    table = document.add_table(rows=2, cols=2)
    table.rows[0].cells[0].text, table.rows[0].cells[1].text = "Поле", "Тип"
    table.rows[1].cells[0].text, table.rows[1].cells[1].text = "weight", "number | null"
    path = tmp_path / "spec.docx"
    document.save(str(path))
    return path


def test_docx_is_read_through_markdown(tmp_path: Path) -> None:
    document = read_document(_docx(tmp_path))
    assert document.format is DocumentFormat.DOCX
    kinds = [f.kind for f in document.fragments]
    assert kinds.count(FragmentKind.HEADING) == 2
    assert FragmentKind.LIST_ITEM in kinds and FragmentKind.TABLE_ROW in kinds
    paragraph = next(f for f in document.fragments if f.kind is FragmentKind.PARAGRAPH)
    assert paragraph.text == "Система должна списывать позицию за 2 секунды."
    assert paragraph.section == ("Склад", "Списание")


def test_broken_docx_is_input_error(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="не читается как DOCX"):
        read_document(write(tmp_path, "a.docx", b"PK\x03\x04"))
    with pytest.raises(InputError, match="расширение .docx"):
        read_document(write(tmp_path, "b.docx", b"%PDF-1.7 not a docx"))


def test_unknown_extension(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="Неподдерживаемый"):
        read_document(write(tmp_path, "a.txt", "текст"))


def test_encrypted_pdf(tmp_path: Path) -> None:
    src = make_pdf([["Hello world"]], tmp_path / "src.pdf")
    doc = pymupdf.open(src)
    enc = tmp_path / "enc.pdf"
    doc.save(
        enc,
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner",
        user_pw="user",
    )
    doc.close()
    with pytest.raises(InputError, match="зашифрован"):
        read_document(enc)


def test_truncated_pdf(tmp_path: Path) -> None:
    src = make_pdf([["Hello world"]], tmp_path / "src.pdf")
    data = src.read_bytes()
    with pytest.raises(InputError):
        read_document(write(tmp_path, "cut.pdf", data[:40]))


def test_scan_pdf_without_text(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="текстовым слоем"):
        read_document(make_pdf([[], []], tmp_path / "scan.pdf"))


def test_whitespace_only_markdown(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="нет текста"):
        read_document(write(tmp_path, "a.md", "\n\n   \n"))


# --- Markdown --------------------------------------------------------------------


def test_markdown_lines_and_kinds(tmp_path: Path) -> None:
    path = write(tmp_path, "spec.md", MD)
    doc = read_document(path)
    assert doc.format is DocumentFormat.MARKDOWN
    assert doc.name == "spec.md"
    assert doc.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert [f.id for f in doc.fragments] == [f"F-{i:04d}" for i in range(1, len(doc.fragments) + 1)]
    by_text = {f.text: f for f in doc.fragments}

    heading = by_text["Склад"]
    assert heading.kind is FragmentKind.HEADING
    assert (heading.location.line_start, heading.location.line_end) == (1, 1)
    assert heading.section == ("Склад",)

    para = by_text["Поставщик передаёт накладную\nна приёмку товара."]
    assert para.kind is FragmentKind.PARAGRAPH
    assert (para.location.line_start, para.location.line_end) == (3, 4)
    assert para.section == ("Склад",)

    item = by_text["- Система печатает этикетку."]
    assert item.kind is FragmentKind.LIST_ITEM
    assert item.location.line_start == 9
    assert item.section == ("Склад", "Приёмка")

    code = by_text["ship --all"]
    assert code.kind is FragmentKind.CODE
    assert code.location.page is None


def test_markdown_table_rows_raw(tmp_path: Path) -> None:
    doc = read_document(write(tmp_path, "spec.md", MD))
    rows = [f for f in doc.fragments if f.kind is FragmentKind.TABLE_ROW]
    assert [r.text for r in rows] == [
        "| Поле | Значение |",
        "|------|----------|",
        "| Срок | 2 часа |",
        "| Статус | Принято |",
    ]
    assert [r.location.line_start for r in rows] == [11, 12, 13, 14]
    assert all(r.location.line_end == r.location.line_start for r in rows)
    raw = MD.splitlines()
    assert all(r.text == raw[r.location.line_start - 1] for r in rows)


def test_paragraph_keeps_newlines(tmp_path: Path) -> None:
    doc = read_document(write(tmp_path, "a.md", "# T\n\nстрока раз  \nстрока два\nстрока три\n"))
    para = doc.fragments[1]
    assert (para.location.line_start, para.location.line_end) == (3, 5)
    assert para.text.count("\n") == 2
    assert para.text == "строка раз\nстрока два\nстрока три"


def test_table_row_keeps_empty_cells(tmp_path: Path) -> None:
    doc = read_document(
        write(tmp_path, "a.md", "| ID | Текст | Прим |\n|--|--|--|\n| X-1 | Что-то | |\n")
    )
    assert doc.fragments[-1].text == "| X-1 | Что-то | |"


def test_markdown_no_empty_fragments(tmp_path: Path) -> None:
    doc = read_document(write(tmp_path, "a.md", "#\n\ntext here\n"))
    assert all(f.text.strip() for f in doc.fragments)
    assert [f.text for f in doc.fragments] == ["text here"]


def test_heading_path_pops_on_same_level(tmp_path: Path) -> None:
    doc = read_document(write(tmp_path, "a.md", "# A\n\n## B\n\nx\n\n## C\n\ny\n\n# D\n\nz\n"))
    paths = {f.text: f.section for f in doc.fragments}
    assert paths["x"] == ("A", "B")
    assert paths["y"] == ("A", "C")
    assert paths["z"] == ("D",)


def test_cp1251(tmp_path: Path) -> None:
    text = "# Заголовок\n\nСистема должна отвечать быстро.\n"
    doc = read_document(write(tmp_path, "a.md", text.encode("cp1251")))
    assert doc.fragments[1].text == "Система должна отвечать быстро."


@pytest.mark.parametrize("encoding", ["utf-8-sig", "utf-16"])
def test_bom_encodings(tmp_path: Path, encoding: str) -> None:
    text = "# Заголовок\r\n\r\nПервый абзац.\r\n"
    doc = read_document(write(tmp_path, "a.md", text.encode(encoding)))
    assert doc.fragments[1].text == "Первый абзац."
    assert doc.fragments[1].location.line_start == 3


# --- PDF -------------------------------------------------------------------------


def test_pdf_two_pages(tmp_path: Path) -> None:
    path = make_pdf(
        [
            ["First page paragraph line one", "line two of the same block"],
            ["Second page text"],
        ],
        tmp_path / "two.pdf",
    )
    doc = read_document(path)
    assert doc.format is DocumentFormat.PDF
    assert doc.unread_pages == ()
    assert doc.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    pages = [f.location.page for f in doc.fragments]
    assert set(pages) == {1, 2}
    first = [f for f in doc.fragments if f.location.page == 1]
    assert first[0].location.line_start == 1
    assert max(f.location.line_end for f in first) == 2
    second = [f for f in doc.fragments if f.location.page == 2]
    assert second[0].text == "Second page text"
    assert (second[0].location.line_start, second[0].location.line_end) == (1, 1)


def test_pdf_partially_empty(tmp_path: Path) -> None:
    path = make_pdf([["Readable text"], [], ["More readable text"]], tmp_path / "part.pdf")
    doc = read_document(path)
    assert doc.unread_pages == (2,)
    assert doc.degraded
    assert {f.location.page for f in doc.fragments} == {1, 3}


def test_pdf_headings_and_list_items(tmp_path: Path) -> None:
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((60, 80), "Section Title", fontsize=20, fontname="helv")
    y = 110
    for line in ["Body text goes here and continues.", "- first item", "- second item"]:
        page.insert_text((60, y), line, fontsize=11, fontname="helv")
        y += 16
    path = tmp_path / "h.pdf"
    pdf.save(path)
    doc = read_document(path)
    kinds = [f.kind for f in doc.fragments]
    assert kinds == [
        FragmentKind.HEADING,
        FragmentKind.PARAGRAPH,
        FragmentKind.LIST_ITEM,
        FragmentKind.LIST_ITEM,
    ]
    assert doc.fragments[1].section == ("Section Title",)
    assert doc.fragments[3].location.line_start == 4


def test_pdf_table_rows(tmp_path: Path) -> None:
    pdf = pymupdf.open()
    page = pdf.new_page()
    xs = [60, 200, 340]
    ys = [100, 130, 160, 190]
    for y in ys:
        page.draw_line((xs[0], y), (xs[-1], y))
    for x in xs:
        page.draw_line((x, ys[0]), (x, ys[-1]))
    cells = [["Field", "Value"], ["Timeout", "30 sec"], ["Retries", "3"]]
    for r, row in enumerate(cells):
        for c, text in enumerate(row):
            page.insert_text((xs[c] + 6, ys[r] + 20), text, fontsize=11, fontname="helv")
    path = tmp_path / "t.pdf"
    pdf.save(path)
    doc = read_document(path)
    rows = [f.text for f in doc.fragments if f.kind is FragmentKind.TABLE_ROW]
    assert rows == ["| Field | Value |", "| Timeout | 30 sec |", "| Retries | 3 |"]


def test_fenced_code_line_is_the_line_of_its_text(tmp_path: Path) -> None:
    text = '# Склад\n\nПример:\n\n```json\n\n{"a": 1}\n{"b": 2}\n```\n'
    document = read_document(write(tmp_path, "a.md", text.encode()))
    code = next(f for f in document.fragments if f.kind is FragmentKind.CODE)
    assert code.text.startswith('{"a": 1}')
    assert code.location.line_start == 7


def test_docx_keeps_review_insertions_nested_tables_and_literal_marks(tmp_path: Path) -> None:
    import docx
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    document = docx.Document()
    para = document.add_paragraph("Начало ")
    inserted, run, text = OxmlElement("w:ins"), OxmlElement("w:r"), OxmlElement("w:t")
    text.text = "вставка рецензента"
    run.append(text)
    inserted.append(run)
    para._p.append(inserted)
    document.add_paragraph("# не заголовок")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "снаружи"
    table.rows[0].cells[1].add_table(rows=1, cols=1).rows[0].cells[0].text = "вложенная"
    body = document.element.body
    sdt, content = OxmlElement("w:sdt"), OxmlElement("w:sdtContent")
    field = document.add_paragraph("Текст поля шаблона")._p
    content.append(field)
    sdt.append(content)
    body.insert(len(body) - 1, sdt)
    assert sdt.find(qn("w:sdtContent")) is not None
    path = tmp_path / "review.docx"
    document.save(str(path))
    read = read_document(path)
    texts = " / ".join(f.text for f in read.fragments)
    for expected in ("вставка рецензента", "вложенная", "Текст поля шаблона", "# не заголовок"):
        assert expected in texts
    assert not [f for f in read.fragments if f.kind is FragmentKind.HEADING]
