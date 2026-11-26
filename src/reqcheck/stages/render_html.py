"""Отчёт в HTML и PDF (FR-22): тот же текст, что в Markdown, с подсветкой цитат.

Формат выбирается по расширению файла отчёта. HTML самодостаточен (стили внутри, без
сети). PDF собирается из того же HTML средствами PyMuPDF, без внешних программ.
"""

import html
import re
from pathlib import Path

import pymupdf
from markdown_it import MarkdownIt

from reqcheck.models import Report
from reqcheck.stages.render import render_markdown

# HTML из отчёта не исполняется: текст документа попадает в отчёт цитатами.
_MD = MarkdownIt("commonmark", {"html": False}).enable("table")

_CSS = """
body { font-family: sans-serif; font-size: 11pt; line-height: 1.45; color: #1c1c1c;
       max-width: 62em; margin: 2em auto; padding: 0 1em; }
h1 { font-size: 20pt; } h2 { font-size: 15pt; border-bottom: 1px solid #bbbbbb; }
h3 { font-size: 12.5pt; } h4 { font-size: 11.5pt; }
table { border-collapse: collapse; width: 100%; margin: 0.6em 0; font-size: 9.5pt; }
th, td { border: 1px solid #bbbbbb; padding: 3px 6px; text-align: left; vertical-align: top; }
th { background-color: #eeeeee; }
blockquote { margin: 0.5em 0; padding: 0.3em 0.8em; border-left: 4px solid #e0a800;
             background-color: #fff6d6; }
code { font-family: monospace; font-size: 9.5pt; background-color: #f1f1f1; }
.blocker { color: #b00020; font-weight: bold; } .critical { color: #c25400; font-weight: bold; }
.major { color: #8a6d00; font-weight: bold; } .minor { color: #555555; font-weight: bold; }
"""

_SEVERITY_CLASS = {
    "блокирующее": "blocker",
    "критичное": "critical",
    "существенное": "major",
    "незначительное": "minor",
}
_SEVERITY_HEAD = re.compile(r"(<h4>D-\d+ — )(" + "|".join(_SEVERITY_CLASS) + r")\b")
PAGE_MARGIN = 36


def render_html(report: Report) -> str:
    body = _MD.render(render_markdown(report))
    body = _SEVERITY_HEAD.sub(
        lambda m: f'{m.group(1)}<span class="{_SEVERITY_CLASS[m.group(2)]}">{m.group(2)}</span>',
        body,
    )
    title = html.escape(f"Проверка требований: {report.passport.document_name}")
    return (
        '<!DOCTYPE html>\n<html lang="ru">\n<head>\n<meta charset="utf-8">\n'
        f"<title>{title}</title>\n<style>{_CSS}</style>\n</head>\n<body>\n{body}</body>\n</html>\n"
    )


def write_pdf(report: Report, path: Path) -> None:
    """Сверстать HTML отчёта в PDF формата A4."""
    body = render_html(report).split("<body>", 1)[1].rsplit("</body>", 1)[0]
    story = pymupdf.Story(html=body, user_css=_CSS)
    page_box = pymupdf.paper_rect("a4")
    content = page_box + (PAGE_MARGIN, PAGE_MARGIN, -PAGE_MARGIN, -PAGE_MARGIN)
    writer = pymupdf.DocumentWriter(str(path))
    more = True
    while more:
        device = writer.begin_page(page_box)
        more, _ = story.place(content)
        story.draw(device)
        writer.end_page()
    writer.close()
