"""Чтение документа и preflight (FR-2, FR-3, FR-20)."""

import hashlib
from collections.abc import Callable
from itertools import count
from pathlib import Path

from reqcheck.errors import InputError
from reqcheck.models import Document, DocumentFormat
from reqcheck.stages.reader_docx import docx_to_markdown
from reqcheck.stages.reader_md import parse_markdown
from reqcheck.stages.reader_pdf import parse_pdf

_MD_EXTENSIONS = {".md", ".markdown"}
_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"%PDF-", "PDF"),
    (b"\x89PNG\r\n\x1a\n", "изображение PNG"),
    (b"\xff\xd8\xff", "изображение JPEG"),
    (b"GIF8", "изображение GIF"),
    (b"PK\x03\x04", "ZIP-архив (в том числе DOCX/XLSX)"),
    (b"\x1f\x8b", "архив gzip"),
    (b"Rar!", "архив RAR"),
    (b"7z\xbc\xaf\x27\x1c", "архив 7z"),
    (b"\xd0\xcf\x11\xe0", "документ Microsoft Office старого формата"),
)
_PDF_HEAD_WINDOW = 1024


def _id_factory() -> Callable[[], str]:
    counter = count(1)
    return lambda: f"F-{next(counter):04d}"


def _detect(data: bytes) -> str | None:
    return next((name for sig, name in _SIGNATURES if data.startswith(sig)), None)


def decode_text(data: bytes, name: str) -> str:
    """UTF-8 (с BOM и без), UTF-16 по BOM, иначе CP1251 при невалидном UTF-8."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError as exc:
            raise InputError(f"Файл «{name}» с маркером UTF-16 не декодируется.") from exc
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    if b"\x00" in data:
        raise InputError(f"Файл «{name}» содержит бинарные данные и не похож на текст Markdown.")
    try:
        return data.decode("cp1251")
    except UnicodeDecodeError as exc:
        raise InputError(
            f"Не удалось определить кодировку файла «{name}». Сохраните его в UTF-8."
        ) from exc


def _check_signature(data: bytes, suffix: str, name: str) -> None:
    detected = _detect(data)
    if suffix == ".docx":
        if not data.startswith(b"PK\x03\x04"):
            what = detected or "не DOCX"
            raise InputError(
                f"Файл «{name}» имеет расширение .docx, но по содержимому это {what}. "
                "Передайте настоящий DOCX, PDF или Markdown."
            )
    elif suffix == ".pdf":
        if b"%PDF-" not in data[:_PDF_HEAD_WINDOW]:
            what = detected or "не PDF"
            raise InputError(
                f"Файл «{name}» имеет расширение .pdf, но по содержимому это {what}. "
                "Передайте настоящий PDF или Markdown."
            )
    elif detected is not None:
        raise InputError(
            f"Файл «{name}» имеет расширение {suffix}, но по содержимому это {detected}. "
            "Передайте текстовый Markdown или PDF."
        )


def _check_extension(suffix: str, name: str) -> None:
    if suffix not in _MD_EXTENSIONS and suffix not in (".pdf", ".docx"):
        shown = suffix or "без расширения"
        raise InputError(
            f"Неподдерживаемый формат файла «{name}» ({shown}). "
            "Поддерживаются PDF, DOCX и Markdown (.md)."
        )


def read_document(path: Path) -> Document:
    """Разобрать файл на фрагменты с номерами строк и страниц.

    Файл, который нельзя читать (пустой, чужая сигнатура, зашифрованный PDF,
    скан без текста), — reqcheck.errors.InputError с причиной на русском.
    """
    name = path.name
    if not path.is_file():
        raise InputError(f"Файл «{path}» не найден.")
    suffix = path.suffix.lower()
    _check_extension(suffix, name)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise InputError(f"Не удалось прочитать файл «{name}»: {exc.strerror or exc}.") from exc
    if not data:
        raise InputError(f"Файл «{name}» пустой (0 байт).")
    _check_signature(data, suffix, name)
    next_id = _id_factory()
    sha256 = hashlib.sha256(data).hexdigest()
    if suffix == ".pdf":
        fragments, unread = parse_pdf(data, next_id)
        fmt = DocumentFormat.PDF
    else:
        is_docx = suffix == ".docx"
        text = docx_to_markdown(data, name) if is_docx else decode_text(data, name)
        fragments, unread = parse_markdown(text, next_id), ()
        fmt = DocumentFormat.DOCX if is_docx else DocumentFormat.MARKDOWN
        if not fragments:
            raise InputError(f"В файле «{name}» нет текста для проверки.")
    return Document(
        name=name, sha256=sha256, format=fmt, fragments=tuple(fragments), unread_pages=unread
    )
