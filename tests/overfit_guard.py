"""Сканер анти-подгонки (FR-12): ищет в коде признаки выданного ТЗ.

Признаки извлекаются из документа на лету и нигде не хранятся: в репозитории лежит только
стоп-лист общей лексики, который к конкретному ТЗ отношения не имеет.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

ENV_VAR = "REQCHECK_ISSUED_DOC"
SCAN_SUFFIXES = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml"}
SCAN_DIRS = ("src", "tests")
# Сам тест и сканер не сканируются; каталоги прогонов на выданном ТЗ — явный список исключений.
EXCLUDED_FILES = {"test_no_overfit.py", "overfit_guard.py"}
EXCLUDED_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache", "issued_doc_runs"}

SHINGLE_SIZE = 4
MIN_CONTENT_WORDS = 2  # минимум значимых (не служебных) слов в шингле
MIN_ID_LEN = 6
MIN_NUMBER_DIGITS = 5

# Стоп-лист: служебные слова и общая лексика требований. К конкретному ТЗ не относится.
STOP_WORDS = frozenset(
    """
    и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее
    мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был
    него до вас нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней
    для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот
    того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда
    зачем всех никогда можно при наконец два об другой хоть после над больше тот через эти нас
    про них какая много разве три эту моя впрочем хорошо свою этой перед иногда лучше чуть том
    нельзя такой им более всегда конечно всю между это также либо при который которая которые
    которого которой которых которому котором должен должна должно должны должна
    система системы системе систему системой пользователь пользователя пользователю клиент
    клиента клиенту сервис сервиса сервису сервисе данные данных данным данными возможность
    возможности необходимо необходима необходимы случае случай требование требования требований
    требованию обеспечивать обеспечить обеспечивает должна осуществлять выполнять поддерживать
    быть быть может могут иметь время ответа запрос запроса запросов обращении обращению
    значение значения значений функция функции функций информация информации информацию
    указанный указанного указанных следующие следующих следующий соответствии соответствие
    раздел раздела разделе пункт пункта пункте документ документа документе
    the a an of to in is and or for be with on by as at
    """.split()
)
# Общие идентификаторы протокола/формата, которые законно встречаются в любом коде.
GENERIC_IDENTIFIERS = frozenset(
    {"application/json", "text/plain", "text/html", "utf-8", "content-type", "n/a", "and/or"}
)
# Те же протокольные обозначения, разобранные на слова шинглами (заголовки HTTP).
GENERIC_PHRASES = frozenset({"content type application json", "accept application json"})

_WORD_RE = re.compile(r"[0-9a-zа-яё]+")
_ID_TOKEN_RE = re.compile(r"[A-Za-z0-9_/.:\-]+")
_NUMBER_RE = re.compile(r"(?<![\d])\d{1,3}(?:[   ]\d{3})+(?![\d])")
_CODE_RE = re.compile(r"(?<![\w-])[А-ЯЁA-Z]{1,3}-\d{1,3}(?![\w-])")


@dataclass(frozen=True)
class Hit:
    path: Path
    line: int
    fragment: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line} -> {self.fragment!r}"


@dataclass
class Signatures:
    identifiers: set[str] = field(default_factory=set)
    numbers: set[str] = field(default_factory=set)  # только цифры
    codes: set[str] = field(default_factory=set)
    shingles: set[tuple[str, ...]] = field(default_factory=set)

    def __bool__(self) -> bool:
        return bool(self.identifiers or self.numbers or self.codes or self.shingles)


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower().replace("ё", "е"))


def _is_identifier(tok: str) -> bool:
    if len(tok) < MIN_ID_LEN or not re.search(r"[A-Za-z]", tok):
        return False
    if tok.lower() in GENERIC_IDENTIFIERS:
        return False
    return bool(re.search(r"[\d_/]", tok) or re.search(r"[a-z][A-Z]", tok))


def _is_content_shingle(sh: tuple[str, ...]) -> bool:
    return sum(1 for w in sh if w not in STOP_WORDS) >= MIN_CONTENT_WORDS


def _id_tokens(text: str) -> list[str]:
    """Токен целиком и его части между «/»: имя операции внутри пути тоже признак."""
    found: list[str] = []
    for m in _ID_TOKEN_RE.finditer(text):
        tok = m.group(0).strip(".:-")
        found.append(tok)
        if "/" in tok:
            found += [part.strip(".:-") for part in tok.split("/") if part]
    return found


def extract_signatures(text: str) -> Signatures:
    sig = Signatures()
    for tok in _id_tokens(text):
        if _is_identifier(tok):
            sig.identifiers.add(tok.lower())
    for m in _NUMBER_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if len(digits) >= MIN_NUMBER_DIGITS:
            sig.numbers.add(digits)
    sig.codes.update(m.group(0).lower() for m in _CODE_RE.finditer(text))
    ws = words(text)
    for i in range(len(ws) - SHINGLE_SIZE + 1):
        sh = tuple(ws[i : i + SHINGLE_SIZE])
        if _is_content_shingle(sh):
            sig.shingles.add(sh)
    return sig


def load_issued_docs(env: str | None = None) -> list[Path]:
    raw = os.environ.get(ENV_VAR, "") if env is None else env
    return [Path(p) for p in raw.split(os.pathsep) if p.strip()]


def merge(sigs: list[Signatures]) -> Signatures:
    out = Signatures()
    for s in sigs:
        out.identifiers |= s.identifiers
        out.numbers |= s.numbers
        out.codes |= s.codes
        out.shingles |= s.shingles
    return out


def iter_scan_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for d in SCAN_DIRS:
        base = root / d
        if not base.is_dir():
            continue
        for p in sorted(base.rglob("*")):
            if not p.is_file() or p.suffix.lower() not in SCAN_SUFFIXES:
                continue
            if p.name in EXCLUDED_FILES or EXCLUDED_DIRS & set(p.relative_to(base).parts):
                continue
            files.append(p)
    return files


def scan_text(text: str, sig: Signatures, path: Path) -> list[Hit]:
    hits: list[Hit] = []
    for n, line in enumerate(text.splitlines(), start=1):
        for tok in _id_tokens(line):
            if tok.lower() in sig.identifiers:
                hits.append(Hit(path, n, tok))
        for m in _NUMBER_RE.finditer(line):
            if re.sub(r"\D", "", m.group(0)) in sig.numbers:
                hits.append(Hit(path, n, m.group(0)))
        for m in _CODE_RE.finditer(line):
            if m.group(0).lower() in sig.codes:
                hits.append(Hit(path, n, m.group(0)))
    hits.extend(_scan_shingles(text, sig, path))
    return hits


def _scan_shingles(text: str, sig: Signatures, path: Path) -> list[Hit]:
    """Шинглы ищутся по всему тексту файла, так что перенос строки фразу не прячет."""
    toks = [(m.group(0), text.count("\n", 0, m.start()) + 1) for m in _word_iter(text)]
    hits: list[Hit] = []
    for i in range(len(toks) - SHINGLE_SIZE + 1):
        sh = tuple(t for t, _ in toks[i : i + SHINGLE_SIZE])
        if sh in sig.shingles and " ".join(sh) not in GENERIC_PHRASES:
            hits.append(Hit(path, toks[i][1], " ".join(sh)))
    return hits


def _word_iter(text: str):  # type: ignore[no-untyped-def]
    low = text.lower().replace("ё", "е")
    return _WORD_RE.finditer(low)


def scan_tree(root: Path, sig: Signatures) -> list[Hit]:
    hits: list[Hit] = []
    for p in iter_scan_files(root):
        text = p.read_text(encoding="utf-8", errors="replace")
        hits.extend(scan_text(text, sig, p.relative_to(root)))
    return hits
