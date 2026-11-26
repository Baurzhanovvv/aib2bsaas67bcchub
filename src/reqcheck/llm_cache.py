"""Дисковый кэш ответов модели (NFR-4): воспроизводимость повторных прогонов.

Хранит сырой текст ответа, прошедшего схему; разбор и проверка — на стороне клиента.
Любая проблема с кэшем (нет прав, битый файл) — это промах, а не ошибка прогона.
"""

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

CACHE_FORMAT_VERSION = 1


@dataclass(frozen=True)
class CacheEntry:
    text: str
    model: str


def make_cache_key(*, provider: str, model: str, system: str, user: str, schema: str) -> str:
    """sha256 от всего, что влияет на ответ, плюс версия формата кэша."""
    payload = json.dumps(
        [CACHE_FORMAT_VERSION, provider, model, system, user, schema],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DiskCache:
    def __init__(self, directory: Path) -> None:
        self._dir = directory

    def _path(self, key: str) -> Path:
        return self._dir / key[:2] / f"{key}.json"

    def get(self, key: str) -> CacheEntry | None:
        try:
            raw = json.loads(self._path(key).read_text(encoding="utf-8"))
            if raw.get("format") != CACHE_FORMAT_VERSION:
                return None
            text, model = raw["text"], raw["model"]
        except (OSError, ValueError, KeyError, AttributeError):
            return None
        if not isinstance(text, str) or not isinstance(model, str):
            return None
        return CacheEntry(text=text, model=model)

    def put(self, key: str, entry: CacheEntry) -> None:
        """Атомарная запись; ошибки диска не роняют прогон."""
        path = self._path(key)
        data = {"format": CACHE_FORMAT_VERSION, "model": entry.model, "text": entry.text}
        tmp: str | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False)
            os.replace(tmp, path)
        except OSError:
            if tmp is not None:
                Path(tmp).unlink(missing_ok=True)
