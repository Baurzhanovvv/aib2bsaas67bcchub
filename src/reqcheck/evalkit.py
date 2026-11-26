"""Оценка отчёта по золотому набору (FR-13): «найдено k из n» и замечания вне набора.

Золотой набор — JSON, который размечает не автор промптов:

    {"expected": [{"id": "G1", "title": "GET меняет состояние",
                   "quote_contains": ["Удаление позиции"], "types": ["interface_misuse"]}]}

Пункт найден, если у какого-то замечания есть цитата, содержащая один из фрагментов
(сравнение после той же нормализации, что при сверке цитат, без учёта регистра).
Длина цитаты ограничена: иначе модель «попадала» бы, цитируя абзац целиком. `types` —
необязательный список допустимых типов дефекта.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from reqcheck.models import Finding, Report
from reqcheck.text_norm import norm_text

MIN_QUOTE_CAP = 400
CAP_FACTOR = 4


class Expected(BaseModel):
    id: str
    title: str = ""
    quote_contains: list[str] = Field(min_length=1)
    types: list[str] = []


class Golden(BaseModel):
    expected: list[Expected] = Field(min_length=1)


@dataclass(frozen=True)
class Score:
    found: dict[str, list[str]]
    """id пункта набора → id замечаний, которые его закрыли."""
    missed: list[Expected]
    extra: list[Finding]
    """Замечания вне набора: не обязательно ложные — набор не исчерпывает дефекты."""
    total: int

    @property
    def recall(self) -> float:
        return len(self.found) / self.total


def load_golden(path: Path) -> Golden:
    return Golden.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _key(text: str) -> str:
    return norm_text(text).casefold()


def _hits(finding: Finding, item: Expected) -> bool:
    if item.types and finding.defect_type.value not in item.types:
        return False
    for fragment in map(_key, item.quote_contains):
        cap = max(MIN_QUOTE_CAP, CAP_FACTOR * len(fragment))
        for quote in finding.quotes:
            text = _key(quote.text)
            if fragment and fragment in text and len(text) <= cap:
                return True
    return False


def score(report: Report, golden: Golden) -> Score:
    found: dict[str, list[str]] = {}
    used: set[str] = set()
    for item in golden.expected:
        ids = [f.id for f in report.findings if _hits(f, item)]
        if ids:
            found[item.id] = ids
            used.update(ids)
    missed = [item for item in golden.expected if item.id not in found]
    extra = [f for f in report.findings if f.id not in used]
    return Score(found=found, missed=missed, extra=extra, total=len(golden.expected))


def render(result: Score, report: Report) -> str:
    p = report.passport
    lines = [
        f"Найдено {len(result.found)} из {result.total} ({result.recall:.0%}). "
        f"Замечаний в отчёте: {len(report.findings)}, вне набора: {len(result.extra)}.",
        f"Модель: {p.model or 'без модели'}; цитаты: проверено {p.quotes_checked}, "
        f"подтверждено {p.quotes_confirmed}; стоимость ${p.cost_usd:.2f}.",
    ]
    for item_id, ids in result.found.items():
        lines.append(f"  + {item_id}: {', '.join(ids)}")
    for item in result.missed:
        lines.append(f"  - {item.id}: не найдено" + (f" — {item.title}" if item.title else ""))
    if result.extra:
        lines.append("Вне набора (проверить вручную: верное, дубль или придирка):")
        lines += [
            f"  ? {f.id} {f.defect_type.value}: {f.quotes[0].text[:80]}" for f in result.extra
        ]
    return "\n".join(lines)
