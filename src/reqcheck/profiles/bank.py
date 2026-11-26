"""Профиль «банк» (FR-19): что банковское требование обязано задать явно.

Перечень общий для платёжных и учётных систем; строк из какого-либо конкретного
документа здесь нет (FR-12).
"""

import re

from reqcheck.models import (
    MIN_QUOTE_LENGTH,
    DefectType,
    Document,
    FindingSource,
    FragmentKind,
    Impact,
    RawFinding,
    Requirement,
    Scope,
)
from reqcheck.profiles import Profile

CHECKLIST = """\
Профиль домена: банк. Дополнительно к общей задаче проверь, задано ли в документе явно
то, без чего банковскую функцию нельзя реализовать однозначно:
- валюта сумм, точность (число знаков) и правило округления;
- часовой пояс и границы расчётного периода (сутки, месяц), момент обнуления счётчиков;
- поведение при повторной доставке одного события: операция не должна учитываться дважды;
- учёт отмен, возвратов и сторнирования в расчётах и счётчиках;
- поведение ровно на границе лимита или порога (равно, больше, меньше);
- маскирование номера карты и персональных данных в сообщениях, ответах и журналах;
- журнал аудита изменения настроек: кто, когда, старое и новое значение;
- поведение при недоступности внешней системы и повторные попытки.
Если пункт относится к функциям документа и не задан — это missing_requirement (или
incomplete, если задан наполовину) с цитатой места, где он должен быть задан. Пункт,
который к функциям документа не относится, не упоминай."""

_FLAGS = re.IGNORECASE
_COUNTED = r"(?:лимит\w*|порог\w*|расход\w*|оборот\w*|сч[её]тчик\w*)"
_GAP = r"[^.;!?\n]{0,40}"
_PERIOD_WORDS = (
    r"(?:в\s+сутки|за\s+сутки|суточн\w+|в\s+день|за\s+день|ежедневн\w+|дневн\w+"
    r"|в\s+месяц|за\s+месяц|ежемесячн\w+|месячн\w+|календарн\w+\s+месяц\w*)"
)
# Счётчик и период рядом в одной фразе: «лимит … за сутки», «месячный порог».
_PERIOD = re.compile(
    rf"\b{_COUNTED}\b{_GAP}\b(?P<a>{_PERIOD_WORDS})"
    rf"|\b(?P<b>{_PERIOD_WORDS}){_GAP}\b{_COUNTED}",
    _FLAGS,
)
_TIMEZONE = re.compile(
    r"часов\w+\s+пояс\w*|\bUTC\b|\bGMT\b|time\s?zone|местн\w+\s+врем\w+|по\s+времени\s+\w+",
    _FLAGS,
)


def rule_period_without_timezone(
    document: Document, requirements: tuple[Requirement, ...]
) -> list[RawFinding]:
    """Счётчик за сутки или месяц есть, а часовой пояс не назван нигде в документе."""
    texts = [f.text for f in document.fragments if f.kind is not FragmentKind.CODE]
    if any(_TIMEZONE.search(text) for text in texts):
        return []
    req_of = {r.fragment_id: r.id for r in requirements}
    for fragment in document.fragments:
        if fragment.kind in (FragmentKind.CODE, FragmentKind.HEADING):
            continue
        for line in fragment.text.splitlines():
            period = _PERIOD.search(line)
            if not period or len(line.strip()) < MIN_QUOTE_LENGTH:
                continue
            ids = (req_of[fragment.id],) if fragment.id in req_of else ()
            found = period.group("a") or period.group("b")
            return [
                RawFinding(
                    defect_type=DefectType.MISSING_REQUIREMENT,
                    requirement_ids=ids,
                    quotes=(line.strip(),),
                    explanation=(
                        f"Величина считается за период («{found}»), "
                        "но часовой пояс "
                        "и момент начала периода в документе не названы. Операция около "
                        "полуночи попадёт в разные периоды в зависимости от того, чьё время "
                        "взято: сервера, клиента или процессинга (ISO/IEC/IEEE 29148, 5.2.6: "
                        "полнота набора)."
                    ),
                    suggestion=(
                        "Вопрос заказчику: по какому времени считается период? Варианты: "
                        "1) единый часовой пояс банка; 2) часовой пояс клиента; 3) UTC. "
                        "Отдельно задать момент обнуления счётчика."
                    ),
                    scope=Scope.REQUIREMENT_SET,
                    impact=Impact.MEDIUM,
                    source=FindingSource.RULE,
                    rule_id="RULE-BANK-PERIOD-TIMEZONE",
                )
            ]
    return []


PROFILE = Profile(
    name="bank",
    title="банк",
    checklist=CHECKLIST,
    rules=(rule_period_without_timezone,),
)
