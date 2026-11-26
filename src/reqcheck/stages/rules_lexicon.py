"""Словари и регулярные выражения общих правил (FR-14).

Здесь только общие для любых ТЗ слова и шаблоны. Строк из конкретного
документа тут быть не должно (FR-12).
"""

import re

_I = re.IGNORECASE


def _rx(pattern: str, flags: int = _I) -> re.Pattern[str]:
    return re.compile(pattern, flags)


# --- RULE-VAGUE ---------------------------------------------------------------

# Расплывчатые слова. True — оценочное качество без порога (не проверить),
# False — открытый перечень или необязательность (неоднозначно).
VAGUE_PATTERNS: tuple[tuple[re.Pattern[str], bool], ...] = (
    (_rx(r"\bбыстр(?:о|ее|ый|ая|ое|ые)\b"), True),
    (_rx(r"\bудобн\w*"), True),
    (_rx(r"\bоперативн\w*"), True),
    (_rx(r"\bсвоевременн\w*"), True),
    (_rx(r"\bнадёжн\w*|\bнадежн\w*"), True),
    (_rx(r"\bинтуитивн\w*"), True),
    (
        _rx(
            r"\bдостаточн(?:о|ый|ая|ое|ые|ых)\b(?!\s+(?:средств|прав|полномочи|остат|баланс|денег))"
        ),
        True,
    ),
    (_rx(r"\bкак можно \w+"), True),
    (_rx(r"\buser[- ]friendly\b|\beasy to use\b|\bfast\b|\bquickly\b|\breliabl[ey]\b"), True),
    (_rx(r"\bпри необходимости\b|\bпо необходимости\b|\bпо возможности\b"), False),
    (_rx(r"\bи т\.\s?д\.|\bи т\.\s?п\.|\bи др\.|\bи пр\.|\betc\b\.?"), False),
    (_rx(r"\bas appropriate\b|\bas needed\b|\bif necessary\b|\bwhere possible\b"), False),
)

# Слова, расплывчатые только пока в предложении нет числа.
VAGUE_WITHOUT_NUMBER = _rx(r"\b(?:минимальн|максимальн)\w*")

_UNITS = (
    r"%|мс|мсек|миллисек\w*|с\b|сек\w*|мин\w*|ч\b|час\w*|сут\w*|дн\w*|день|недел\w*|месяц\w*"
    r"|год\w*|лет\b|₸|тенге|тг|руб\w*|\$|€|usd|kzt|eur|кб|мб|гб|тб|байт\w*|шт\w*|раз\w*"
    r"|запрос\w*|пользовател\w*|симв\w*|ms|s\b|sec\w*|seconds?|minutes?|min\b|hours?|h\b"
    r"|days?|rps|tps|mb|gb|kb|tb|requests?|users?|chars?|bytes?"
)
NUMBER_WITH_UNIT = _rx(rf"\d+(?:[\s ]\d{{3}})*(?:[.,]\d+)?\s*(?:{_UNITS})|[$€₸]\s*\d")
ANY_DIGIT = re.compile(r"\d")

# --- RULE-TBD -----------------------------------------------------------------

TBD_PATTERNS: tuple[re.Pattern[str], ...] = (
    _rx(r"\b(?:TBD|TBA|TBC|TODO)\b", 0),
    _rx(r"\bуточняется\b"),
    _rx(r"\b(?:будет|будут)\s+(?:определен|уточнен|определён)\w*"),
    _rx(r"\?{3,}|…"),
)

# --- HTTP ---------------------------------------------------------------------

HTTP_SAFE_METHOD = re.compile(r"\b(GET|HEAD)\b")
HTTP_CONTEXT = _rx(
    r"\b(?:REST|HTTP|API|URL|endpoint|метод\w*|method|запрос\w*|request|эндпоинт\w*)\b"
)
HTTP_PATH_AFTER_METHOD = re.compile(r"\b(?:GET|HEAD)\s+`?\S*/")

# Глаголы-действия, меняющие состояние. Причастия и прилагательные
# («удалённых», «созданные») сюда не входят намеренно.
MUTATION_RU = _rx(
    r"\b(?:сохран(?:ить|яет|ять|яют|ение)|созда(?:ть|ёт|ет|ют|ние)"
    r"|измен(?:ить|яет|ять|яют|ение)|удал(?:ить|яет|ять|яют|ение)"
    r"|обнов(?:ить|ляет|лять|ляют|ление)|установ(?:ить|ит|ят|ление))\b"
)
MUTATION_EN = _rx(
    r"\b(?:save|saves|create|creates|update|updates|delete|deletes|set|sets|remove|removes)\b"
)
# saveLimit, create_order, /items/delete — действие внутри идентификатора.
MUTATION_IDENT = re.compile(r"(?<![A-Za-z])(?:save|create|update|delete|set|remove)(?=[A-Z_/\-])")

STATUS_200 = re.compile(r"(?<![\d.,/-])200(?![\d.,]*\d)(?!\s*(?:₸|тг|тенге|руб|\$|%|шт|мс|ms))")
HTTP_STATUS_CONTEXT = _rx(
    r"\b(?:HTTP|status|статус\w*|код\w*|ответ\w*|response|REST|API|метод\w*)\b|\|\s*200\s*\|"
)
OTHER_ERROR_STATUS = re.compile(r"(?<![\d.,])[45]\d\d(?![\d.,]*\d)")
ERROR_MARKER = _rx(
    r"(?<!без\s)\b(?:errors?|ошибк\w*|fail(?:s|ed|ure)?|неуспе\w*)\b"
    r"|\bstatus\W{0,3}ERROR\b|\"result\"\s*:\s*\"ERROR\""
)
