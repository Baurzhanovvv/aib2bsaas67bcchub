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
# setPrice, create_order, /items/delete — действие внутри идентификатора.
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

# --- Контракт API между фрагментами (rules_api) -------------------------------

# Объявление метода: «Метод: `GET`», «HTTP method — HEAD», строка таблицы с типом запроса.
API_METHOD_DECL = re.compile(r"(?i:REST\s*type|HTTP[- ]?method|метод|method)\W{0,12}(GET|HEAD)\b")
# Назначение операции начинается со слова-действия: «Description | Сохранение …».
API_PURPOSE_MUTATION = _rx(
    r"^\W*(?:description|описание|назначение|purpose|summary)\W+"
    rf"({MUTATION_RU.pattern}|{MUTATION_EN.pattern})"
)
# Действие в имени операции внутри пути: /items/setPrice, /orders/delete-all.
API_PATH_MUTATION = re.compile(
    r"/[\w\-/{}]*?(?<![A-Za-z])((?:save|create|update|delete|set|remove)(?=[A-Z_/\-])\w*)"
)
NEGATION_BEFORE = _rx(r"\b(?:не|not)\s+$")

# Контекст HTTP без общих слов («ответ», «код», «метод» встречаются в любом ТЗ).
HTTP_STRICT = _rx(r"\b(?:HTTP|REST|API|status|статус\w*|response)\b")
SUCCESS_STATUS_HEAD = re.compile(r"^\W*2\d\d\b")
SUCCESS_STATUS = re.compile(r"(?<![\d.,/-])2\d\d(?![\d.,]*\d)")
ERROR_BODY = _rx(r"\"(?:result|status|state|outcome)\"\s*:\s*\"(?:error|fail(?:ed|ure)?)\"")

_KNOWN_STATUS = (
    "200|201|202|204|301|302|304|400|401|403|404|405|409|410|412|415|422|429|500|502|503|504"
)
# После числа стоит единица или счётное слово — это количество, а не код ответа.
_NOT_A_STATUS = (
    r"₸|тг|тенге|руб|\$|€|%|kzt|usd|eur|шт|мс|ms|сек|мин|час|дн|раз|строк|запис|симв|знак"
    r"|байт|бит|bit|px|rps|кб|мб|гб|kb|mb|gb|кг|км|позиц|пользовател|клиент|заказ|документ"
    r"|элемент|объект|запрос|попыт|items?|users?|records?|rows?|chars?|bytes?|requests?"
)
_STATUS = rf"({_KNOWN_STATUS})(?![\d.,]*\d)(?![\s ]*(?:\d{{3}}\b|{_NOT_A_STATUS}))"

# Стандартные фразы кодов ответа (RFC 9110). «OK» не берём: слово слишком общее.
REASON_PHRASES: dict[str, int] = {
    "created": 201,
    "accepted": 202,
    "no content": 204,
    "bad request": 400,
    "unauthorized": 401,
    "forbidden": 403,
    "not found": 404,
    "conflict": 409,
    "internal server error": 500,
}
# Фраза стоит сразу за кодом; между ними — знаки и не больше одного слова с тире:
# «204 — Not Found», «202. Done — Created», «`201` | **Created**».
STATUS_WITH_PHRASE = _rx(
    rf"(?<![\d.,/:-])(?<!\d[\s ]){_STATUS}[\s.|:*_`—–-]{{0,8}}"
    r"(?:[A-Za-zА-Яа-яЁё]+[\s*_`]*[—–-][\s*_`]*)?\b(" + "|".join(REASON_PHRASES) + r")\b"
)

# Перечень статусов: код в начале строки с описанием, столбец кодов, ключи спецификации.
STATUS_LIST_HEAD = re.compile(r"^[\s|*_`#>-]*([1-5]\d\d)[*_`]*\s*(?:[.:—–-]|\|)[*_`]*\s*\S")
STATUS_COLUMN = _rx(r"http|status|статус|код|код ответа|код состояния|status code|http status")
STATUS_CELL = re.compile(r"[1-5]\d\d")
STATUS_SPEC_KEY = re.compile(r"^\s*['\"]?([1-5]\d\d)['\"]?\s*:", re.MULTILINE)
STATUS_HTTP_LINE = re.compile(r"HTTP/\d(?:\.\d)?\s+([1-5]\d\d)\b")
# Упоминание кода в тексте: слово-указатель прямо перед числом, дальше — перечисление.
STATUS_MENTION = _rx(
    r"\b(?:код\w*|статус\w*|status|HTTP|результат\w*|возвраща\w*|верн\w*|отвеча\w*"
    r"|returns?|responds?)\s+(?:(?:код\w*|статус\w*|status|HTTP)\s+)?" + _STATUS
)
STATUS_NEXT = _rx(r"\s*(?:,|или|и|or|and|/)\s*" + _STATUS)

# Таблицы параметров: столбец имени и столбец типа; пара «запрос — ответ» одного метода.
TABLE_SEPARATOR = re.compile(r"^\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?$")
CELL_SPLIT = re.compile(r"(?<!\\)\|")
NAME_COLUMN = _rx(r"parameter|param|параметр|поле|field|name|имя|атрибут|attribute")
TYPE_COLUMN = _rx(r"format|формат|тип|type|тип данных|data type")
TYPE_WORD = _rx(r"[a-zа-яё]+")
TYPE_SYNONYMS: dict[str, str] = {
    "int": "integer",
    "long": "integer",
    "целое": "integer",
    "bool": "boolean",
    "булево": "boolean",
    "str": "string",
    "строка": "string",
    "text": "string",
    "число": "number",
    "float": "number",
    "double": "number",
    "decimal": "number",
}
REQUEST_PART = _rx(r"запрос|request|query|body|парамет|входн")
RESPONSE_PART = _rx(r"ответ|response|результат|выходн")

# Машиночитаемая спецификация и пути, объявленные в тексте.
SPEC_MARK = _rx(r"^\s*(?:openapi|swagger)\s*:", _I | re.MULTILINE)
SPEC_PATHS_LINE = _rx(r"^paths\s*:", _I | re.MULTILINE)
SPEC_PATH_KEY = re.compile(r"^\s*(/[\w\-./{}]*)\s*:", re.MULTILINE)
PATH_DECL = _rx(r"(?:путь|path|endpoint|эндпоинт)\W{0,12}(/[\w\-./{}]+)")
PATH_PARAM = re.compile(r"\{[^}]*\}")
