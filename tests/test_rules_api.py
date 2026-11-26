"""Тесты правил контракта API (FR-14). Документы синтетические, разбираются настоящим читателем."""

from pathlib import Path

from reqcheck.models import DefectType, FindingSource, RawFinding, Scope
from reqcheck.stages import reader, registry
from reqcheck.stages.rules_api import apply_api_rules

FENCE = "```"


def run(tmp_path: Path, markdown: str) -> list[RawFinding]:
    path = tmp_path / "doc.md"
    path.write_text(markdown, encoding="utf-8")
    document = reader.read_document(path)
    found = apply_api_rules(document, registry.build_registry(document, None))
    for item in found:
        assert item.source is FindingSource.RULE and item.rule_id
        assert item.explanation and item.suggestion
        for quote in item.quotes:
            assert len(quote) >= 8
            assert quote.strip("| ") in markdown
    return found


def by_rule(found: list[RawFinding], rule_id: str) -> list[RawFinding]:
    return [f for f in found if f.rule_id == rule_id]


# --- RULE-API-METHOD-MUTATES ----------------------------------------------------

METHOD_DOC = """# Склад

## Метод списания

| Ключ | Что |
| --- | --- |
| Назначение | {purpose} |
| HTTP method | {method} |
| Адрес | `/stock/{ident}` |
"""


def test_get_with_mutating_description(tmp_path: Path) -> None:
    doc = METHOD_DOC.format(purpose="Удаление позиции со склада", method="GET", ident="item")
    found = by_rule(run(tmp_path, doc), "RULE-API-METHOD-MUTATES")
    assert len(found) == 1
    assert found[0].defect_type is DefectType.INTERFACE_MISUSE
    assert found[0].scope is Scope.REQUIREMENT_SET
    assert len(found[0].quotes) == 2


def test_get_with_mutating_identifier_only(tmp_path: Path) -> None:
    doc = METHOD_DOC.format(purpose="Позиция склада", method="GET", ident="removeItem")
    assert len(by_rule(run(tmp_path, doc), "RULE-API-METHOD-MUTATES")) == 1


def test_get_reading_and_post_writing_are_clean(tmp_path: Path) -> None:
    reading = METHOD_DOC.format(purpose="Список позиций склада", method="GET", ident="items")
    writing = METHOD_DOC.format(purpose="Удаление позиции со склада", method="POST", ident="item")
    negated = METHOD_DOC.format(purpose="Метод не изменяет остатки", method="GET", ident="items")
    sorting = METHOD_DOC.format(
        purpose="Returns orders sorted by last update date", method="GET", ident="items"
    )
    rights = METHOD_DOC.format(
        purpose="Список позиций. Требуется право на создание заказов", method="GET", ident="items"
    )
    for doc in (reading, writing, negated, sorting, rights):
        assert not by_rule(run(tmp_path, doc), "RULE-API-METHOD-MUTATES")


# --- RULE-API-ERROR-UNDER-SUCCESS, RULE-API-STATUS-PHRASE -----------------------


def _status_doc(title: str, body: str) -> str:
    return f"# Склад\n\n## Ответы\n\n### {title}\n\n{FENCE}json\n{body}\n{FENCE}\n"


def test_error_body_under_success_heading(tmp_path: Path) -> None:
    doc = _status_doc("200. Успешный ответ метода", '{"status":"error","text":"Нет остатка"}')
    found = by_rule(run(tmp_path, doc), "RULE-API-ERROR-UNDER-SUCCESS")
    assert len(found) == 1
    assert found[0].quotes == ('{"status":"error","text":"Нет остатка"}',)


def test_error_body_under_error_heading_is_clean(tmp_path: Path) -> None:
    under_error = _status_doc("422. Ошибка проверки данных", '{"status":"error"}')
    success = _status_doc("200. Успешный ответ метода", '{"status":"done","errors":[]}')
    not_a_status = _status_doc("Лимит 250 обращений в минуту", '{"status":"error"}')
    for doc in (under_error, success, not_a_status):
        assert not by_rule(run(tmp_path, doc), "RULE-API-ERROR-UNDER-SUCCESS")


def test_status_with_foreign_reason_phrase(tmp_path: Path) -> None:
    doc = "# Склад\n\n## Ответы\n\nHTTP-код 204 — Not Found, если позиции нет на складе.\n"
    found = by_rule(run(tmp_path, doc), "RULE-API-STATUS-PHRASE")
    assert len(found) == 1
    assert "404" in found[0].explanation


def test_plain_numbers_near_reason_words_are_clean(tmp_path: Path) -> None:
    for line in (
        "Максимум 500 символов, иначе Bad Request.",
        "До 400 записей в выдаче; при превышении — conflict.",
        "Склад на 204 места, доступ forbidden для посторонних.",
    ):
        doc = f"# Склад\n\n## Ответы API\n\n{line}\n"
        assert not by_rule(run(tmp_path, doc), "RULE-API-STATUS-PHRASE")


def test_status_with_own_reason_phrase_is_clean(tmp_path: Path) -> None:
    doc = (
        "# Склад\n\n## Ответы\n\n"
        "| Код | Фраза | Когда |\n| --- | --- | --- |\n"
        "| `201` | **Created** | Позиция заведена |\n"
        "| `409` | **Conflict** | Позиция уже есть |\n"
    )
    assert not by_rule(run(tmp_path, doc), "RULE-API-STATUS-PHRASE")


# --- RULE-API-STATUS-UNLISTED ---------------------------------------------------

STATUS_LIST = """# Склад

## Статусы ответа

**200.** Позиция найдена на складе.
**400.** Запрос составлен неверно.
**401.** Нет входа в систему.

{extra}
"""


def test_status_used_but_not_listed(tmp_path: Path) -> None:
    doc = STATUS_LIST.format(extra="Для закрытого склада метод отвечает кодом 403.")
    found = by_rule(run(tmp_path, doc), "RULE-API-STATUS-UNLISTED")
    assert len(found) == 1
    assert "403" in found[0].explanation


def test_listed_statuses_and_plain_numbers_are_clean(tmp_path: Path) -> None:
    listed = STATUS_LIST.format(extra="При неверном запросе метод отвечает кодом 400.")
    numbers = STATUS_LIST.format(extra="Ответ содержит до 500 записей и не больше 1 404 строк.")
    counts = STATUS_LIST.format(
        extra="Ответ содержит не более 500 позиций.\n\nМетод возвращает до 404 пользователей."
    )
    for doc in (listed, numbers, counts):
        assert not by_rule(run(tmp_path, doc), "RULE-API-STATUS-UNLISTED")


def test_status_enumeration_and_table_column(tmp_path: Path) -> None:
    several = STATUS_LIST.format(
        extra="При успехе метод отвечает кодом 200 / 201, при отказе — статус 400, 409."
    )
    found = by_rule(run(tmp_path, several), "RULE-API-STATUS-UNLISTED")
    assert sorted(f.explanation.split()[2] for f in found) == ["201", "409"]
    table = (
        "# Склад\n\n## Ошибки\n\n"
        "| Ошибка | HTTP | Описание |\n| --- | --- | --- |\n"
        "| E001 | 404 | Позиции нет |\n| E002 | 409 | Позиция занята |\n"
        "| E003 | 422 | Неверная масса |\n\n"
        "Если позиции нет, метод отвечает кодом 404, а на прочее — кодом 410.\n"
    )
    found = by_rule(run(tmp_path, table), "RULE-API-STATUS-UNLISTED")
    assert len(found) == 1 and "410" in found[0].explanation


def test_short_status_list_disables_rule(tmp_path: Path) -> None:
    doc = "# Склад\n\n## Статусы ответа\n\n**200.** Позиция найдена.\n\nИначе код 404.\n"
    assert not by_rule(run(tmp_path, doc), "RULE-API-STATUS-UNLISTED")


# --- RULE-API-TYPE-MISMATCH -----------------------------------------------------

TYPES_DOC = """# Склад

## Метод списания

### Запрос

| Параметр | Тип | Описание |
| --- | --- | --- |
| `weight` | number | Масса позиции |
| `active` | boolean | Признак учёта |

### Ответ

| Параметр | Тип | Описание |
| --- | --- | --- |
| `weight` | {weight} | Масса позиции |
| `active` | boolean | Признак учёта |
"""


def test_same_field_with_different_types(tmp_path: Path) -> None:
    found = by_rule(run(tmp_path, TYPES_DOC.format(weight="string")), "RULE-API-TYPE-MISMATCH")
    assert len(found) == 1
    assert found[0].defect_type is DefectType.INCONSISTENT_TERMS
    assert len(found[0].quotes) == 2


def test_same_types_synonyms_and_other_entities_are_clean(tmp_path: Path) -> None:
    entities = (
        "# Склад\n\n## Сущности\n\n### Позиции\n\n"
        "| Поле | Тип |\n| --- | --- |\n| `id` | integer |\n\n"
        "### Накладные\n\n| Поле | Тип |\n| --- | --- |\n| `id` | uuid |\n"
    )
    for doc in (TYPES_DOC.format(weight="number"), TYPES_DOC.format(weight="float"), entities):
        assert not by_rule(run(tmp_path, doc), "RULE-API-TYPE-MISMATCH")


# --- RULE-API-SPEC-* ------------------------------------------------------------

SPEC_HEAD = "# Склад\n\n## Методы\n\n- **Путь:** `/stock/items`\n- **Путь:** `/stock/audit`\n\n"
SPEC_OK = (
    "openapi: 3.0.3\ninfo:\n  title: Stock\npaths:\n"
    "  /stock/items:\n    get:\n      summary: List\n"
)
SPEC_FLAT = (
    "openapi: 3.0.3\ninfo:\ntitle: Stock\npaths:\n/stock/items:\nget:\nsummary: List\n"
    "/stock/audit:\nget:\nsummary: Audit\n"
)


def _spec_doc(body: str) -> str:
    return f"{SPEC_HEAD}## Спецификация\n\n{FENCE}yaml\n{body}\n{FENCE}\n"


def test_path_missing_from_spec(tmp_path: Path) -> None:
    found = run(tmp_path, _spec_doc(SPEC_OK))
    missing = by_rule(found, "RULE-API-SPEC-PATH")
    assert len(missing) == 1
    assert "/stock/audit" in missing[0].explanation
    assert missing[0].defect_type is DefectType.MISSING_REQUIREMENT
    assert not by_rule(found, "RULE-API-SPEC-STRUCTURE")


def test_spec_without_indentation(tmp_path: Path) -> None:
    found = run(tmp_path, _spec_doc(SPEC_FLAT))
    assert len(by_rule(found, "RULE-API-SPEC-STRUCTURE")) == 1
    assert not by_rule(found, "RULE-API-SPEC-PATH")


def test_server_prefix_and_foreign_yaml_are_clean(tmp_path: Path) -> None:
    prefixed = (
        "openapi: 3.0.3\nservers:\n  - url: /stock\npaths:\n"
        "  /items:\n    get:\n      summary: List\n  /audit:\n    get:\n      summary: Audit\n"
    )
    pipeline = "artifacts:\n  paths:\n    - build/\n    - dist/\ncache:\n  key: stock\n"
    for body in (prefixed, pipeline):
        found = run(tmp_path, _spec_doc(body))
        assert not by_rule(found, "RULE-API-SPEC-PATH")
        assert not by_rule(found, "RULE-API-SPEC-STRUCTURE")
    found = run(tmp_path, SPEC_HEAD)
    assert not by_rule(found, "RULE-API-SPEC-PATH")
    assert not by_rule(found, "RULE-API-SPEC-STRUCTURE")


def test_foreign_phrase_and_same_code_unlisted_is_one_finding(tmp_path: Path) -> None:
    doc = STATUS_LIST.format(
        extra="HTTP-код 200 — Created, если позиция есть.\n\nПри успехе метод отвечает кодом 201."
    )
    found = run(tmp_path, doc)
    phrase = by_rule(found, "RULE-API-STATUS-PHRASE")
    assert len(phrase) == 1 and len(phrase[0].quotes) == 2
    assert not by_rule(found, "RULE-API-STATUS-UNLISTED")
