"""Тесты детерминированных правил (FR-14). Тексты синтетические."""

import pytest

from reqcheck.models import (
    DefectType,
    Document,
    DocumentFormat,
    FindingSource,
    Fragment,
    FragmentKind,
    Location,
    RawFinding,
    Requirement,
    Scope,
)
from reqcheck.stages.rules import apply_rules, make_quote


def build(
    texts: list[str],
    source_ids: list[str | None] | None = None,
    kind: FragmentKind = FragmentKind.PARAGRAPH,
) -> tuple[Document, tuple[Requirement, ...]]:
    fragments, reqs = [], []
    for i, text in enumerate(texts, start=1):
        loc = Location(line_start=i, line_end=i)
        fragments.append(Fragment(id=f"F-{i:04d}", kind=kind, text=text, location=loc))
        reqs.append(
            Requirement(
                id=f"R-{i:03d}",
                source_id=source_ids[i - 1] if source_ids else None,
                summary="требование",
                quote=text,
                location=loc,
                fragment_id=f"F-{i:04d}",
            )
        )
    doc = Document(
        name="d.md", sha256="0" * 64, format=DocumentFormat.MARKDOWN, fragments=tuple(fragments)
    )
    return doc, tuple(reqs)


def run(texts: list[str], **kw: object) -> list[RawFinding]:
    doc, reqs = build(texts, **kw)  # type: ignore[arg-type]
    return apply_rules(doc, reqs)


def by_rule(findings: list[RawFinding], rule_id: str) -> list[RawFinding]:
    return [f for f in findings if f.rule_id == rule_id]


def check_invariants(texts: list[str], findings: list[RawFinding]) -> None:
    for f in findings:
        assert f.source is FindingSource.RULE and f.rule_id
        assert f.explanation and f.suggestion
        text = texts[int(f.requirement_ids[0][2:]) - 1]
        for q in f.quotes:
            assert len(q) >= 8
            assert q in text


# --- VAGUE --------------------------------------------------------------------


def test_vague_single_finding_for_many_words() -> None:
    texts = ["Склад должен быстро обрабатывать заказы и быть удобным, при необходимости и т. д."]
    found = by_rule(run(texts), "RULE-VAGUE")
    assert len(found) == 1
    assert found[0].defect_type is DefectType.UNVERIFIABLE
    assert found[0].scope is Scope.REQUIREMENT
    assert len(found[0].quotes) >= 3
    check_invariants(texts, found)


def test_vague_short_word_gets_neighbours_in_quote() -> None:
    texts = ["Склад должен работать быстро."]
    (f,) = by_rule(run(texts), "RULE-VAGUE")
    assert f.quotes == ("работать быстро",)


def test_vague_skipped_when_number_with_unit_in_sentence() -> None:
    assert not by_rule(run(["Заказ обрабатывается быстро: не более 5 секунд."]), "RULE-VAGUE")
    assert not by_rule(run(["Отклик оперативный, до 200 мс при нагрузке."]), "RULE-VAGUE")
    assert not by_rule(run(["Ответ быстрее 1 500 ms на 95% запросов."]), "RULE-VAGUE")


def test_vague_number_in_other_sentence_does_not_exempt() -> None:
    texts = ["Склад хранит 500 коробок. Выдача должна быть оперативной."]
    assert len(by_rule(run(texts), "RULE-VAGUE")) == 1


def test_vague_open_list_is_ambiguous() -> None:
    texts = ["Поддерживаются форматы PDF, DOCX и др. по возможности."]
    (f,) = by_rule(run(texts), "RULE-VAGUE")
    assert f.defect_type is DefectType.AMBIGUOUS


def test_vague_min_max_only_without_digits() -> None:
    assert by_rule(run(["Размер партии не должен быть максимальным для клиента."]), "RULE-VAGUE")
    assert not by_rule(run(["Максимальный размер партии — 40 коробок."]), "RULE-VAGUE")


def test_vague_english_words() -> None:
    texts = ["The portal must be user-friendly and respond fast, etc."]
    (f,) = by_rule(run(texts), "RULE-VAGUE")
    check_invariants(texts, [f])


def test_vague_no_false_positives_on_clean_text() -> None:
    clean = [
        "Врач вносит диагноз не позднее 24 часов после приёма.",
        "Если на счёте достаточно средств, списание выполняется.",
        "Система быстрых переводов не используется.",
    ]
    assert not by_rule(run(clean), "RULE-VAGUE")


def test_vague_same_text_gives_finding_per_requirement() -> None:
    texts = ["Приложение должно быть удобным для курьера."] * 2
    found = by_rule(run(texts), "RULE-VAGUE")
    assert [f.requirement_ids for f in found] == [("R-001",), ("R-002",)]


def test_vague_skipped_in_code_fragment() -> None:
    assert not run(["fast = True  # etc. TODO"], kind=FragmentKind.CODE)


# --- TBD ----------------------------------------------------------------------


def test_tbd_markers() -> None:
    texts = ["Срок хранения карточки сотрудника: TBD, лимит уточняется ???"]
    (f,) = by_rule(run(texts), "RULE-TBD")
    assert f.defect_type is DefectType.INCOMPLETE
    assert len(f.quotes) == 3
    check_invariants(texts, [f])


def test_tbd_will_be_defined_and_ellipsis() -> None:
    assert by_rule(run(["Размер штрафа будет определён позднее."]), "RULE-TBD")
    assert by_rule(run(["Список ролей: врач, медсестра, …"]), "RULE-TBD")


def test_tbd_not_a_word_part() -> None:
    assert not by_rule(run(["Поле TODOLIST хранит список задач сотрудника."]), "RULE-TBD")


def test_tbd_empty_cell_in_table_row() -> None:
    texts = ["| Кабинет | Врач | | Вторник |"]
    (f,) = by_rule(run(texts, kind=FragmentKind.TABLE_ROW), "RULE-TBD")
    assert f.quotes == (texts[0],)


def test_tbd_empty_last_cell_and_separator_are_ignored() -> None:
    assert not by_rule(
        run(["| Кабинет | Врач | Примечание | "], kind=FragmentKind.TABLE_ROW), "RULE-TBD"
    )
    assert not by_rule(run(["| :--- | :--- | :--- |"], kind=FragmentKind.TABLE_ROW), "RULE-TBD")


# --- HTTP ---------------------------------------------------------------------


def test_http_get_with_mutation_is_flagged() -> None:
    texts = ["GET /patients/update — изменить телефон пациента."]
    (f,) = by_rule(run(texts), "RULE-HTTP-SAFE-METHOD")
    assert f.defect_type is DefectType.INTERFACE_MISUSE
    check_invariants(texts, [f])


def test_http_get_with_identifier_verb() -> None:
    texts = ["REST type: GET, метод /orders/deleteOrder."]
    assert by_rule(run(texts), "RULE-HTTP-SAFE-METHOD")


def test_http_get_reading_deleted_items_is_not_flagged() -> None:
    texts = [
        "GET /orders/archive — получить список удалённых заказов.",
        "GET /orders — вернуть созданные сегодня заказы.",
        "POST /orders — создать заказ.",
        "Кнопка GET STARTED удаляет черновик.",
    ]
    assert not by_rule(run(texts), "RULE-HTTP-SAFE-METHOD")


def test_http_error_with_200_is_flagged() -> None:
    texts = ['Ответ 200: {"status": "ERROR", "message": "нет места"}']
    (f,) = by_rule(run(texts), "RULE-HTTP-ERROR-200")
    assert f.defect_type is DefectType.INTERFACE_MISUSE
    check_invariants(texts, [f])


def test_http_error_200_table_row() -> None:
    texts = ["| 200 | При ошибке валидации возвращается описание |"]
    assert by_rule(run(texts, kind=FragmentKind.TABLE_ROW), "RULE-HTTP-ERROR-200")


@pytest.mark.parametrize(
    "text",
    [
        "Ответ 200 при успехе; при ошибке ввода возвращается 400.",
        "Штраф за ошибку водителя составляет 200 рублей.",
        "Статус ответа 200, ошибок нет, время ответа 2 с.",
        "Код 2000 означает ошибку склада.",
    ],
)
def test_http_error_200_false_positive_guards(text: str) -> None:
    assert not by_rule(run([text]), "RULE-HTTP-ERROR-200")


# --- UNDEFINED-REF ------------------------------------------------------------


def test_undefined_ref_flagged_with_set_scope() -> None:
    texts = ["FR-1. Курьер видит заказ.", "FR-2. Статус меняется по FR-9."]
    found = by_rule(run(texts, source_ids=["FR-1", "FR-2"]), "RULE-UNDEFINED-REF")
    assert len(found) == 1
    f = found[0]
    assert f.scope is Scope.REQUIREMENT_SET
    assert f.defect_type is DefectType.UNDEFINED_REFERENCE
    assert f.requirement_ids == ("R-002",)
    assert f.quotes == ("меняется по FR-9",) or all("FR-9" in q for q in f.quotes)
    check_invariants(texts, found)


def test_undefined_ref_defined_and_foreign_prefixes_ok() -> None:
    texts = [
        "FR-1. Курьер видит заказ.",
        "FR-2. Как в FR-1, шифрование по SHA-256 и ISO-8601, версия UTF-8.",
    ]
    assert not by_rule(run(texts, source_ids=["FR-1", "FR-2"]), "RULE-UNDEFINED-REF")


def test_undefined_ref_zero_padding_and_cyrillic_homoglyph() -> None:
    texts = ["Т-01. Базовое.", "Т-02. Уточняет Т-1 и T-1."]
    assert not by_rule(run(texts, source_ids=["T-01", "Т-02"]), "RULE-UNDEFINED-REF")


def test_undefined_ref_own_label_not_flagged_without_source_id() -> None:
    texts = ["**FR-5.** Курьер видит заказ.", "FR-6. Расширяет FR-5."]
    found = by_rule(run(texts, source_ids=[None, "FR-6"]), "RULE-UNDEFINED-REF")
    assert [f.requirement_ids for f in found] == [("R-002",)]


def test_no_registry_ids_means_no_ref_rule() -> None:
    assert not by_rule(run(["Смотри FR-9 и FR-10."]), "RULE-UNDEFINED-REF")


# --- Общее --------------------------------------------------------------------


def test_one_finding_per_rule_per_requirement() -> None:
    texts = ["TBD: быстро, удобно, оперативно, и т. д., TODO, ???"]
    findings = run(texts)
    ids = [f.rule_id for f in findings]
    assert len(ids) == len(set(ids))


def test_passive_and_conjunction_are_not_flagged() -> None:
    texts = ["Заказ должен быть подтверждён менеджером и должен быть отправлен на склад."]
    assert run(texts) == []


def test_make_quote_extends_until_unique() -> None:
    text = "alpha beta. gamma beta. delta"
    quote = make_quote(text, 6, 10)
    assert text.count(quote) == 1
    assert len(quote) >= 8


def test_multiline_sentence_number_on_next_line_exempts() -> None:
    assert not by_rule(run(["Система должна отвечать быстро,\nне дольше 2 с."]), "RULE-VAGUE")


def test_multiline_marker_in_short_tail_is_quoted() -> None:
    texts = ["Сохранение выполняется\nбыстро."]
    (f,) = by_rule(run(texts), "RULE-VAGUE")
    check_invariants(texts, [f])


def test_blank_line_separates_sentences() -> None:
    texts = ["Срок хранения 5 лет.\n\nДоступ должен быть удобным."]
    assert by_rule(run(texts), "RULE-VAGUE")


def test_impact_per_rule() -> None:
    from reqcheck.models import Impact

    assert (
        by_rule(run(["GET /a/update — изменить запись."]), "RULE-HTTP-SAFE-METHOD")[0].impact
        is Impact.HIGH
    )
    assert (
        by_rule(run(['Ответ 200: {"status": "ERROR"}']), "RULE-HTTP-ERROR-200")[0].impact
        is Impact.HIGH
    )
    assert by_rule(run(["Срок: TBD, уточняется."]), "RULE-TBD")[0].impact is Impact.MEDIUM
    refs = run(["FR-1. Базовое.", "FR-2. Как FR-7."], source_ids=["FR-1", "FR-2"])
    assert by_rule(refs, "RULE-UNDEFINED-REF")[0].impact is Impact.MEDIUM
    assert (
        by_rule(run(["Отклик должен приходить быстро."]), "RULE-VAGUE")[0].impact is Impact.MEDIUM
    )
    low = by_rule(run(["Форматы: PDF, DOCX и др."]), "RULE-VAGUE")
    assert low[0].impact is Impact.LOW
