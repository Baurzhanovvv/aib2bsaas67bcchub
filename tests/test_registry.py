"""Тесты реестра требований: код решает состав, модель уточняет kind и summary."""

import pytest

from reqcheck.errors import ModelError
from reqcheck.llm import LLMResult, LLMUsage, RegistryRefinement, RequirementRefinement
from reqcheck.models import (
    Document,
    DocumentFormat,
    Fragment,
    FragmentKind,
    Location,
    RequirementKind,
)
from reqcheck.stages.registry import build_registry

SHA = "1" * 64
K = FragmentKind


def frag(n: int, kind: FragmentKind, text: str, *section: str) -> Fragment:
    return Fragment(
        id=f"F-{n:04d}",
        kind=kind,
        text=text,
        location=Location(line_start=n, line_end=n),
        section=tuple(section),
    )


def doc(*fragments: Fragment) -> Document:
    return Document(name="doc.md", sha256=SHA, format=DocumentFormat.MARKDOWN, fragments=fragments)


class FakeLLM:
    def __init__(self, items: tuple[RequirementRefinement, ...] = (), fail: bool = False) -> None:
        self.items = items
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def complete(self, *, system: str, user: str, response_model: type) -> LLMResult:
        self.calls.append((system, user))
        if self.fail:
            raise ModelError("модель недоступна")
        return LLMResult(data=RegistryRefinement(items=self.items), model="fake")

    def usage(self) -> LLMUsage:
        return LLMUsage()


SAMPLE = doc(
    frag(1, K.HEADING, "Складской модуль", "Складской модуль"),
    frag(2, K.PARAGRAPH, "Документ описывает работу склада.", "Складской модуль"),
    frag(3, K.LIST_ITEM, "ФТ-1. Кладовщик может принять поставку.", "Функции"),
    frag(4, K.LIST_ITEM, "Система должна печатать накладную. Формат — A4.", "Функции"),
    frag(5, K.CODE, "must = True", "Функции"),
)


def test_deterministic() -> None:
    assert build_registry(SAMPLE, None) == build_registry(SAMPLE, None)


def test_ids_follow_document_order_and_copy_fragment_data() -> None:
    reqs = build_registry(SAMPLE, None)
    assert [r.id for r in reqs] == ["R-001", "R-002"]
    assert [r.fragment_id for r in reqs] == ["F-0003", "F-0004"]
    assert reqs[0].quote == "ФТ-1. Кладовщик может принять поставку."
    assert reqs[0].location.line_start == 3


def test_source_id_taken_from_text() -> None:
    reqs = build_registry(SAMPLE, None)
    assert reqs[0].source_id == "ФТ-1"
    assert reqs[1].source_id is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("REQ-12 Отчёт формируется ежедневно.", "REQ-12"),
        ("NFR-3: срок хранения не менее года.", "NFR-3"),
        ("Т-2.1 Список заказов сортируется.", "Т-2.1"),
        ("[R5] Отчёт выгружается в файл.", "R5"),
        ("US-04 Как курьер, я вижу маршрут.", "US-04"),
        ("**AC-1** Заказ виден в списке.", "AC-1"),
    ],
)
def test_common_id_formats(text: str, expected: str) -> None:
    reqs = build_registry(doc(frag(1, K.PARAGRAPH, text)), None)
    assert [r.source_id for r in reqs] == [expected]


def test_encoding_names_are_not_ids() -> None:
    fragments = doc(frag(1, K.PARAGRAPH, "UTF-8 используется для обмена."))
    assert build_registry(fragments, None) == ()


@pytest.mark.parametrize(
    "text",
    [
        "Врач должен подтвердить назначение.",
        "Необходимо хранить историю визитов.",
        "Оператор обязан указать причину.",
        "Требуется журнал изменений.",
        "Удаление не допускается.",
        "Система позволяет искать пациента.",
        "The clinic shall keep records.",
        "Staff must sign in.",
    ],
)
def test_modality_makes_candidate(text: str) -> None:
    assert len(build_registry(doc(frag(1, K.PARAGRAPH, text)), None)) == 1


def test_plain_paragraph_and_heading_are_not_requirements() -> None:
    fragments = doc(
        frag(1, K.HEADING, "Система должна быть быстрой", "Система должна быть быстрой"),
        frag(2, K.PARAGRAPH, "Это общее описание склада и его задач."),
    )
    assert build_registry(fragments, None) == ()


def test_table_rows_in_requirement_section_without_header() -> None:
    fragments = doc(
        frag(1, K.TABLE_ROW, "| Поле | Описание |", "Требования"),
        frag(2, K.TABLE_ROW, "| --- | --- |", "Требования"),
        frag(3, K.TABLE_ROW, "| Вес | Масса посылки в граммах |", "Требования"),
    )
    reqs = build_registry(fragments, None)
    assert [r.fragment_id for r in reqs] == ["F-0003"]
    assert reqs[0].summary == "Вес — Масса посылки в граммах"


def test_endpoint_row_is_interface() -> None:
    fragments = doc(
        frag(1, K.TABLE_ROW, "| POST /parcels | Создаёт посылку |", "Интерфейсы API"),
    )
    assert build_registry(fragments, None)[0].kind is RequirementKind.INTERFACE


def test_acceptance_by_section() -> None:
    fragments = doc(frag(1, K.LIST_ITEM, "Посылка появляется в списке.", "Критерии приёмки"))
    assert build_registry(fragments, None)[0].kind is RequirementKind.ACCEPTANCE


def test_acceptance_by_given_when_then() -> None:
    en = doc(frag(1, K.PARAGRAPH, "Given a parcel, when it is scanned, then status changes."))
    ru = doc(frag(1, K.PARAGRAPH, "Дано посылка. Когда её сканируют, тогда статус меняется."))
    assert build_registry(en, None)[0].kind is RequirementKind.ACCEPTANCE
    assert build_registry(ru, None)[0].kind is RequirementKind.ACCEPTANCE


def test_non_functional_and_default_kind() -> None:
    fragments = doc(
        frag(1, K.PARAGRAPH, "Время отклика должно быть приемлемым."),
        frag(2, K.PARAGRAPH, "Система должна создавать заказ."),
    )
    kinds = [r.kind for r in build_registry(fragments, None)]
    assert kinds == [RequirementKind.NON_FUNCTIONAL, RequirementKind.FUNCTIONAL]


def test_summary_first_sentence_without_id() -> None:
    reqs = build_registry(SAMPLE, None)
    assert reqs[0].summary == "Кладовщик может принять поставку"
    assert reqs[1].summary == "Система должна печатать накладную"


def test_summary_truncated_on_word_boundary() -> None:
    text = "Система должна " + " ".join(["обрабатывать"] * 30)
    summary = build_registry(doc(frag(1, K.PARAGRAPH, text)), None)[0].summary
    assert len(summary) <= 120
    assert summary.endswith("…")
    assert summary[:-1].split()[-1] == "обрабатывать"


def test_llm_refines_kind_and_summary() -> None:
    llm = FakeLLM(
        (RequirementRefinement(id="R-002", kind=RequirementKind.CONSTRAINT, summary="Печать A4"),)
    )
    plain = build_registry(SAMPLE, None)
    refined = build_registry(SAMPLE, llm)
    assert len(llm.calls) == 1
    assert refined[1].kind is RequirementKind.CONSTRAINT
    assert refined[1].summary == "Печать A4"
    assert refined[0] == plain[0]
    assert refined[1].quote == plain[1].quote


def test_llm_unknown_id_is_ignored() -> None:
    llm = FakeLLM(
        (RequirementRefinement(id="R-099", kind=RequirementKind.OTHER, summary="Лишнее"),)
    )
    assert build_registry(SAMPLE, llm) == build_registry(SAMPLE, None)


def test_llm_batches_at_most_three_calls() -> None:
    many = doc(*(frag(n, K.PARAGRAPH, f"Система должна делать шаг {n}.") for n in range(1, 301)))
    llm = FakeLLM()
    reqs = build_registry(many, llm)
    assert len(reqs) == 300
    assert len(llm.calls) == 3


def test_llm_not_called_for_empty_registry() -> None:
    llm = FakeLLM()
    assert build_registry(doc(frag(1, K.HEADING, "Заголовок")), llm) == ()
    assert llm.calls == []


def test_model_error_propagates() -> None:
    with pytest.raises(ModelError):
        build_registry(SAMPLE, FakeLLM(fail=True))


def test_llm_blank_summary_keeps_old_summary_but_takes_kind() -> None:
    llm = FakeLLM(
        (RequirementRefinement(id="R-002", kind=RequirementKind.CONSTRAINT, summary="  "),)
    )
    plain = build_registry(SAMPLE, None)
    refined = build_registry(SAMPLE, llm)
    assert refined[1].summary == plain[1].summary
    assert refined[1].kind is RequirementKind.CONSTRAINT


def test_raw_markdown_table_rows() -> None:
    fragments = doc(
        frag(1, K.TABLE_ROW, "| ID | Требование | Приоритет |", "Требования"),
        frag(2, K.TABLE_ROW, "|---|---|---|", "Требования"),
        frag(3, K.TABLE_ROW, "| FR-1 | Время отклика должно быть малым | |", "Требования"),
    )
    reqs = build_registry(fragments, None)
    assert [r.fragment_id for r in reqs] == ["F-0003"]
    assert reqs[0].source_id == "FR-1"
    assert reqs[0].summary == "Время отклика должно быть малым"
    assert reqs[0].kind is RequirementKind.NON_FUNCTIONAL


def test_multiline_paragraph_summary() -> None:
    text = "Система должна печатать\nнакладную. Формат\nA4."
    req = build_registry(doc(frag(1, K.PARAGRAPH, text)), None)[0]
    assert req.summary == "Система должна печатать накладную"
    assert req.quote == text


def test_source_id_inherited_from_heading() -> None:
    fragments = doc(
        frag(1, K.HEADING, "US-7 Возврат товара", "Истории", "US-7 Возврат товара"),
        frag(2, K.PARAGRAPH, "Покупатель оформляет возврат.", "Истории", "US-7 Возврат товара"),
    )
    reqs = build_registry(fragments, None)
    assert [(r.fragment_id, r.source_id) for r in reqs] == [("F-0002", "US-7")]


def test_id_in_non_first_table_cell() -> None:
    fragments = doc(frag(1, K.TABLE_ROW, "| 1 | **REQ-9** | Отчёт выгружается |", "Требования"))
    assert build_registry(fragments, None)[0].source_id == "REQ-9"


def test_review_checklist_sections_are_skipped() -> None:
    fragments = doc(
        frag(1, K.TABLE_ROW, "| Требование проверено | Да / Нет |", "Ревью требований"),
        frag(2, K.PARAGRAPH, "Согласующий должен подписать.", "Блок согласующих"),
    )
    assert build_registry(fragments, None) == ()


def test_parameter_rows_in_api_section_are_interface() -> None:
    fragments = doc(
        frag(
            1,
            K.TABLE_ROW,
            "| `weight` | integer | Да | Масса в граммах |",
            "Интерфейс складского сервиса (REST)",
        ),
    )
    assert build_registry(fragments, None)[0].kind is RequirementKind.INTERFACE


def test_any_fragment_in_api_section_is_interface() -> None:
    fragments = doc(
        frag(1, K.PARAGRAPH, "Заголовок авторизации должен присутствовать.", "Склад REST"),
        frag(2, K.LIST_ITEM, "Система должна вернуть код ошибки.", "Склад REST", "Ответ"),
        frag(3, K.LIST_ITEM, "Дано токен. Когда запрос, тогда ответ.", "REST API", "Критерии"),
    )
    kinds = [r.kind for r in build_registry(fragments, None)]
    assert kinds == [
        RequirementKind.INTERFACE,
        RequirementKind.INTERFACE,
        RequirementKind.ACCEPTANCE,
    ]


def test_llm_cannot_turn_interface_into_functional() -> None:
    fragments = doc(frag(1, K.PARAGRAPH, "Система должна вернуть код ошибки.", "REST API"))
    llm = FakeLLM(
        (RequirementRefinement(id="R-001", kind=RequirementKind.FUNCTIONAL, summary="Код ошибки"),)
    )
    req = build_registry(fragments, llm)[0]
    assert req.kind is RequirementKind.INTERFACE
    assert req.summary == "Код ошибки"


def test_business_sections_with_method_or_ui_words_are_not_interface() -> None:
    fragments = doc(
        frag(1, K.PARAGRAPH, "Система должна списывать комиссию.", "ТЗ", "Методы оплаты"),
        frag(2, K.PARAGRAPH, "Кнопка должна быть активна.", "ТЗ", "Пользовательский интерфейс"),
        frag(3, K.PARAGRAPH, "Система должна вернуть ошибку.", "ТЗ", "Методы API"),
        frag(4, K.PARAGRAPH, "Система должна вернуть ошибку.", "ТЗ", "Программный интерфейс"),
    )
    kinds = [r.kind for r in build_registry(fragments, None)]
    assert kinds == [
        RequirementKind.FUNCTIONAL,
        RequirementKind.FUNCTIONAL,
        RequirementKind.INTERFACE,
        RequirementKind.INTERFACE,
    ]


def test_regulation_designations_are_not_ids_or_requirements() -> None:
    fragments = doc(
        frag(1, K.TABLE_ROW, "| ФЗ-152 | персональные данные |", "Нормативная база"),
        frag(2, K.PARAGRAPH, "ФЗ-152 регулирует обработку данных.", "Нормативная база"),
    )
    assert build_registry(fragments, None) == ()


def test_document_passport_rows_are_not_requirements_but_content_rows_are() -> None:
    section = ("Интерфейс API", "Общие параметры")
    rows = (
        "| Author | Школа аналитиков |",
        "| Gitlab project | internal |",
        "| Framework | Spring |",
        "| Разработчик | Команда склада |",
        "| Developer | Команда должна согласовать схему |",
        "| Статус | Активен при наличии оплаты и подтверждения |",
        "| Timeout | 30 секунд |",
    )
    fragments = [frag(n, FragmentKind.TABLE_ROW, text, *section) for n, text in enumerate(rows, 1)]
    quotes = [r.quote for r in build_registry(doc(*fragments), None)]
    assert quotes == list(rows[4:])


def test_paragraph_under_requirements_heading_is_requirement() -> None:
    long = "Склад выдаёт груз курьеру за десять минут с момента прихода машины."
    fragments = (
        frag(1, FragmentKind.PARAGRAPH, long, "Склад", "Нефункциональные требования"),
        frag(2, FragmentKind.PARAGRAPH, "заполняется позже", "Склад", "Требования к тестам"),
        frag(3, FragmentKind.PARAGRAPH, long, "Склад", "Схема процесса"),
    )
    assert [r.quote for r in build_registry(doc(*fragments), None)] == [long]
