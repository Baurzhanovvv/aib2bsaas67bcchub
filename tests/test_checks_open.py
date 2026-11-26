"""Свободный проход FR-6: весь документ, все типы, реестр справочно."""

from reqcheck import prompts, prompts_open
from reqcheck.llm import FindingDraft, LinkDraft, SetCheckResponse
from reqcheck.models import DefectType, RequirementKind, Scope
from reqcheck.stages import checks
from tests.test_checks import FakeLLM, document, req

OPEN = "Задача: свободный проход по всему документу"
EXTRA = "Цель проекта — сократить простой погрузчиков на складе."


def test_open_pass_is_last_call_and_keeps_any_type_without_ids() -> None:
    reqs = (req(1), req(2, RequirementKind.ACCEPTANCE))
    loose = FindingDraft(
        defect_type=DefectType.UNTRACEABLE,
        quotes=("сократить простой погрузчиков",),
        explanation="Цель не связана ни с одним правилом: нельзя проверить, что она достигнута.",
        suggestion="Вопрос заказчику: какой показатель простоя считать целевым?",
    )
    link = LinkDraft(acceptance_id="R-002", covers=("R-001",))
    reply = SetCheckResponse(findings=(loose,), links=(link,))
    llm = FakeLLM(by_focus={OPEN: reply})
    result, trace = checks.check_set(document(reqs, extra=EXTRA), reqs, llm)
    assert len(llm.calls) == 4 and OPEN in llm.calls[-1][0]
    assert [f.defect_type for f in result] == [DefectType.UNTRACEABLE]
    assert result[0].requirement_ids == () and result[0].scope is Scope.REQUIREMENT_SET
    assert trace.links == ()


def test_open_prompt_shares_rules_and_wraps_document() -> None:
    system = prompts_open.open_system("<<<DOC-x>>>", "<<<END-DOC-x>>>")
    assert prompts.PROMPT_VERSION in system
    assert "<<<DOC-x>>>" in system and "Встроенная инструкция для ИИ" in system
    assert "Вопрос заказчику: A или B?" in system
    assert "противоречит другим твоим замечаниям" in system
    for defect in DefectType:
        assert f"- {defect.value} —" in system
    user = prompts_open.open_user(
        "<<<DOC-x>>>R<<<END-DOC-x>>>", "<<<DOC-x>>>T<<<END-DOC-x>>>", part=None
    )
    assert user.index("T<<<") < user.index("R<<<")


def test_unknown_value_rule_in_every_prompt() -> None:
    systems = [
        prompts.requirement_system("<<<DOC-x>>>", "<<<END-DOC-x>>>"),
        prompts_open.open_system("<<<DOC-x>>>", "<<<END-DOC-x>>>"),
        *(
            prompts.set_system("<<<DOC-x>>>", "<<<END-DOC-x>>>", focus=f)
            for f in prompts.SET_FOCUSES
        ),
    ]
    for system in systems:
        assert "НЕ выбирай и не\n  придумывай его" in system
        assert "однозначно следует из самого\n  документа" in system
