"""Профиль домена (FR-19) и адресаты вопросов (FR-17)."""

from pathlib import Path

import pytest

from reqcheck import pipeline, profiles
from reqcheck.errors import InputError
from reqcheck.models import DefectType, RequirementKind
from reqcheck.prompts_open import open_system
from reqcheck.stages.render import render_markdown
from reqcheck.stages.render_tables import ROLE_ARCHITECT, ROLE_PRODUCT, ROLE_QA, question_role

DOC = """# Копилка

## Правила

Система должна ограничивать пополнение копилки: лимит 300 000 единиц за сутки.

Система должна показывать остаток копилки на главном экране.
"""


def _doc(tmp_path: Path, text: str = DOC) -> Path:
    path = tmp_path / "doc.md"
    path.write_text(text, encoding="utf-8")
    return path


def _bank_findings(report) -> list:
    return [f for f in report.findings if f.rule_id == "RULE-BANK-PERIOD-TIMEZONE"]


def test_bank_rule_only_under_flag(tmp_path: Path) -> None:
    plain = pipeline.build(_doc(tmp_path), None)
    assert not _bank_findings(plain) and plain.passport.profile is None
    bank = pipeline.build(_doc(tmp_path), None, profile=profiles.load("bank"))
    found = _bank_findings(bank)
    assert len(found) == 1 and "за сутки" in found[0].quotes[0].text
    assert found[0].defect_type is DefectType.MISSING_REQUIREMENT
    assert bank.passport.profile == "банк"
    assert "| Профиль домена | банк |" in render_markdown(bank)


def test_bank_rule_silent_when_timezone_is_named(tmp_path: Path) -> None:
    text = DOC + "\nСутки считаются по часовому поясу банка, обнуление в 00:00.\n"
    report = pipeline.build(_doc(tmp_path, text), None, profile=profiles.load("bank"))
    assert not _bank_findings(report)


def test_bank_rule_silent_on_plain_period_words(tmp_path: Path) -> None:
    for line in (
        "Система хранит остаток на счёте и выводит его ежедневно в личном кабинете.",
        "Ежедневно в 09:00 отправляется отчёт об остатках на счёте руководителю.",
        "Пользователь видит итоговую сумму в календарном месяце оформления заявки.",
    ):
        report = pipeline.build(
            _doc(tmp_path, f"# Копилка\n\n{line}\n"), None, profile=profiles.load("bank")
        )
        assert not _bank_findings(report)


def test_profile_text_reaches_only_open_prompt_and_unknown_name_fails() -> None:
    bank = profiles.load("bank")
    assert bank is not None and profiles.load(None) is None
    assert "Профиль домена: банк" in open_system("<a>", "</a>", profile=bank.checklist)
    assert "Профиль домена" not in open_system("<a>", "</a>")
    with pytest.raises(InputError, match="Неизвестный профиль"):
        profiles.load("медицина")


class _F:
    def __init__(self, defect_type: DefectType, ids: tuple[str, ...]) -> None:
        self.defect_type, self.requirement_ids = defect_type, ids
        self.suggestion = None


def test_question_role_by_type_and_requirement_kind() -> None:
    kinds = {
        "R-001": RequirementKind.FUNCTIONAL,
        "R-002": RequirementKind.INTERFACE,
        "R-003": RequirementKind.ACCEPTANCE,
    }
    assert question_role(_F(DefectType.CONTRADICTION, ("R-001",)), kinds) == ROLE_PRODUCT
    assert question_role(_F(DefectType.CONTRADICTION, ("R-001", "R-003")), kinds) == ROLE_PRODUCT
    assert question_role(_F(DefectType.INCOMPLETE, ("R-002",)), kinds) == ROLE_ARCHITECT
    assert question_role(_F(DefectType.INTERFACE_MISUSE, ()), kinds) == ROLE_ARCHITECT
    assert question_role(_F(DefectType.UNVERIFIABLE, ("R-001",)), kinds) == ROLE_QA
    assert question_role(_F(DefectType.INCOMPLETE, ("R-003",)), kinds) == ROLE_QA
    assert question_role(_F(DefectType.MISSING_REQUIREMENT, ()), kinds) == ROLE_PRODUCT
