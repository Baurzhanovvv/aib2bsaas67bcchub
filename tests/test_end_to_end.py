"""Сквозной прогон на настоящих этапах, без подмен внутри конвейера.

Модульные тесты этапов собирают входы руками и однажды уже проверяли формат,
которого соседний этап на деле не выдаёт. Здесь вход — файл, выход — отчёт;
подменена только модель.
"""

from pathlib import Path

import pytest
from pydantic import BaseModel

from reqcheck import pipeline
from reqcheck.errors import ExitCode
from reqcheck.llm import (
    FindingDraft,
    LLMResult,
    LLMUsage,
    RegistryRefinement,
    RequirementCheckResponse,
    SetCheckResponse,
)
from reqcheck.models import DefectType, FindingSource, RunMode, Scope

DOC = """# Сервис записи к врачу

## Функциональные требования

Система должна подтверждать запись
пациенту по SMS
быстро.

| ID | Требование | Срок | Приоритет |
|---|---|---|---|
| ЗП-1 | Пациент выбирает врача и время приёма. | | высокий |
| ЗП-2 | Отмена записи возможна по правилам из ЗП-9. | 1 день | высокий |
"""


@pytest.fixture
def doc_path(tmp_path: Path) -> Path:
    path = tmp_path / "clinic.md"
    path.write_text(DOC, encoding="utf-8")
    return path


def _rule_findings(report, rule_id: str):
    return [f for f in report.findings if f.rule_id == rule_id]


def test_rules_only_run_on_real_stages(doc_path: Path) -> None:
    report = pipeline.build(doc_path, None)

    sources = {r.source_id for r in report.requirements}
    assert {"ЗП-1", "ЗП-2"} <= sources
    assert report.passport.mode is RunMode.RULES_ONLY

    vague = _rule_findings(report, "RULE-VAGUE")
    assert vague, "«быстро» без числа должно дать замечание"
    assert vague[0].quotes[0].location.line_end == 7  # слово стоит на седьмой строке файла

    tbd = _rule_findings(report, "RULE-TBD")
    assert tbd and tbd[0].quotes[0].location.line_start == 11  # строка с пустой ячейкой

    undefined = _rule_findings(report, "RULE-UNDEFINED-REF")
    assert undefined and "ЗП-9" in undefined[0].quotes[0].text


class FakeModel:
    """Отвечает по схеме запроса; одно замечание хорошее, одно битое."""

    def __init__(self) -> None:
        self.requests = 0

    def complete(self, *, system: str, user: str, response_model: type[BaseModel]):
        self.requests += 1
        if response_model is RegistryRefinement:
            data = RegistryRefinement()
        elif response_model is RequirementCheckResponse:
            data = RequirementCheckResponse(
                findings=(
                    FindingDraft(
                        defect_type=DefectType.INCOMPLETE,
                        requirement_ids=("R-002",),
                        quotes=("Пациент выбирает врача и время приёма.",),
                        explanation="Не сказано, как выбирается филиал.",
                    ),
                    FindingDraft(  # битое: короткая цитата, чужой id, нет объяснения
                        defect_type=DefectType.AMBIGUOUS,
                        requirement_ids=("FR-1",),
                        quotes=("быстро",),
                    ),
                )
            )
        else:
            data = SetCheckResponse()
        return LLMResult(data=data, model="fake-model")

    def usage(self) -> LLMUsage:
        return LLMUsage(model="fake-model", requests=self.requests)


def test_model_run_keeps_good_findings_and_drops_broken(doc_path: Path) -> None:
    report = pipeline.build(doc_path, FakeModel())

    assert report.passport.mode is RunMode.FULL
    assert report.passport.model_error is None
    model_findings = [f for f in report.findings if f.source is FindingSource.MODEL]
    assert len(model_findings) == 1
    finding = model_findings[0]
    assert finding.scope is Scope.REQUIREMENT
    assert finding.quotes[0].location.line_start == 11


def test_cli_writes_report_without_model(doc_path: Path, tmp_path: Path) -> None:
    output = tmp_path / "out" / "report.md"

    code = pipeline.run(doc_path, output, llm=None)

    text = output.read_text(encoding="utf-8")
    assert code in (ExitCode.OK, ExitCode.NO_GO)
    assert "## 6. Итог" in text
    assert "строки 6–7" in text  # цитата «SMS быстро.» на двух строках
