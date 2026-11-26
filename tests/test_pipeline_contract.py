"""Конвейер и этапы не расходятся: сигнатуры, порядок, деградация при сбое модели.

Этапы пишут разные люди параллельно. Если кто-то поменяет сигнатуру своего
этапа, этот тест упадёт сразу, а не на финальном прогоне.
"""

import inspect
from pathlib import Path

import pytest

from reqcheck import pipeline
from reqcheck.errors import InputError, ModelError
from reqcheck.llm import LLMUsage
from reqcheck.models import GroundingResult, TraceReport, Verdict
from reqcheck.stages import checks, grounding, reader, registry, report, rules
from tests.factories import make_document, make_report, make_requirement

EXPECTED_PARAMS = {
    reader.read_document: ["path"],
    registry.build_registry: ["document", "llm"],
    rules.apply_rules: ["document", "requirements"],
    checks.check_requirements: ["requirements", "llm"],
    checks.check_set: ["document", "requirements", "llm"],
    grounding.verify_and_classify: ["document", "requirements", "raw"],
    report.build_report: [
        "document",
        "requirements",
        "grounded",
        "trace",
        "usage",
        "duration_s",
        "model_error",
    ],
}


@pytest.mark.parametrize("stage", list(EXPECTED_PARAMS), ids=lambda f: f.__qualname__)
def test_stage_signature_matches_pipeline(stage) -> None:
    assert list(inspect.signature(stage).parameters) == EXPECTED_PARAMS[stage]


class FakeClient:
    def complete(self, **kwargs):  # pragma: no cover — этапы с моделью подменены
        raise AssertionError("не должен вызываться напрямую")

    def usage(self) -> LLMUsage:
        return LLMUsage(model="fake-model", requests=3)


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    seen: dict[str, object] = {}
    monkeypatch.setattr(reader, "read_document", lambda path: make_document())
    monkeypatch.setattr(registry, "build_registry", lambda doc, llm: (make_requirement(),))
    monkeypatch.setattr(rules, "apply_rules", lambda doc, reqs: [])
    monkeypatch.setattr(checks, "check_requirements", lambda reqs, llm: [])
    monkeypatch.setattr(checks, "check_set", lambda doc, reqs, llm: ([], TraceReport()))
    monkeypatch.setattr(grounding, "verify_and_classify", lambda doc, reqs, raw: GroundingResult())

    def fake_build_report(document, requirements, grounded, trace, usage, duration_s, **kw):
        seen.update(trace=trace, usage=usage, model_error=kw.get("model_error"))
        return make_report(Verdict.GO)

    monkeypatch.setattr(report, "build_report", fake_build_report)
    return seen


def test_full_run_passes_usage_and_trace(calls: dict[str, object]) -> None:
    pipeline.build(Path("doc.md"), FakeClient())

    assert calls["usage"] == LLMUsage(model="fake-model", requests=3)
    assert calls["trace"] == TraceReport()
    assert calls["model_error"] is None


def test_without_client_usage_is_none(calls: dict[str, object]) -> None:
    pipeline.build(Path("doc.md"), None)

    assert calls["usage"] is None
    assert calls["trace"] is None


def test_model_failure_degrades_instead_of_crashing(
    calls: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    def quota(doc, reqs, llm):
        raise ModelError("исчерпана квота модели")

    monkeypatch.setattr(checks, "check_set", quota)

    pipeline.build(Path("doc.md"), FakeClient())

    assert calls["model_error"] == "исчерпана квота модели"
    assert calls["trace"] is None


def test_registry_model_failure_falls_back_to_code_only(
    calls: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    def registry_stage(doc, llm):
        if llm is not None:
            raise ModelError("сеть недоступна")
        return (make_requirement(),)

    monkeypatch.setattr(registry, "build_registry", registry_stage)

    pipeline.build(Path("doc.md"), FakeClient())

    assert calls["model_error"] == "сеть недоступна"


def test_unwritable_report_path_is_a_user_error(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")

    with pytest.raises(InputError, match="Не удалось записать отчёт"):
        pipeline._write_atomically(blocker / "report.md", "# Отчёт\n")
