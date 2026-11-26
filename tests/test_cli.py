"""CLI и связка конвейера: коды выхода, сообщения без traceback, запись отчёта."""

from pathlib import Path

import pytest

from reqcheck import pipeline
from reqcheck.cli import main
from reqcheck.errors import ExitCode
from reqcheck.models import GroundingResult, Verdict
from reqcheck.stages import checks, grounding, reader, registry, render, report, rules
from tests.factories import make_document, make_report, make_requirement


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "doc.md"
    path.write_text("# Требования\n\nСистема должна работать быстро.\n", encoding="utf-8")
    return path


def _stub_stages(monkeypatch: pytest.MonkeyPatch, verdict: Verdict) -> None:
    monkeypatch.setattr(reader, "read_document", lambda path: make_document())
    monkeypatch.setattr(registry, "build_registry", lambda doc, llm: (make_requirement(),))
    monkeypatch.setattr(rules, "apply_rules", lambda doc, reqs: [])
    monkeypatch.setattr(grounding, "verify_and_classify", lambda doc, reqs, raw: GroundingResult())
    monkeypatch.setattr(report, "build_report", lambda *args, **kwargs: make_report(verdict))
    monkeypatch.setattr(render, "render_markdown", lambda rep: "# Отчёт\n")


def test_missing_file_is_reported_without_traceback(tmp_path: Path, capsys) -> None:
    code = main([str(tmp_path / "нет.md"), str(tmp_path / "out.md")])

    err = capsys.readouterr().err
    assert code == ExitCode.FAILED
    assert "Файл не найден" in err
    assert "Traceback" not in err


def test_directory_as_input_is_rejected(tmp_path: Path, capsys) -> None:
    assert main([str(tmp_path), str(tmp_path / "out.md")]) == ExitCode.FAILED
    assert "не файл" in capsys.readouterr().err


def test_report_path_equal_to_input_is_rejected(source: Path, capsys) -> None:
    code = main([str(source), str(source)])

    assert code == ExitCode.FAILED
    assert "затёрт" in capsys.readouterr().err
    assert source.read_text(encoding="utf-8").startswith("# Требования")


def test_report_path_that_is_a_directory_is_rejected(source: Path, tmp_path: Path, capsys) -> None:
    assert main([str(source), str(tmp_path)]) == ExitCode.FAILED
    assert "каталог" in capsys.readouterr().err


def test_unexpected_error_is_reported_without_traceback(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    def boom(path: Path) -> None:
        raise RuntimeError("сломалось")

    monkeypatch.delenv("REQCHECK_DEBUG", raising=False)
    monkeypatch.setattr(reader, "read_document", boom)

    code = main([str(source), str(tmp_path / "out.md")])

    err = capsys.readouterr().err
    assert code == ExitCode.FAILED
    assert "Внутренняя ошибка" in err
    assert "Traceback" not in err


def test_run_writes_report_and_creates_folder(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_stages(monkeypatch, Verdict.GO)
    output = tmp_path / "reports" / "nested" / "report.md"

    code = main([str(source), str(output)])

    assert code == ExitCode.OK
    assert output.read_text(encoding="utf-8") == "# Отчёт\n"


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        (Verdict.GO, ExitCode.OK),
        (Verdict.CONDITIONAL_GO, ExitCode.OK),
        (Verdict.NO_GO, ExitCode.NO_GO),
    ],
)
def test_exit_code_follows_verdict(
    source: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    verdict: Verdict,
    expected: ExitCode,
) -> None:
    _stub_stages(monkeypatch, verdict)

    assert pipeline.run(source, tmp_path / "out.md", llm=None) == expected


def test_model_stages_are_skipped_without_client(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_stages(monkeypatch, Verdict.GO)

    def must_not_run(*args: object) -> None:
        raise AssertionError("этап с моделью вызван без клиента")

    monkeypatch.setattr(checks, "check_requirements", must_not_run)
    monkeypatch.setattr(checks, "check_set", must_not_run)

    assert pipeline.run(source, tmp_path / "out.md", llm=None) == ExitCode.OK
