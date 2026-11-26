"""Eval по золотому набору (FR-13) и форматы отчёта (FR-22), прогресс (FR-23)."""

import json
from pathlib import Path

import pymupdf

from reqcheck import evalkit, pipeline
from reqcheck.errors import ExitCode

DOC = """# Склад

## Метод списания

| Ключ | Что |
| --- | --- |
| Назначение | Удаление позиции со склада |
| HTTP method | GET |
| Адрес | `/stock/item` |

### Запрос

| Параметр | Тип | Описание |
| --- | --- | --- |
| `weight` | number | Масса позиции |

### Ответ

| Параметр | Тип | Описание |
| --- | --- | --- |
| `weight` | string | Масса позиции |
"""

GOLDEN = {
    "expected": [
        {"id": "G1", "title": "GET меняет состояние", "quote_contains": ["удаление позиции"]},
        {"id": "G2", "title": "тип массы", "quote_contains": ["`weight` | string"]},
        {"id": "G3", "title": "нет прав доступа", "quote_contains": ["права доступа"]},
        {"id": "G4", "quote_contains": ["HTTP method"], "types": ["ambiguous"]},
    ]
}


def _doc(tmp_path: Path) -> Path:
    path = tmp_path / "stock.md"
    path.write_text(DOC, encoding="utf-8")
    return path


def test_score_counts_found_missed_and_extra(tmp_path: Path) -> None:
    report = pipeline.build(_doc(tmp_path), None)
    result = evalkit.score(report, evalkit.Golden.model_validate(GOLDEN))
    assert sorted(result.found) == ["G1", "G2"]
    assert [item.id for item in result.missed] == ["G3", "G4"]
    assert result.recall == 0.5
    text = evalkit.render(result, report)
    assert "Найдено 2 из 4 (50%)" in text and "- G3: не найдено — нет прав доступа" in text


def test_whole_paragraph_quote_does_not_count(tmp_path: Path) -> None:
    report = pipeline.build(_doc(tmp_path), None)
    long_quote = report.findings[0].quotes[0].model_copy(update={"text": "масса " + "х" * 500})
    finding = report.findings[0].model_copy(update={"quotes": (long_quote,)})
    item = evalkit.Expected(id="G", quote_contains=["масса"])
    assert not evalkit._hits(finding, item)
    short = finding.model_copy(
        update={"quotes": (long_quote.model_copy(update={"text": "Масса позиции"}),)}
    )
    assert evalkit._hits(short, item)


def test_score_script_exit_codes(tmp_path: Path, capsys) -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "score", Path(__file__).parents[1] / "eval" / "score.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    golden = tmp_path / "golden.json"
    golden.write_text(json.dumps(GOLDEN, ensure_ascii=False), encoding="utf-8")
    doc = str(_doc(tmp_path))
    assert module.main([doc, str(golden), "--no-llm", "--min-recall", "0.5"]) == 0
    assert module.main([doc, str(golden), "--no-llm", "--min-recall", "0.9"]) == 1
    assert module.main([doc, str(tmp_path / "нет.json"), "--no-llm"]) == 2
    assert "Найдено 2 из 4" in capsys.readouterr().out


def test_html_and_pdf_reports_and_progress(tmp_path: Path) -> None:
    doc = _doc(tmp_path)
    steps: list[str] = []
    assert pipeline.run(doc, tmp_path / "r.html", None, steps.append) is ExitCode.OK
    html = (tmp_path / "r.html").read_text(encoding="utf-8")
    assert html.startswith("<!DOCTYPE html>") and "<blockquote>" in html
    assert 'class="critical"' in html and "<script" not in html
    assert steps[0].startswith("[1/6]") and steps[-1].startswith("[6/6]")
    pipeline.run(doc, tmp_path / "r.pdf", None)
    pdf = pymupdf.open(tmp_path / "r.pdf")
    assert "Отчёт о проверке требований" in pdf[0].get_text()
