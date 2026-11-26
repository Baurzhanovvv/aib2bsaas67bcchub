"""FR-12: в коде, промптах, правилах и тестах не должно быть фрагментов выданного ТЗ.

Запуск: REQCHECK_ISSUED_DOC=/путь/к/ТЗ.md uv run pytest tests/test_no_overfit.py
(несколько документов — через os.pathsep). Без переменной тест пропускается.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests import overfit_guard as g

ROOT = Path(__file__).resolve().parent.parent


def _load() -> g.Signatures:
    docs = g.load_issued_docs()
    if not docs:
        pytest.skip(f"не задана переменная {g.ENV_VAR}: нет выданного ТЗ для сверки")
    missing = [str(p) for p in docs if not p.is_file()]
    if missing:
        pytest.skip(f"{g.ENV_VAR}: файл не найден: {', '.join(missing)}")
    sig = g.merge([g.extract_signatures(p.read_text(encoding="utf-8")) for p in docs])
    if not sig:
        pytest.skip("из выданного ТЗ не извлечено ни одного отличительного признака")
    return sig


def test_no_issued_doc_fragments_in_repo() -> None:
    sig = _load()
    hits = g.scan_tree(ROOT, sig)
    assert not hits, "Найдены фрагменты выданного ТЗ (подгонка, FR-12):\n" + "\n".join(
        f"  {h}" for h in hits
    )


def test_scanner_catches_injected_fragments(tmp_path: Path) -> None:
    """Самопроверка: вставленные в файл признаки из ТЗ сканер обязан найти."""
    sig = _load()
    ident = sorted(sig.identifiers)[0]
    number = sorted(sig.numbers)[0]
    shingle = " ".join(sorted(sig.shingles)[0])
    code_dir = tmp_path / "src"
    code_dir.mkdir()
    cases = {
        "ident.py": f'ENDPOINT = "{ident}"\n',
        "number.md": f"порог {f'{int(number):,}'.replace(',', ' ')} единиц\n",
        "shingle.txt": f"# {shingle}\n",
        "wrapped.txt": "\n".join(shingle.split(" ", 2)) + "\n",
    }
    for name, body in cases.items():
        (code_dir / name).write_text(body, encoding="utf-8")
    for i, extra in enumerate(sorted(sig.identifiers)):
        name = f"id_{i}.py"
        cases[name] = f"X = '{extra}'\n"
        (code_dir / name).write_text(cases[name], encoding="utf-8")
    found = {h.path.name for h in g.scan_tree(tmp_path, sig)}
    assert found == set(cases), f"сканер пропустил: {set(cases) - found}"


def test_scanner_ignores_generic_text(tmp_path: Path) -> None:
    sig = _load()
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "ok.md").write_text(
        "Система должна возвращать время ответа сервиса при обращении к реестру "
        "складских остатков в случае если партия найдена.\n",
        encoding="utf-8",
    )
    assert g.scan_tree(tmp_path, sig) == []
