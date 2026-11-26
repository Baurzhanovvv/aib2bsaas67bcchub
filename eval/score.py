"""Eval-команда (FR-13): прогон агента на документе и сверка с золотым набором.

    uv run python eval/score.py <документ> <золотой_набор.json> [--no-llm] [--min-recall 0.7]

Печатает «найдено k из n», пропущенные пункты и замечания вне набора. Код выхода 1,
если доля найденного ниже порога; 2 — если не читается документ или набор. Золотой
набор выданного документа в репозиторий не кладётся; формат — в reqcheck.evalkit,
пример — tests/test_evalkit.py.
"""

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from reqcheck import evalkit, pipeline
from reqcheck.errors import ReqcheckError
from reqcheck.llm_client import create_client_from_env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Сверка отчёта агента с золотым набором")
    parser.add_argument("document", type=Path)
    parser.add_argument("golden", type=Path)
    parser.add_argument("--no-llm", action="store_true", help="только правила, бесплатно")
    parser.add_argument("--min-recall", type=float, default=0.0)
    args = parser.parse_args(argv)
    try:
        golden = evalkit.load_golden(args.golden)
        llm = None if args.no_llm else create_client_from_env()
        report = pipeline.build(args.document, llm)
    except (OSError, ValueError, ValidationError, ReqcheckError) as error:
        print(f"Ошибка: {error}", file=sys.stderr)
        return 2
    result = evalkit.score(report, golden)
    print(evalkit.render(result, report))
    return 0 if result.recall >= args.min_recall else 1


if __name__ == "__main__":
    sys.exit(main())
