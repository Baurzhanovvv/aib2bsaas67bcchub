"""Точка входа: reqcheck <путь к документу> <путь к отчёту> (FR-1)."""

import argparse
import os
import sys
from pathlib import Path

from reqcheck import __version__, pipeline, profiles
from reqcheck.errors import ExitCode, InputError, ReqcheckError
from reqcheck.llm_client import create_client_from_env

DEBUG_ENV = "REQCHECK_DEBUG"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="reqcheck",
        description="Статическая проверка документа с требованиями. Отчёт пишется в файл.",
    )
    parser.add_argument(
        "input", type=Path, help="путь к документу с требованиями (PDF, DOCX, Markdown)"
    )
    parser.add_argument(
        "output", type=Path, help="путь к файлу отчёта: .md, .html или .pdf (формат по расширению)"
    )
    parser.add_argument("--quiet", action="store_true", help="не печатать ход работы")
    parser.add_argument(
        "--profile",
        choices=profiles.names(),
        help="профиль домена: дополнительные проверки (bank — банковские требования)",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="прогон только правилами, без обращения к модели",
    )
    parser.add_argument("--version", action="version", version=f"reqcheck {__version__}")
    return parser


def _check_paths(source: Path, output: Path) -> None:
    if not source.exists():
        raise InputError(f"Файл не найден: {source}")
    if not source.is_file():
        raise InputError(f"Это не файл: {source}")
    if output.exists() and output.is_dir():
        raise InputError(f"Путь отчёта — каталог, нужен путь к файлу: {output}")
    if output.resolve() == source.resolve():
        raise InputError("Путь отчёта совпадает с входным документом — документ был бы затёрт.")


def _progress(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def _silent(_: str) -> None:
    return None


def _debug() -> bool:
    return os.environ.get(DEBUG_ENV, "").strip().lower() in {"1", "true", "yes"}


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _check_paths(args.input, args.output)
        llm = None if args.no_llm else create_client_from_env()
        if llm is None:
            print(
                "Ключ модели не задан (GEMINI_API_KEY) — прогон только правилами, "
                "проверка набора не выполняется.",
                file=sys.stderr,
            )
        progress = _silent if args.quiet else _progress
        profile = profiles.load(args.profile)
        return int(
            pipeline.run(args.input, args.output, llm=llm, progress=progress, profile=profile)
        )
    except ReqcheckError as error:
        print(f"Ошибка: {error.message}", file=sys.stderr)
        return int(error.exit_code)
    except KeyboardInterrupt:
        print("Прервано пользователем.", file=sys.stderr)
        return 130
    except Exception as error:  # noqa: BLE001 — пользователю не показываем traceback (NFR-6)
        if _debug():
            raise
        print(
            f"Внутренняя ошибка: {type(error).__name__}: {error}\n"
            f"Подробности: запустите с переменной {DEBUG_ENV}=1.",
            file=sys.stderr,
        )
        return int(ExitCode.FAILED)


if __name__ == "__main__":
    sys.exit(main())
