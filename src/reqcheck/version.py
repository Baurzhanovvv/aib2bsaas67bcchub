"""Версия агента для паспорта прогона: тег или короткий хеш коммита."""

import subprocess
from pathlib import Path

from reqcheck import __version__

GIT_TIMEOUT_S = 5.0
PACKAGE_DIR = Path(__file__).resolve().parent


def _git(*args: str) -> str | None:
    """Вывод git в каталоге пакета; None, если git недоступен или это не репозиторий."""
    try:
        done = subprocess.run(
            ["git", *args],
            cwd=PACKAGE_DIR,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.strip()


def agent_version() -> tuple[str, bool]:
    """Вернуть (версия, есть ли незакоммиченные изменения).

    Версия — `git describe --tags --always`; вне git-репозитория — __version__.
    Незакоммиченным считается любой непустой `git status --porcelain` репозитория,
    в том числе новые файлы.
    """
    described = _git("describe", "--tags", "--always")
    if not described:
        return __version__, False
    status = _git("status", "--porcelain")
    return described, bool(status)
