"""Контрольный прогон «голая модель»: тот же документ, та же модель, один простой запрос.

Нужен, чтобы отделить вклад агента от силы модели. Без реестра, правил, проверки
цитат, справочника серьёзности и узких проверок набора — только модель и промпт,
который написал бы человек без нашей архитектуры.

Запуск (провайдер и модель — из тех же переменных, что у агента):

    uv run python eval/baseline.py <документ.md> <отчёт.md>
"""

import sys
from pathlib import Path

from pydantic import BaseModel

from reqcheck.llm_client import create_client_from_env

SYSTEM = (
    "Ты аналитик требований. Проверь документ с требованиями по ISO/IEC/IEEE 29148 "
    "и IEEE 830: найди дефекты (неоднозначность, непроверяемость, неполнота, "
    "противоречия, пропуски). Для каждого дефекта дай цитату из документа, описание "
    "и серьёзность (blocker, critical, major, minor). Пиши по-русски."
)


class NaiveFinding(BaseModel):
    quote: str
    problem: str
    severity: str


class NaiveResponse(BaseModel):
    findings: list[NaiveFinding] = []


def render(findings: list[NaiveFinding], model: str, cost: float) -> str:
    lines = [
        "# Контрольный прогон: голая модель",
        "",
        f"Модель: {model}. Стоимость: ${cost:.2f}. Один запрос, без архитектуры агента.",
        "",
    ]
    for n, f in enumerate(findings, start=1):
        quote = " ".join(f.quote.split())
        lines += [f"## B-{n:03d} — {f.severity}", "", f"> {quote}", "", f.problem.strip(), ""]
    return "\n".join(lines)


def main() -> int:
    source, target = Path(sys.argv[1]), Path(sys.argv[2])
    llm = create_client_from_env()
    if llm is None:
        print("Ключ модели не задан.", file=sys.stderr)
        return 2
    text = source.read_text(encoding="utf-8")
    result = llm.complete(system=SYSTEM, user=text, response_model=NaiveResponse)
    usage = llm.usage()
    cost = getattr(usage, "cost_usd", 0.0)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render(result.data.findings, usage.model or "?", cost), encoding="utf-8")
    print(f"замечаний: {len(result.data.findings)}, стоимость: ${cost:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
