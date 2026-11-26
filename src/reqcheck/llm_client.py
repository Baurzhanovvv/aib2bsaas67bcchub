"""Клиент модели: Gemini + фолбэк на OpenAI-совместимый endpoint + дисковый кэш.

Реализует протокол reqcheck.llm.LLMClient (FR-8, NFR-3, NFR-4). Транспорт вынесен
в «провайдеры» с одним методом generate — в тестах подменяется фейком.
"""

import json
import math
import os
import random
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlparse

from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, ValidationError

from reqcheck.errors import ModelError
from reqcheck.llm import LLMClient, LLMResult, LLMUsage
from reqcheck.llm_cache import CacheEntry, DiskCache, make_cache_key
from reqcheck.llm_providers import (
    EFFORTS,
    AnthropicProvider,
    GeminiProvider,
    OpenAICompatProvider,
    Provider,
    ProviderError,
    ProviderReply,
)

T = TypeVar("T", bound=BaseModel)

DEFAULT_MODELS = "gemini-3.1-flash-lite,gemini-3.5-flash-lite,gemini-3.8-flash"
DEFAULT_CACHE_DIR = ".reqcheck_cache"
TRANSIENT_RETRIES = 2
BACKOFF_BASE_S = 2.0
BACKOFF_BUDGET_S = 30.0
RETRY_AFTER_CAP_S = 20.0
PROVIDER_BUDGET_S = 90.0
STRIKES_TO_SKIP = 3
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
DEFAULT_MAX_COST_USD = 2.0

# Цены, USD за 1 млн токенов: (вход, выход, чтение из кэша). Запись в кэш — вход × 1.25.
PRICES: dict[str, tuple[float, float, float]] = {
    "claude-opus-5-5": (4.0, 20.0, 0.20),
    "claude-sonnet-5-5": (2.0, 10.0, 0.10),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}
CACHE_WRITE_FACTOR = 1.25


_WARNED: set[str] = set()


def _warn_once(text: str) -> None:
    if text not in _WARNED:
        _WARNED.add(text)
        print(f"Предупреждение: {text}.", file=sys.stderr)


def reply_cost_usd(reply: ProviderReply) -> float:
    """Стоимость одного ответа; модель вне таблицы (Gemini free, фолбэк) считается бесплатной."""
    price = next((v for k, v in PRICES.items() if reply.model.startswith(k)), None)
    if price is None:
        if not reply.model.startswith("claude"):
            return 0.0
        price = max(PRICES.values())  # неизвестная claude-модель — по самой дорогой
        _warn_once(f"нет цены для модели {reply.model}, стоимость считается по самой дорогой")
    p_in, p_out, p_read = price
    fresh = max(0, reply.tokens_in - reply.cache_read_tokens - reply.cache_write_tokens)
    total = (
        fresh * p_in
        + reply.cache_write_tokens * p_in * CACHE_WRITE_FACTOR
        + reply.cache_read_tokens * p_read
        + reply.tokens_out * p_out
    )
    return total / 1_000_000


_MAX_ERROR_CHARS = 1500


class GeminiClient:
    """Основной провайдер + необязательный фолбэк + необязательный кэш."""

    def __init__(
        self,
        primary: Provider | Sequence[Provider],
        *,
        fallback: Provider | None = None,
        cache: DiskCache | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: random.Random | None = None,
        clock: Callable[[], float] = time.monotonic,
        max_cost_usd: float | None = None,
    ) -> None:
        self._max_cost = max_cost_usd
        self._cost = 0.0
        self._primaries: list[Provider] = (
            list(primary) if isinstance(primary, Sequence) else [primary]
        )
        self._primary = self._primaries[0]
        self._clock = clock
        self._skip: set[int] = set()
        self._strikes: dict[int, int] = {}
        self._models: list[str] = []
        self._fallback = fallback
        self._cache = cache
        self._sleep = sleep
        self._rng = rng or random.Random()
        self._lock = threading.Lock()
        self._usage = {
            "model": None,
            "requests": 0,
            "cache_hits": 0,
            "tokens_in": 0,
            "tokens_out": 0,
            "invalid_items": 0,
        }

    def usage(self) -> LLMUsage:
        with self._lock:
            usage = {
                **self._usage,
                "model": ",".join(self._models) or None,
                "cost_usd": round(self._cost, 6),
            }
            return LLMUsage(**usage)  # type: ignore[arg-type]

    def complete(self, *, system: str, user: str, response_model: type[T]) -> LLMResult[T]:
        schema = response_model.model_json_schema()
        key = make_cache_key(
            provider=self._primary.name,
            model=",".join(getattr(p, "cache_tag", p.model) for p in self._primaries),
            system=system,
            user=user,
            schema=json.dumps(schema, ensure_ascii=False, sort_keys=True),
        )
        hit = self._cache_lookup(key, response_model)
        if hit is not None:
            return hit
        reasons: list[str] = []
        chain = [*self._primaries, *([self._fallback] if self._fallback else [])]
        live = [p for p in chain if id(p) not in self._skip] or chain
        for index, provider in enumerate(live):
            try:
                result = self._run_provider(
                    provider, system, user, schema, response_model, last=index == len(live) - 1
                )
            except ProviderError as exc:
                reasons.append(f"{provider.model}: {exc.reason}")
                self._register_failure(provider, exc)
                continue
            self._strikes[id(provider)] = 0
            if self._cache is not None and provider in self._primaries and result[1] is not None:
                self._cache.put(key, CacheEntry(text=result[1], model=result[0].model))
            return result[0]
        raise ModelError(self._failure_text(reasons))

    def _register_failure(self, provider: Provider, exc: ProviderError) -> None:
        """Дневная квота или серия сбоев подряд — модель в этом прогоне больше не пробуем."""
        strikes = self._strikes.get(id(provider), 0) + 1
        self._strikes[id(provider)] = strikes
        if exc.exhausted or strikes >= STRIKES_TO_SKIP:
            self._skip.add(id(provider))

    # --- кэш ---

    def _cache_lookup(self, key: str, response_model: type[T]) -> LLMResult[T] | None:
        entry = self._cache.get(key) if self._cache is not None else None
        if entry is None:
            return None
        try:
            data, dropped = validate_lenient(response_model, entry.text)
        except ValidationError:
            return None
        with self._lock:
            self._usage["invalid_items"] += dropped
            self._usage["cache_hits"] += 1
            self._note_model(entry.model)
        return LLMResult(data=data, model=entry.model, cached=True)

    # --- вызов провайдера ---

    def _run_provider(
        self,
        provider: Provider,
        system: str,
        user: str,
        schema: dict[str, Any],
        response_model: type[T],
        *,
        last: bool,
    ) -> tuple[LLMResult[T], str | None]:
        """Запрос с повтором при невалидном ответе; ProviderError — провайдер не помог."""
        prompt = user
        started = self._clock()
        for attempt in (1, 2):
            reply = self._call_with_retries(provider, system, prompt, schema, started, last)
            try:
                data, dropped = validate_lenient(response_model, reply.text)
            except ValidationError as exc:
                if attempt == 2:
                    raise ProviderError(
                        f"модель дважды вернула ответ не по схеме ({_short_validation(exc)[:300]})",
                        retryable=False,
                    ) from None
                prompt = (
                    f"{user}\n\nТвой предыдущий ответ не прошёл проверку схемы:\n"
                    f"{_short_validation(exc)}\nВерни исправленный JSON строго по схеме."
                )
                continue
            with self._lock:
                self._usage["invalid_items"] += dropped
            result = LLMResult(
                data=data, model=reply.model, tokens_in=reply.tokens_in, tokens_out=reply.tokens_out
            )
            return result, reply.text
        raise AssertionError("недостижимо")  # pragma: no cover

    def _call_with_retries(
        self,
        provider: Provider,
        system: str,
        user: str,
        schema: dict[str, Any],
        started: float,
        last: bool,
    ) -> ProviderReply:
        """Повторы с паузой — только у последнего в цепочке; остальные сразу уступают следующему."""
        slept = 0.0
        retries = TRANSIENT_RETRIES if last else 0
        for retry in range(retries + 1):
            self._check_cost_limit()
            if self._clock() - started > PROVIDER_BUDGET_S:
                raise ProviderError("превышено время ожидания ответа модели", retryable=False)
            try:
                reply = provider.generate(system=system, user=user, schema=schema)
            except ProviderError as exc:
                self._count_request(exc.billed, note_model=False)
                delay = self._backoff(exc, retry, slept)
                if not exc.retryable or retry == retries or delay is None:
                    raise
                slept += delay
                self._sleep(delay)
                continue
            self._count_request(reply)
            return reply
        raise AssertionError("недостижимо")  # pragma: no cover

    def _backoff(self, exc: ProviderError, retry: int, slept: float) -> float | None:
        """Пауза перед повтором; None — ждать нельзя (retry-after дольше лимита)."""
        if exc.retry_after is not None:
            return exc.retry_after if exc.retry_after <= RETRY_AFTER_CAP_S else None
        delay = BACKOFF_BASE_S * (2**retry) * (0.5 + self._rng.random())
        remaining = BACKOFF_BUDGET_S - slept
        return min(delay, remaining) if remaining > 0 else None

    def _check_cost_limit(self) -> None:
        if self._max_cost is not None and self._cost >= self._max_cost:
            raise ModelError(
                f"Превышен лимит расходов на прогон ${self._max_cost:.2f} "
                f"(израсходовано ${self._cost:.2f}). Повысьте REQCHECK_MAX_COST_USD "
                "или смените модель."
            )

    def _note_model(self, model: str) -> None:
        if model not in self._models:
            self._models.append(model)

    def _count_request(
        self, reply: ProviderReply | None = None, *, note_model: bool = True
    ) -> None:
        with self._lock:
            self._usage["requests"] += 1
            if reply is not None:
                if note_model:
                    self._note_model(reply.model)
                self._usage["tokens_in"] += reply.tokens_in
                self._usage["tokens_out"] += reply.tokens_out
                self._cost += reply_cost_usd(reply)

    @staticmethod
    def _failure_text(reasons: list[str]) -> str:
        detail = "; ".join(reasons)
        return (
            f"Модель недоступна ({detail}). "
            "Проверьте ключ GEMINI_API_KEY, квоту и сеть либо настройте резервный провайдер "
            "(REQCHECK_FALLBACK_BASE_URL, REQCHECK_FALLBACK_API_KEY, REQCHECK_FALLBACK_MODEL)."
        )


def validate_lenient(response_model: type[T], text: str) -> tuple[T, int]:
    """Разобрать ответ; битые элементы списков отбросить, если целых не меньше.

    Одно замечание с выдуманным типом не должно стоить всего ответа: слабые модели
    ошибаются в одном поле. Сломан конверт или большинство элементов — ValidationError,
    и вызывающий переспрашивает модель. Возвращает данные и число отброшенных элементов.
    """
    try:
        return response_model.model_validate_json(text), 0
    except ValidationError as exc:
        try:
            data = json.loads(text)
        except ValueError:
            raise exc from None
        bad: dict[str, set[int]] = {}
        for error in exc.errors():
            loc = error["loc"]
            in_list = len(loc) >= 2 and isinstance(loc[1], int) and isinstance(data, dict)
            if not in_list or not isinstance(data.get(loc[0]), list):
                raise
            bad.setdefault(str(loc[0]), set()).add(loc[1])
        dropped = sum(len(indexes) for indexes in bad.values())
        if dropped > sum(len(data[field]) for field in bad) - dropped:
            raise
        for field, indexes in bad.items():
            data[field] = [item for i, item in enumerate(data[field]) if i not in indexes]
        return response_model.model_validate(data), dropped


def _short_validation(exc: ValidationError) -> str:
    return str(exc)[:_MAX_ERROR_CHARS]


def create_client_from_env() -> LLMClient | None:
    """Клиент по переменным окружения и .env; нет ключа выбранного провайдера — None."""
    load_dotenv(find_dotenv(usecwd=True), override=False)
    provider = os.environ.get("REQCHECK_PROVIDER", "").strip().lower() or "gemini"
    if provider not in ("gemini", "anthropic"):
        raise ModelError("REQCHECK_PROVIDER должен быть gemini или anthropic.")
    key_name = "GEMINI_API_KEY" if provider == "gemini" else "ANTHROPIC_API_KEY"
    api_key = os.environ.get(key_name, "").strip()
    if not api_key:
        return None
    default = DEFAULT_MODELS if provider == "gemini" else DEFAULT_ANTHROPIC_MODEL
    raw_models = os.environ.get("REQCHECK_MODEL", "").strip() or default
    models = _models_for(provider, raw_models, default)
    effort = os.environ.get("REQCHECK_EFFORT", "").strip().lower() or "medium"
    if effort not in EFFORTS:
        print(
            f"Предупреждение: REQCHECK_EFFORT={effort!r} неизвестен, взято medium.", file=sys.stderr
        )
        effort = "medium"
    primary: list[Provider] = [
        GeminiProvider(api_key, m)
        if provider == "gemini"
        else AnthropicProvider(api_key, m, effort)
        for m in models
    ]
    cache = None
    if os.environ.get("REQCHECK_NO_CACHE", "").strip().lower() not in ("1", "true", "yes"):
        cache = DiskCache(
            Path(os.environ.get("REQCHECK_CACHE_DIR", "").strip() or DEFAULT_CACHE_DIR)
        )
    return GeminiClient(
        primary, fallback=_fallback_from_env(), cache=cache, max_cost_usd=_max_cost_from_env()
    )


def _models_for(provider: str, raw: str, default: str) -> list[str]:
    """Список моделей из REQCHECK_MODEL; имена чужого провайдера отбрасываем с предупреждением."""
    names = list(dict.fromkeys(m.strip() for m in raw.split(",") if m.strip()))
    mine = [m for m in names if m.startswith("claude") == (provider == "anthropic")]
    if len(mine) != len(names):
        print(
            f"Предупреждение: в REQCHECK_MODEL есть модели не провайдера {provider} — пропущены.",
            file=sys.stderr,
        )
    return mine or [m.strip() for m in default.split(",")]


def _max_cost_from_env() -> float:
    raw = os.environ.get("REQCHECK_MAX_COST_USD", "").strip()
    if not raw:
        return DEFAULT_MAX_COST_USD
    try:
        value = float(raw)
    except ValueError:
        value = float("nan")
    if not (math.isfinite(value) and value > 0):
        print(
            "Предупреждение: REQCHECK_MAX_COST_USD должен быть положительным числом, взято 2.0.",
            file=sys.stderr,
        )
        return DEFAULT_MAX_COST_USD
    return value


def _fallback_from_env() -> Provider | None:
    base = os.environ.get("REQCHECK_FALLBACK_BASE_URL", "").strip()
    key = os.environ.get("REQCHECK_FALLBACK_API_KEY", "").strip()
    model = os.environ.get("REQCHECK_FALLBACK_MODEL", "").strip()
    if not (base and key and model):
        return None
    try:
        provider = OpenAICompatProvider(base, key, model)
    except ModelError as exc:
        print(f"Предупреждение: резервный провайдер отключён. {exc.message}", file=sys.stderr)
        return None
    if urlparse(base).hostname not in ("localhost", "127.0.0.1", "::1") and base.lower().startswith(
        "http://"
    ):
        print(
            "Предупреждение: резервный провайдер использует http:// без шифрования — "
            "ключ уйдёт по сети открытым текстом.",
            file=sys.stderr,
        )
    return provider
