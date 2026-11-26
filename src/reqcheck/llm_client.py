"""Клиент модели: Gemini + фолбэк на OpenAI-совместимый endpoint + дисковый кэш.

Реализует протокол reqcheck.llm.LLMClient (FR-8, NFR-3, NFR-4). Транспорт вынесен
в «провайдеры» с одним методом generate — в тестах подменяется фейком.
"""

import json
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
    ) -> None:
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
        }

    def usage(self) -> LLMUsage:
        with self._lock:
            usage = {**self._usage, "model": ",".join(self._models) or None}
            return LLMUsage(**usage)  # type: ignore[arg-type]

    def complete(self, *, system: str, user: str, response_model: type[T]) -> LLMResult[T]:
        schema = response_model.model_json_schema()
        key = make_cache_key(
            provider=self._primary.name,
            model=",".join(p.model for p in self._primaries),
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
            data = response_model.model_validate_json(entry.text)
        except ValidationError:
            return None
        with self._lock:
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
                data = response_model.model_validate_json(reply.text)
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
            if self._clock() - started > PROVIDER_BUDGET_S:
                raise ProviderError("превышено время ожидания ответа модели", retryable=False)
            try:
                reply = provider.generate(system=system, user=user, schema=schema)
            except ProviderError as exc:
                self._count_request()
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

    def _note_model(self, model: str) -> None:
        if model not in self._models:
            self._models.append(model)

    def _count_request(self, reply: ProviderReply | None = None) -> None:
        with self._lock:
            self._usage["requests"] += 1
            if reply is not None:
                self._note_model(reply.model)
                self._usage["tokens_in"] += reply.tokens_in
                self._usage["tokens_out"] += reply.tokens_out

    @staticmethod
    def _failure_text(reasons: list[str]) -> str:
        detail = "; ".join(reasons)
        return (
            f"Модель недоступна ({detail}). "
            "Проверьте ключ GEMINI_API_KEY, квоту и сеть либо настройте резервный провайдер "
            "(REQCHECK_FALLBACK_BASE_URL, REQCHECK_FALLBACK_API_KEY, REQCHECK_FALLBACK_MODEL)."
        )


def _short_validation(exc: ValidationError) -> str:
    return str(exc)[:_MAX_ERROR_CHARS]


def create_client_from_env() -> LLMClient | None:
    """Клиент по переменным окружения и .env; нет GEMINI_API_KEY — None (режим без модели)."""
    load_dotenv(find_dotenv(usecwd=True), override=False)
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return None
    raw_models = os.environ.get("REQCHECK_MODEL", "").strip() or DEFAULT_MODELS
    models = list(dict.fromkeys(m.strip() for m in raw_models.split(",") if m.strip()))
    primary = [GeminiProvider(api_key, m) for m in models]
    fallback = _fallback_from_env()
    cache = None
    if os.environ.get("REQCHECK_NO_CACHE", "").strip().lower() not in ("1", "true", "yes"):
        cache = DiskCache(
            Path(os.environ.get("REQCHECK_CACHE_DIR", "").strip() or DEFAULT_CACHE_DIR)
        )
    return GeminiClient(primary, fallback=fallback, cache=cache)


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
