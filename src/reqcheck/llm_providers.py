"""Транспорт к моделям: Gemini и OpenAI-совместимый фолбэк (FR-8).

Провайдер — объект с методом generate; ошибки приводятся к ProviderError с безопасным
текстом (без ключей и сырых сообщений SDK).
"""

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol

from reqcheck.errors import ModelError

TEMPERATURE = 0.1
REQUEST_TIMEOUT_S = 60.0
MAX_RESPONSE_BYTES = 10 * 1024 * 1024
_MAX_ERROR_CHARS = 1500


@dataclass(frozen=True)
class ProviderReply:
    text: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0


class ProviderError(Exception):
    """Сбой транспорта. reason — безопасный текст для пользователя (без ключей)."""

    def __init__(
        self,
        reason: str,
        *,
        retryable: bool,
        retry_after: float | None = None,
        exhausted: bool = False,
    ) -> None:
        super().__init__(reason)
        self.reason = reason
        self.retryable = retryable
        self.retry_after = retry_after
        self.exhausted = exhausted
        """Дневная квота модели исчерпана: в этом прогоне её больше не пробуем."""


class Provider(Protocol):
    name: str
    model: str

    def generate(self, *, system: str, user: str, schema: dict[str, Any]) -> ProviderReply: ...


def _redact(text: str, secret: str) -> str:
    cleaned = text.replace(secret, "***") if secret else text
    return cleaned[:_MAX_ERROR_CHARS]


def _classify_status(
    code: int, message: str, retry_after: float | None, *, daily: bool = False
) -> ProviderError:
    if code in (401, 403) or (code == 400 and "api key" in message.lower()):
        return ProviderError("ключ API неверен или не имеет доступа", retryable=False)
    if code == 429:
        return ProviderError(
            "исчерпана квота или превышен лимит запросов (429)",
            retryable=True,
            retry_after=retry_after,
            exhausted=daily,
        )
    if code == 408 or code >= 500:
        return ProviderError(
            f"сервис модели временно недоступен ({code})", retryable=True, retry_after=retry_after
        )
    return ProviderError(f"запрос отклонён сервисом модели ({code})", retryable=False)


def _parse_delay(value: object) -> float | None:
    """Секунды из '12', '12.5' или '34s' (формат RetryInfo); иначе None."""
    try:
        return max(0.0, float(str(value).strip().rstrip("s")))
    except ValueError:
        return None


GEMINI_API_VERSION = "v1beta"
"""JSON-режим (response_json_schema) есть только в v1beta; SDK по умолчанию ходит в v1 → 400."""


def _gemini_http_options() -> Any:
    """Явная версия API и без встроенных повторов SDK: повторы и фолбэк делает GeminiClient.

    Иначе SDK сам повторяет 503 с паузами, и одна «попытка» клиента длится минуту.
    """
    from google.genai import types

    return types.HttpOptions(
        api_version=GEMINI_API_VERSION,
        timeout=int(REQUEST_TIMEOUT_S * 1000),
        retry_options=types.HttpRetryOptions(attempts=1),
    )


class GeminiProvider:
    name = "gemini"

    def __init__(self, api_key: str, model: str) -> None:
        from google import genai

        self.model = model
        self._key = api_key
        self._client = genai.Client(api_key=api_key, http_options=_gemini_http_options())

    def generate(self, *, system: str, user: str, schema: dict[str, Any]) -> ProviderReply:
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=TEMPERATURE,
            response_mime_type="application/json",
            response_json_schema=simplify_schema(schema),
            http_options=_gemini_http_options(),
        )
        try:
            resp = self._client.models.generate_content(
                model=self.model, contents=user, config=config
            )
        except errors.APIError as exc:
            raise _classify_status(
                int(exc.code or 0),
                _redact(str(exc.message or ""), self._key),
                _gemini_retry_after(exc),
                daily=_is_daily_quota(exc),
            ) from None
        except Exception:  # noqa: BLE001 — сеть, таймаут, SSL — детали SDK могут содержать URL с ключом
            raise ProviderError("нет связи с сервисом модели (сеть)", retryable=True) from None
        usage = resp.usage_metadata
        return ProviderReply(
            text=resp.text or "",
            model=resp.model_version or self.model,
            tokens_in=(usage.prompt_token_count or 0) if usage else 0,
            tokens_out=((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0))
            if usage
            else 0,
        )


def _is_daily_quota(exc: Any) -> bool:
    """429 из-за суточной квоты модели (идентификатор квоты содержит PerDay)."""
    if int(exc.code or 0) != 429:
        return False
    text = f"{exc.message} {exc.details}".lower()
    return "perday" in text or "per day" in text or "daily" in text


_DROPPED_SCHEMA_KEYS = {"title", "default", "$defs", "examples"}


def simplify_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Схема pydantic → совместимая со всеми моделями: без $ref/$defs, title и default."""
    defs = schema.get("$defs", {})

    def walk(node: Any, depth: int = 0) -> Any:
        if isinstance(node, list):
            return [walk(item, depth) for item in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node and depth < 20:
            target = defs.get(str(node["$ref"]).rsplit("/", 1)[-1], {})
            return walk({**target, **{k: v for k, v in node.items() if k != "$ref"}}, depth + 1)
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key in _DROPPED_SCHEMA_KEYS:
                continue
            if key == "properties" and isinstance(value, dict):  # имена полей не фильтруем
                out[key] = {name: walk(sub, depth) for name, sub in value.items()}
            else:
                out[key] = walk(value, depth)
        return out

    result = walk(schema)
    return result if isinstance(result, dict) else schema


def _gemini_retry_after(exc: Any) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if headers is not None:
        try:
            delay = _parse_delay(headers.get("retry-after"))
        except AttributeError:
            delay = None
        if delay is not None:
            return delay
    details = (
        exc.details.get("error", {}).get("details", []) if isinstance(exc.details, dict) else []
    )
    for item in details:
        if isinstance(item, dict) and "retryDelay" in item:
            return _parse_delay(item["retryDelay"])
    return None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Не ходим по редиректам: заголовок Authorization не должен уехать на другой хост."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


_OPENER = urllib.request.build_opener(_NoRedirect)


class OpenAICompatProvider:
    name = "openai-compat"

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        if not base_url.lower().startswith(("http://", "https://")):
            raise ModelError("REQCHECK_FALLBACK_BASE_URL должен начинаться с http:// или https://.")
        self.model = model
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._key = api_key

    def generate(self, *, system: str, user: str, schema: dict[str, Any]) -> ProviderReply:
        schema_text = json.dumps(schema, ensure_ascii=False)
        body = {
            "model": self.model,
            "temperature": TEMPERATURE,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": f"{user}\n\nОтветь одним JSON-объектом строго по этой "
                    f"JSON-схеме, без пояснений и markdown:\n{schema_text}",
                },
            ],
        }
        request = urllib.request.Request(  # noqa: S310 — схема http(s) проверена в конструкторе
            self._url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self._key}"},
            method="POST",
        )
        try:
            with _OPENER.open(request, timeout=REQUEST_TIMEOUT_S) as raw:
                data = raw.read(MAX_RESPONSE_BYTES + 1)
            if len(data) > MAX_RESPONSE_BYTES:
                raise ProviderError(
                    "резервный сервис вернул слишком большой ответ", retryable=False
                )
            payload = json.loads(data.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if 300 <= exc.code < 400:
                raise ProviderError(
                    "резервный сервис ответил редиректом — переходы не поддерживаются, "
                    "укажите итоговый URL",
                    retryable=False,
                ) from None
            retry_after = _parse_delay(exc.headers.get("Retry-After")) if exc.headers else None
            raise _classify_status(exc.code, "", retry_after) from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise ProviderError(
                "нет связи с резервным сервисом модели (сеть)", retryable=True
            ) from None
        except ValueError:  # включая UnicodeDecodeError
            raise ProviderError("резервный сервис вернул не JSON", retryable=False) from None
        return _parse_openai_payload(payload, self.model)


def _parse_openai_payload(payload: Any, default_model: str) -> ProviderReply:
    try:
        text = payload["choices"][0]["message"]["content"] or ""
        if not isinstance(text, str):
            raise TypeError("content")
        model = str(payload.get("model") or default_model)
        usage = payload.get("usage")
    except (KeyError, IndexError, TypeError, AttributeError):
        raise ProviderError(
            "резервный сервис вернул ответ неожиданного формата", retryable=False
        ) from None
    return ProviderReply(
        text=text,
        model=model,
        tokens_in=_safe_count(usage, "prompt_tokens"),
        tokens_out=_safe_count(usage, "completion_tokens"),
    )


def _safe_count(usage: object, field: str) -> int:
    """Токены из usage; всё нечисловое или отрицательное — 0."""
    if not isinstance(usage, dict):
        return 0
    value = usage.get(field)
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0
