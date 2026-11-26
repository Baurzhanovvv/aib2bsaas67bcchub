"""Провайдер Anthropic, стоимость и предохранитель расходов — на фейковом SDK, без сети."""

from pathlib import Path
from types import SimpleNamespace

import anthropic
import httpx2 as httpx
import pytest
from pydantic import BaseModel

from reqcheck.errors import ModelError
from reqcheck.llm_client import GeminiClient, create_client_from_env, reply_cost_usd
from reqcheck.llm_providers import AnthropicProvider, ProviderError, ProviderReply, strict_schema

SECRET = "sk-ant-secret-777"
GOOD = '{"title": "Склад", "priority": 1}'


class Ticket(BaseModel):
    title: str
    priority: int = 0


def sdk_response(
    text: str = GOOD, stop: str = "end_turn", read: int = 0, write: int = 0
) -> SimpleNamespace:
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="..."),
            SimpleNamespace(type="text", text=text),
        ],
        stop_reason=stop,
        model="claude-opus-5-5",
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            cache_read_input_tokens=read,
            cache_creation_input_tokens=write,
        ),
    )


class FakeSDK:
    def __init__(self, *items: object) -> None:
        self.items = list(items)
        self.kwargs: list[dict] = []
        self.messages = self

    def create(self, **kwargs: object) -> object:
        self.kwargs.append(kwargs)
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def provider(*items: object) -> tuple[AnthropicProvider, FakeSDK]:
    prov = AnthropicProvider(SECRET, "claude-opus-5-5", "medium")
    sdk = FakeSDK(*items)
    prov._client = sdk  # type: ignore[assignment]
    return prov, sdk


def status_error(cls: type[anthropic.APIStatusError], code: int) -> anthropic.APIStatusError:
    req = httpx.Request("POST", "https://api.example/v1/messages")
    resp = httpx.Response(code, request=req, headers={"retry-after": "3"})
    return cls(f"boom {SECRET}", response=resp, body=None)


def test_success_request_shape_and_usage() -> None:
    prov, sdk = provider(sdk_response(read=40, write=10))
    reply = prov.generate(system="s", user="u", schema=Ticket.model_json_schema())
    assert reply.text == GOOD and reply.model == "claude-opus-5-5"
    assert (reply.tokens_in, reply.tokens_out) == (150, 50)
    assert (reply.cache_read_tokens, reply.cache_write_tokens) == (40, 10)
    kw = sdk.kwargs[0]
    assert "thinking" not in kw and "temperature" not in kw
    assert kw["cache_control"] == {"type": "ephemeral"}
    assert kw["output_config"]["effort"] == "medium"
    fmt = kw["output_config"]["format"]
    assert fmt["type"] == "json_schema" and fmt["schema"]["additionalProperties"] is False


def test_strict_schema_closes_nested_objects_and_inlines_refs() -> None:
    class Outer(BaseModel):
        title: str
        items: list[Ticket] = []

    out = strict_schema(Outer.model_json_schema())
    text = str(out)
    assert "$ref" not in text and "$defs" not in text
    assert out["additionalProperties"] is False
    assert out["properties"]["items"]["items"]["additionalProperties"] is False
    assert "title" in out["properties"]


def test_refusal_and_truncation_are_non_retryable_but_billed() -> None:
    prov, _ = provider(sdk_response(text="", stop="refusal"), sdk_response(stop="max_tokens"))
    with pytest.raises(ProviderError, match="refusal") as r1:
        prov.generate(system="s", user="u", schema={})
    assert not r1.value.retryable and r1.value.billed is not None
    with pytest.raises(ProviderError, match="обрезан") as r2:
        prov.generate(system="s", user="u", schema={})
    assert not r2.value.retryable


def test_429_is_retryable_with_retry_after_and_secret_not_leaked() -> None:
    prov, _ = provider(status_error(anthropic.RateLimitError, 429))
    with pytest.raises(ProviderError) as info:
        prov.generate(system="s", user="u", schema={})
    assert info.value.retryable and info.value.retry_after == 3
    assert SECRET not in info.value.reason


def test_auth_error_not_retryable() -> None:
    prov, _ = provider(status_error(anthropic.AuthenticationError, 401))
    with pytest.raises(ProviderError, match="ключ") as info:
        prov.generate(system="s", user="u", schema={})
    assert not info.value.retryable


def test_429_moves_to_next_provider() -> None:
    a, _ = provider(status_error(anthropic.RateLimitError, 429))
    b, _ = provider(sdk_response())
    b.model = "claude-sonnet-5-5"
    client = GeminiClient([a, b], sleep=lambda _: None)
    res = client.complete(system="s", user="u", response_model=Ticket)
    assert res.data.title == "Склад"


def test_cost_with_cache() -> None:
    reply = ProviderReply(
        text="",
        model="claude-opus-5-5",
        tokens_in=1_000_000 + 400_000 + 100_000,
        tokens_out=100_000,
        cache_read_tokens=400_000,
        cache_write_tokens=100_000,
    )
    # вход 1M*4 + запись 0.1M*5 + чтение 0.4M*0.2 + выход 0.1M*20 = 4 + 0.5 + 0.08 + 2
    assert reply_cost_usd(reply) == pytest.approx(6.58)
    free = ProviderReply(text="", model="gemini-3.1-flash-lite", tokens_in=10**6, tokens_out=10**6)
    assert reply_cost_usd(free) == 0.0


def test_client_accumulates_cost_and_stops_at_limit() -> None:
    prov, sdk = provider(*[sdk_response(read=0, write=0) for _ in range(3)])
    # одна пара (100 вх, 50 вых) ≈ $0.0014; лимит чуть выше одного вызова
    client = GeminiClient(prov, sleep=lambda _: None, max_cost_usd=0.002)
    client.complete(system="s", user="1", response_model=Ticket)
    client.complete(system="s", user="2", response_model=Ticket)
    assert client.usage().cost_usd == pytest.approx(0.0028)
    with pytest.raises(ModelError, match="лимит расходов"):
        client.complete(system="s", user="3", response_model=Ticket)
    assert len(sdk.kwargs) == 2


def test_truncated_answer_cost_is_counted() -> None:
    prov, _ = provider(sdk_response(stop="max_tokens"))
    client = GeminiClient(prov, sleep=lambda _: None)
    with pytest.raises(ModelError):
        client.complete(system="s", user="u", response_model=Ticket)
    assert client.usage().cost_usd > 0


def test_factory_selects_anthropic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("GEMINI_API_KEY", "ANTHROPIC_API_KEY", "REQCHECK_MODEL", "REQCHECK_MAX_COST_USD"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("REQCHECK_PROVIDER", "anthropic")
    assert create_client_from_env() is None  # нет ANTHROPIC_API_KEY
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setenv("REQCHECK_EFFORT", "high")
    client = create_client_from_env()
    assert isinstance(client, GeminiClient)
    assert client._primary.model == "claude-opus-5-5"  # type: ignore[attr-defined]
    assert client._primary._effort == "high"  # type: ignore[attr-defined]
    assert client._max_cost == 2.0  # type: ignore[attr-defined]
    monkeypatch.setenv("REQCHECK_PROVIDER", "nope")
    with pytest.raises(ModelError):
        create_client_from_env()


def test_models_of_foreign_provider_are_dropped(capsys: pytest.CaptureFixture[str]) -> None:
    from reqcheck.llm_client import _models_for

    assert _models_for("anthropic", "gemini-3.1-flash-lite,claude-sonnet-5-5", "d") == [
        "claude-sonnet-5-5"
    ]
    assert "Предупреждение" in capsys.readouterr().err
    assert _models_for("anthropic", "gemini-x", "claude-opus-5-5") == ["claude-opus-5-5"]
    assert _models_for("gemini", "gemini-a,gemini-b", "d") == ["gemini-a", "gemini-b"]


def test_unknown_claude_model_priced_as_most_expensive_and_warns(
    capsys: pytest.CaptureFixture[str],
) -> None:
    reply = ProviderReply(text="", model="claude-future-9", tokens_in=1_000_000, tokens_out=0)
    assert reply_cost_usd(reply) == pytest.approx(4.0)
    reply_cost_usd(reply)
    assert capsys.readouterr().err.count("нет цены") == 1  # предупреждение один раз


@pytest.mark.parametrize("raw", ["nan", "inf", "-1", "0", "abc"])
def test_bad_max_cost_falls_back_with_warning(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], raw: str
) -> None:
    from reqcheck.llm_client import _max_cost_from_env

    monkeypatch.setenv("REQCHECK_MAX_COST_USD", raw)
    assert _max_cost_from_env() == 2.0
    assert "Предупреждение" in capsys.readouterr().err
    monkeypatch.setenv("REQCHECK_MAX_COST_USD", "0.5")
    assert _max_cost_from_env() == 0.5


def test_effort_is_part_of_cache_key(tmp_path: Path) -> None:
    from reqcheck.llm_cache import DiskCache

    def client(effort: str) -> tuple[GeminiClient, FakeSDK]:
        prov = AnthropicProvider(SECRET, "claude-opus-5-5", effort)
        sdk = FakeSDK(sdk_response())
        prov._client = sdk  # type: ignore[assignment]
        return GeminiClient(prov, cache=DiskCache(tmp_path)), sdk

    low, sdk_low = client("low")
    low.complete(system="s", user="u", response_model=Ticket)
    high, sdk_high = client("high")
    res = high.complete(system="s", user="u", response_model=Ticket)
    assert not res.cached and len(sdk_high.kwargs) == 1
    again, sdk_again = client("low")
    assert again.complete(system="s", user="u", response_model=Ticket).cached
    assert sdk_again.kwargs == []


def test_bad_effort_warns_and_uses_medium(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("REQCHECK_MODEL", raising=False)
    monkeypatch.setenv("REQCHECK_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", SECRET)
    monkeypatch.setenv("REQCHECK_EFFORT", "turbo")
    client = create_client_from_env()
    assert client._primary._effort == "medium"  # type: ignore[attr-defined]
    assert "REQCHECK_EFFORT" in capsys.readouterr().err
