"""Клиент модели: схема, повторы, фолбэк, кэш, учёт. Сеть не используется."""

import json
from pathlib import Path
from random import Random
from typing import Literal

import pytest
from pydantic import BaseModel

from reqcheck.errors import ModelError
from reqcheck.llm_cache import DiskCache
from reqcheck.llm_client import GeminiClient, create_client_from_env
from reqcheck.llm_providers import ProviderError, ProviderReply

SECRET = "sk-super-secret-key-123"


class Ticket(BaseModel):
    title: str
    priority: int


GOOD = '{"title": "Доставка", "priority": 2}'


class FakeProvider:
    def __init__(self, script: list[object], name: str = "fake", model: str = "cfg-model") -> None:
        self.name, self.model = name, model
        self.script = list(script)
        self.calls: list[str] = []

    def generate(self, *, system: str, user: str, schema: dict) -> ProviderReply:
        self.calls.append(user)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return ProviderReply(text=str(item), model="api-model-1", tokens_in=10, tokens_out=5)


def make(primary: FakeProvider, **kw: object) -> GeminiClient:
    sleeps: list[float] = []
    client = GeminiClient(primary, sleep=sleeps.append, rng=Random(1), **kw)  # type: ignore[arg-type]
    client.sleeps = sleeps  # type: ignore[attr-defined]
    return client


def ask(client: GeminiClient):
    return client.complete(system="sys", user="usr", response_model=Ticket)


def test_success_and_usage() -> None:
    client = make(FakeProvider([GOOD]))
    res = ask(client)
    assert res.data == Ticket(title="Доставка", priority=2)
    assert res.model == "api-model-1" and not res.cached
    u = client.usage()
    assert (u.model, u.requests, u.cache_hits, u.tokens_in, u.tokens_out) == (
        "api-model-1",
        1,
        0,
        10,
        5,
    )


def test_invalid_json_retries_with_error_text() -> None:
    p = FakeProvider(["не json", GOOD])
    res = ask(make(p))
    assert res.data.priority == 2
    assert len(p.calls) == 2
    assert "не прошёл проверку схемы" in p.calls[1] and p.calls[1].startswith("usr")


def test_two_invalid_answers_raise_model_error() -> None:
    p = FakeProvider(['{"title": 1}', "{}"])
    with pytest.raises(ModelError, match="дважды"):
        ask(make(p))
    assert len(p.calls) == 2


def test_two_invalid_answers_go_to_fallback_and_are_not_cached(tmp_path: Path) -> None:
    primary = FakeProvider(["x", "y"])
    fallback = FakeProvider([GOOD], name="alt")
    cache = DiskCache(tmp_path)
    assert ask(make(primary, fallback=fallback, cache=cache)).data.title == "Доставка"
    assert len(fallback.calls) == 1
    again = FakeProvider([GOOD])
    assert not ask(make(again, cache=cache)).cached  # ответ фолбэка в кэш не попал


def test_openai_payload_garbage_is_provider_error() -> None:
    from reqcheck.llm_providers import _parse_openai_payload

    ok = {"choices": [{"message": {"content": GOOD}}], "usage": "oops"}
    assert _parse_openai_payload(ok, "m").tokens_in == 0
    ok["usage"] = {"prompt_tokens": "x", "completion_tokens": -3}
    r = _parse_openai_payload(ok, "m")
    assert (r.tokens_in, r.tokens_out, r.model) == (0, 0, "m")
    for bad in (None, [], {"choices": []}, {"choices": [{"message": {"content": 5}}]}):
        with pytest.raises(ProviderError):
            _parse_openai_payload(bad, "m")


def test_bad_fallback_url_warns_and_works_without_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("REQCHECK_FALLBACK_BASE_URL", "ftp://x")
    monkeypatch.setenv("REQCHECK_FALLBACK_API_KEY", "k2")
    monkeypatch.setenv("REQCHECK_FALLBACK_MODEL", "m")
    client = create_client_from_env()
    assert client is not None and client._fallback is None  # type: ignore[attr-defined]
    assert "Предупреждение" in capsys.readouterr().err


def test_redirects_are_refused() -> None:
    import urllib.request

    from reqcheck.llm_providers import _OPENER

    req = urllib.request.Request("https://a.example/x")
    handler = next(h for h in _OPENER.handlers if h.__class__.__name__ == "_NoRedirect")
    assert handler.redirect_request(req, None, 302, "Found", {}, "https://b.example/") is None


def test_cache_write_failure_leaves_no_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import os

    def boom(*_: object) -> None:
        raise OSError("disk")

    monkeypatch.setattr(os, "replace", boom)
    DiskCache(tmp_path).put(
        "ab" * 32, __import__("reqcheck.llm_cache").llm_cache.CacheEntry("t", "m")
    )
    assert list(tmp_path.rglob("*.tmp")) == []


def test_429_falls_back_to_second_provider() -> None:
    quota = ProviderError("исчерпана квота (429)", retryable=True)
    primary = FakeProvider([quota])
    fallback = FakeProvider([GOOD], name="alt")
    client = make(primary, fallback=fallback)
    assert ask(client).data.title == "Доставка"
    assert len(primary.calls) == 1 and len(fallback.calls) == 1
    assert client.sleeps == []  # type: ignore[attr-defined]  # есть куда уйти — не ждём
    assert client.usage().requests == 2


def test_last_provider_retries_with_bounded_pauses() -> None:
    quota = ProviderError("429", retryable=True)
    client = make(FakeProvider([quota, quota, quota]))
    with pytest.raises(ModelError):
        ask(client)
    assert len(client.sleeps) == 2 and sum(client.sleeps) <= 30  # type: ignore[attr-defined]


def test_retry_after_respected_and_long_one_skips_to_fallback() -> None:
    p = FakeProvider([ProviderError("429", retryable=True, retry_after=7), GOOD])
    client = make(p)
    ask(client)
    assert client.sleeps == [7]  # type: ignore[attr-defined]

    long = FakeProvider([ProviderError("429", retryable=True, retry_after=300)])
    fb = FakeProvider([GOOD], name="alt")
    c2 = make(long, fallback=fb)
    ask(c2)
    assert c2.sleeps == [] and len(long.calls) == 1  # type: ignore[attr-defined]


def test_non_retryable_error_does_not_sleep() -> None:
    p = FakeProvider([ProviderError("ключ API неверен", retryable=False)])
    client = make(p)
    with pytest.raises(ModelError, match="ключ API неверен"):
        ask(client)
    assert client.sleeps == []  # type: ignore[attr-defined]


def test_both_providers_down_gives_clear_error() -> None:
    err = ProviderError("нет связи (сеть)", retryable=False)
    client = make(
        FakeProvider([err], model="m-a"), fallback=FakeProvider([err], name="alt", model="m-b")
    )
    with pytest.raises(ModelError) as info:
        ask(client)
    assert "m-a: нет связи" in info.value.message and "m-b: нет связи" in info.value.message


def test_cache_second_call_skips_provider(tmp_path: Path) -> None:
    p = FakeProvider([GOOD])
    client = make(p, cache=DiskCache(tmp_path))
    first, second = ask(client), ask(client)
    assert not first.cached and second.cached and second.data == first.data
    assert second.model == "api-model-1"
    assert len(p.calls) == 1
    u = client.usage()
    assert (u.requests, u.cache_hits, u.tokens_in) == (1, 1, 10)


def test_cache_key_depends_on_prompt(tmp_path: Path) -> None:
    p = FakeProvider([GOOD, GOOD])
    client = make(p, cache=DiskCache(tmp_path))
    client.complete(system="a", user="u", response_model=Ticket)
    client.complete(system="b", user="u", response_model=Ticket)
    assert len(p.calls) == 2


def test_invalid_answer_is_not_cached(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    with pytest.raises(ModelError):
        ask(make(FakeProvider(["x", "y"]), cache=cache))
    p = FakeProvider([GOOD])
    assert not ask(make(p, cache=cache)).cached


def test_factory_without_key_returns_none(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)  # нет .env
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert create_client_from_env() is None


def test_factory_does_not_override_env_and_respects_no_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / ".env").write_text("GEMINI_API_KEY=from-file\nREQCHECK_MODEL=from-file\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    monkeypatch.delenv("REQCHECK_MODEL", raising=False)
    monkeypatch.setenv("REQCHECK_NO_CACHE", "1")
    client = create_client_from_env()
    assert isinstance(client, GeminiClient)
    assert client._cache is None
    assert client._primary._key == "from-env"  # type: ignore[attr-defined]
    assert client._primary.model == "from-file"
    monkeypatch.delenv("REQCHECK_NO_CACHE")
    monkeypatch.setenv("REQCHECK_CACHE_DIR", str(tmp_path / "c"))
    assert create_client_from_env()._cache is not None  # type: ignore[attr-defined]


def test_secret_not_in_error_message() -> None:
    from reqcheck.llm_providers import GeminiProvider

    prov = GeminiProvider(SECRET, "m")

    class Boom:
        models = None

        def generate_content(self, **_: object) -> object:
            raise RuntimeError(f"https://x?key={SECRET}")

    boom = Boom()
    boom.models = boom  # type: ignore[assignment]
    prov._client = boom  # type: ignore[assignment]
    client = make(prov)  # type: ignore[arg-type]
    with pytest.raises(ModelError) as info:
        ask(client)
    assert SECRET not in info.value.message
    assert SECRET not in repr(info.value)


def test_plain_http_fallback_warns_except_localhost(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("REQCHECK_FALLBACK_API_KEY", "k2")
    monkeypatch.setenv("REQCHECK_FALLBACK_MODEL", "m")
    monkeypatch.setenv("REQCHECK_FALLBACK_BASE_URL", "http://llm.example.com/v1")
    assert create_client_from_env()._fallback is not None  # type: ignore[attr-defined]
    assert "http://" in capsys.readouterr().err
    monkeypatch.setenv("REQCHECK_FALLBACK_BASE_URL", "http://localhost:8000/v1")
    create_client_from_env()
    assert capsys.readouterr().err == ""


def test_oversized_fallback_response_is_provider_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from reqcheck import llm_providers
    from reqcheck.llm_providers import OpenAICompatProvider

    class Resp:
        def __enter__(self) -> "Resp":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self, n: int) -> bytes:
            return b"x" * n

    monkeypatch.setattr(llm_providers, "MAX_RESPONSE_BYTES", 10)
    monkeypatch.setattr(llm_providers._OPENER, "open", lambda *a, **k: Resp())
    prov = OpenAICompatProvider("https://x.example/v1", "k", "m")
    with pytest.raises(ProviderError, match="слишком большой"):
        prov.generate(system="s", user="u", schema={})


def test_model_chain_skips_daily_exhausted_model_and_reports_models() -> None:
    daily = ProviderError("429 суточная", retryable=True, exhausted=True)
    a = FakeProvider([daily], model="m-a")
    b = FakeProvider([GOOD, GOOD], model="m-b")
    client = GeminiClient([a, b], sleep=lambda _: None)
    client.complete(system="s", user="1", response_model=Ticket)
    client.complete(system="s", user="2", response_model=Ticket)
    assert len(a.calls) == 1  # вторую модель дёрнули, первую больше не пробуем
    assert client.usage().model == "api-model-1"


def test_usage_lists_every_model_that_answered() -> None:
    class Named(FakeProvider):
        def generate(self, **kw: object) -> ProviderReply:  # type: ignore[override]
            r = super().generate(**kw)  # type: ignore[arg-type]
            return ProviderReply(r.text, self.model, 1, 1)

    a = Named(
        [
            ProviderError("503", retryable=True),
        ],
        model="m-a",
    )
    b = Named([GOOD], model="m-b")
    a2 = Named([GOOD], model="m-a")
    client = GeminiClient([a, b], sleep=lambda _: None)
    client.complete(system="s", user="1", response_model=Ticket)
    client._primaries[0] = a2  # type: ignore[attr-defined]
    client.complete(system="s", user="2", response_model=Ticket)
    assert client.usage().model == "m-b,m-a"


def test_three_failures_in_a_row_skip_model() -> None:
    bad = ProviderError("503", retryable=True)
    a = FakeProvider([bad, bad, bad], model="m-a")
    b = FakeProvider([GOOD] * 4, model="m-b")
    client = GeminiClient([a, b], sleep=lambda _: None)
    for i in range(4):
        client.complete(system="s", user=str(i), response_model=Ticket)
    assert len(a.calls) == 3


def test_long_retry_after_moves_to_next_model_without_sleep() -> None:
    a = FakeProvider([ProviderError("429", retryable=True, retry_after=21)], model="m-a")
    b = FakeProvider([GOOD], model="m-b")
    client = make(a, fallback=b)
    ask(client)
    assert client.sleeps == []  # type: ignore[attr-defined]


def test_provider_time_budget_is_enforced() -> None:
    now = [0.0]

    def clock() -> float:
        return now[0]

    def tick(delay: float) -> None:
        now[0] += delay

    transient = ProviderError("сеть", retryable=True, retry_after=15)
    p = FakeProvider([transient, transient, GOOD])
    client = GeminiClient(p, sleep=lambda d: tick(d + 50), clock=clock)
    with pytest.raises(ModelError, match="превышено время"):
        ask(client)
    assert len(p.calls) == 2


def test_simplify_schema_inlines_refs_keeps_field_named_title() -> None:
    from reqcheck.llm_providers import simplify_schema

    out = simplify_schema(Ticket2.model_json_schema())
    text = str(out)
    assert "$ref" not in text and "$defs" not in text and "'default'" not in text
    assert "title" in out["properties"]  # поле с именем title не потеряно
    assert out["properties"]["inner"]["properties"]["x"]["type"] == "integer"


class Inner(BaseModel):
    x: int = 0


class Ticket2(BaseModel):
    title: str
    inner: Inner


# --- мягкий разбор: битый элемент списка не стоит всего ответа ---


class _Item(BaseModel):
    kind: Literal["a", "b"]
    text: str


class _Bag(BaseModel):
    items: list[_Item]
    note: str = ""


def _bag(*kinds: str) -> str:
    return json.dumps({"items": [{"kind": k, "text": "t"} for k in kinds]})


def test_broken_list_item_is_dropped_and_counted() -> None:
    p = FakeProvider([_bag("a", "zzz", "b")])
    client = make(p)
    res = client.complete(system="sys", user="usr", response_model=_Bag)
    assert [i.kind for i in res.data.items] == ["a", "b"]
    assert len(p.calls) == 1
    assert client.usage().invalid_items == 1


def test_mostly_broken_answer_is_asked_again() -> None:
    p = FakeProvider([_bag("zzz", "yyy", "a"), _bag("a")])
    client = make(p)
    res = client.complete(system="sys", user="usr", response_model=_Bag)
    assert len(p.calls) == 2 and len(res.data.items) == 1
    assert client.usage().invalid_items == 0


def test_broken_envelope_is_not_salvaged() -> None:
    p = FakeProvider(['{"items": [], "note": 5}', "[]"])
    with pytest.raises(ModelError, match="дважды"):
        make(p).complete(system="sys", user="usr", response_model=_Bag)
