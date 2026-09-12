"""Carried Slack contracts against real adapter imports and a fake SDK boundary.

No Slack network, credentials, or production state. The SDK fake models append-only
streams: a failed stop can leave an old visible message; delivery is not exactly-once.
"""
import asyncio
import logging
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from plugins.platforms.slack.adapter import SlackAdapter


class Client:
    def __init__(self, fail=()):
        self.calls = []
        self.fail = set(fail)
        self.messages = {}
        self.open = set()

    async def call(self, method, **kwargs):
        self.calls.append((method, kwargs))
        if method in self.fail:
            raise RuntimeError(method + " unavailable")
        if method == "chat_startStream":
            self.messages["100.1"] = kwargs.get("markdown_text", "")
            self.open.add("100.1")
            return {"ok": True, "ts": "100.1"}
        if method == "chat_stopStream":
            self.open.discard(kwargs["ts"])
            self.messages[kwargs["ts"]] += kwargs.get("markdown_text", "")
        if method == "chat_update":
            self.messages[kwargs["ts"]] = kwargs["text"]
        if method == "chat_postMessage":
            self.messages["200.1"] = kwargs["text"]
            return {"ok": True, "ts": "200.1"}
        return {"ok": True}

    async def api_call(self, method, **kwargs):
        return await self.call(method, **kwargs)

    def __getattr__(self, name):
        async def invoke(**kwargs):
            return await self.call(name, **kwargs)
        return invoke


def make_adapter(fail=()):
    client = Client(fail)
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="test-token", extra={}))
    adapter._app = SimpleNamespace(client=Client())
    # Real workspace routing must choose T1, not the unrelated default client.
    adapter._team_clients = {"T1": client}
    adapter._channel_team["D1"] = "T1"
    return adapter, client


META = {"thread_id": "99.1", "team_id": "T1", "user_id": "U1"}


@pytest.mark.asyncio
@pytest.mark.parametrize("modern_fails", [False, True])
async def test_sessions_status_and_title_use_modern_first_with_per_call_fallback(modern_fails):
    failures = {"agents.sessions.setStatus", "agents.sessions.rename"} if modern_fails else ()
    adapter, client = make_adapter(failures)
    adapter._status_text = {"D1": "Checking facts"}
    await adapter.send_typing("D1", META)
    await adapter.stop_typing("D1", META)
    await adapter._set_assistant_thread_title("D1", "99.1", "  Test\n title  ", team_id="T1")
    modern = [(m, k["json"]) for m, k in client.calls if m.startswith("agents.")]
    assert modern == [
        ("agents.sessions.setStatus", {"channel_id": "D1", "thread_ts": "99.1", "status": "processing"}),
        ("agents.sessions.setStatus", {"channel_id": "D1", "thread_ts": "99.1", "status": "active"}),
        ("agents.sessions.rename", {"channel_id": "D1", "thread_ts": "99.1", "title": "Test title"}),
    ]
    legacy = [(m, k) for m, k in client.calls if m.startswith("assistant_")]
    assert [k.get("status", k.get("title")) for _, k in legacy] == (
        ["Checking facts", "", "Test title"] if modern_fails else [])
    assert not adapter._active_status_threads
    assert adapter._titled_assistant_threads
    client.fail.clear()
    await adapter.send_typing("D1", META)
    assert client.calls[-1][0] == "agents.sessions.setStatus"
    assert not adapter._app.client.calls


@pytest.mark.asyncio
@pytest.mark.parametrize("sent,final,fail", [
    ("old prefix", "different final", ()),
    ("", "different final", ()),
    ("old prefix", "old prefix final", ()),
    ("old prefix", "different final", ("chat_stopStream",)),
    ("old prefix", "old prefix final", ("chat_stopStream",)),
    ("old prefix", "different final", ("chat_update",)),
])
async def test_authoritative_final_reconciles_stream_or_delivers_fallback(sent, final, fail):
    adapter, client = make_adapter(fail)
    await adapter.send_draft("D1", 7, sent, metadata=META)
    result = await adapter.send("D1", final, metadata=META)
    assert result.success
    assert "D1" not in adapter._active_streams
    assert client.messages[result.message_id] == final
    if not fail:
        assert list(client.messages.values()) == [final]
        assert not client.open
    else:
        # Successful fallback does NOT prove the old remote stream disappeared.
        assert "200.1" in client.messages
        assert len(client.messages) == 2
        assert bool(client.open) == ("chat_stopStream" in fail)


@pytest.mark.asyncio
@pytest.mark.parametrize("content", ["old prefix", "old prefix extended", "unrelated notice"])
@pytest.mark.parametrize("status_notice", [False, True])
async def test_interim_send_does_not_seal_live_stream(content, status_notice):
    adapter, client = make_adapter()
    await adapter.send_draft("D1", 7, "old prefix", metadata=META)
    if status_notice:
        result = await adapter.send_or_update_status("D1", "progress", content, metadata=META)
    else:
        result = await adapter.send("D1", content, metadata={**META, "_interim_send": True})
    assert result.success
    assert adapter._active_streams["D1"]["sent"] == "old prefix"
    assert client.open == {"100.1"}
    assert not any(m == "chat_stopStream" for m, _ in client.calls)


class Bolt:
    def __init__(self, client):
        self.client = client
        self.listeners = []

    def event(self, selector):
        def register(fn):
            self.listeners.append((selector, fn))
            return fn
        return register

    def action(self, selector):
        return lambda fn: fn

    command = action

    async def dispatch(self, event, body):
        for selector, fn in self.listeners:
            if selector == event["type"] or (
                    not isinstance(selector, str) and selector.fullmatch(event["type"])):
                args = {"event": event, "say": None, "body": body, "logger": logging.getLogger(__name__)}
                await fn(**{k: args[k] for k in fn.__code__.co_varnames[:fn.__code__.co_argcount]})
                return


@pytest.mark.asyncio
@pytest.mark.parametrize("user,channel,allowed", [("U1", "C1", True), ("U2", "C1", False), ("U1", "C2", False)])
async def test_native_stop_runs_inline_through_authorized_message_pipeline(user, channel, allowed):
    from gateway.platforms.event import MessageEvent, MessageType

    adapter, client = make_adapter()
    adapter._app = Bolt(client)
    adapter._bot_user_id = "UBOT"
    adapter.config.extra.update({"allowed_channels": ["C1"], "strict_mention": True, "thread_require_mention": True})
    adapter.set_authorization_check(lambda uid, chat_type, chat_id: uid == "U1")
    received = []

    async def handler(event):
        received.append(event)
        assert event.get_command() == "stop"
        assert not adapter._pending_messages

    adapter.set_message_handler(handler)
    source = adapter.build_source(chat_id=channel, user_id=user, chat_type="group", thread_id="99.1", scope_id="T1")
    key = adapter._event_session_key(MessageEvent(text="/stop", message_type=MessageType.COMMAND, source=source))
    adapter._active_sessions[key] = asyncio.Event()
    adapter._register_bolt_handlers()
    event = {"type": "agent_session_stopped", "channel_id": channel, "user_id": user, "thread_ts": "99.1", "event_ts": "101.1"}
    await adapter._app.dispatch(event, {"authorizations": [{"team_id": "T1"}]})
    assert len(received) == int(allowed)
    assert not adapter._pending_messages
    if allowed:
        assert received[0].source.scope_id == "T1"
        assert received[0].source.thread_id == "99.1"
        assert key not in adapter._active_sessions
        await adapter._app.dispatch(event, {"team_id": "T1"})
        assert len(received) == 1  # native replay goes through normal dedup
    else:
        assert key in adapter._active_sessions
        assert not any(m.startswith("agents.sessions") for m, _ in client.calls)


@pytest.mark.asyncio
async def test_failed_final_fallback_does_not_claim_delivery():
    adapter, client = make_adapter({"chat_stopStream", "chat_postMessage"})
    await adapter.send_draft("D1", 7, "old prefix", metadata=META)
    result = await adapter.send("D1", "different final", metadata=META)
    assert not result.success
    assert client.open == {"100.1"}
    assert list(client.messages.values()) == ["old prefix"]


@pytest.mark.asyncio
async def test_sessions_total_failure_is_retryable_and_title_not_cached():
    adapter, client = make_adapter({"agents.sessions.setStatus", "agents.sessions.rename",
                                   "assistant_threads_setStatus", "assistant_threads_setTitle"})
    await adapter.send_typing("D1", META)
    await adapter.stop_typing("D1", META)
    await adapter._set_assistant_thread_title("D1", "99.1", "title", team_id="T1")
    assert not adapter._active_status_threads
    assert not adapter._titled_assistant_threads
    client.fail.clear()
    await adapter._set_assistant_thread_title("D1", "99.1", "title", team_id="T1")
    assert adapter._titled_assistant_threads


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata", [{**META, "thread_id": "other"}, {**META, "team_id": "T2"}])
async def test_final_does_not_seal_another_workspace_or_thread(metadata):
    adapter, client = make_adapter()
    await adapter.send_draft("D1", 7, "old prefix", metadata=META)
    await adapter.send("D1", "different final", metadata=metadata)
    assert client.open == {"100.1"}
    assert "D1" in adapter._active_streams


@pytest.mark.asyncio
async def test_ignored_egress_cannot_seal_stream():
    adapter, client = make_adapter()
    await adapter.send_draft("D1", 7, "old prefix", metadata=META)
    adapter.config.extra["ignored_channels"] = ["D1"]
    result = await adapter.send("D1", "different final", metadata=META)
    assert not result.success
    assert client.open == {"100.1"}
    assert len(client.calls) == 1
