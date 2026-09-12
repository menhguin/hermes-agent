"""Real gateway producers + Slack adapter; only the SDK/LLM/storage are fakes.

A generic send is not proof of turn-final ownership, even in the same thread.
The transport double models stopStream's append-only behavior.
"""
import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from gateway.config import PlatformConfig
from gateway.stream_consumer import GatewayStreamConsumer, StreamConsumerConfig
from plugins.platforms.slack.adapter import SlackAdapter
from tests.gateway.test_slack_adapter_carries import Client, META


class StreamClient(Client):
    def __init__(self, fail=()):
        super().__init__(fail)
        self.started = asyncio.Event()
        self.appended = asyncio.Event()

    async def call(self, method, **kwargs):
        result = await super().call(method, **kwargs)
        if method == "chat_startStream":
            self.started.set()
        elif method == "chat_appendStream":
            self.messages[kwargs["ts"]] += kwargs.get("markdown_text", "")
            self.appended.set()
        return result


def make_adapter(fail=()):
    adapter = SlackAdapter(PlatformConfig(enabled=True, token="test-token", extra={}))
    client = StreamClient(fail)
    adapter._app = SimpleNamespace(client=StreamClient())
    adapter._team_clients = {"T1": client, "T2": StreamClient()}
    adapter._channel_team["D1"] = "T1"
    return adapter, client


@asynccontextmanager
async def running_consumer(adapter, client, text="Main response under construction"):
    consumer = GatewayStreamConsumer(
        adapter, "D1", StreamConsumerConfig(
            transport="auto", chat_type="dm", edit_interval=0.01,
            buffer_threshold=1, cursor=""), metadata=dict(META))
    task = asyncio.create_task(consumer.run())
    consumer.on_delta(text)
    try:
        await asyncio.wait_for(client.started.wait(), 3)
        yield consumer, task
    finally:
        if not task.done():
            task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("metadata,reply_to,known_team,reply_in_thread,seals,thread", [
    ({"team_id": "T1"}, "other-thread", True, True, False, "other-thread"),
    (None, "other-thread", True, True, False, "other-thread"),
    (None, None, True, True, False, None),
    ({}, None, True, True, False, None),
    ({"team_id": "T1"}, None, True, True, False, None),
    ({"thread_id": "99.1"}, None, False, True, False, "99.1"),
    ({**META, "team_id": "T2"}, None, True, True, False, "99.1"),
    ({**META, "thread_id": "other-thread"}, None, True, True, False, "other-thread"),
    ({"team_id": "T1"}, "99.1", True, True, True, "99.1"),
    (None, "99.1", True, True, True, "99.1"),
    ({"thread_ts": "99.1"}, None, True, True, True, "99.1"),
    (META, "child-ts", True, True, True, "99.1"),
    (META, "99.1", True, False, False, None),
])
async def test_consumer_final_only_seals_its_effective_destination(
        metadata, reply_to, known_team, reply_in_thread, seals, thread):
    adapter, client = make_adapter()
    async with running_consumer(adapter, client) as (consumer, task):
        # Exercise the real final-send producer with a changed/partial route.
        # Neither the test nor a fake caller injects a final ownership marker.
        consumer.metadata = dict(metadata) if metadata is not None else None
        consumer._initial_reply_to_id = reply_to
        adapter.config.extra["reply_in_thread"] = reply_in_thread
        if not known_team:
            adapter._channel_team.clear()
        final = "Authoritative rewritten final"
        consumer.finish(final)
        await asyncio.wait_for(task, 3)
        target = adapter._team_clients["T2"] if (metadata or {}).get("team_id") == "T2" else (
            adapter._app.client if not known_team else client)
        if seals:
            assert not client.open
            assert not adapter._active_streams
            assert client.messages == {"100.1": final}
        else:
            assert client.open == {"100.1"}
            assert adapter._active_streams["D1"]["sent"] == "Main response under construction"
            assert client.messages["100.1"] == "Main response under construction"
            posts = [kw for name, kw in target.calls if name == "chat_postMessage"]
            assert len(posts) == 1
            assert posts[0].get("thread_ts") == thread
            assert posts[0]["text"] == final
        assert consumer.delivered_final_matches(final) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("sender", [
    "btw", "btw_error", "platform_notice", "background_update",
    "unmarked_same_text", "unmarked_prefix", "no_stream_consumer", "other_consumer",
])
async def test_side_callers_cannot_finalize_another_consumers_stream(sender, monkeypatch):
    from unittest.mock import AsyncMock

    from agent import side_question
    from gateway.platforms.base import (
        _reply_anchor_for_event as reply_anchor,
        _thread_metadata_for_source as thread_metadata,
    )
    from gateway.platforms.event import MessageEvent, MessageType
    from gateway.run_notifications import GatewayNotificationsMixin
    from gateway.slash_commands import GatewaySlashCommandsMixin

    adapter, client = make_adapter()
    source = adapter.build_source(
        chat_id="D1", user_id="U1", chat_type="dm", thread_id="99.1", scope_id="T1")

    class Handler(GatewaySlashCommandsMixin, GatewayNotificationsMixin):
        # Use production routing/metadata helpers, not hand-authored send metadata.
        _reply_anchor_for_event = staticmethod(reply_anchor)
        _thread_metadata_for_source = staticmethod(thread_metadata)

        def _adapter_for_source(self, source):
            return adapter

        def _resolve_session_agent_runtime(self, **kwargs):
            return "test/model", {"api_key": "synthetic-test-value", "provider": "custom"}

        def _session_key_for_source(self, source):
            return "test-session"

    handler = Handler()
    handler._background_tasks = set()
    handler.async_session_store = SimpleNamespace(
        get_or_create_session=AsyncMock(return_value=SimpleNamespace(session_id="test-session")),
        load_transcript=AsyncMock(return_value=[{"role": "user", "content": "Synthetic context"}]))

    def answer(*args, **kwargs):
        if sender == "btw_error":
            raise RuntimeError("Side provider unavailable")
        return "Side answer, not the main turn final"

    monkeypatch.setattr(side_question, "answer_side_question", answer)
    async with running_consumer(adapter, client) as (consumer, task):
        if sender.startswith("btw"):
            event = MessageEvent(
                text="/btw explain a side issue", message_type=MessageType.COMMAND,
                source=source, message_id="101.1")
            await handler._handle_btw_command(event)
            await asyncio.gather(*handler._background_tasks)
        elif sender == "platform_notice":
            await handler._deliver_platform_notice(source, "A separate operational notice")
        elif sender == "background_update":
            target = handler._UpdateTarget(
                adapter, "D1", "test-session", handler._thread_metadata_for_source(source), source.platform)
            await target.send("A separate background update")
        elif sender in {"no_stream_consumer", "other_consumer"}:
            other = GatewayStreamConsumer(adapter, "D1", metadata=dict(META))
            if sender == "other_consumer":
                # A different consumer owns a draft ID, but not this live stream.
                other.cfg.transport = "auto"
                await other._start_transports()
            other.on_delta("Final from another turn")
            other.finish("Final from another turn")
            await other.run()
        else:
            text = "Main response under construction"
            if sender == "unmarked_prefix":
                text += " -- independently quoted"
            await adapter.send("D1", text, metadata=dict(META))

        assert client.open == {"100.1"}
        assert adapter._active_streams["D1"]["sent"] == "Main response under construction"
        assert client.messages["100.1"] == "Main response under construction"
        posts = [kw for name, kw in client.calls if name == "chat_postMessage"]
        assert len(posts) == 1
        assert posts[0]["thread_ts"] == "99.1"
        side_text = posts[0]["text"]
        assert client.messages["200.1"] == side_text

        # The owner can continue and finish, replacing the stream, not the side answer.
        consumer.on_delta(" -- continuing the main turn")
        await asyncio.wait_for(client.appended.wait(), 3)
        assert client.messages["100.1"].endswith(" -- continuing the main turn")
        final = "The owner's authoritative rewritten final"
        consumer.finish(final)
        await asyncio.wait_for(task, 3)
        assert not client.open
        assert not adapter._active_streams
        assert client.messages == {"100.1": final, "200.1": side_text}
        assert consumer.message_id == "100.1"
        assert consumer.delivered_final_matches(final) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("final,fail", [
    ("Main response under construction", ()),
    ("Main response under construction, now complete with footer", ()),
    ("Rewritten final", ()),
    ("Main response under construction, now complete", ("chat_stopStream",)),
    ("Rewritten final", ("chat_stopStream",)),
    ("Rewritten final", ("chat_update",)),
    ("Rewritten final", ("chat_stopStream", "chat_postMessage")),
])
async def test_owner_final_preserves_reconciliation_and_honest_fallback(final, fail):
    adapter, client = make_adapter(fail)
    async with running_consumer(adapter, client) as (consumer, task):
        consumer.finish(final)
        await asyncio.wait_for(task, 3)
        assert not adapter._active_streams
        stops = [kw for method, kw in client.calls if method == "chat_stopStream"]
        assert len(stops) == 1
        if "chat_postMessage" in fail:
            assert client.messages == {"100.1": "Main response under construction"}
            assert not consumer.final_content_delivered
            assert not consumer.final_response_sent
        else:
            assert consumer.delivered_final_matches(final) is True
            assert client.messages[consumer.message_id] == final
            if not fail:
                assert client.messages == {"100.1": final}
                # stopStream may append only the missing suffix, never a rewrite.
                assert stops[0].get("markdown_text", "") == (
                    final.removeprefix("Main response under construction")
                    if final.startswith("Main response under construction") else "")
            else:
                assert consumer.message_id == "200.1"
                assert len(client.messages) == 2
        # A failed remote stop still leaves a visible/open old message: no
        # exactly-once or disappearance claim follows from a successful fallback.
        assert bool(client.open) == ("chat_stopStream" in fail)
