"""Fake-client lifecycle contracts for the carried renderer (no Slack API calls)."""
import asyncio

import pytest

from gateway.slack_task_stream import SlackTaskStream
from tests.gateway.test_rich_slack_task_cards import FakeClient


class SDKClient(FakeClient):
    async def chat_startStream(self, **kwargs):
        return await self.api_call("chat.startStream", json=kwargs)

    async def chat_appendStream(self, **kwargs):
        return await self.api_call("chat.appendStream", json=kwargs)

    async def chat_stopStream(self, **kwargs):
        return await self.api_call("chat.stopStream", json=kwargs)


@pytest.mark.asyncio
async def test_disabled_stream_is_still_stopped_and_cannot_reopen():
    client = SDKClient()
    stream = SlackTaskStream(client, "C1", "thread")
    await stream.task_started("tool-a", "terminal")
    stream.disabled = True
    await stream.stop()
    await stream.stop()
    await stream.task_started("late", "terminal")
    assert [m for m, _ in client.calls].count("chat.stopStream") == 1
    assert client.opens == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("code,recover", [
    ("message_not_found", True),
    ("message_not_in_streaming_state", True),
    ("msg_too_long", True),
    ("stopped_by_user", False),
    ("invalid_auth", False),
    ("channel_not_found", False),
    ("message_not_owned_by_app", False),
    ("messaging_processing_failed", False),
    ("request_timeout", False),
])
async def test_recovery_uses_exact_slack_error_codes(code, recover):
    from slack_sdk.errors import SlackApiError
    client = SDKClient()
    stream = SlackTaskStream(client, "C1", "thread")
    await stream.task_started("first", "read_file")
    await stream.task_finished("first", "read_file")
    original = client.api_call
    rejected = False

    async def reject_once(method, *, json):
        nonlocal rejected
        if method == "chat.appendStream" and not rejected:
            rejected = True
            raise SlackApiError("synthetic rejection", {"ok": False, "error": code})
        return await original(method, json=json)

    client.api_call = reject_once
    await stream.task_started("next", "write_file", details="new body")
    assert rejected
    assert client.opens == (2 if recover else 1)
    assert stream.disabled is not recover
    delivered = [c for _, p in client.calls for c in p.get("chunks", []) if c.get("id") == "next"]
    assert "".join(c.get("details", "") for c in delivered) == ("new body" if recover else "")
    await stream.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("message", ["unknown acceptance, prior message_not_found mentioned", "message_not_found"])
async def test_ambiguous_exception_mentioning_missing_message_is_not_replayed(message):
    client = SDKClient()
    stream = SlackTaskStream(client, "C1", "thread")
    await stream.task_started("first", "read_file")

    attempts = []

    async def ambiguous(method, *, json):
        attempts.append(method)
        raise TimeoutError(message)

    client.api_call = ambiguous
    await stream.task_started("next", "write_file", details="do not replay")
    assert attempts == ["chat.appendStream"]
    assert client.opens == 1
    assert stream.disabled


@pytest.mark.asyncio
async def test_missing_message_recovery_logs_identity_not_body(caplog):
    import logging
    caplog.set_level(logging.INFO, logger="gateway.slack_task_stream")
    client = SDKClient()
    stream = SlackTaskStream(client, "C1", "thread")
    await stream.task_started("first", "read_file")
    await stream.task_finished("first", "read_file")
    original = client.api_call
    rejected = False

    async def reject_once(method, *, json):
        nonlocal rejected
        if method == "chat.appendStream" and not rejected:
            rejected = True
            raise RuntimeError("message_not_found")
        return await original(method, json=json)

    client.api_call = reject_once
    await stream.task_started("next", "write_file", details="private-test-body")
    assert "code=message_not_found" in caplog.text
    assert "channel=C1" in caplog.text and "thread=thread" in caplog.text
    assert "ts=stream-1" in caplog.text and "chunk=task_update/next" in caplog.text
    assert "last_append_age_s=" in caplog.text
    assert "private-test-body" not in caplog.text
    await stream.stop()


async def _stop_during_blocked_send(stream, operation, entered, release):
    publishing = asyncio.create_task(operation)
    stopping = None
    stop_entered = asyncio.Event()

    async def stop_stream():
        stop_entered.set()
        await stream.stop(flush_reasoning=False)

    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert stream._send_lock.locked()
        stopping = asyncio.create_task(stop_stream())
        await asyncio.wait_for(stop_entered.wait(), timeout=5)
        # stop(False) latches synchronously before waiting for the send lock.
        # The blocked request cannot resume until we release its Event below.
        assert stream._stopped
        assert stream._send_lock.locked()
        assert not stopping.done()
    finally:
        release.set()
        await asyncio.wait_for(
            asyncio.gather(publishing, *([stopping] if stopping else [])), timeout=5,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("pause_at", [
    "rejected_append", "settlement", "settlement_rejected", "old_stop",
    "old_stop_rejected", "fresh_start", "header_replay", "task_replay",
])
async def test_concurrent_stop_prevents_missing_message_continuation(pause_at):
    client = SDKClient()
    stream = SlackTaskStream(client, "C1", "thread")
    await stream.task_started("seed", "read_file", details="accepted body")
    await stream.task_started("seed-2", "read_file", details="second accepted body")
    original = client.api_call
    entered, release = asyncio.Event(), asyncio.Event()
    attempts = []
    rejected = False

    async def pause():
        entered.set()
        await release.wait()

    async def blocked_recovery(method, *, json):
        nonlocal rejected
        attempts.append((method, json, stream._stopped))
        chunk = json.get("chunks", [{}])[0]
        if method == "chat.appendStream" and not rejected and chunk.get("id") == "next":
            rejected = True
            if pause_at == "rejected_append":
                await pause()
            raise RuntimeError("message_not_found")
        boundary = None
        if method == "chat.appendStream":
            if json["ts"] == "stream-1" and chunk.get("id") == "seed":
                boundary = "settlement"
            elif json["ts"] == "stream-2":
                if chunk["type"] == "plan_update":
                    boundary = "header_replay"
                elif chunk.get("id") == "seed":
                    boundary = "task_replay"
        elif method == "chat.stopStream" and not stream._stopped:
            boundary = "old_stop"
        elif method == "chat.startStream":
            boundary = "fresh_start"
        if pause_at == f"{boundary}_rejected":
            await pause()
            raise RuntimeError("message_not_found")
        # Accept this in-flight write before pausing its response: stop cannot
        # unsend it, but must prevent every subsequent append/start/continuation.
        result = await original(method, json=json)
        if pause_at == boundary:
            await pause()
        return result

    client.api_call = blocked_recovery
    await _stop_during_blocked_send(
        stream, stream.task_started("next", "write_file", details="rejected body"),
        entered, release,
    )
    assert rejected
    fresh_started = pause_at in {"fresh_start", "header_replay", "task_replay"}
    assert client.opens == (2 if fresh_started else 1)
    after_stop = [(method, payload) for method, payload, stopped in attempts if stopped]
    assert [method for method, _ in after_stop] == ["chat.stopStream"]
    assert after_stop[0][1]["ts"] == ("stream-2" if fresh_started else "stream-1")
    assert "continued below" not in str(after_stop)
    assert not any(c.get("id") == "next" for _, p in client.calls for c in p.get("chunks", []))
    assert "next" not in stream._in_progress
    assert not stream.disabled


@pytest.mark.asyncio
@pytest.mark.parametrize("recover", [False, True])
async def test_graceful_stop_flushes_reasoning_before_latching_even_with_recovery(recover):
    client = SDKClient()
    stream = SlackTaskStream(client, "C1", "thread")
    await stream.task_started("seed", "read_file")
    await stream.task_finished("seed", "read_file")
    tail = "Final unpunctuated thought"
    await stream.reasoning_update(tail)
    original = client.api_call
    rejected = False
    append_stopped = []

    async def reject_reasoning_once(method, *, json):
        nonlocal rejected
        if method == "chat.appendStream":
            append_stopped.append(stream._stopped)
            if recover and not rejected:
                rejected = True
                raise RuntimeError("message_not_found")
        return await original(method, json=json)

    client.api_call = reject_reasoning_once
    await stream.stop("final answer", flush_reasoning=True)
    thoughts = [c for _, p in client.calls for c in p.get("chunks", [])
                if str(c.get("id", "")).startswith("think")]
    assert "".join(c.get("details", "") for c in thoughts) == tail + " "
    assert thoughts[-1]["status"] == "complete"
    assert append_stopped and not any(append_stopped)
    assert rejected is recover
    assert client.opens == (2 if recover else 1)
    method, payload = client.calls[-1]
    assert method == "chat.stopStream"
    assert payload["ts"] == stream.ts
    assert payload["markdown_text"] == "final answer"
    assert stream._stopped and not stream.disabled


@pytest.mark.asyncio
async def test_completed_reasoning_cannot_publish_after_stream_stop():
    client = SDKClient()
    stream = SlackTaskStream(client, "C1", "thread")
    await stream.task_started("seed", "read_file")
    await stream.stop()
    before = list(client.calls)
    await stream.reasoning_update("**Late heading**", completed=True)
    await stream.stop()
    assert client.calls == before


@pytest.mark.asyncio
async def test_reasoning_tuning_caps_wire_details_not_only_local_buffer():
    client = SDKClient()
    stream = SlackTaskStream(client, "C1", "thread", reasoning_chars=80)
    await stream.reasoning_update("This is a long thought. " * 30)
    await stream.task_started("call", "terminal")
    await stream.stop()
    thoughts = [c for _, p in client.calls for c in p.get("chunks", []) if str(c.get("id", "")).startswith("think")]
    assert thoughts
    assert all(len(c.get("details", "")) <= 80 for c in thoughts)


@pytest.mark.asyncio
async def test_reasoning_only_turn_never_opens_a_progress_stream():
    client = SDKClient()
    stream = SlackTaskStream(client, "C1", "thread")
    await stream.reasoning_update("This is a substantial thought with no tool call in the turn.")
    await stream.stop()
    assert client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("rollover_by", ["age", "chars"])
async def test_rollover_replays_only_prior_tasks_and_never_reappends_old_details(rollover_by):
    client = SDKClient()
    stream = SlackTaskStream(client, "C1", "thread", rollover_age_s=1,
                             rollover_chars=1 if rollover_by == "chars" else None)
    await stream.task_started("tool-a", "write_file", details="original payload")
    if rollover_by == "age":
        stream._stream_opened_at -= 2
    await stream.task_started("tool-b", "write_file", details="new payload")
    await stream.stop()
    old = [c for _, p in client.calls if p.get("ts") == "stream-1" for c in p.get("chunks", [])]
    new = [c for _, p in client.calls if p.get("ts") == "stream-2" for c in p.get("chunks", [])]
    assert "".join(c.get("details", "") for c in old if c.get("id") == "tool-a") == "original payload"
    assert not any(c.get("id") == "tool-b" for c in old)
    assert "".join(c.get("details", "") for c in new if c.get("id") == "tool-b") == "new payload"
    assert any(c.get("id") == "tool-a" and c["status"] == "in_progress" for c in new)
    assert client.opens == 2