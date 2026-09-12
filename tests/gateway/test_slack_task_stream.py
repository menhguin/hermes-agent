"""Fake-client lifecycle contracts for the carried renderer (no Slack API calls)."""
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
