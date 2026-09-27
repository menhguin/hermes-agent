"""Warning/media composition with a real consumer declaring the owned Slack final.

The native warning-composition probe hand-calls send() without an owner; generic
sends must no longer seal a live draft. Keep that boundary explicit while proving
that production consumer finalization still composes with the warning producers.
"""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from agent.status_output import StatusOutputMixin
from gateway.run_turn_runner import TurnRunner
from gateway.turn_context import TurnContext
from tests.gateway.test_slack_finalization_ownership import META, make_adapter, running_consumer


@pytest.mark.asyncio
@pytest.mark.parametrize("setting", [None, False, True])
async def test_warning_and_failed_media_leave_the_consumer_owned_final_intact(
        tmp_path, monkeypatch, setting):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(tmp_path / "managed"))
    cfg = {} if setting is None else {"display": {"suppress_warning_notifications": setting}}
    (tmp_path / "config.yaml").write_text(json.dumps(cfg))
    adapter, client = make_adapter()
    source = adapter.build_source(
        chat_id="D1", chat_type="dm", user_id="U1",
        thread_id=META["thread_id"], scope_id=META["team_id"])
    ctx = TurnContext(
        source=source, user_config=cfg, _run_still_current=lambda: True,
        _status_adapter=adapter, _status_chat_id="D1", _status_thread_metadata=dict(META))
    turn = TurnRunner(SimpleNamespace(), ctx)
    loop = asyncio.get_running_loop()
    scheduled = []

    def schedule(coro, *args):
        future = asyncio.run_coroutine_threadsafe(coro, loop)
        scheduled.append(future)
        return future

    def count(method):
        return sum(name == method for name, _ in client.calls)

    turn._schedule = schedule
    agent = StatusOutputMixin()
    agent.suppress_status_output = True
    agent.status_callback = turn._status_callback_sync
    async with running_consumer(adapter, client, text="Requested") as (consumer, task):
        await asyncio.to_thread(agent._warn_uncompressed_context_overflow, 200, 100)
        await asyncio.gather(*(asyncio.wrap_future(f) for f in scheduled))
        assert bool(agent._last_ctx_overflow_warn)
        assert count("chat_postMessage") == (0 if setting is True else 1)
        assert count("chat_stopStream") == 0

        client.files_upload_v2 = AsyncMock(side_effect=RuntimeError("fixture upload failure"))
        document = tmp_path / "fixture.pdf"
        document.write_bytes(b"fixture document")
        media = await adapter.send_document(
            "D1", str(document), metadata={**META, "_interim_send": True})
        assert media.success is not (setting is True)
        assert count("chat_postMessage") == (0 if setting is True else 2)
        assert count("chat_stopStream") == 0
        assert client.open == {"100.1"}
        assert adapter._active_streams["D1"]["sent"] == "Requested"

        consumer.on_delta(" final")
        await asyncio.wait_for(client.appended.wait(), 3)
        consumer.finish("Requested final")
        await asyncio.wait_for(task, 3)
        assert count("chat_startStream") == 1
        assert count("chat_stopStream") == 1
        assert count("chat_postMessage") == (0 if setting is True else 2)
        assert not adapter._active_streams
        assert not client.open
        assert client.messages["100.1"] == "Requested final"
        assert consumer.message_id == "100.1"
        assert consumer.delivered_final_matches("Requested final") is True
