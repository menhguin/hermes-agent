"""C009: exact registered-vault sanitation precedes lossy named-field masks.

Promoted from the independent source-port review's real caller/transport probe.
"""
import copy
import json

import pytest

from agent import redact
from gateway.slack_task_stream import RichTaskCardSession
from tests.gateway.slack_task_renderer import RenderingSlackClient
from tests.gateway.test_rich_slack_task_cards import direct_adapter, make_turn


@pytest.mark.asyncio
@pytest.mark.parametrize("redaction_enabled", [False, True])
@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.parametrize("transport", ["direct", "queued_rich", "queued_fallback"])
async def test_named_mask_preserves_exact_vault_policy_on_wire(monkeypatch, redaction_enabled, registered, transport):
    monkeypatch.setattr(redact, "_REDACT_ENABLED", redaction_enabled)
    from tests.gateway.test_rich_slack_task_cards import drain_turn
    adapter, _ = direct_adapter()
    client = RenderingSlackClient()
    adapter._team_clients["T2"] = client
    config = {"display": {"platforms": {"slack": {"tool_progress": "all", "tool_progress_native": True, "tool_progress_native_output_chars": 300}}}}
    ctx, turn = await make_turn(monkeypatch, config, adapter)
    fallback_wire = []
    original = client.api_call
    async def api(method, *, json):
        if transport == "queued_fallback" and method == "chat.appendStream":
            raise RuntimeError("synthetic fallback trigger")
        return await original(method, json=json)
    async def post(**kwargs):
        fallback_wire.append(("post", copy.deepcopy(kwargs)))
        return {"ok": True, "ts": "synthetic-fallback"}
    async def edit(**kwargs):
        fallback_wire.append(("edit", copy.deepcopy(kwargs)))
        return {"ok": True, "ts": "synthetic-fallback"}
    client.api_call, client.chat_postMessage, client.chat_update = api, post, edit
    value = "ZYXWVU-long-synthetic-vault-only-middle-Q987"
    args = {"query": {"password": value}}
    before = copy.deepcopy(args)
    redact.clear_vault_redaction_values()
    if registered:
        redact.register_vault_redaction_value(value)
    try:
        if transport == "direct":
            session = RichTaskCardSession(adapter, ctx)
            assert (await session.publish([
                {"type": "tool.started", "tool_call_id": "vault", "tool_name": "web_search", "args": args},
                {"type": "tool.completed", "tool_call_id": "vault", "tool_name": "web_search", "args": {}, "result": {"password": value}},
            ])).success
            await session.stop()
        else:
            turn.native_tool_start_callback("vault", "web_search", args)
            turn.native_tool_complete_callback("vault", "web_search", args, {"password": value})
            turn.native_tool_start_callback("later", "write_file", {"path": "LATER_PUBLICATION"})
            turn.native_tool_complete_callback("later", "write_file", {}, "saved")
            await drain_turn(turn)
            assert ctx.progress_queue.empty()
            if transport == "queued_fallback":
                assert fallback_wire and "LATER_PUBLICATION" in json.dumps(fallback_wire)
            else:
                assert not fallback_wire
        assert args == before
        wire = json.dumps([client.calls, fallback_wire])
        # Native generic named masks may keep head/tail. Exact registered vault
        # values have an earlier no-secret-material replacement which must win.
        if registered:
            assert value[:6] not in wire and value[-4:] not in wire, wire
        else:
            assert value not in wire
    finally:
        redact.clear_vault_redaction_values()
