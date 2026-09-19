"""A failed rich transport must not expose secrets through its text fallback."""
import json

import pytest

from tests.gateway.test_rich_slack_task_cards import direct_adapter, drain_turn, make_turn


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["prefixed", "opaque_long"])
async def test_rich_transport_failure_keeps_send_and_edit_fallbacks_redacted(monkeypatch, request, kind):
    from agent.redact import clear_vault_redaction_values, register_vault_redaction_value
    adapter, client = direct_adapter()
    ctx, turn = await make_turn(monkeypatch, {"display": {"platforms": {"slack": {
        "tool_progress": "all", "tool_progress_native": True,
    }}}}, adapter)
    fixture = "sk-" + "A" * 48 if kind == "prefixed" else "opaque-sensitive-" + "b" * 80
    register_vault_redaction_value(fixture)
    request.addfinalizer(clear_vault_redaction_values)
    # Opaque values lose their recognizable full value if preview is clipped first.
    forbidden = fixture if kind == "prefixed" else fixture[:40]
    wire = []
    original = client.api_call

    async def fail_append(method, *, json):
        if method == "chat.appendStream":
            raise RuntimeError("synthetic rich transport failure")
        return await original(method, json=json)

    async def post(**kwargs):
        wire.append(("post", kwargs))
        return {"ok": True, "ts": "fallback-1"}

    async def edit(**kwargs):
        wire.append(("edit", kwargs))
        return {"ok": True, "ts": "fallback-1"}

    client.api_call = fail_append
    client.chat_postMessage = post
    client.chat_update = edit
    args = {"command": f"echo {fixture}"}
    turn.native_tool_start_callback("secret-call", "terminal", args)
    turn.native_tool_complete_callback("secret-call", "terminal", args, "done")
    await drain_turn(turn)
    assert {kind for kind, _ in wire} == {"post", "edit"}
    assert forbidden not in json.dumps(wire)
    assert ctx.progress_queue.empty()
    assert args["command"] == f"echo {fixture}"  # display scrubbing must not mutate execution
