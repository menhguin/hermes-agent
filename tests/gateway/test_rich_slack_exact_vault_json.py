"""Whole JSON-shaped vault values must be scrubbed before structural decoding."""
import json

import pytest

from tests.gateway.test_rich_slack_task_cards import direct_adapter, drain_turn, make_turn


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["object", "array"])
async def test_registered_json_vault_value_is_scrubbed_before_decode(monkeypatch, request, shape):
    from agent import redact
    from gateway.slack_task_stream import _redact_card_value

    monkeypatch.setattr(redact, "_REDACT_ENABLED", False)
    interior = "SyntheticWholeVaultInterior0123456789"
    container = {"ordinary": interior} if shape == "object" else [interior]
    secret = json.dumps(container)
    redact.clear_vault_redaction_values()
    request.addfinalizer(redact.clear_vault_redaction_values)
    redact.register_vault_redaction_value(secret)
    native = redact.redact_for_egress(secret)
    assert interior not in native
    assert _redact_card_value(secret) == native

    # A JSON string nested under an ordinary key must retain the exact-value
    # boundary too, even when the enclosing tool argument is serialized JSON.
    args = {"path": "public.txt", "content": json.dumps({"ordinary": secret, "public": "visible"})}
    safe_args = _redact_card_value(args)
    assert json.loads(safe_args["content"]) == {"ordinary": native, "public": "visible"}
    assert json.loads(args["content"])["ordinary"] == secret

    adapter, client = direct_adapter()
    config = {"display": {"platforms": {"slack": {
        "tool_progress": "all", "tool_progress_native": True,
        "tool_progress_native_output_chars": 300,
    }}}}
    _, turn = await make_turn(monkeypatch, config, adapter)
    turn.native_tool_start_callback("synthetic-vault-json", "write_file", args)
    turn.native_tool_complete_callback("synthetic-vault-json", "write_file", args, secret)
    await drain_turn(turn)
    wire = json.dumps(client.calls, ensure_ascii=False)
    assert interior not in wire
    assert native in wire
    assert adapter._app is not None
    assert adapter._app.client.calls == []
    assert all(payload["channel"] == "C1" for _, payload in client.calls)
    tasks = [chunk for _, payload in client.calls for chunk in payload.get("chunks", [])]
    assert any(c.get("id") == "synthetic-vault-json" and c.get("status") == "complete" for c in tasks)
