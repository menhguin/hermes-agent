"""Rich display egress preserves native policy before clipping or JSON unwrap.

Exercise real TurnRunner -> rich cards -> guarded client with synthetic secrets
and fake Slack I/O (the native test runner isolates HERMES_HOME).
"""
import json

import pytest

from tests.gateway.test_rich_slack_task_cards import direct_adapter, drain_turn, make_turn


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["reasoning", "args", "result", "child"])
async def test_display_leaves_use_forced_native_egress_before_clipping(monkeypatch, surface):
    from agent import redact
    from gateway.slack_task_stream import _redact_card_value

    monkeypatch.setattr(redact, "_REDACT_ENABLED", False)
    token = "SyntheticOpaqueCredential0123456789abc"
    text = "The diagnostic reported Bearer " + token + " during the request."
    safe = redact.redact_for_egress(text)
    assert token not in safe
    public = "I am the bearer of bad news about the test fixture."
    assert _redact_card_value(public) == redact.redact_for_egress(public) == public
    adapter, client = direct_adapter()
    config = {"display": {"platforms": {"slack": {
        "tool_progress": "all", "tool_progress_native": True,
        "tool_progress_native_output_chars": 300,
    }}}}
    _, turn = await make_turn(monkeypatch, config, adapter)
    if surface == "reasoning":
        split = text.index(token) + 10
        turn.native_reasoning_callback(text[:split])
        turn.native_reasoning_callback(text[split:])
    if surface == "child":
        turn.progress_callback("subagent.start", subagent_id="synthetic-child", goal=text)
    args = {"path": "safe.txt", "content": text if surface == "args" else "public"}
    turn.native_tool_start_callback("synthetic-call", "write_file", args)
    result = json.dumps({"output": text}) if surface == "result" else "saved"
    turn.native_tool_complete_callback("synthetic-call", "write_file", args, result)
    await drain_turn(turn)

    wire = json.dumps(client.calls)
    assert token not in wire
    assert "Bearer [redacted]" in wire
    assert adapter._app is not None
    assert adapter._app.client.calls == []  # workspace resolution is unchanged
    assert all(payload["channel"] == "C1" for _, payload in client.calls)
    tasks = [chunk for _, payload in client.calls for chunk in payload.get("chunks", [])]
    assert any(c.get("id") == "synthetic-call" and c.get("status") == "complete" for c in tasks)


@pytest.mark.asyncio
@pytest.mark.parametrize("output_chars", [0, 300])
@pytest.mark.parametrize("registered", [False, True])
@pytest.mark.parametrize("serialized", [False, True], ids=["dict", "json"])
async def test_json_display_matches_structural_policy_before_result_unwrap(
    monkeypatch, request, serialized, registered, output_chars,
):
    from agent import redact
    from gateway.slack_task_stream import _redact_card_value

    monkeypatch.setattr(redact, "_REDACT_ENABLED", False)
    secret = 'synthetic-prefix-"PrivateInterior0123456789-opaque-ending'
    redact.clear_vault_redaction_values()
    request.addfinalizer(redact.clear_vault_redaction_values)
    if registered:
        redact.register_vault_redaction_value(secret)
    record = {"password": secret, "ordinary": 'public "quoted" \\path', "token": "CPU"}
    result = json.dumps(record) if serialized else record
    # A serialized payload remains a string; structural masks and public fields
    # match the dictionary representation, without mutating the original result.
    safe_record = _redact_card_value(record)
    safe = _redact_card_value(result)
    assert safe_record["ordinary"] == record["ordinary"]
    assert safe_record["token"] == record["token"]
    assert record["password"] == secret

    adapter, client = direct_adapter()
    config = {"display": {"platforms": {"slack": {
        "tool_progress": "all", "tool_progress_native": True,
        "tool_progress_native_output_chars": output_chars,
    }}}}
    _, turn = await make_turn(monkeypatch, config, adapter)
    turn.native_tool_start_callback("call-json", "terminal", {"command": "diagnostic"})
    turn.native_tool_complete_callback("call-json", "terminal", {}, result)
    await drain_turn(turn)
    wire = json.dumps(client.calls)
    assert "PrivateInterior0123456789" not in wire
    if registered:
        assert secret[:6] not in wire and secret[-4:] not in wire
    complete = [c for _, p in client.calls for c in p.get("chunks", [])
                if c.get("id") == "call-json" and c.get("status") == "complete"]
    assert complete and "Exec" in complete[-1]["title"]
    assert bool(complete[-1].get("output")) is bool(output_chars)
    if serialized:
        assert isinstance(safe, str)
        assert json.loads(safe) == safe_record
    else:
        assert safe == safe_record


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["decode", "sanitize", "encode"])
async def test_json_sanitization_failure_discards_display_without_stalling_turn(
    monkeypatch, caplog, failure,
):
    from agent import redact

    secret = 'synthetic-prefix-"PrivateInterior0123456789-opaque-ending'
    result = json.dumps({"password": secret})
    adapter, client = direct_adapter()
    config = {"display": {"platforms": {"slack": {
        "tool_progress": "all", "tool_progress_native": True,
        "tool_progress_native_output_chars": 300,
    }}}}
    _, turn = await make_turn(monkeypatch, config, adapter)
    turn.native_tool_start_callback("call-json", "terminal", {"command": "diagnostic"})
    turn.native_tool_complete_callback("call-json", "terminal", {}, result)
    turn.native_tool_start_callback("later", "write_file", {"path": "public.txt"})
    turn.native_tool_complete_callback("later", "write_file", {}, "saved")

    # Faults contain tainted exception text. Fail only the new display boundary,
    # leaving all other JSON operations and the native transport intact.
    if failure == "decode":
        original = json.loads

        def broken_decode(value, *args, **kwargs):
            if value == result:
                raise RecursionError(secret)
            return original(value, *args, **kwargs)

        monkeypatch.setattr(json, "loads", broken_decode)
    elif failure == "encode":
        original = json.dumps

        def broken_encode(value, *args, **kwargs):
            if isinstance(value, dict) and "password" in value:
                raise ValueError(secret)
            return original(value, *args, **kwargs)

        monkeypatch.setattr(json, "dumps", broken_encode)
    else:
        def broken_policy(*args, **kwargs):
            raise ValueError(secret)

        monkeypatch.setattr(redact, "_should_redact_assignment", broken_policy)

    await drain_turn(turn)
    wire = json.dumps(client.calls)
    assert "PrivateInterior0123456789" not in wire + caplog.text
    assert "display unavailable" in wire or "redaction-unavailable" in wire
    tasks = [c for _, p in client.calls for c in p.get("chunks", [])]
    assert any(c.get("id") == "later" and c.get("status") == "complete" for c in tasks)
    assert client.calls[-1][0] == "chat.stopStream"
