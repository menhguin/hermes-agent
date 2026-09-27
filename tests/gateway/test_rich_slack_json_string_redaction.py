"""Escaped JSON strings must not bypass rich-card exact-vault sanitation (Q1)."""
import copy
import json

import pytest

from agent import redact
from gateway.slack_task_stream import RichTaskCardSession, _redact_card_value
from tests.gateway.test_rich_slack_task_cards import direct_adapter, make_turn


@pytest.mark.asyncio
@pytest.mark.parametrize("redaction_enabled", [True, False])
@pytest.mark.parametrize("nested", [False, True], ids=["scalar", "nested_scalar"])
@pytest.mark.parametrize("rail", ["details", "reasoning", "output", "output_off"])
async def test_escaped_json_string_secret_is_scrubbed_before_wire_clipping(
    monkeypatch, request, redaction_enabled, nested, rail,
):
    secret = 'synthetic-json-string-credential-"private\\vault-suffix'
    redact.clear_vault_redaction_values()
    request.addfinalizer(redact.clear_vault_redaction_values)
    redact.register_vault_redaction_value(secret)
    monkeypatch.setattr(redact, "_REDACT_ENABLED", redaction_enabled)
    # Keep reasoning above its existing display threshold after redaction.
    encoded = json.dumps(secret + " — public display context remains visible after redaction.")
    if nested:
        encoded = json.dumps({"ordinary": encoded})
    adapter, client = direct_adapter()
    config = {"display": {"platforms": {"slack": {
        "tool_progress": "all", "tool_progress_native": True,
        "tool_progress_native_output_chars": 0 if rail == "output_off" else 300,
    }}}}
    ctx, _ = await make_turn(monkeypatch, config, adapter)
    session = RichTaskCardSession(adapter, ctx)
    try:
        assert (await session.publish([
            {"type": "tool.started", "tool_call_id": "seed", "tool_name": "terminal"},
        ])).success
        if rail == "details":
            events = [{"type": "tool.started", "tool_call_id": "probe", "tool_name": "write_file",
                       "args": {"path": "fixture.json", "content": encoded}}]
        elif rail == "reasoning":
            events = [{"type": "reasoning.delta", "text": encoded},
                      {"type": "tool.started", "tool_call_id": "probe", "tool_name": "terminal"}]
        else:
            events = [{"type": "tool.completed", "tool_call_id": "seed", "tool_name": "terminal",
                       "result": encoded}]
        before = copy.deepcopy(events)
        assert (await session.publish(events)).success
        await session.stop()
        assert events == before  # Never change the execution/replay input.
        chunks = [c for _, p in client.calls for c in p.get("chunks", [])]
        fields = [str(v) for c in chunks for k, v in c.items() if k in {"details", "title", "output"}]
        assert fields
        # Assert against accepted display fields, including long prefixes: JSON
        # escaping and summary clipping must not hide a recoverable credential.
        for forbidden in (secret, json.dumps(secret), "synthetic-json-string-credential", "private", "vault-suffix"):
            assert all(forbidden not in field for field in fields), fields
        assert not any("display unavailable" in field for field in fields)
        if rail == "output_off":
            assert not any("output" in c for c in chunks)
            assert any("→" in c.get("title", "") for c in chunks)
        elif rail == "reasoning":
            assert any(c.get("title", "").startswith("💭") and c.get("details") for c in chunks)
        else:
            assert any(c.get(rail) for c in chunks)
    finally:
        await session.stop(flush_reasoning=False)


@pytest.mark.parametrize("redaction_enabled", [True, False])
@pytest.mark.parametrize("depth", [0, 1, 4, 12])
def test_json_string_layers_preserve_public_data_and_whole_vault_boundaries(
    monkeypatch, request, redaction_enabled, depth,
):
    # Whole JSON object/array/string secrets must match BEFORE each decode, not
    # only after splitting them into otherwise ordinary fields and strings.
    secrets = [
        json.dumps({"ordinary": "synthetic-whole-object-private-interior"}),
        json.dumps(["synthetic-whole-array-private-interior"]),
        json.dumps("synthetic-whole-string-private-interior"),
        'synthetic-opaque-"private\\interior',
    ]
    redact.clear_vault_redaction_values()
    request.addfinalizer(redact.clear_vault_redaction_values)
    for secret in secrets:
        redact.register_vault_redaction_value(secret)
    monkeypatch.setattr(redact, "_REDACT_ENABLED", redaction_enabled)
    safe_secrets = [redact.redact_for_egress(secret) for secret in secrets]
    public = ["", "null", "true", "1234", '"ordinary public string"',
              'public "quoted" \\path\n\ttext ☃', json.dumps({"ordinary": ["visible"]})]
    for _ in range(depth):
        secrets = [json.dumps(value, ensure_ascii=False) for value in secrets]
        safe_secrets = [json.dumps(value, ensure_ascii=False) for value in safe_secrets]
        public = [json.dumps(value, ensure_ascii=False) for value in public]
    value = {"ordinary": secrets, "public": public}
    before = copy.deepcopy(value)
    expected = {"ordinary": safe_secrets, "public": public}
    assert _redact_card_value(value) == expected
    serialized = json.dumps(value, ensure_ascii=False)
    safe = _redact_card_value(serialized)
    assert json.loads(safe) == expected
    assert "private-interior" not in safe
    assert "display unavailable" not in safe
    assert _redact_card_value(safe) == safe
    assert value == before
