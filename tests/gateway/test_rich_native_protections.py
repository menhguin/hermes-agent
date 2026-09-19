"""v2026.9.14 destination/intent protections apply before rich or basic I/O."""
import copy
import json

import pytest

from gateway.platforms.base import SendResult
from tests.gateway.test_rich_slack_task_cards import direct_adapter, drain_turn, make_turn


@pytest.mark.asyncio
@pytest.mark.parametrize("rich", [False, True])
@pytest.mark.parametrize("explicit", [False, True])
async def test_uncardable_preflight_precedes_transport_and_text_fallback(monkeypatch, rich, explicit):
    adapter, client = direct_adapter()
    adapter.config.extra["native_task_cards"] = True
    adapter._resolve_thread_ts = lambda *args, **kwargs: None
    sends = []

    async def send(**kwargs):
        sends.append(kwargs)
        return SendResult(success=True, message_id="fallback")

    async def forbidden(*args, **kwargs):
        pytest.fail("un-cardable destination must be resolved before any native I/O")

    adapter.send = send
    adapter.send_native_task_card_progress = forbidden
    client.api_call = forbidden
    config = {"display": {"tool_progress_native": rich}}
    if explicit:
        config["display"]["tool_progress"] = "all"
    ctx, turn = await make_turn(monkeypatch, config, adapter)
    turn.native_tool_start_callback("call", "terminal", {"command": "date"})
    await drain_turn(turn)
    assert bool(sends) is explicit
    assert ctx.progress_queue.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("rich", [False, True])
async def test_explicit_off_leaves_no_native_callbacks_or_card_publication(monkeypatch, rich):
    from types import SimpleNamespace
    adapter, client = direct_adapter()
    adapter.config.extra["native_task_cards"] = True
    ctx, turn = await make_turn(monkeypatch, {"display": {
        "tool_progress": "off", "tool_progress_native": rich,
    }}, adapter)
    agent = SimpleNamespace(reasoning_callback=lambda text: pytest.fail("stale sink"))
    turn._wire_turn_agent_callbacks(agent, {}, None, None, None, False)
    assert not ctx._native_slack_task_cards and not ctx._rich_slack_task_cards
    assert agent.tool_start_callback is None and agent.tool_complete_callback is None
    assert agent.reasoning_callback is None
    assert client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("rich", [False, True])
async def test_uncardable_explicit_fallback_redacts_before_preview_clipping(monkeypatch, request, rich):
    from agent.redact import clear_vault_redaction_values, register_vault_redaction_value
    adapter, client = direct_adapter()
    adapter.config.extra["native_task_cards"] = True
    adapter._resolve_thread_ts = lambda *args, **kwargs: None
    secret = "synthetic-opaque-" + "c" * 100
    register_vault_redaction_value(secret)
    request.addfinalizer(clear_vault_redaction_values)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
    sends = []

    async def send(**kwargs):
        sends.append(kwargs)
        return SendResult(success=True, message_id="fallback")

    adapter.send = send
    _, turn = await make_turn(monkeypatch, {"display": {
        "tool_progress": "all", "tool_progress_native": rich,
    }}, adapter)
    args = {"path": secret, "content": "safe"}
    original = copy.deepcopy(args)
    turn.native_tool_start_callback("call", "write_file", args)
    await drain_turn(turn)
    assert sends
    assert secret[:40] not in json.dumps(sends)
    assert args == original
    assert client.calls == []
