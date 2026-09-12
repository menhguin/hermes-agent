"""Whole-burst sanitation must precede lossless, API-sized reasoning appends."""
import json

import pytest

from agent.redact import (
    clear_vault_redaction_values,
    redact_sensitive_text,
    register_vault_redaction_value,
)
from gateway.slack_task_stream import RichTaskCardSession, SlackTaskStream
from tests.gateway.slack_task_renderer import RenderingSlackClient
from tests.gateway.test_rich_slack_task_cards import direct_adapter, make_turn


async def session_with_renderer(monkeypatch, cap=0):
    adapter, _ = direct_adapter()
    client = RenderingSlackClient()
    adapter._team_clients["T2"] = client
    config = {"display": {"platforms": {"slack": {
        "tool_progress": "all", "tool_progress_native": True,
        "tool_progress_native_reasoning_chars": cap,
        "tool_progress_native_rollover_chars": 10_000,
    }}}}
    ctx, _ = await make_turn(monkeypatch, config, adapter)
    return RichTaskCardSession(adapter, ctx), client


async def tool(session, identity):
    assert (await session.publish([
        {"type": "tool.started", "tool_call_id": identity, "tool_name": "terminal"},
        {"type": "tool.completed", "tool_call_id": identity, "tool_name": "terminal"},
    ])).success


def thoughts(client):
    return {key: card for key, card in client.cards.items() if key[1].startswith("think")}


def details(client):
    return "".join(card.get("details", "") for card in thoughts(client).values())


async def burst(session, client, text, boundary):
    if boundary != "first_tool":
        await tool(session, "seed")
    # Mimic provider deltas, but do not let chunk/timer boundaries expose partial secrets.
    for offset in range(0, len(text), 4093):
        assert (await session.publish([
            {"type": "reasoning.delta", "text": text[offset:offset + 4093]},
        ])).success
        assert details(client) == ""
    if boundary != "turn_close":
        await tool(session, "next")
    await session.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["first_tool", "later_tool", "turn_close"])
@pytest.mark.parametrize("shape", ["sentences", "long_tail", "no_sentence"])
@pytest.mark.parametrize("cap", [0, 1, 20, 80])
async def test_large_safe_burst_conserves_text_and_positive_budget(monkeypatch, boundary, shape, cap):
    session, client = await session_with_renderer(monkeypatch, cap)
    if shape == "sentences":
        middle = " ".join(f"Sentence {i:04d} describes a safe deterministic observation." for i in range(1300))
        text = "UNIQUE_HEAD begins this thought. " + middle + " UNIQUE_TAIL ends this thought."
    else:
        prefix = "UNIQUE_HEAD begins this thought. " if shape == "long_tail" else "UNIQUE_HEAD "
        text = prefix + " ".join(f"word{i:05d}" for i in range(8000)) + " UNIQUE_TAIL"
    assert len(text) > 2 * SlackTaskStream.SLACK_FIELD_CEILING
    assert redact_sensitive_text(text, force=True) == text
    expected = (text + " ")[:cap] if cap else text + " "
    await burst(session, client, text, boundary)
    assert details(client) == expected
    assert all(card["status"] == "complete" for card in thoughts(client).values())
    wire_details = [c["details"] for _, p in client.calls for c in p.get("chunks", [])
                    if c.get("id", "").startswith("think") and "details" in c]
    assert all(len(value) <= SlackTaskStream.SLACK_FIELD_CEILING for value in wire_details)
    assert session.main._reasoning_sent_chars == len(expected)
    assert not session.main._in_progress
    if not cap:
        assert client.open_count > 1
        # Continuations retain one burst identity, and never replay its prefix.
        assert len({tid for _, tid in thoughts(client)}) == 1
        assert details(client).count("UNIQUE_HEAD") == details(client).count("UNIQUE_TAIL") == 1
    else:
        assert all(len(card.get("details", "")) <= cap for _, card in client.snapshots)


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["first_tool", "later_tool", "turn_close"])
@pytest.mark.parametrize("kind", ["opaque", "prefixed"])
async def test_large_burst_redacts_split_secret_before_prospective_chunk_boundary(monkeypatch, request, boundary, kind):
    session, client = await session_with_renderer(monkeypatch)
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
    secret = "opaque-vault-fixture-value-with-a-private-suffix" if kind == "opaque" else "sk-" + "SyntheticCredential" * 3
    if kind == "opaque":
        register_vault_redaction_value(secret)
        request.addfinalizer(clear_vault_redaction_values)
    prefix = "UNIQUE_HEAD " + "safe word " * 4000
    prefix = prefix[:SlackTaskStream.SLACK_FIELD_CEILING - 12] + " "
    text = prefix + secret + " " + "safe tail " * 4100 + "UNIQUE_TAIL."
    safe = " ".join(redact_sensitive_text(text, force=True).split())
    assert secret not in safe
    if boundary != "first_tool":
        await tool(session, "seed")
    # Split both the credential and its future output-chunk boundary. Child events
    # intentionally do not flush the main model's partially recognized credential.
    cut = len(prefix) + 14
    for part in (text[:cut], text[cut:]):
        assert (await session.publish([{"type": "reasoning.delta", "text": part}])).success
        assert details(client) == ""
        assert (await session.publish([{"type": "subagent.tool", "subagent_id": "background",
                                        "tool_name": "terminal"}])).success
        assert details(client) == ""
    if boundary != "turn_close":
        await tool(session, "next")
    await session.stop()
    assert details(client) == safe + " "
    assert client.open_count > 2  # child plus main continuations
    wire = json.dumps(client.calls)
    assert secret not in wire
    assert secret[:14] not in wire
    assert secret[14:] not in wire
    assert all(card["status"] == "complete" for card in thoughts(client).values())
