"""Rich cards: real wiring, canonical redaction and yielding Slack wire boundaries."""
import asyncio
import copy
import json
from types import SimpleNamespace
from typing import Callable

import pytest

from agent.redact import redact_sensitive_text
from gateway.slack_task_stream import RichTaskCardSession
from tests.gateway.test_rich_slack_task_cards import direct_adapter, make_turn


class YieldingSlack:
    """Record accepted wire writes; gate either side of acceptance deterministically."""

    def __init__(self):
        self.calls = []
        self.requests = []
        self.opens = 0
        self.gate: Callable[[str, dict], bool] | None = None
        self.accept_first = False
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.returned = asyncio.Event()

    async def api_call(self, method, *, json):
        payload = copy.deepcopy(json)
        self.requests.append((method, payload))
        gated = self.gate is not None and self.gate(method, payload)
        if gated:
            self.gate = None
        if gated and not self.accept_first:
            self.entered.set()
            await self.release.wait()
        # Every request yields, including ungated calls and content-free closure.
        await asyncio.sleep(0)
        self.calls.append((method, payload))
        if method == "chat.startStream":
            self.opens += 1
            result = {"ok": True, "ts": f"s{self.opens}"}
        else:
            result = {"ok": True}
        if gated and self.accept_first:
            self.entered.set()
            await self.release.wait()
        if gated:
            self.returned.set()
        return result


def chunks(client):
    return [c for _, payload in client.calls for c in payload.get("chunks", [])]


async def wired_turn(monkeypatch):
    adapter, _ = direct_adapter()
    client = YieldingSlack()
    adapter._team_clients["T2"] = client
    config = {"display": {"platforms": {"slack": {
        "tool_progress": "all", "tool_progress_native": True,
        "tool_progress_native_output_chars": 300,
    }}}}
    ctx, turn = await make_turn(monkeypatch, config, adapter)
    ctx.agent_holder[0] = SimpleNamespace(is_interrupted=False)
    return adapter, client, ctx, turn


@pytest.mark.asyncio
@pytest.mark.parametrize("tool,field", [("write_file", "content"), ("patch", "new_string"), ("execute_code", "code")])
@pytest.mark.parametrize("split", [False, True])
async def test_wire_text_uses_canonical_redaction_before_clipping(monkeypatch, tool, field, split):
    adapter, client, ctx, turn = await wired_turn(monkeypatch)
    session = RichTaskCardSession(adapter, ctx)
    # Real outbound policy is forced even when storage/tool redaction is disabled.
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
    secret = "sk-" + "syntheticCardCredential" * 2
    thought = f"Inspect the complete credential safely before the next tool: {secret}."
    safe = redact_sensitive_text(thought, force=True)
    assert safe != thought
    await session.publish([{"type": "tool.started", "tool_call_id": "seed", "tool_name": "terminal"}])
    parts = [thought] if not split else [thought[:thought.index(secret) + 5], thought[thought.index(secret) + 5:]]
    for part in parts:
        # Cross the actual session's timed flush boundary on every partial delta.
        session.reasoning_last = 0
        await session.publish([{"type": "reasoning.delta", "text": part}])
    body = f"payload prefix {secret}\nSERVICE_TOKEN=opaque_synthetic_fixture_value"
    # Also test canonical handling of control-split tokens in complete argument payloads.
    if split:
        body = body.replace(secret, secret[:8] + "\x1b" + secret[8:])
    safe_body = redact_sensitive_text(body, force=True)
    events = [
        {"type": "tool.started", "tool_call_id": "call", "tool_name": tool,
         "preview": secret, "args": {field: body}},
        {"type": "tool.completed", "tool_call_id": "call", "tool_name": tool,
         "args": {field: body}, "result": {"output": secret}},
        {"type": "subagent.start", "subagent_id": "child", "goal": secret},
        {"type": "subagent.complete", "subagent_id": "child", "summary": secret, "status": "completed"},
        {"type": "tool.started", "tool_call_id": "clip", "tool_name": tool,
         "args": {field: "x" * 270 + " " + secret}},
    ]
    # The real callback has already truncated its preview. Rebuild that title
    # from complete redacted args; scrubbing the clipped prefix is too late.
    preview_args = {"path": "x" * 53 + " " + secret, field: "x" * 53 + " " + secret}
    assert redact_sensitive_text(preview_args[field], force=True) != preview_args[field]
    turn.native_tool_start_callback("preview", tool, preview_args)
    events.append(ctx.progress_queue.get_nowait())
    original = copy.deepcopy(events)
    assert (await session.publish(events)).success
    await session.stop()
    wire = json.dumps(client.calls)
    assert secret not in wire, [c for c in chunks(client) if secret in str(c)]
    assert "opaque_synthetic_fixture_value" not in wire
    rendered_thought = "".join(c.get("details", "") for c in chunks(client) if c.get("id", "").startswith("think"))
    assert rendered_thought.strip() == safe
    detail = next(c["details"] for c in chunks(client) if c.get("id") == "call" and "details" in c)
    assert detail == safe_body
    clipped = next(c["details"] for c in chunks(client) if c.get("id") == "clip")
    assert clipped == redact_sensitive_text("x" * 270 + " " + secret, force=True)[:300]
    from agent.display import build_tool_preview
    safe_args = {k: redact_sensitive_text(v, force=True) for k, v in preview_args.items()}
    expected_preview = build_tool_preview(tool, safe_args, max_len=64)
    title = next(c["title"] for c in chunks(client) if c.get("id") == "preview")
    assert title.endswith(expected_preview)
    assert events == original  # Display sanitation must not mutate tool inputs / history.


@pytest.mark.asyncio
async def test_multiline_reasoning_and_wire_content_preserve_identity(monkeypatch):
    adapter, client, ctx, _ = await wired_turn(monkeypatch)
    session = RichTaskCardSession(adapter, ctx)
    identity = "ghp_" + "opaqueIdentityNotACredential"
    pem = "-----BEGIN PRIVATE KEY-----\nsynthetic body with spaces\n-----END PRIVATE KEY-----"
    thought = "Inspect this multiline fixture before continuing: " + pem + " done."
    assert redact_sensitive_text(thought, force=True) != thought
    await session.publish([{"type": "tool.started", "tool_call_id": identity, "tool_name": "terminal"}])
    cut = thought.index("PRIVATE") + 3
    for text in [thought[:cut], thought[cut:]]:
        session.reasoning_last = 0
        await session.publish([{"type": "reasoning.delta", "text": text}])
        # Background child progress is NOT a boundary of the main model's
        # reasoning stream; it can arrive in the middle of a credential.
        await session.publish([{"type": "subagent.tool", "subagent_id": "background", "tool_name": "terminal"}])
    await session.stop()
    actual = "".join(c.get("details", "") for c in chunks(client) if c.get("id", "").startswith("think"))
    assert actual.strip() == redact_sensitive_text(thought, force=True)
    assert any(c.get("id") == identity for c in chunks(client))


async def cleanup_turn(turn, progress):
    """Use the real normal turn-finally cancellation and join path."""
    turn._ctx.session_key = None  # no live session store / lease to release
    turn._runner._draining = False
    tracking = asyncio.create_task(asyncio.Event().wait())
    await turn._runner._run_agent_cleanup_turn_tasks(
        turn._ctx, progress_task=progress, log_task=None, interrupt_monitor=None,
        _notify_task=None, tracking_task=tracking, stream_task=None,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["start", "completion", "child_summary"])
@pytest.mark.parametrize("accept_first", [False, True])
async def test_normal_cleanup_joins_owned_publication_exactly_once(monkeypatch, boundary, accept_first):
    adapter, client, ctx, turn = await wired_turn(monkeypatch)
    summary = "The child found the verified answer."
    client.accept_first = accept_first
    def gate(method, payload):
        if boundary == "start":
            return method == "chat.startStream"
        return method == "chat.appendStream" and any(
            (c.get("id") == "call" and c.get("status") == "complete")
            if boundary == "completion" else c.get("details") == summary
            for c in payload.get("chunks", [])
        )
    client.gate = gate
    turn.native_tool_start_callback("call", "write_file", {"content": "main body"})
    turn.native_tool_complete_callback("call", "write_file", {}, "saved")
    if boundary == "child_summary":
        turn.progress_callback("subagent.start", subagent_id="child", goal="investigate")
        turn.progress_callback("subagent.complete", subagent_id="child", status="completed", summary=summary)
    progress = asyncio.create_task(turn.send_progress_messages())
    await asyncio.wait_for(client.entered.wait(), 3)
    cleanup = asyncio.create_task(cleanup_turn(turn, progress))
    # Run cancellation before resuming the awaited transport boundary.
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    client.release.set()
    await asyncio.wait_for(cleanup, 3)
    assert client.returned.is_set(), "normal cleanup canceled the only publication owner"
    completed = [c for c in chunks(client) if c.get("id") == "call" and c.get("status") == "complete"]
    assert len(completed) == 1
    assert sum(c.get("details") == "main body" for c in chunks(client)) == 1
    if boundary == "child_summary":
        detail = [c for c in chunks(client) if c.get("details") == summary]
        assert len(detail) == 1
        assert any(c.get("id") == detail[0]["id"] and c.get("status") == "complete" for c in chunks(client))
    stopped = [p["ts"] for m, p in client.calls if m == "chat.stopStream"]
    assert len(stopped) == client.opens == len(set(stopped))
    assert ctx.progress_queue.empty()
    turn.native_tool_start_callback("late", "write_file", {"content": "must not enqueue"})
    assert ctx.progress_queue.empty()


@pytest.mark.asyncio
async def test_cleanup_timeout_never_replays_an_ambiguously_accepted_append(monkeypatch):
    adapter, client, ctx, turn = await wired_turn(monkeypatch)
    before = asyncio.all_tasks()
    client.accept_first = True
    client.gate = lambda method, payload: method == "chat.appendStream" and any(
        c.get("id") == "call" and c.get("status") == "complete" for c in payload.get("chunks", [])
    )
    turn.native_tool_start_callback("call", "write_file", {"content": "one body"})
    turn.native_tool_complete_callback("call", "write_file", {}, "saved")
    progress = asyncio.create_task(turn.send_progress_messages())
    await asyncio.wait_for(client.entered.wait(), 3)
    turn.native_tool_start_callback("later", "write_file", {"content": "must be abandoned"})
    # Exercise the real bounded join, not a no-yield timeout mock.
    await asyncio.wait_for(cleanup_turn(turn, progress), 8)
    assert not client.returned.is_set()
    assert sum(c.get("id") == "call" and c.get("status") == "complete" for c in chunks(client)) == 1
    assert not any(c.get("id") == "later" for c in chunks(client))
    assert [p["ts"] for m, p in client.calls if m == "chat.stopStream"] == ["s1"]
    assert ctx.progress_queue.empty()
    assert asyncio.all_tasks() <= before  # no shielded publication left behind
    mark = len(client.requests)
    client.release.set()
    await asyncio.sleep(0)
    assert len(client.requests) == mark


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["start", "child_start", "append"])
@pytest.mark.parametrize("invalidation", ["stop", "stale"])
@pytest.mark.parametrize("via_runner", [False, True])
async def test_invalidated_turn_allows_only_content_free_closure(monkeypatch, boundary, invalidation, via_runner):
    adapter, client, ctx, turn = await wired_turn(monkeypatch)
    events = [{"type": "tool.started", "tool_call_id": "call", "tool_name": "write_file",
               "args": {"content": "file body must not arrive after stop"}}]
    if boundary == "child_start":
        events.append({"type": "subagent.start", "subagent_id": "child", "goal": "child body"})
    def gate(method, payload):
        if boundary == "append":
            return method == "chat.appendStream"
        return method == "chat.startStream" and (boundary == "start" or client.opens == 1)
    client.gate = gate
    client.accept_first = True
    async def forbidden(**kwargs):
        pytest.fail("turn invalidation is not permission to fall back to text")
    adapter.send = forbidden
    if via_runner:
        for event in events:
            ctx.progress_queue.put(event)
        publication = asyncio.create_task(turn.send_progress_messages())
        session = None
    else:
        session = RichTaskCardSession(adapter, ctx)
        publication = asyncio.create_task(session.publish(events))
    await asyncio.wait_for(client.entered.wait(), 3)
    mark = len(client.requests)
    if invalidation == "stop":
        ctx.agent_holder[0].is_interrupted = True
    else:
        ctx._run_still_current = lambda: False
    if via_runner:
        cleanup = asyncio.create_task(cleanup_turn(turn, publication))
        await asyncio.sleep(0)
    client.release.set()
    if via_runner:
        await asyncio.wait_for(cleanup, 3)
    else:
        await asyncio.wait_for(publication, 3)
        # Even the default normal stop must re-check the now-invalid turn.
        await session.stop()
    new_requests = client.requests[mark:]
    assert new_requests
    assert all(m == "chat.stopStream" for m, _ in new_requests)
    assert all(set(p) == {"channel", "ts"} for _, p in new_requests)
    stopped = [p["ts"] for m, p in client.calls if m == "chat.stopStream"]
    assert len(stopped) == client.opens == len(set(stopped))
    assert client.returned.is_set()  # the accepted start/append wasn't orphaned


@pytest.mark.asyncio
@pytest.mark.parametrize("opened", [False, True])
async def test_egress_refusal_is_terminal_even_for_closure(monkeypatch, opened):
    from gateway.platforms.base import SendResult
    from gateway.relay.egress import declined_send
    adapter, client, ctx, _ = await wired_turn(monkeypatch)
    session = RichTaskCardSession(adapter, ctx)
    event = {"type": "tool.started", "tool_call_id": "call", "tool_name": "write_file", "args": {"content": "body"}}
    if opened:
        assert (await session.publish([event])).success
    mark = len(client.requests)
    refused = SendResult(success=False, error="fixture refusal", raw_response={"code": "egress_declined"})
    adapter._outbound_blocked = lambda *args: refused
    result = await session.publish([event])
    assert declined_send(result)
    adapter._outbound_blocked = lambda *args: None
    assert declined_send(await session.publish([event]))
    ctx.agent_holder[0].is_interrupted = True
    await session.stop(flush_reasoning=False)
    assert client.requests[mark:] == []
