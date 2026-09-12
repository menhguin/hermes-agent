"""Carried rich cards use the real gateway wiring, one queue and fake Slack I/O."""
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.display_config import resolve_display_setting
from gateway.run import GatewayRunner
from gateway.session import SessionSource


class NativeAdapter:
    config = SimpleNamespace(extra={"reply_in_thread": True, "mention_patterns": ["carnie"]})
    supports_status_text = False

    def __init__(self, basic=False):
        self.basic = basic

    def native_task_cards_enabled(self):
        return self.basic


async def make_turn(monkeypatch, config, adapter, platform=Platform.SLACK):
    import gateway.run as run
    monkeypatch.setattr(run, "_load_gateway_config", lambda: config)
    runner = object.__new__(GatewayRunner)
    runner.adapters = {platform: adapter}
    runner._adapter_for_source = lambda source: adapter
    runner._resolve_turn_toolsets = lambda *a: ([], [])
    runner._run_still_current_fn = lambda *a: lambda: True
    runner._service_tier = None
    runner._consume_pending_turn_sidecar_notes = lambda key: []
    runner.hooks = SimpleNamespace(loaded_hooks=[])
    source = SessionSource(platform=platform, chat_id="C1", thread_id="123.4", user_id="U2", scope_id="T2")
    display = runner._run_agent_display_settings(source)
    ctx, turn, _ = runner._run_agent_build_turn_context(
        display, SimpleNamespace, message="hello", source=source,
        session_key="session", session_id="session-abcdef", run_generation=1,
    )
    runner._run_agent_bind_turn_wiring(ctx, turn, source, "123.4", display._native_slack_task_cards)
    turn._make_bg_review_callbacks = lambda: (None, None)
    turn._attach_session_title_callback = lambda *a: None
    return ctx, turn


@pytest.mark.asyncio
async def test_legacy_opt_in_wires_reasoning_and_clears_cached_agent_on_next_turn(monkeypatch):
    config = {"display": {"platforms": {"slack": {"tool_progress": "all", "tool_progress_native": True}}}}
    ctx, turn = await make_turn(monkeypatch, config, NativeAdapter())
    assert ctx._native_slack_task_cards is True
    agent = SimpleNamespace()
    turn._wire_turn_agent_callbacks(agent, {}, None, None, None, False)
    assert callable(agent.reasoning_callback)
    agent.reasoning_callback("A substantial reasoning burst before the tool call.")
    assert ctx.progress_queue.get_nowait()["type"] == "reasoning.delta"
    ctx2, turn2 = await make_turn(monkeypatch, {}, NativeAdapter())
    turn2._wire_turn_agent_callbacks(agent, {}, None, None, None, False)
    assert agent.reasoning_callback is None
    assert agent.tool_start_callback is None
    assert agent.tool_complete_callback is None
    assert ctx2.progress_queue is None


class FakeClient:
    def __init__(self):
        self.calls = []
        self.opens = 0

    async def api_call(self, method, *, json):
        self.calls.append((method, json))
        if method == "chat.startStream":
            self.opens += 1
            return {"ok": True, "ts": f"stream-{self.opens}"}
        return {"ok": True}


def direct_adapter():
    from gateway.config import PlatformConfig
    from plugins.platforms.slack.adapter import SlackAdapter
    adapter = SlackAdapter(PlatformConfig(enabled=True, extra={"mention_patterns": ["carnie"]}))
    client = FakeClient()
    adapter._app = SimpleNamespace(client=FakeClient())
    adapter._team_clients["T2"] = client
    # Stale channel cache must not win over the inbound workspace.
    adapter._channel_team["C1"] = "T1"
    return adapter, client


async def drain_turn(turn):
    import asyncio
    task = asyncio.create_task(turn.send_progress_messages())
    await asyncio.sleep(0)
    task.cancel()
    await task


@pytest.mark.asyncio
async def test_real_wiring_publishes_rich_payloads_once_in_order(monkeypatch):
    adapter, client = direct_adapter()
    adapter.config.extra["native_task_cards"] = True
    config = {"display": {"platforms": {"slack": {
        "tool_progress": "all", "tool_progress_native": True,
        "tool_progress_native_mode": "timeline", "tool_progress_native_output_chars": 300,
    }}}}
    ctx, turn = await make_turn(monkeypatch, config, adapter)
    agent = SimpleNamespace()
    turn._wire_turn_agent_callbacks(agent, {}, None, None, None, False)
    thought = "I will inspect the two sources before writing the verified result."
    agent.reasoning_callback(thought)
    args = {"path": "test.txt", "content": "the complete file payload"}
    agent.tool_start_callback("call-a", "write_file", args)
    # Legacy name-only callbacks must not make duplicate progress cards.
    agent.tool_progress_callback("tool.started", "write_file", "test.txt", args)
    agent.tool_start_callback("call-b", "write_file", args)
    agent.tool_complete_callback("call-b", "write_file", args, '{"error":"denied"}')
    agent.tool_complete_callback("call-a", "write_file", args, '{"output":"saved content"}')
    web_args = {"urls": ["https://example.com/source"]}
    agent.tool_start_callback("call-web", "web_extract", web_args)
    agent.tool_complete_callback("call-web", "web_extract", web_args, '{"content":"source content"}')
    await drain_turn(turn)
    assert client.opens == 1
    assert adapter._app.client.calls == []
    start = client.calls[0][1]
    assert start["task_display_mode"] == "timeline"
    assert start["recipient_team_id"] == "T2" and start["recipient_user_id"] == "U2"
    chunks = [c for _, p in client.calls for c in p.get("chunks", [])]
    tasks = [c for c in chunks if c["type"] == "task_update"]
    assert tasks[0]["details"].strip() == thought
    assert tasks[1]["details"] == "the complete file payload"
    assert tasks[1]["id"] == "call-a"
    finished = {c["id"]: c for c in tasks if c["status"] == "complete"}
    assert "failed" in finished["call-b"]["title"]
    assert "failed" not in finished["call-a"]["title"]
    assert finished["call-a"]["output"] == "saved content"
    assert finished["call-web"]["sources"][0]["url"] == web_args["urls"][0]
    assert any("carnie · abcdef" in c["title"] for c in chunks if c["type"] == "plan_update")
    assert sum(c["id"] == "call-a" and c["status"] == "in_progress" for c in tasks) == 1
    assert client.calls[-1][0] == "chat.stopStream"
    assert all(not ("chunks" in p and "markdown_text" in p) for _, p in client.calls)


@pytest.mark.asyncio
async def test_child_streams_are_ordered_and_closed_on_completion_and_turn_cleanup(monkeypatch):
    adapter, client = direct_adapter()
    config = {"display": {"platforms": {"slack": {
        "tool_progress": "all", "tool_progress_native": True, "tool_progress_native_mode": "dense",
    }}}}
    ctx, turn = await make_turn(monkeypatch, config, adapter)
    turn.native_tool_start_callback("parent", "delegate_task", {})
    from tools.delegate_tool_progress import _build_child_progress_callback
    parent = SimpleNamespace(tool_progress_callback=turn.progress_callback)
    child_a = _build_child_progress_callback(0, "first goal", parent,
        session_ref={"session_id": "session-a", "delegation_id": "batch-a"})
    child_b = _build_child_progress_callback(0, "second goal", parent,
        session_ref={"session_id": "session-b", "delegation_id": "batch-b"})
    child_a("subagent.start")
    child_a("tool.started", "write_file", "child.txt", {"content": "child payload"})
    child_b("subagent.start")
    child_a("subagent.complete", status="completed", summary="## Found the result", duration_seconds=12)
    await drain_turn(turn)
    assert client.opens == 3
    headers = [(p["ts"], c["title"]) for _, p in client.calls for c in p.get("chunks", []) if c["type"] == "plan_update"]
    assert any(ts == "stream-2" and "SUBAGENT #1" in title for ts, title in headers)
    assert any(ts == "stream-3" and "SUBAGENT #2" in title for ts, title in headers)
    child_chunks = [c for _, p in client.calls if p.get("ts") == "stream-2" for c in p.get("chunks", [])]
    assert any(c.get("details") == "child payload" for c in child_chunks)
    assert any(c.get("details") == "Found the result" for c in child_chunks)
    stopped = [p["ts"] for method, p in client.calls if method == "chat.stopStream"]
    assert sorted(stopped) == ["stream-1", "stream-2", "stream-3"]
    # Legacy dense config is accepted, but Slack documents only plan/timeline.
    assert all(p["task_display_mode"] == "plan" for m, p in client.calls if m == "chat.startStream")
    # Saved callbacks from the completed turn cannot reopen orphan streams or queue forever.
    turn.progress_callback("subagent.tool", "terminal", "late", subagent_id="child-b")
    turn.native_reasoning_callback("late reasoning")
    assert ctx.progress_queue.empty()


@pytest.mark.asyncio
async def test_transport_failure_falls_back_as_interim_and_stops_open_stream(monkeypatch):
    from gateway.platforms.base import SendResult
    adapter, client = direct_adapter()
    config = {"display": {"platforms": {"slack": {"tool_progress": "all", "tool_progress_native": True}}}}
    ctx, turn = await make_turn(monkeypatch, config, adapter)
    original = client.api_call
    async def failing(method, *, json):
        if method == "chat.appendStream":
            raise RuntimeError("synthetic transport failure")
        return await original(method, json=json)
    client.api_call = failing
    fallbacks = []
    async def send(**kwargs):
        fallbacks.append(kwargs)
        return SendResult(success=True, message_id="fallback")
    async def edit(**kwargs):
        return SendResult(success=True, message_id="fallback")
    adapter.send, adapter.edit_message = send, edit
    turn.native_tool_start_callback("call-a", "terminal", {"command": "date"})
    turn.native_tool_complete_callback("call-a", "terminal", {}, "done")
    await drain_turn(turn)
    assert len(fallbacks) == 1
    assert fallbacks[0]["metadata"]["_interim_send"] is True
    assert [m for m, _ in client.calls].count("chat.stopStream") == 1


@pytest.mark.asyncio
async def test_egress_decline_remains_terminal_for_all_rich_updates(monkeypatch):
    from gateway.platforms.base import SendResult
    adapter, client = direct_adapter()
    config = {"display": {"platforms": {"slack": {"tool_progress": "all", "tool_progress_native": True}}}}
    ctx, turn = await make_turn(monkeypatch, config, adapter)
    adapter._outbound_blocked = lambda *a: SendResult(success=False, error="declined", raw_response={"code": "egress_declined"})
    async def forbidden(**kwargs):
        pytest.fail("declined progress must never fall back to text")
    adapter.send = forbidden
    turn.native_tool_start_callback("call-a", "terminal", {"command": "date"})
    turn.native_tool_complete_callback("call-a", "terminal", {}, "done")
    turn.progress_callback("subagent.start", subagent_id="blocked-child")
    await drain_turn(turn)
    assert client.calls == []


@pytest.mark.asyncio
async def test_interrupted_turn_closes_children_without_flushing_pending_work(monkeypatch):
    import asyncio
    adapter, client = direct_adapter()
    config = {"display": {"platforms": {"slack": {"tool_progress": "all", "tool_progress_native": True}}}}
    ctx, turn = await make_turn(monkeypatch, config, adapter)
    ctx.agent_holder[0] = SimpleNamespace(is_interrupted=False)
    child_open = asyncio.Event()
    original = client.api_call
    async def observed(method, *, json):
        result = await original(method, json=json)
        if method == "chat.appendStream" and json.get("ts") == "stream-2":
            child_open.set()
        return result
    client.api_call = observed
    turn.native_tool_start_callback("first", "delegate_task", {})
    turn.progress_callback("subagent.start", subagent_id="child-a")
    task = asyncio.create_task(turn.send_progress_messages())
    await asyncio.wait_for(child_open.wait(), 3)
    turn.native_reasoning_callback("Pending reasoning must not be delivered after interruption.")
    async def reasoning_consumed():
        while not ctx.progress_queue.empty():
            await asyncio.sleep(0)
    await asyncio.wait_for(reasoning_consumed(), 3)
    turn.native_tool_start_callback("late", "terminal", {"command": "do not show"})
    ctx.agent_holder[0].is_interrupted = True
    task.cancel()
    await task
    assert client.opens == 2
    assert sorted(p["ts"] for m, p in client.calls if m == "chat.stopStream") == ["stream-1", "stream-2"]
    assert not any(c.get("id") == "late" for _, p in client.calls for c in p.get("chunks", []))
    assert "Pending reasoning" not in str(client.calls)
    assert ctx.progress_queue.empty()


@pytest.mark.asyncio
@pytest.mark.parametrize("platform,progress,legacy,basic,native,rich", [
    (Platform.SLACK, "off", True, False, False, False),
    (Platform.SLACK, "off", False, True, True, False),
    (Platform.SLACK, "all", False, True, True, False),
    (Platform.DISCORD, "all", True, True, False, False),
    (Platform.WEBHOOK, "all", True, True, False, False),
])
async def test_stock_and_non_slack_gates_are_preserved(monkeypatch, platform, progress, legacy, basic, native, rich):
    config = {"display": {"tool_progress": progress, "tool_progress_native": legacy}}
    ctx, _ = await make_turn(monkeypatch, config, NativeAdapter(basic), platform)
    assert ctx._native_slack_task_cards is native
    assert ctx._rich_slack_task_cards is rich


@pytest.mark.asyncio
async def test_reactive_rollover_recovers_without_switching_to_text_fallback(monkeypatch):
    from gateway.slack_task_stream import RichTaskCardSession
    adapter, client = direct_adapter()
    ctx, _ = await make_turn(monkeypatch, {}, adapter)
    session = RichTaskCardSession(adapter, ctx)
    original = client.api_call
    failed = False
    async def expired(method, *, json):
        nonlocal failed
        if method == "chat.appendStream" and not failed:
            failed = True
            raise RuntimeError("message_not_in_streaming_state")
        return await original(method, json=json)
    client.api_call = expired
    result = await session.publish([{"type": "tool.started", "tool_call_id": "real-call", "tool_name": "terminal"}])
    await session.stop()
    assert result.success is True
    assert client.opens == 2


@pytest.mark.asyncio
async def test_basic_only_connector_keeps_its_guarded_native_seam(monkeypatch):
    from gateway.platforms.base import SendResult
    class Connector(NativeAdapter):
        def __init__(self):
            super().__init__(basic=True)
            self.publications = 0
        async def send_native_task_card_progress(self, **kwargs):
            self.publications += 1
            return SendResult(success=False, error="declined", raw_response={"code": "egress_declined"})
        async def stop_native_task_card_progress(self, *a, **kwargs):
            pass
        async def send(self, **kwargs):
            pytest.fail("connector refusal must not become rich/text bypass")
    adapter = Connector()
    config = {"display": {"tool_progress": "all", "tool_progress_native": True}}
    _, turn = await make_turn(monkeypatch, config, adapter)
    turn.native_tool_start_callback("call", "terminal", {})
    await drain_turn(turn)
    assert adapter.publications == 1


def test_native_display_tuning_is_normalised():
    config = {"display": {"platforms": {"slack": {
        "tool_progress_native": "off", "tool_progress_native_mode": "TIMELINE",
        "tool_progress_native_output_chars": -1, "tool_progress_native_rollover_age_s": "invalid",
    }}}}
    assert resolve_display_setting(config, "slack", "tool_progress_native") is False
    assert resolve_display_setting(config, "slack", "tool_progress_native_mode") == "timeline"
    assert resolve_display_setting(config, "slack", "tool_progress_native_output_chars") == 0
    assert resolve_display_setting(config, "slack", "tool_progress_native_rollover_age_s") > 0
