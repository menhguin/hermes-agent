"""Joint contract: real reasoning delivery -> real gateway wiring -> rich cards.

Only Slack I/O and unrelated setup callbacks are faked. No model/API call runs.
"""
import asyncio
from types import MethodType, SimpleNamespace

import pytest

from agent.agent_runtime_helpers import extract_reasoning
from agent.chat_completion_helpers import _assistant_reasoning_text
from agent.stream_delivery import StreamDeliveryMixin
from gateway.config import Platform, PlatformConfig
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from plugins.platforms.slack.adapter import SlackAdapter
from tests.gateway.slack_task_renderer import RenderingSlackClient


@pytest.mark.asyncio
async def test_streamed_reasoning_materializes_once_and_disabled_reuse_cannot_reopen_cards(monkeypatch):
    import gateway.run as gateway_run
    import agent.plugin_stream_hooks as hooks

    monkeypatch.setattr(hooks, "stream_reasoning_deltas_enabled", lambda: False)
    config = {"display": {"platforms": {"slack": {
        "tool_progress": "all", "tool_progress_native": True,
    }}}}
    monkeypatch.setattr(gateway_run, "_load_gateway_config", lambda: config)
    adapter = SlackAdapter(PlatformConfig(enabled=True, extra={"mention_patterns": ["test-agent"]}))
    client = RenderingSlackClient()
    adapter._app = SimpleNamespace(client=client)
    adapter._team_clients["T_JOINT"] = client
    runner = object.__new__(GatewayRunner)
    runner.adapters = {Platform.SLACK: adapter}
    runner._adapter_for_source = lambda source: adapter
    runner._resolve_turn_toolsets = lambda *args: ([], [])
    runner._run_still_current_fn = lambda *args: lambda: True
    runner._service_tier = None
    runner._consume_pending_turn_sidecar_notes = lambda key: []
    runner.hooks = SimpleNamespace(loaded_hooks=[])
    source = SessionSource(platform=Platform.SLACK, chat_id="C_JOINT", thread_id="123.4",
                           user_id="U_JOINT", scope_id="T_JOINT")

    def make_turn():
        display = runner._run_agent_display_settings(source)
        ctx, turn, _ = runner._run_agent_build_turn_context(
            display, SimpleNamespace, message="test request", source=source,
            session_key="joint-session", session_id="joint-session", run_generation=1,
        )
        runner._run_agent_bind_turn_wiring(ctx, turn, source, "123.4", display._native_slack_task_cards)
        turn._make_bg_review_callbacks = lambda: (None, None)
        turn._attach_session_title_callback = lambda *args: None
        return ctx, turn

    ctx, turn = make_turn()
    agent = StreamDeliveryMixin()
    agent._stream_callback = None
    agent.verbose_logging = False
    agent._extract_reasoning = MethodType(extract_reasoning, agent)
    turn._wire_turn_agent_callbacks(agent, {}, None, None, None, False)
    assert callable(getattr(agent, "reasoning_callback", None)), "Rich-card reasoning sink must be wired"
    thought = ("This is synthetic reasoning used to verify the producer and gateway together. "
               "Materializing the same completed response must not deliver this paragraph twice.")
    agent._fire_reasoning_delta(thought)
    message = SimpleNamespace(reasoning=thought, reasoning_content=None, reasoning_details=[], content="answer")
    assert _assistant_reasoning_text(agent, message) == thought
    assert _assistant_reasoning_text(agent, message) == thought
    assert ctx.progress_queue.qsize() == 1
    event = ctx.progress_queue.get_nowait()
    assert event["type"] == "reasoning.delta"
    ctx.progress_queue.put(event)
    agent.tool_start_callback("joint-tool", "terminal", {"command": "synthetic fixture"})
    agent.tool_complete_callback("joint-tool", "terminal", {}, '{"output":"fixture complete"}')
    progress = asyncio.create_task(turn.send_progress_messages())
    await asyncio.sleep(0)
    progress.cancel()
    await progress
    # Details append on (stream ts, task id), while title/status replace.
    # A repeated snapshot body would be a real duplicate, including at finalize.
    assert sum(card.get("details", "").count(thought) for card in client.cards.values()) == 1
    assert client.open_count == 1
    assert client.calls[-1][0] == "chat.stopStream"

    config.clear()
    disabled_ctx, disabled_turn = make_turn()
    disabled_turn._wire_turn_agent_callbacks(agent, {}, None, None, None, False)
    calls_before = len(client.calls)
    assert agent.reasoning_callback is None
    agent._fire_reasoning_delta("A later disabled turn must not retain the old card callback.")
    assert disabled_ctx.progress_queue is None
    assert ctx.progress_queue.empty()
    assert len(client.calls) == calls_before
