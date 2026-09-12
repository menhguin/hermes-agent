"""C-006: degraded child transport must not turn failure into success."""
import pytest

from gateway.slack_task_stream import RichTaskCardSession
from tests.gateway.slack_task_renderer import RenderingSlackClient
from tests.gateway.test_rich_slack_task_cards import direct_adapter, make_turn


class FailingChildClient(RenderingSlackClient):
    def __init__(self, failure_at):
        super().__init__()
        self.failure_at = failure_at
        self.fail_child_append = False
        self.attempts = []

    async def api_call(self, method, *, json):
        self.attempts.append((method, json))
        if method == "chat.startStream" and self.open_count and self.failure_at == "start":
            raise RuntimeError("synthetic child start failure")
        if method == "chat.appendStream" and json["ts"] != "stream-1" and self.fail_child_append:
            raise RuntimeError("synthetic child append failure")
        return await super().api_call(method, json=json)


async def make_session(monkeypatch, failure_at="start"):
    adapter, _ = direct_adapter()
    client = FailingChildClient(failure_at)
    adapter._team_clients["T2"] = client
    ctx, _ = await make_turn(monkeypatch, {}, adapter)
    return RichTaskCardSession(adapter, ctx), client


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "error", "timeout"])
@pytest.mark.parametrize("failure_at", ["start", "tool", "completion"])
async def test_child_fallback_preserves_failure_and_terminal_bookkeeping(monkeypatch, status, failure_at):
    session, client = await make_session(monkeypatch, failure_at)
    child = {"subagent_id": "child-a", "goal": "fixture child goal"}
    assert (await session.publish([{"type": "subagent.start", **child}])).success
    client.fail_child_append = failure_at == "tool"
    assert (await session.publish([{"type": "subagent.tool", "tool_name": "terminal", **child}])).success
    client.fail_child_append = failure_at != "start"
    result = await session.publish([{"type": "subagent.complete", "status": status, **child}])
    assert result.success
    card = client.cards.get((session.main.ts, "sub_child-a"))
    assert card is not None, "A failed dedicated stream must render its terminal event on the main stream"
    assert card["status"] == "complete"
    assert "✗ failed" in card["title"], "Transport degradation must not report the child as successful"
    assert "#1" in card["title"]
    assert "child-a" in session.child_completed
    assert session.children["child-a"]._stopped
    before = len(client.attempts)
    await session.publish([
        {"type": "subagent.tool", "tool_name": "write_file", **child},
        {"type": "subagent.complete", "status": "completed", **child},
    ])
    assert len(client.attempts) == before, "A terminal child cannot be revived or overwritten by late callbacks"
    await session.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("identity", ["subagent_id", "child_session_id", "delegation_id"])
async def test_fallback_child_numbering_and_completion_are_identity_scoped(monkeypatch, identity):
    session, client = await make_session(monkeypatch)
    for index, value in enumerate(("batch-a", "batch-b"), 1):
        raw = {identity: value, "task_index": 0, "goal": f"fixture goal {index}"}
        key = f"{value}:0" if identity == "delegation_id" else value
        await session.publish([{"type": "subagent.start", **raw}])
        await session.publish([{"type": "subagent.tool", "tool_name": "terminal", **raw}])
        await session.publish([{"type": "subagent.complete", "status": "completed", **raw}])
        card = client.cards[session.main.ts, f"sub_{key}"]
        assert card["status"] == "complete" and "failed" not in card["title"]
        assert f"#{index}" in card["title"]
        assert key in session.child_completed
    assert len(session.child_completed) == 2, "Equal task indices in different batches must remain independent"
    await session.stop()
