"""Parent cleanup ends live display, not the background child's work.

Offline regression for the parent-close false-red probe. Only transport is fake;
real callback/queue/cleanup and Slack's append/replace contract are exercised.
"""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from gateway.slack_task_stream import RichTaskCardSession
from tests.gateway.slack_task_renderer import RenderingSlackClient
from tests.gateway.test_rich_slack_task_cards import direct_adapter, drain_turn, make_turn


class ClosingSlack(RenderingSlackClient):
    def __init__(self, fail_child_start=False):
        super().__init__()
        self.fail_child_start = fail_child_start
        self.at_close = {}

    async def api_call(self, method, *, json):
        if method == "chat.startStream" and self.open_count and self.fail_child_start:
            raise RuntimeError("synthetic child-start failure")
        if method == "chat.stopStream":
            self.at_close[json["ts"]] = {
                task: deepcopy(card) for (ts, task), card in self.cards.items() if ts == json["ts"]
            }
        return await super().api_call(method, json=json)


@pytest.mark.asyncio
@pytest.mark.parametrize("fallback", [False, True])
async def test_parent_close_hands_off_unfinished_child_without_success(monkeypatch, fallback):
    adapter, _ = direct_adapter()
    client = ClosingSlack(fallback)
    adapter._team_clients["T2"] = client
    ctx, turn = await make_turn(monkeypatch, {"display": {
        "tool_progress": "all", "tool_progress_native": True,
    }}, adapter)
    turn.progress_callback("subagent.start", subagent_id="child", goal="background investigation")
    turn.progress_callback("subagent.tool", "write_file", args={"content": "unique child body"}, subagent_id="child")
    await drain_turn(turn)
    cards = client.at_close["stream-1" if fallback else "stream-2"]
    assert cards, "the child must have been rendered before parent cleanup"
    # Pending is a documented Slack status. It is neither terminal success nor
    # the in_progress-at-stop shape responsible for false error triangles.
    assert all(c["status"] == "pending" for c in cards.values())
    assert all("result pending" in c["title"].lower() for c in cards.values())
    assert all("completed" not in c["title"].lower() and "failed" not in c["title"].lower() for c in cards.values())
    if not fallback:
        assert cards["t1"]["details"] == "unique child body"  # no append-only replay
    stopped = [p["ts"] for m, p in client.calls if m == "chat.stopStream"]
    assert len(stopped) == len(set(stopped)) == client.open_count
    before = len(client.calls)
    turn.progress_callback("subagent.complete", subagent_id="child", status="completed", summary="late result")
    assert ctx.progress_queue.empty()
    assert len(client.calls) == before


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["completed", "failed"])
async def test_real_child_completion_is_not_relabelled_by_parent_close(monkeypatch, status):
    adapter, _ = direct_adapter()
    client = ClosingSlack()
    adapter._team_clients["T2"] = client
    ctx, _ = await make_turn(monkeypatch, {}, adapter)
    session = RichTaskCardSession(adapter, ctx)
    common = {"subagent_id": "child", "goal": "background investigation"}
    await session.publish([
        dict(common, type="subagent.start"),
        dict(common, type="subagent.tool", tool_name="read_file"),
        dict(common, type="subagent.complete", status=status, summary="unique terminal result"),
    ])
    before = deepcopy(client.calls)
    await session.publish([dict(common, type="subagent.complete", status=status)])
    assert client.calls == before
    await session.stop()
    await session.stop()
    cards = client.at_close["stream-2"]
    assert all(c["status"] == "complete" for c in cards.values())
    assert all("result pending" not in c["title"].lower() for c in cards.values())
    assert ("failed" in cards["t2"]["title"]) is (status == "failed")
    assert cards["t2"]["details"] == "unique terminal result"
    assert session.child_completed == {"child"}
    assert sum(m == "chat.stopStream" and p["ts"] == "stream-2" for m, p in client.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("rollover", ["age", "msg_too_long"])
@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("terminal", [None, "completed", "failed"])
async def test_child_rollover_stays_pending_until_real_completion(monkeypatch, rollover, fallback, terminal):
    adapter, _ = direct_adapter()
    client = ClosingSlack()
    original = client.api_call
    fail_child_start = fallback
    reject_append = False

    async def fail_once(method, *, json):
        nonlocal fail_child_start, reject_append
        if fail_child_start and method == "chat.startStream" and client.open_count == 1:
            fail_child_start = False
            raise RuntimeError("synthetic child-start failure")
        if reject_append and method == "chat.appendStream":
            reject_append = False
            raise RuntimeError("msg_too_long")
        return await original(method, json=json)

    client.api_call = fail_once
    adapter._team_clients["T2"] = client
    ctx, _ = await make_turn(monkeypatch, {}, adapter)
    session = RichTaskCardSession(adapter, ctx)
    common = {"subagent_id": "child", "goal": "background investigation"}
    await session.publish([
        dict(common, type="subagent.start"),
        dict(common, type="subagent.tool", tool_name="write_file", args={"content": "unique child body"}),
    ])
    stream = session.main if fallback else session.children["child"]
    assert stream is not None
    for _ in range(2):
        old_ts = stream.ts
        before = {tid: deepcopy(card) for (ts, tid), card in client.cards.items() if ts == old_ts}
        if rollover == "age":
            stream._stream_opened_at -= stream.ROLLOVER_MAX_AGE_S + 1
        else:
            reject_append = True
        await session.publish([dict(common, type="subagent.tool", tool_name="read_file")])
        assert stream.ts != old_ts
        archived = client.at_close[old_ts]
        assert archived.keys() == before.keys()
        assert all(card["status"] == "pending" for card in archived.values())
        assert all(card["title"].endswith("⤵") for card in archived.values())
        for tid, card in before.items():
            assert archived[tid].get("details") == card.get("details")  # settlement never appends
            assert client.cards[stream.ts, tid]["status"] == "in_progress"  # same continuation identity
        assert session.child_completed == set()
    assert all(card["status"] != "complete" for _, card in client.snapshots)
    if not fallback:
        assert client.cards[stream.ts, "t1"]["details"] == "unique child body"
    archived_before_completion = deepcopy(client.at_close)
    if terminal:
        await session.publish([dict(common, type="subagent.complete", status=terminal, summary="unique result")])
        before = deepcopy(client.calls)
        await session.publish([dict(common, type="subagent.complete", status=terminal)])
        assert client.calls == before
    await session.stop()
    await session.stop()
    for ts, cards in archived_before_completion.items():
        assert client.at_close[ts] == cards  # completion must not rewrite old segments
    current = client.at_close[stream.ts]
    assert all(card["status"] == ("complete" if terminal else "pending") for card in current.values())
    assert all(("result pending" in card["title"]) is (terminal is None) for card in current.values())
    if terminal:
        result = current["sub_child" if fallback else "t4"]
        assert ("failed" in result["title"]) is (terminal == "failed")
        assert session.child_completed == {"child"}
    else:
        assert session.child_completed == set()
    stopped = [p["ts"] for m, p in client.calls if m == "chat.stopStream"]
    assert len(stopped) == len(set(stopped)) == client.open_count
    before = deepcopy(client.calls)
    await session.publish([dict(common, type="subagent.complete", status="completed")])
    assert client.calls == before


@pytest.mark.asyncio
async def test_muted_notification_clears_rich_reasoning_callback(monkeypatch):
    adapter, client = direct_adapter()
    ctx, turn = await make_turn(monkeypatch, {"display": {
        "tool_progress": "all", "tool_progress_native": True,
    }}, adapter)
    agent = SimpleNamespace()
    turn._wire_turn_agent_callbacks(agent, {}, None, None, None, False)
    saved = agent.reasoning_callback
    ctx.mute_notification_reply = True
    turn._wire_turn_agent_callbacks(agent, {}, None, None, None, False)
    assert agent.reasoning_callback is None
    saved("must not escape via a previous callback")
    assert ctx.progress_queue.empty()
    assert client.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("invalidate", ["stop", "stale", "decline"])
async def test_handoff_rechecks_egress_after_each_accepted_update(monkeypatch, invalidate):
    from gateway.platforms.base import SendResult
    adapter, _ = direct_adapter()
    client = ClosingSlack()
    adapter._team_clients["T2"] = client
    ctx, _ = await make_turn(monkeypatch, {}, adapter)
    ctx.agent_holder[0] = SimpleNamespace(is_interrupted=False)
    session = RichTaskCardSession(adapter, ctx)
    common = {"subagent_id": "child", "goal": "background investigation"}
    await session.publish([
        dict(common, type="subagent.start"),
        dict(common, type="subagent.tool", tool_name="read_file"),
    ])
    original = client.api_call
    after_handoff = []
    invalidated = False

    async def invalidating(method, *, json):
        nonlocal invalidated
        if invalidated:
            after_handoff.append((method, deepcopy(json)))
        result = await original(method, json=json)
        if any(c.get("status") == "pending" for c in json.get("chunks", [])):
            invalidated = True
            if invalidate == "stop":
                ctx.agent_holder[0].is_interrupted = True
            elif invalidate == "stale":
                ctx._run_still_current = lambda: False
            else:
                adapter._outbound_blocked = lambda *a: SendResult(
                    success=False, error="refused", raw_response={"code": "egress_declined"})
        return result

    client.api_call = invalidating
    await session.stop()
    assert invalidated, "this test must cross the new handoff publication boundary"
    if invalidate == "decline":
        assert after_handoff == []
    else:
        assert after_handoff
        assert all(m == "chat.stopStream" and set(p) == {"channel", "ts"} for m, p in after_handoff)
    assert "child" not in session.child_completed
    child_tasks = [c for _, p in client.calls if p.get("ts") == "stream-2" for c in p.get("chunks", [])]
    assert not any(c.get("status") == "complete" for c in child_tasks)
