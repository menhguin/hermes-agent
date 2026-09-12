"""Rendered-content regressions for C-004/C-005 (real renderer, fake I/O)."""
import pytest

from gateway.slack_task_stream import SlackTaskStream
from tests.gateway.slack_task_renderer import RenderingSlackClient


def thoughts(client):
    return {key: card for key, card in client.cards.items() if key[1].startswith("think")}


@pytest.mark.asyncio
async def test_fake_slack_appends_details_but_replaces_other_fields_per_stream():
    client = RenderingSlackClient()
    first = (await client.chat_startStream(channel="C1", thread_ts="thread"))["ts"]
    await client.chat_appendStream(ts=first, chunks=[{
        "type": "task_update", "id": "same", "title": "old", "status": "in_progress",
        "details": "AAA", "output": "first output",
    }])
    await client.chat_appendStream(ts=first, chunks=[{
        "type": "task_update", "id": "same", "title": "new", "status": "complete",
        "details": "BBB", "output": "last output",
    }])
    second = (await client.chat_startStream(channel="C1", thread_ts="thread"))["ts"]
    await client.chat_appendStream(ts=second, chunks=[{
        "type": "task_update", "id": "same", "title": "other", "status": "in_progress",
        "details": "CCC",
    }])
    card = client.cards[first, "same"]
    assert card["details"] == "AAABBB"
    assert (card["title"], card["status"], card["output"]) == ("new", "complete", "last output")
    assert client.cards[second, "same"]["details"] == "CCC"


@pytest.mark.asyncio
@pytest.mark.parametrize("cap", [1, 20, 80])
@pytest.mark.parametrize("buffered", [False, True])
async def test_reasoning_budget_bounds_rendered_card_not_each_delta(cap, buffered):
    client = RenderingSlackClient()
    stream = SlackTaskStream(client, "C1", "thread", reasoning_chars=cap)
    if not buffered:
        await stream.task_started("initial", "terminal")
    parts = [f"Segment {index:02d} explains the next deterministic fixture check." for index in range(10)]
    for part in parts:
        await stream.reasoning_update(part)
    await stream.task_started("boundary", "terminal")
    await stream.stop()
    cards = thoughts(client)
    assert cards, "Small positive budgets must not disappear below the default 40-character hold"
    assert len(cards) == 1
    card = next(iter(cards.values()))
    assert len(card["details"]) <= cap, "Slack appends every details delta for this card"
    assert card["details"] == (" ".join(parts) + " ")[:cap]
    assert card["status"] == "complete"
    assert all(len(card.get("details", "")) <= cap for (_, tid), card in client.snapshots if tid.startswith("think"))


@pytest.mark.asyncio
async def test_each_reasoning_burst_gets_a_fresh_budget():
    cap = 20
    client = RenderingSlackClient()
    stream = SlackTaskStream(client, "C1", "thread", reasoning_chars=cap)
    parts = ["First burst has enough text to fill the configured reasoning budget.",
             "Second burst has independent text and must get its own reasoning budget."]
    for index, part in enumerate(parts):
        await stream.reasoning_update(part)
        await stream.task_started(f"boundary-{index}", "terminal")
    await stream.stop()
    assert [card.get("details", "") for card in thoughts(client).values()] == [part[:cap] for part in parts]


@pytest.mark.asyncio
async def test_equal_reasoning_deltas_roll_over_without_replaying_delivered_text():
    client = RenderingSlackClient()
    stream = SlackTaskStream(client, "C1", "thread", rollover_chars=350)
    await stream.task_started("initial", "terminal")
    await stream.task_finished("initial", "terminal")
    parts = [f"Segment {index:02d} explains the next deterministic fixture check." for index in range(20)]
    for part in parts:
        await stream.reasoning_update(part)
    await stream.stop()
    assert client.open_count > 1, "Equal-sized details deltas still grow the rendered message"
    assert "".join(card.get("details", "") for card in thoughts(client).values()) == " ".join(parts) + " "
    assert all(card["status"] == "complete" for card in thoughts(client).values())


@pytest.mark.asyncio
@pytest.mark.parametrize("rollover", ["age", "message_not_in_streaming_state", "msg_too_long"])
@pytest.mark.parametrize("boundary", ["delta", "finalize"])
async def test_reasoning_text_is_conserved_across_rollover_and_finalization(rollover, boundary):
    client = RenderingSlackClient()
    stream = SlackTaskStream(client, "C1", "thread", rollover_age_s=100)
    await stream.task_started("initial", "terminal")
    await stream.task_finished("initial", "terminal")
    first = "First complete sentence has enough text to render immediately."
    tail = "Second unfinished fragment belongs exactly once to this burst"
    await stream.reasoning_update(first)
    await stream.reasoning_update(tail)
    if rollover == "age":
        stream._stream_opened_at -= 101
    else:
        original = client.chat_appendStream
        rejected = False

        async def reject_once(**kwargs):
            nonlocal rejected
            if not rejected:
                rejected = True
                raise RuntimeError(rollover)
            return await original(**kwargs)

        client.chat_appendStream = reject_once
    if boundary == "delta":
        await stream.reasoning_update("and ends here.")
        tail += " and ends here."
    await stream.stop()
    assert client.open_count == 2
    assert "".join(card.get("details", "") for card in thoughts(client).values()) == first + " " + tail + " "
    assert all(card["status"] == "complete" for card in thoughts(client).values())


@pytest.mark.asyncio
async def test_size_accounting_adds_details_and_replaces_only_present_fields():
    client = RenderingSlackClient()
    stream = SlackTaskStream(client, "C1", "thread")
    await stream.ensure_started()
    # Equal deltas, omitted output, growing then shrinking replacement fields.
    for title, status, details, output in [
        ("Long original title", "in_progress", "AAA", "original output"),
        ("Long original title", "in_progress", "BBB", None),
        ("Short", "complete", "CCC", "last"),
    ]:
        await stream._append_raw_task("tool", title, status=status, details=details, output=output)
        rendered = client.cards[stream.ts, "tool"]
        assert stream._sent_chars == sum(len(str(value)) for value in rendered.values())
    await stream.stop()


@pytest.mark.asyncio
async def test_reasoning_budget_does_not_clip_provider_tool_ids_starting_with_think():
    client = RenderingSlackClient()
    stream = SlackTaskStream(client, "C1", "thread", reasoning_chars=1)
    body = "This is file content, not a reasoning burst."
    await stream.task_started("thinking-provider-tool", "write_file", details=body)
    await stream.task_finished("thinking-provider-tool", "write_file")
    await stream.stop()
    assert client.cards[stream.ts, "thinking-provider-tool"]["details"] == body


@pytest.mark.asyncio
async def test_reasoning_budget_is_not_replenished_or_replayed_by_rollover():
    cap = 100
    client = RenderingSlackClient()
    stream = SlackTaskStream(client, "C1", "thread", reasoning_chars=cap, rollover_age_s=100)
    await stream.task_started("initial", "terminal")
    parts = [f"Segment {index:02d} explains the next deterministic fixture check." for index in range(3)]
    await stream.reasoning_update(parts[0])
    stream._stream_opened_at -= 101
    for part in parts[1:]:
        await stream.reasoning_update(part)
    await stream.stop()
    assert client.open_count == 2
    assert "".join(card.get("details", "") for card in thoughts(client).values()) == (" ".join(parts) + " ")[:cap]
    assert all(len(card.get("details", "")) <= cap for (_, tid), card in client.snapshots if tid.startswith("think"))


@pytest.mark.asyncio
async def test_default_reasoning_budget_remains_uncapped_across_many_deltas():
    client = RenderingSlackClient()
    stream = SlackTaskStream(client, "C1", "thread", rollover_chars=100_000)
    await stream.task_started("initial", "terminal")
    parts = [f"Segment {index:04d} explains the next deterministic fixture check." for index in range(600)]
    for part in parts:
        await stream.reasoning_update(part)
    await stream.stop()
    expected = " ".join(parts) + " "
    assert len(expected) > stream.SLACK_FIELD_CEILING
    assert "".join(card.get("details", "") for card in thoughts(client).values()) == expected
    assert client.open_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("configured,wire", [("dense", "plan"), ("plan", "plan"), ("timeline", "timeline")])
async def test_legacy_dense_config_is_a_supported_plan_wire_alias(configured, wire):
    from gateway.display_config import resolve_display_setting

    config = {"display": {"platforms": {"slack": {"tool_progress_native_mode": configured}}}}
    mode = resolve_display_setting(config, "slack", "tool_progress_native_mode")
    assert mode == configured, "Legacy config remains accepted without rewriting user settings"
    client = RenderingSlackClient()
    stream = SlackTaskStream(client, "C1", "thread", task_display_mode=mode, rollover_age_s=100)
    await stream.task_started("initial", "terminal")
    stream._stream_opened_at -= 101
    await stream.task_started("next", "terminal")
    await stream.stop()
    assert client.open_count == 2
    assert stream.task_display_mode == wire
    assert all(payload["task_display_mode"] == wire for method, payload in client.calls if method == "chat.startStream")
