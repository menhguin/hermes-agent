"""Escaped named secrets cannot corrupt display copies or kill the card drain."""
import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from gateway.session_state import SessionState
from gateway.slack_task_stream import _redact_card_event
from tests.gateway.slack_task_renderer import RenderingSlackClient
from tests.gateway.test_rich_slack_task_cards import direct_adapter, make_turn


NAMED_FIELDS = (
    "apiKey", "api_key", "password", "access_token", "refresh_token", "auth_token",
    "bearer", "secret", "secret_value", "raw_secret", "secret_input", "key_material",
)
VALUES = [
    'synthetic-quote-"private-suffix',
    "synthetic-backslash-\\private-suffix\\",
    "synthetic-control-\n\t\r\x00\x1bprivate-suffix",
    'synthetic-mixed-\\"\n\t\x00private-suffix',
]


def nested_args(value):
    # These opaque values are NOT registered and have no provider prefix. Only
    # the canonical named-field policy can recognize them as secrets.
    return {"query": {"accounts": [{field: value for field in NAMED_FIELDS}],
                      "ordinary": 'public "quoted" \\path\ntext',
                      "token": "CPU", "token_count": 128, "secretary": "public"}}


@pytest.mark.parametrize("value", VALUES, ids=["quote", "backslash", "controls", "mixed"])
def test_structural_named_secrets_mask_whole_values_without_mutating_arguments(monkeypatch, value):
    monkeypatch.setattr("agent.redact._REDACT_ENABLED", False)
    args = nested_args(value)
    event = {"type": "tool.started", "tool_call_id": "call", "tool_name": "web_search", "args": args}
    before = copy.deepcopy(event)
    safe = _redact_card_event(event)
    assert event == before
    record = safe["args"]["query"]["accounts"][0]
    assert all(record[field] != value for field in NAMED_FIELDS)
    assert "private-suffix" not in json.dumps(safe)
    for field in ("ordinary", "token", "token_count", "secretary"):
        assert safe["args"]["query"][field] == args["query"][field]
    assert safe["args"] is not args
    # The value's original JSON escaping was valid all along.
    assert json.loads(json.dumps(args)) == args


@pytest.mark.asyncio
@pytest.mark.parametrize("value", VALUES, ids=["quote", "backslash", "controls", "mixed"])
@pytest.mark.parametrize("mode", [
    "rich", "fallback", "redactor_failure", "redactor_failure_fallback",
    "named_policy_failure", "named_policy_failure_fallback",
    "preview_failure", "preview_failure_fallback",
])
async def test_escaped_secret_or_failed_redactor_keeps_later_tool_publishing(monkeypatch, caplog, value, mode):
    from agent import redact

    adapter, _ = direct_adapter()
    client = RenderingSlackClient()
    adapter._team_clients["T2"] = client
    config = {"display": {"platforms": {"slack": {
        "tool_progress": "all", "tool_progress_native": True,
        "tool_progress_native_output_chars": 300,
    }}}}
    ctx, turn = await make_turn(monkeypatch, config, adapter)
    # Real generation/slot lifecycle, not the older helper's always-current stub.
    runner = turn._runner
    del runner.__dict__["_run_still_current_fn"]
    state = SessionState()
    runner._sessions = {"session": state}
    state.persistent.run_generation = 1
    state.turn.agent = ctx.agent_holder[0] = SimpleNamespace(is_interrupted=False)
    ctx._run_still_current = runner._run_still_current_fn("session", 1)
    runner._persist_active_agents = lambda: None  # no runtime status writes
    runner._draining = False
    monkeypatch.setattr(redact, "_REDACT_ENABLED", False)

    wire = []
    later = asyncio.Event()
    ordinary = "later_ordinary_publication"
    original_api = client.api_call

    async def api(method, *, json):
        if "fallback" in mode and method == "chat.appendStream":
            raise RuntimeError("synthetic card transport failure")
        result = await original_api(method, json=json)
        if any(c.get("id") == "later" and c.get("status") == "complete" for c in json.get("chunks", [])):
            later.set()
        return result

    async def post(**kwargs):
        wire.append(("post", copy.deepcopy(kwargs)))
        return {"ok": True, "ts": "fallback-1"}

    async def edit(**kwargs):
        wire.append(("edit", copy.deepcopy(kwargs)))
        if f"- write_file - {ordinary} - complete" in kwargs.get("text", ""):
            later.set()
        return {"ok": True, "ts": "fallback-1"}

    client.api_call = api
    client.chat_postMessage = post
    client.chat_update = edit
    if mode.startswith("redactor_failure"):
        original_redact = redact.redact_sensitive_text

        def fail_sensitive(text, **kwargs):
            if "redactor-bomb" in text:
                # Exception strings are tainted too; they must not reach fallback/logs.
                raise ValueError(value)
            return original_redact(text, **kwargs)

        monkeypatch.setattr(redact, "redact_sensitive_text", fail_sensitive)
        args = {"query": "redactor-bomb " + value, "nested": nested_args(value)}
    elif mode.startswith("named_policy_failure"):
        def fail_policy(*args, **kwargs):
            raise ValueError(value)

        monkeypatch.setattr(redact, "_should_redact_assignment", fail_policy)
        # Failure of the dictionary's named-field pass can replace the WHOLE
        # args value. An old, clipped preview must not survive that replacement.
        args = {"password": value, "query": value}
    elif mode.startswith("preview_failure"):
        args = {"text": value, "password": value}
    else:
        args = nested_args(value)
    before = copy.deepcopy(args)
    name = "browser_type" if mode.startswith("preview_failure") else "web_search"
    turn.native_tool_start_callback("secret-call", name, args)
    turn.native_tool_complete_callback("secret-call", name, args, {"nested": nested_args(value)})
    if mode.startswith("preview_failure"):
        def fail_preview_redactor(*args, **kwargs):
            raise ValueError(value)

        # The preview builder has its own forced browser-input redaction pass.
        # Fail it in the display consumer, after the real callbacks enqueue.
        monkeypatch.setattr("agent.display.redact_sensitive_text", fail_preview_redactor)
    turn.native_tool_start_callback("later", "write_file", {"path": ordinary, "content": ordinary})
    turn.native_tool_complete_callback("later", "write_file", {}, "saved")
    progress = asyncio.create_task(turn.send_progress_messages())
    waiter = asyncio.create_task(later.wait())
    cleanup_result = []
    try:
        done, _ = await asyncio.wait({progress, waiter}, timeout=3, return_when=asyncio.FIRST_COMPLETED)
        error = progress.exception() if progress in done and not progress.cancelled() else None
        assert waiter in done, f"later event not delivered; consumer error: {type(error).__name__}"
        assert not progress.done(), "bad display data terminated the progress consumer"
    finally:
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        tracking = asyncio.create_task(asyncio.Event().wait())
        cleanup_result = await asyncio.gather(runner._run_agent_cleanup_turn_tasks(
            ctx, progress_task=progress, log_task=None, interrupt_monitor=None,
            _notify_task=None, tracking_task=tracking, stream_task=None,
        ), return_exceptions=True)
        await asyncio.gather(progress, return_exceptions=True)
    assert not any(isinstance(result, BaseException) for result in cleanup_result)
    assert args == before
    serialized = json.dumps([client.calls, wire])
    assert "private-suffix" not in serialized
    assert "private-suffix" not in caplog.text
    assert "redactor-bomb" not in serialized
    assert ordinary in serialized
    if "fallback" in mode:
        assert {kind for kind, _ in wire} == {"post", "edit"}
    else:
        assert not wire
    assert ctx.progress_queue.empty()
    assert ctx._task_cards_closed
    assert state.turn.agent is None
    assert state.persistent.run_generation == 1
