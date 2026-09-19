"""An explicit caller's supervisor requirement survives registry lifecycle changes."""
import asyncio
import threading
from contextvars import copy_context
from types import SimpleNamespace

import pytest

from tools import browser_supervisor as bs, browser_vault_tool as vault


@pytest.fixture
def connection(monkeypatch):
    registry = bs._SupervisorRegistry()
    monkeypatch.setattr(bs, "SUPERVISOR_REGISTRY", registry)
    supervisor = bs.CDPSupervisor("bound", "ws://local.invalid", target_id="selected",
                                  instrument_dialogs=False)
    supervisor._set_active(True)
    registry._by_task["bound"] = supervisor
    return registry, supervisor


@pytest.mark.parametrize("kind", ["inspection", "secret"])
@pytest.mark.parametrize("loss", ["missing", "replaced", "stopped"])
def test_scope_refuses_lost_registry_without_native_fallback(connection, monkeypatch, kind, loss):
    registry, supervisor = connection
    from tools import browser_tool_session
    calls = []
    monkeypatch.setattr(browser_tool_session, "_run_browser_command",
                        lambda *a: calls.append("fallback") or {"success": True})
    monkeypatch.setattr(registry, "get_or_start", lambda **k: calls.append("restart"))
    evaluate = vault._eval_js if kind == "inspection" else vault._eval_js_secret
    scope = getattr(bs, "require_supervisor", None)
    assert scope is not None, "Missing same-supervisor invocation scope"
    with pytest.raises(bs.SupervisorBindingError):
        with scope("bound", supervisor):
            if loss == "missing":
                registry._pop("bound")
            elif loss == "replaced":
                registry._by_task["bound"] = SimpleNamespace(
                    evaluate_runtime=lambda *a: calls.append("replacement") or {"ok": True})
            else:
                supervisor.stop()
            assert evaluate("bound", "nonsecret canary")["success"] is False
    assert calls == []


@pytest.mark.parametrize("race", ["during_guard", "after_scheduling"])
def test_scope_rechecks_actual_dispatch_not_only_lookup(connection, monkeypatch, race):
    registry, supervisor = connection
    sent = []
    class Wire:
        async def send(self, payload):
            sent.append(payload)
            raise AssertionError("Revoked command reached websocket")
    supervisor._ws = Wire()
    supervisor._page_session_id = "selected-session"
    supervisor._loop = SimpleNamespace(is_running=lambda: True)
    def evict():
        registry._pop("bound")
        return True
    supervisor.before_evaluate = evict if race == "during_guard" else lambda: True
    def schedule(coro, loop, **kwargs):
        if race == "after_scheduling":
            evict()
        return asyncio.run(coro)
    monkeypatch.setattr(bs, "_schedule", schedule)
    scope = getattr(bs, "require_supervisor", None)
    assert scope is not None, "Missing same-supervisor invocation scope"
    with pytest.raises(bs.SupervisorBindingError):
        with scope("bound", supervisor):
            assert vault._eval_js_secret("bound", "nonsecret canary")["success"] is False
    assert sent == []


def test_scope_is_task_profile_context_local_and_restored(connection, tmp_path):
    from hermes_constants import set_hermes_home_override, reset_hermes_home_override
    registry, supervisor = connection
    other = bs.CDPSupervisor("other", "ws://local.invalid")
    registry._by_task["other"] = other
    scope = getattr(bs, "require_supervisor", None)
    assert scope is not None, "Missing same-supervisor invocation scope"
    with pytest.raises(bs.SupervisorBindingError):
        with scope("bound", supervisor):
            results = []
            worker = threading.Thread(target=lambda: results.append(bs.get_scoped_supervisor("other")))
            worker.start(); worker.join(timeout=5)
            assert results == [other]
            with pytest.raises(bs.SupervisorBindingError):
                bs.get_scoped_supervisor("other")
    assert bs.get_scoped_supervisor("other") is other
    with pytest.raises(bs.SupervisorBindingError):
        with scope("bound", supervisor):
            token = set_hermes_home_override(tmp_path / "foreign")
            try:
                with pytest.raises(bs.SupervisorBindingError):
                    bs.get_scoped_supervisor("bound")
            finally:
                reset_hermes_home_override(token)
    assert bs.get_scoped_supervisor("bound") is supervisor


@pytest.mark.parametrize("error", [None, RuntimeError, KeyboardInterrupt])
def test_exited_scope_revokes_captured_context_without_revoking_outer_scope(connection, error):
    _, supervisor = connection
    with bs.require_supervisor("bound", supervisor):
        outer = copy_context()
        captured = None
        try:
            with bs.require_supervisor("bound", supervisor):
                captured = copy_context()
                assert captured.run(bs.get_scoped_supervisor, "bound") is supervisor
                if error is not None:
                    raise error("synthetic handler exit")
        except (RuntimeError, KeyboardInterrupt) as exc:
            assert type(exc) is error
        # Resetting a ContextVar alone leaves the captured invocation authorized.
        assert captured is not None
        with pytest.raises(bs.SupervisorBindingError):
            captured.run(bs.get_scoped_supervisor, "bound")
        assert outer.run(bs.get_scoped_supervisor, "bound") is supervisor
        assert bs.get_scoped_supervisor("bound") is supervisor
    with pytest.raises(bs.SupervisorBindingError):
        outer.run(bs.get_scoped_supervisor, "bound")
    assert bs.get_scoped_supervisor("bound") is supervisor


@pytest.mark.parametrize("target", [None, "selected"])
def test_required_connection_failure_never_falls_back_even_without_target(connection, monkeypatch, target):
    from tools import browser_tool_session
    _, supervisor = connection
    supervisor.target_id = target
    # A stale active snapshot must not grant authority to start another browser
    # when the required connection's loop has gone away.
    calls = []
    monkeypatch.setattr(browser_tool_session, "_run_browser_command",
                        lambda *a: calls.append("fallback") or {"success": True, "data": {}})
    with bs.require_supervisor("bound", supervisor):
        assert vault._eval_js("bound", "nonsecret canary")["success"] is False
    assert calls == []


def test_owned_stop_preserves_replacement_and_reaps_evicted_connection(connection, monkeypatch):
    registry, supervisor = connection
    stopped = []
    replacement = SimpleNamespace(stop=lambda: stopped.append("replacement"))
    registry._by_task["bound"] = replacement
    monkeypatch.setattr(supervisor, "stop", lambda: stopped.append("owned"))
    bs.stop_supervisor("bound", expected=supervisor)
    assert registry.get("bound") is replacement
    assert stopped == ["owned"]
    bs.stop_supervisor("bound")
    assert registry.get("bound") is None
    assert stopped == ["owned", "replacement"]
