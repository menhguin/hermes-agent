"""9.21 reconnect eviction composes with exact-binding resource ownership."""
import threading
from types import SimpleNamespace

import pytest

from tools import browser_supervisor as bs


@pytest.mark.parametrize("replacement_when", ["before", "between_check_and_pop"])
def test_budget_expiry_cannot_evict_another_callers_replacement(monkeypatch, replacement_when):
    registry = bs._SupervisorRegistry()
    monkeypatch.setattr(bs, "SUPERVISOR_REGISTRY", registry)
    old = bs.CDPSupervisor("vault-owner", "ws://127.0.0.1:1", target_id="selected",
                           instrument_dialogs=False)
    replacement = SimpleNamespace(_stop_requested=False)
    registry._by_task[old.task_id] = replacement if replacement_when == "before" else old

    if replacement_when == "between_check_and_pop":
        class ReplaceAfterFirstCriticalSection:
            """Deterministically model publication immediately after lock release."""
            def __init__(self):
                self.lock = threading.Lock()
                self.replaced = False

            def __enter__(self):
                self.lock.acquire()

            def __exit__(self, *exc):
                self.lock.release()
                if not self.replaced:
                    with self.lock:
                        self.replaced = True
                        registry._by_task[old.task_id] = replacement

        registry._lock = ReplaceAfterFirstCriticalSection()

    assert old._reconnect_budget_spent(bs.MAX_POST_ATTACH_RECONNECT_FAILURES, ConnectionError("synthetic"))
    assert registry.get(old.task_id) is replacement
    assert not replacement._stop_requested


def test_budget_expiry_removes_only_its_owned_connection(monkeypatch):
    registry = bs._SupervisorRegistry()
    monkeypatch.setattr(bs, "SUPERVISOR_REGISTRY", registry)
    old = bs.CDPSupervisor("vault-owner", "ws://127.0.0.1:1", target_id="selected",
                           instrument_dialogs=False)
    registry._by_task[old.task_id] = old
    assert not old._reconnect_budget_spent(bs.MAX_POST_ATTACH_RECONNECT_FAILURES - 1, ConnectionError("synthetic"))
    assert registry.get(old.task_id) is old
    assert old._reconnect_budget_spent(bs.MAX_POST_ATTACH_RECONNECT_FAILURES, ConnectionError("synthetic"))
    assert registry.get(old.task_id) is None
