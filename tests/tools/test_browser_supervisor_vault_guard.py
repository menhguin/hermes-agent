"""Generic per-connection evaluation guard required by an explicit-tab consumer."""
import asyncio
import threading
from unittest.mock import patch

from tools.browser_supervisor import CDPSupervisor


def test_bound_evaluation_revalidates_before_sending_expression():
    sup = CDPSupervisor('guard', 'ws://127.0.0.1:1')
    sup.before_evaluate = lambda: False
    with patch.object(sup, '_cdp') as send:
        result = sup.evaluate_runtime('secret-bearing-expression')
    assert result == {'ok': False, 'error': 'Browser binding is no longer valid'}
    send.assert_not_called()


def test_bound_native_inspection_cannot_fall_back_to_another_browser(monkeypatch):
    from tools import browser_vault_tool
    from tools import browser_tool_session
    from tools.browser_supervisor import SUPERVISOR_REGISTRY
    from types import SimpleNamespace
    monkeypatch.setattr(SUPERVISOR_REGISTRY, 'get', lambda task: SimpleNamespace(target_id='exact', evaluate_runtime=lambda expression: {'ok': False, 'error': 'supervisor is not active'}))
    calls = []
    monkeypatch.setattr(browser_tool_session, '_run_browser_command', lambda *a: calls.append(a) or {'success': False})
    assert browser_vault_tool._eval_js('bound', 'location.href')['success'] is False
    assert calls == []


def test_public_session_owner_getter_is_thread_local():
    from agent.vault_backends import unlock
    assert hasattr(unlock, 'get_current_session_id')
    previous = unlock.get_current_session_id()
    unlock.set_current_session_id('parent-owner')
    try:
        owners = []
        def child():
            owners.append(unlock.get_current_session_id())
            unlock.set_current_session_id('child-owner')
            owners.append(unlock.get_current_session_id())
        thread = threading.Thread(target=child); thread.start(); thread.join()
        assert owners == [None, 'child-owner']
        assert unlock.get_current_session_id() == 'parent-owner'
    finally:
        unlock.set_current_session_id(previous)


def test_bound_inspection_exception_cannot_fall_back(monkeypatch):
    from tools import browser_vault_tool, browser_tool_session
    from tools.browser_supervisor import SUPERVISOR_REGISTRY
    from types import SimpleNamespace
    def broken(expression): raise ImportError('private-exception-canary')
    monkeypatch.setattr(SUPERVISOR_REGISTRY, 'get', lambda task: SimpleNamespace(target_id='exact', evaluate_runtime=broken))
    calls = []
    monkeypatch.setattr(browser_tool_session, '_run_browser_command', lambda *a: calls.append(a) or {'success': False})
    result = browser_vault_tool._eval_js('bound', 'location.href')
    assert result['success'] is False
    assert calls == []


def test_guard_exception_does_not_escape_or_leak():
    sup = CDPSupervisor('guard', 'ws://127.0.0.1:1')
    def broken():
        raise RuntimeError('sensitive-error-canary')
    sup.before_evaluate = broken
    assert sup.evaluate_runtime('anything') == {'ok': False, 'error': 'Browser binding is no longer valid'}
