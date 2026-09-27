"""Synthetic gateway turns must not become human Honcho messages.

The Slack fixture preserves the audited transport/notification header shape;
names, IDs and the payload are anonymized. No SDK or network is needed.
"""

from types import SimpleNamespace

import pytest

from agent.memory_manager import MemoryManager
from plugins.memory.honcho import HonchoMemoryProvider


NOTICE = (
    "[ASYNC DELEGATION BATCH COMPLETE — deleg_example]\n"
    "A background fan-out of 1 subagent(s) you dispatched earlier has finished."
)
SLACK = "[Example User | Slack user <@U012ABCDEF>] "


@pytest.fixture
def capture():
    writes, saves, lookups = [], [], []
    session = SimpleNamespace(add_message=lambda role, text, **kw: writes.append((role, text)))

    class Manager:
        def resolve_author_peer_id(self, key, author_id, name=None):
            return author_id

        def get_or_create(self, key):
            lookups.append(key)
            return session

        def save(self, value):
            saves.append(value)

    provider = HonchoMemoryProvider()
    provider._config = SimpleNamespace(save_messages=True, message_max_chars=25000)
    provider._manager = Manager()
    provider._session_key = "test-session"
    provider._session_initialized = True
    yield provider, writes, saves, lookups
    if provider._sync_thread:
        provider._sync_thread.join(timeout=5)
        assert not provider._sync_thread.is_alive()


@pytest.mark.parametrize("text", [NOTICE, SLACK + NOTICE, SLACK + "[CONTEXT SUMMARY]: reference"])
def test_untyped_notification_does_not_write(capture, text):
    provider, writes, saves, lookups = capture
    provider.sync_turn(text, "completion response")
    if provider._sync_thread:
        provider._sync_thread.join(timeout=5)
    assert writes == []
    assert saves == []
    assert lookups == []


@pytest.mark.parametrize("text", [NOTICE, SLACK + NOTICE, "[Example User] " + NOTICE, "Resume wake-up"])
def test_native_internal_origin_reaches_honcho(capture, monkeypatch, text):
    provider, writes, saves, lookups = capture
    manager = MemoryManager()
    manager._providers = [provider]
    monkeypatch.setattr(manager, "_submit_background", lambda fn, **kw: fn())
    manager.sync_all(text, "completion response", messages=[
        {"role": "user", "content": "earlier genuine request"},
        {"role": "assistant", "content": "working"},
        {"role": "user", "content": text, "display_kind": "internal_notification"},
        {"role": "assistant", "content": "completion response"},
    ])
    if provider._sync_thread:
        provider._sync_thread.join(timeout=5)
    assert writes == []
    assert saves == []
    assert lookups == []


@pytest.mark.parametrize("text", [
    SLACK + "Please check the output.",
    SLACK + 'What does "[ASYNC DELEGATION BATCH COMPLETE]" mean?',
    SLACK + "> [ASYNC DELEGATION BATCH COMPLETE]\nPlease explain this quote.",
    '[Example User] I saw [CONTEXT SUMMARY] in the log.',
    '"[ASYNC DELEGATION BATCH COMPLETE]" is a header I am documenting.',
    "Please don't save the temporary output file.",
])
def test_genuine_turn_writes_after_historical_notification(capture, monkeypatch, text):
    provider, writes, saves, lookups = capture
    manager = MemoryManager()
    manager._providers = [provider]
    monkeypatch.setattr(manager, "_submit_background", lambda fn, **kw: fn())
    manager.sync_all(text, "genuine response", messages=[
        {"role": "user", "content": NOTICE, "display_kind": "internal_notification"},
        {"role": "assistant", "content": "previous response"},
        {"role": "user", "content": text},
        {"role": "assistant", "content": "genuine response"},
    ])
    assert provider._sync_thread is not None
    provider._sync_thread.join(timeout=5)
    assert writes == [("user", text), ("assistant", "genuine response")]
    assert len(saves) == 1
    assert lookups == ["test-session"]
