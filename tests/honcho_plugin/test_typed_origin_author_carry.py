"""Typed-origin filtering composes with the target author-routing contract."""
from unittest.mock import Mock

import pytest

from agent.memory_manager import MemoryManager
from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.client import HonchoClientConfig


def provider_and_dispatch(monkeypatch):
    provider = HonchoMemoryProvider()
    provider._config = HonchoClientConfig(save_messages=True, ai_peer="assistant")
    provider._session_key = "shared-session"
    provider._session_initialized = True
    provider._manager = Mock()
    provider._manager.assistant_peer_id.return_value = "assistant"
    provider._manager.resolve_author_peer_id.side_effect = lambda key, id, name=None, **kw: id
    manager = MemoryManager()
    manager._providers = [provider]
    monkeypatch.setattr(manager, "_submit_background", lambda fn, **kw: fn())
    return provider, manager


@pytest.mark.parametrize("author", [{"id": "human", "is_bot": False}, {"id": "bot:coder", "is_bot": True}])
def test_typed_notification_stops_before_initialization_or_author_lookup(monkeypatch, author):
    provider, dispatcher = provider_and_dispatch(monkeypatch)
    ready = Mock(return_value=True)
    spawn = Mock()
    monkeypatch.setattr(provider, "_ready_or_kick_init", ready)
    monkeypatch.setattr(provider, "_spawn_write", spawn)
    dispatcher.sync_all("unrecognized machine event", "response", turn_author=author,
                        messages=[{"role": "user", "display_kind": "internal_notification"},
                                  {"role": "assistant", "content": "response"}])
    ready.assert_not_called()
    assert provider._manager.mock_calls == []
    spawn.assert_not_called()


@pytest.mark.parametrize("is_bot", [False, True])
def test_real_turn_preserves_author_snapshot_and_session_routing(monkeypatch, is_bot):
    provider, dispatcher = provider_and_dispatch(monkeypatch)
    pending = []
    monkeypatch.setattr(provider, "_spawn_write", lambda fn, *a: pending.append(fn))
    author = {"id": "bot:coder" if is_bot else "human-B", "name": "test", "is_bot": is_bot}
    dispatcher.sync_all("real contribution", "answer", turn_author=author,
                        messages=[{"role": "user", "display_kind": "internal_notification"},
                                  {"role": "user", "content": "real contribution"}])
    assert len(pending) == 1
    # A later turn may mutate the stash before the scheduled write runs.
    provider._turn_author = {"id": "unrelated-later-author", "is_bot": False}
    provider._manager.resolve_author_peer_id.reset_mock()
    pending[0]()
    provider._manager.resolve_author_peer_id.assert_not_called()
    if is_bot:
        provider._manager.get_or_create.assert_called_once_with(provider._a2a_session_key(author), user_peer_id=author["id"])
    else:
        provider._manager.get_or_create.assert_called_once_with("shared-session")
    session = provider._manager.get_or_create.return_value
    assert session.add_message.call_args_list[0].args == ("user", "real contribution")
    assert session.add_message.call_args_list[0].kwargs == {"author_peer_id": None if is_bot else "human-B"}
    assert session.add_message.call_args_list[1].args == ("assistant", "answer")
    provider._manager.save.assert_called_once_with(session)
