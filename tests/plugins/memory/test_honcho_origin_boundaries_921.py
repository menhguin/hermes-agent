"""Typed-origin filtering with real peer/session resolution and completed turns."""
from copy import deepcopy
from unittest.mock import Mock

import pytest

from agent.memory_manager import MemoryManager
from plugins.memory.honcho import HonchoMemoryProvider
from plugins.memory.honcho.client import HonchoClientConfig
from plugins.memory.honcho.session import HonchoSessionManager
from run_agent import AIAgent


@pytest.mark.parametrize("is_bot", [False, True])
def test_typed_origin_preserves_resolved_owner_and_never_replays_history(monkeypatch, is_bot):
    config = HonchoClientConfig(api_key="synthetic-key", peer_name="declared owner", ai_peer="assistant",
                                user_peer_aliases={"transport-owner": "declared owner"},
                                write_frequency="turn", save_messages=True)
    sessions = HonchoSessionManager(honcho=Mock(), config=config, runtime_user_peer_name="transport-owner")
    # Substitute only the SDK-facing seams: native peer resolver and session buffer remain real.
    monkeypatch.setattr(sessions, "_get_or_create_peer", Mock())
    monkeypatch.setattr(sessions, "_get_or_create_honcho_session", Mock(return_value=(Mock(), [], None)))
    saved = []
    monkeypatch.setattr(sessions, "save", lambda session: saved.append(session))
    provider = HonchoMemoryProvider()
    provider._config = config
    provider._session_key = "shared-session"
    provider._session_initialized = True
    provider._manager = sessions
    monkeypatch.setattr(provider, "_spawn_write", lambda fn, *a: fn())
    manager = MemoryManager()
    manager._providers = [provider]
    monkeypatch.setattr(manager, "_submit_background", lambda fn, **kw: fn())
    monkeypatch.setattr(manager, "queue_prefetch_all", lambda *a, **kw: None)
    agent = AIAgent.__new__(AIAgent)
    agent._memory_manager = manager
    agent.session_id = "shared-session"
    # A bot whose name collides with the resolved owner still gets its own peer.
    author = {"id": "bot:declared-owner" if is_bot else "transport-owner", "is_bot": is_bot}
    agent._turn_author = author
    history = [{"role": "user", "content": "old private request"},
               {"role": "assistant", "content": "old answer"},
               {"role": "user", "content": "synthetic event", "display_kind": "internal_notification"},
               {"role": "assistant", "content": "synthetic answer"}]
    original = deepcopy(history)
    agent._sync_external_memory_for_turn(original_user_message="synthetic event", final_response="synthetic answer",
                                         interrupted=False, messages=history)
    assert saved == []
    sessions._get_or_create_peer.assert_not_called()
    assert history == original
    for index in range(2):
        text, answer = f"current request {index}", f"current answer {index}"
        history.extend([{"role": "user", "content": text}, {"role": "assistant", "content": answer}])
        agent._sync_external_memory_for_turn(original_user_message=text, final_response=answer,
                                             interrupted=False, messages=history)
    assert len(saved) == 2
    assert saved[0] is saved[1]
    session = saved[-1]
    assert [(row["role"], row["content"]) for row in session.messages] == [
        ("user", "current request 0"), ("assistant", "current answer 0"),
        ("user", "current request 1"), ("assistant", "current answer 1")]
    assert all(row.get("author_peer_id") is None for row in session.messages)
    owner = sessions._resolve_user_peer_id("shared-session")
    assert owner == "declared-owner"
    if is_bot:
        assert session.key == provider._a2a_session_key(author)
        assert session.user_peer_id != owner
        assert "shared-session" not in sessions._cache
    else:
        assert session.key == "shared-session"
        assert session.user_peer_id == owner
