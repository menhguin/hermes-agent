"""Real end-of-turn caller -> memory manager -> Honcho attribution boundary.

Gateway notifications carry display_kind on their transcript row. Start at the
completed-turn synchronization seam; only the SDK/session sink and background
scheduling are replaced. No live memory or network is used.
"""

from types import SimpleNamespace

import pytest


@pytest.mark.parametrize(
    "text,kind,expected_write",
    [
        ("Scheduler completed synthetic job", "internal_notification", False),
        ("[Example User] Synthetic completion", "internal_notification", False),
        (
            "[Example User | Slack user <@U012ABCDEF>] "
            "[ASYNC DELEGATION BATCH COMPLETE] task finished",
            None,
            False,
        ),
        ("Please investigate the synthetic completion", None, True),
        ('What does "[ASYNC DELEGATION BATCH COMPLETE]" mean?', None, True),
    ],
    ids=["typed", "typed-prefixed", "legacy-slack", "human", "human-quote"],
)
def test_completed_turn_preserves_origin_boundary(monkeypatch, text, kind, expected_write):
    from agent.memory_manager import MemoryManager
    from plugins.memory.honcho import HonchoMemoryProvider
    from run_agent import AIAgent

    writes, saves = [], []
    session = SimpleNamespace(
        add_message=lambda role, content, **kw: writes.append((role, content))
    )

    class Sink:
        def resolve_author_peer_id(self, key, author_id, name=None):
            return author_id

        def get_or_create(self, key):
            return session

        def save(self, value):
            saves.append(value)

    provider = HonchoMemoryProvider()
    provider._config = SimpleNamespace(save_messages=True, message_max_chars=25000)
    provider._manager = Sink()
    provider._session_key = "synthetic-session"
    provider._session_initialized = True
    monkeypatch.setattr(provider, "_spawn_write", lambda fn, *a, **kw: fn())
    manager = MemoryManager()
    manager._providers = [provider]
    monkeypatch.setattr(manager, "_submit_background", lambda fn, **kw: fn())
    monkeypatch.setattr(manager, "queue_prefetch_all", lambda *a, **kw: None)
    agent = AIAgent.__new__(AIAgent)
    agent._memory_manager = manager
    agent.session_id = "synthetic-session"
    agent._turn_author = None
    # An old internal row must not suppress a later genuine human turn.
    messages = [
        {"role": "user", "content": "earlier machine event", "display_kind": "internal_notification"},
        {"role": "assistant", "content": "old response"},
        {"role": "user", "content": text},
        {"role": "assistant", "content": "final response"},
    ]
    if kind:
        messages[-2]["display_kind"] = kind
    agent._sync_external_memory_for_turn(
        original_user_message=text,
        final_response="final response",
        interrupted=False,
        messages=messages,
    )
    if expected_write:
        assert writes == [("user", text), ("assistant", "final response")]
        assert saves == [session]
    else:
        assert writes == []
        assert saves == []
