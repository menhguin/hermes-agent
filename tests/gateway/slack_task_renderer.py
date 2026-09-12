"""Offline Slack contract: details append; other task fields replace.

Identity is (stream timestamp, task id), not task id across a whole turn.
This models the carried, historically probed contract, not a live Slack check.
"""
from copy import deepcopy


class RenderingSlackClient:
    def __init__(self):
        self.calls = []
        self.open_count = 0
        self.cards = {}
        self.snapshots = []

    async def api_call(self, method, *, json):
        payload = deepcopy(json)
        self.calls.append((method, payload))
        if method == "chat.startStream":
            self.open_count += 1
            ts = f"stream-{self.open_count}"
        else:
            ts = payload["ts"]
        for chunk in payload.get("chunks", []):
            if chunk.get("type") != "task_update":
                continue
            card = self.cards.setdefault((ts, chunk["id"]), {})
            for field, value in chunk.items():
                if field == "details":
                    card[field] = card.get(field, "") + value
                else:
                    card[field] = deepcopy(value)
            self.snapshots.append(((ts, chunk["id"]), deepcopy(card)))
        return {"ok": True, "ts": ts}

    async def chat_startStream(self, **kwargs):
        return await self.api_call("chat.startStream", json=kwargs)

    async def chat_appendStream(self, **kwargs):
        return await self.api_call("chat.appendStream", json=kwargs)

    async def chat_stopStream(self, **kwargs):
        return await self.api_call("chat.stopStream", json=kwargs)
