"""Pocket admission through real local HTTP; no external delivery or model."""
import asyncio
import hashlib
import hmac
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from multidict import CIMultiDict

from gateway.config import PlatformConfig
from gateway.platforms import webhook

SECRET = "synthetic-pocket-secret"
NOW = 1_800_000_000.0
BODY = b'{ "event": "transcript.edited", "text": "raw spacing" }\n'


def digest(body):
    return hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()


@pytest.mark.parametrize("headers", [
    {"X-HeyPocket-Signature": ""},
    {"X-HeyPocket-Timestamp": ""},
    {"X-HeyPocket-Signature": "", "X-HeyPocket-Timestamp": ""},
    {"X-HeyPocket-Signature": "bad"},
    {"X-HeyPocket-Timestamp": str(int(NOW * 1000))},
    {"X-HeyPocket-Signature": "bad", "X-HeyPocket-Timestamp": "9" * 1000},
])
@pytest.mark.parametrize("fallback", ["gitlab", "v2", "v1"])
def test_any_pocket_header_presence_commits_before_valid_lower_scheme(monkeypatch, headers, fallback):
    monkeypatch.setattr(webhook, "time", SimpleNamespace(time=lambda: NOW))
    lower = {
        "gitlab": {"X-Gitlab-Token": SECRET},
        "v2": {"X-Webhook-Timestamp": str(int(NOW)),
               "X-Webhook-Signature-V2": digest(str(int(NOW)).encode() + b"." + BODY)},
        "v1": {"X-Webhook-Signature": digest(BODY)},
    }[fallback]
    adapter = webhook.WebhookAdapter(PlatformConfig(enabled=True, extra={"routes": {}}))
    request = SimpleNamespace(headers=CIMultiDict({**lower, **headers}), match_info={"route_name": "pocket"})
    assert adapter._validate_signature(request, BODY, SECRET) is False


@pytest.mark.asyncio
async def test_http_raw_body_admits_once_and_rejects_downgrades(monkeypatch):
    monkeypatch.setattr(webhook, "time", SimpleNamespace(time=lambda: NOW))
    adapter = webhook.WebhookAdapter(PlatformConfig(enabled=True, extra={"routes": {
        "pocket": {"secret": SECRET, "prompt": "{text}", "deliver": "log"},
    }}))
    events, admitted = [], asyncio.Event()
    async def capture(event):
        events.append(event)
        admitted.set()
    monkeypatch.setattr(adapter, "handle_message", capture)
    app = web.Application()
    app.router.add_post("/webhooks/{route_name}", adapter._handle_webhook)
    timestamp = str(int(NOW * 1000))
    signed = {"X-HeyPocket-Timestamp": timestamp,
              "X-HeyPocket-Signature": digest(timestamp.encode() + b"." + BODY)}
    async with TestClient(TestServer(app, host="127.0.0.1")) as client:
        for headers, body in [
            ({"X-HeyPocket-Signature": "", "X-Gitlab-Token": SECRET}, BODY),
            ({"X-HeyPocket-Timestamp": "", "X-Webhook-Signature": digest(BODY)}, BODY),
            ({**signed, "X-Hub-Signature-256": "sha256=invalid"}, BODY),
            (signed, BODY.rstrip()),
        ]:
            response = await client.post("/webhooks/pocket", data=body, headers=headers)
            assert response.status == 401
            assert not events
        response = await client.post("/webhooks/pocket", data=BODY, headers={**signed, "X-Gitlab-Token": "wrong"})
        assert response.status == 202
        await asyncio.wait_for(admitted.wait(), 5)
    assert len(events) == 1
    assert events[0].text == "raw spacing"
    assert events[0].source.user_id == "webhook:pocket"
    assert events[0].raw_message == {"event": "transcript.edited", "text": "raw spacing"}
    await adapter.disconnect()
