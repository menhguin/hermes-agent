"""Pocket's position in the existing signature chain is part of the contract."""
import base64
import hashlib
import hmac
from types import SimpleNamespace

import pytest
from multidict import CIMultiDict

from gateway.config import PlatformConfig
from gateway.platforms import webhook

NOW = 1_800_000_000
BODY = b"raw bytes\x00\xff"
SECRET = "synthetic-secret"


def digest(body):
    return hmac.new(SECRET.encode(), body, hashlib.sha256).hexdigest()


def validate(monkeypatch, headers):
    monkeypatch.setattr(webhook.time, "time", lambda: NOW)
    request = SimpleNamespace(headers=CIMultiDict(headers), match_info={"route_name": "test"})
    adapter = webhook.WebhookAdapter(PlatformConfig(enabled=True, extra={"routes": {}}))
    return adapter._validate_signature(request, BODY, SECRET)


@pytest.mark.parametrize("scheme", ["svix", "standard", "linear", "github", "gitlab", "v2", "v1"])
@pytest.mark.parametrize("pocket_valid", [False, True])
def test_scheme_precedence_preserves_target_and_pocket_contract(monkeypatch, scheme, pocket_valid):
    ts = str(NOW * 1000)
    headers = {"X-HeyPocket-Timestamp": ts,
               "X-HeyPocket-Signature": digest(ts.encode() + b"." + BODY) if pocket_valid else ""}
    if scheme in {"svix", "standard"}:
        prefix = "svix" if scheme == "svix" else "webhook"
        signature = base64.b64encode(hmac.new(SECRET.encode(), b"msg." + str(NOW).encode() + b"." + BODY, hashlib.sha256).digest()).decode()
        headers.update({prefix + "-id": "msg", prefix + "-timestamp": str(NOW), prefix + "-signature": "v1," + signature})
    else:
        headers.update({
            "linear": {"linear-signature": digest(BODY)},
            "github": {"X-Hub-Signature-256": "sha256=" + digest(BODY)},
            "gitlab": {"X-Gitlab-Token": "wrong-token" if pocket_valid else SECRET},
            "v2": {"X-Webhook-Timestamp": str(NOW), "X-Webhook-Signature-V2": digest(str(NOW).encode() + b"." + BODY)},
            "v1": {"X-Webhook-Signature": digest(BODY)},
        }[scheme])
    assert validate(monkeypatch, headers) is (pocket_valid or scheme in {"svix", "standard", "linear", "github"})


@pytest.mark.parametrize("timestamp", ["01800000000000", "+1800000000000", " 1800000000000 "])
def test_hmac_binds_exact_timestamp_representation(monkeypatch, timestamp):
    headers = {"X-HeyPocket-Timestamp": timestamp,
               "X-HeyPocket-Signature": digest(timestamp.encode() + b"." + BODY)}
    assert validate(monkeypatch, headers) is True
    headers["X-HeyPocket-Timestamp"] = str(NOW * 1000)
    assert validate(monkeypatch, headers) is False
