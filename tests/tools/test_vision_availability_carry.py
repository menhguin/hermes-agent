"""Image availability must not accidentally promise auxiliary video capability."""
from unittest.mock import Mock

import pytest

from agent import auxiliary_client, image_routing
from tools import vision_tools as vision
from tools.registry import registry


@pytest.mark.asyncio
async def test_native_image_exists_without_auxiliary_client(monkeypatch):
    monkeypatch.setattr(vision, "_should_use_native_vision_fast_path", lambda: True)
    resolver = Mock(return_value=(None, None))
    monkeypatch.setattr(auxiliary_client, "resolve_vision_provider_client", resolver)
    image = registry.get_entry("vision_analyze")
    assert image.check_fn() is True
    resolver.assert_not_called()
    assert registry.get_entry("video_analyze").check_fn() is False
    png = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    result = await image.handler({"image_url": "data:image/png;base64," + png, "question": "What is here?"})
    assert result["_multimodal"] is True
    assert result["meta"]["native_vision"] is True


@pytest.mark.parametrize("native", [False, RuntimeError("capability unavailable")])
@pytest.mark.parametrize("auxiliary", [False, True])
def test_native_probe_failure_keeps_auxiliary_fallback(monkeypatch, native, auxiliary):
    def probe():
        if isinstance(native, Exception):
            raise native
        return native
    monkeypatch.setattr(vision, "_should_use_native_vision_fast_path", probe)
    calls = []
    def resolve(**kwargs):
        calls.append(kwargs)
        return None, object() if auxiliary and kwargs.get("provider") == "auto" else None
    monkeypatch.setattr(auxiliary_client, "resolve_vision_provider_client", resolve)
    assert registry.get_entry("vision_analyze").check_fn() is auxiliary
    assert calls == [{}, {"provider": "auto"}]


def test_profile_media_veto_keeps_native_only_tool_unavailable(monkeypatch):
    monkeypatch.setattr(auxiliary_client, "_read_main_provider", lambda: "xiaomi")
    monkeypatch.setattr(auxiliary_client, "_read_main_model", lambda: "mimo-v2.5")
    monkeypatch.setattr(image_routing, "decide_image_input_mode", lambda *a: "native")
    lookup = Mock(return_value=True)
    monkeypatch.setattr(image_routing, "_lookup_supports_vision", lookup)
    monkeypatch.setattr(auxiliary_client, "resolve_vision_provider_client", lambda **kw: (None, None))
    assert registry.get_entry("vision_analyze").check_fn() is False
    lookup.assert_not_called()
