"""Audit native replacement using real gate/catcher, synthetic credentials and no network."""
import socket
from unittest.mock import Mock

import pytest
from agent import auxiliary_client as aux, image_routing
from tools import vision_tools as vision
from tools.registry import registry


@pytest.fixture(autouse=True)
def deny_network(monkeypatch):
    def denied(*args, **kwargs):
        raise AssertionError('network forbidden in core vision audit')
    monkeypatch.setattr(socket.socket, 'connect', denied)
    monkeypatch.setattr(socket, 'create_connection', denied)


@pytest.mark.asyncio
async def test_native_registry_availability_and_image_execution_without_aux(monkeypatch):
    from hermes_cli import config
    monkeypatch.setattr(config, 'load_config', lambda: {
        'model': {'supports_vision': True}, 'agent': {'image_input_mode': 'native'}})
    aux.set_runtime_main('custom', 'audit-local-vision')
    resolver = Mock(return_value=(None, None))
    monkeypatch.setattr(aux, 'resolve_vision_provider_client', resolver)
    try:
        image, video = registry.get_entry('vision_analyze'), registry.get_entry('video_analyze')
        assert vision._should_use_native_vision_fast_path() is True
        assert image.check_fn() is True
        resolver.assert_not_called()
        assert video.check_fn() is False
        png = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII='
        result = await image.handler({'image_url': 'data:image/png;base64,' + png, 'question': 'Inspect'})
        assert result['_multimodal'] is True and result['meta']['native_vision'] is True
    finally:
        aux.clear_runtime_main()


@pytest.mark.parametrize('condition', ['native-off', 'native-routing-crash', 'profile-media-veto'])
@pytest.mark.parametrize('available', [False, True])
def test_real_native_fail_closed_path_keeps_aux_fallback(monkeypatch, condition, available):
    aux.set_runtime_main('xiaomi' if condition == 'profile-media-veto' else 'custom', 'audit-model')
    def routing(*args):
        if condition == 'native-routing-crash':
            raise RuntimeError('capability unavailable')
        return 'native' if condition == 'profile-media-veto' else 'text'
    monkeypatch.setattr(image_routing, 'decide_image_input_mode', routing)
    lookup = Mock(return_value=True)
    monkeypatch.setattr(image_routing, '_lookup_supports_vision', lookup)
    calls = []
    def resolve(**kwargs):
        assert aux._aux_probe_active(), 'availability must not build SDK clients'
        calls.append(kwargs)
        return None, object() if available and kwargs.get('provider') == 'auto' else None
    monkeypatch.setattr(aux, 'resolve_vision_provider_client', resolve)
    try:
        assert registry.get_entry('vision_analyze').check_fn() is available
        assert calls == [{}, {'provider': 'auto'}]
        lookup.assert_not_called()
    finally:
        aux.clear_runtime_main()
