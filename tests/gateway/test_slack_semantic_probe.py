"""Offline API/ownership probes from the neutral Slack equivalence audit."""
import copy
from types import SimpleNamespace

import pytest
from gateway.config import PlatformConfig
from plugins.platforms.slack import adapter as slack

META = {"thread_id": "123.4", "team_id": "T2", "user_id": "U2"}


class Wire:
    def __init__(self, fail=()):
        self.calls = []
        self.fail = set(fail)

    async def api_call(self, method, *, json):
        payload = copy.deepcopy(json)
        self.calls.append((method, payload))
        if method in self.fail:
            raise RuntimeError('feature_disabled')
        return {'ok': True}

    async def agents_sessions_setStatus(self, **kwargs):
        return await self.api_call('agents.sessions.setStatus', json=kwargs)

    async def agents_sessions_rename(self, **kwargs):
        return await self.api_call('agents.sessions.rename', json=kwargs)

    async def assistant_threads_setStatus(self, **kwargs):
        return await self.api_call('assistant.threads.setStatus', json=kwargs)

    async def assistant_threads_setTitle(self, **kwargs):
        return await self.api_call('assistant.threads.setTitle', json=kwargs)

def adapter(fail=()):
    a = slack.SlackAdapter(PlatformConfig(enabled=True, extra={'native_task_cards': True, 'mention_patterns':['carnie']}))
    w = Wire(fail)
    a._app = SimpleNamespace(client=Wire())
    a._team_clients = {'T2': w}
    a._channel_team['C1'] = 'T2'
    return a, w

@pytest.mark.asyncio
@pytest.mark.parametrize('case', ['lifecycle', 'rename', 'fallback', 'legacy_sdk'])
async def test_session_contract(case, monkeypatch):
    monkeypatch.setattr(slack, '_AGENT_SESSIONS_SUPPORTED', case != 'legacy_sdk')
    a, w = adapter({'agents.sessions.setStatus','agents.sessions.rename'} if case == 'fallback' else ())
    a._status_text = {'C1': 'Checking facts'}
    await a.send_typing('C1', META)
    await a.stop_typing('C1', META)
    await a._set_assistant_thread_title('C1','123.4','  Test\n title  ',team_id='T2')
    assert a._app.client.calls == []
    if case == 'lifecycle':
        assert [p['status'] for m,p in w.calls if m == 'agents.sessions.setStatus'] == ['processing','active']
    elif case == 'rename':
        assert [p for m,p in w.calls if m == 'agents.sessions.rename'] == [{'channel_id':'C1','thread_ts':'123.4','title':'Test title'}]
        assert a._titled_assistant_threads
    else:
        assert [(m,p.get('status',p.get('title'))) for m,p in w.calls if m.startswith('assistant.')] == [('assistant.threads.setStatus','Checking facts'),('assistant.threads.setStatus',''),('assistant.threads.setTitle','Test title')]
        assert a._titled_assistant_threads

@pytest.mark.asyncio
@pytest.mark.parametrize('route', ['other_thread','other_workspace','missing_thread','unowned'])
async def test_matching_prefix_never_claims_unowned_destination(route):
    from tests.gateway.test_slack_adapter_carries import make_adapter, META, FINAL_META
    a,w = make_adapter()
    await a.send_draft('D1',7,'Main prefix',metadata=META)
    md = dict(FINAL_META)
    if route == 'other_thread':
        md['thread_id'] = 'other'
    elif route == 'other_workspace':
        md['team_id'] = 'T2'
    elif route == 'missing_thread':
        md.pop('thread_id')
    else:
        md.pop('_finalize_draft_id')
    await a.send('D1','Main prefix also quoted independently',metadata=md)
    assert w.open == {'100.1'}
    assert a._active_streams['D1']['sent'] == 'Main prefix'
