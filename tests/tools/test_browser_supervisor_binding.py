"""Explicit-target supervision must not retarget or instrument vault-only pages."""

import asyncio
import json

import pytest

from tools import browser_supervisor as bs


class CDPTrace:
    """Protocol-only fake; exercise the supervisor's actual attachment logic."""

    def __init__(self, targets):
        self.targets = targets
        self.calls = []

    async def __call__(self, method, params=None, *, session_id=None, timeout=10.0):
        params = params or {}
        self.calls.append((method, params, session_id))
        if method == "Target.getTargets":
            return {"result": {"targetInfos": self.targets}}
        if method == "Target.attachToTarget":
            return {"result": {"sessionId": "session-" + params["targetId"]}}
        if method == "Target.createTarget":
            return {"result": {"targetId": "created"}}
        return {"result": {}}


def page(target_id, url="https://example.test/login"):
    return {"type": "page", "targetId": target_id, "url": url}


@pytest.mark.asyncio
async def test_initial_attachment_uses_exact_target(monkeypatch):
    trace = CDPTrace([page("unrelated"), page("selected")])
    supervisor = bs.CDPSupervisor("vault-task", "ws://test", target_id="selected")
    monkeypatch.setattr(supervisor, "_cdp", trace)

    await supervisor._attach_initial_page()

    assert supervisor._page_session_id == "session-selected"
    assert [params["targetId"] for method, params, _ in trace.calls
            if method == "Target.attachToTarget"] == ["selected"]


@pytest.mark.asyncio
@pytest.mark.parametrize("targets", [[], [page("unrelated")], [
    {"type": "worker", "targetId": "selected", "url": "https://example.test"}
]])
async def test_missing_bound_page_never_creates_or_attaches_another(monkeypatch, targets):
    trace = CDPTrace(targets)
    supervisor = bs.CDPSupervisor("vault-task", "ws://test", target_id="selected")
    monkeypatch.setattr(supervisor, "_cdp", trace)

    with pytest.raises(RuntimeError, match="target"):
        await supervisor._attach_initial_page()

    assert not any(method in {"Target.createTarget", "Target.attachToTarget"}
                   for method, _, _ in trace.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("origin,accept,selected_url,ok", [
    ("", None, "https://example.test/login", True),
    ("https://example.test", None, "https://example.test/login", True),
    ("https://other.test", None, "https://example.test/login", False),
    ("", "false", "https://example.test/login", False),
    ("", None, "about:blank", False),
])
async def test_bound_focus_never_probes_or_attaches_another_target(
    monkeypatch, origin, accept, selected_url, ok,
):
    trace = CDPTrace([page("unrelated", "https://other.test/login"),
                      page("selected", selected_url)])
    supervisor = bs.CDPSupervisor("vault-task", "ws://test", target_id="selected")
    supervisor._loop = asyncio.get_running_loop()
    supervisor._page_session_id = "session-selected"
    monkeypatch.setattr(supervisor, "_cdp", trace)

    result = await asyncio.to_thread(supervisor.focus_page, origin, accept=accept)

    assert result["ok"] is ok
    assert supervisor._page_session_id == "session-selected"
    assert all(params["targetId"] == "selected" for method, params, _ in trace.calls
               if method == "Target.attachToTarget")
    assert all(sid == "session-selected" for method, _, sid in trace.calls
               if method == "Runtime.evaluate")


@pytest.mark.asyncio
async def test_vault_only_attachment_and_focus_do_not_instrument_pages(monkeypatch):
    trace = CDPTrace([page("selected")])
    supervisor = bs.CDPSupervisor("vault-task", "ws://test", target_id="selected",
                                  instrument_dialogs=False)
    supervisor._loop = asyncio.get_running_loop()
    monkeypatch.setattr(supervisor, "_cdp", trace)

    await supervisor._attach_initial_page()
    result = await asyncio.to_thread(supervisor.focus_page, "")

    assert result["ok"] is True
    # No Page/Runtime enable, auto-attach, binding, script injection or Fetch interceptor.
    assert {method for method, _, _ in trace.calls} == {
        "Target.getTargets", "Target.attachToTarget",
    }


@pytest.mark.asyncio
async def test_vault_only_reader_ignores_instrumentation_events(monkeypatch):
    supervisor = bs.CDPSupervisor("vault-task", "ws://test", target_id="selected",
                                  instrument_dialogs=False)
    seen = []
    monkeypatch.setattr(supervisor, "_EVENT_HANDLERS", {
        method: lambda *args: seen.append(args) for method in supervisor._EVENT_HANDLERS
    })
    reply = {"id": 1, "result": {"result": {"type": "string", "value": "selected"}}}

    class Messages:
        async def __aiter__(self):
            for method in supervisor._EVENT_HANDLERS:
                yield json.dumps({"method": method, "params": {}, "sessionId": "session-selected"})
            yield json.dumps(reply)

    supervisor._ws = Messages()
    pending = asyncio.get_running_loop().create_future()
    supervisor._pending_calls[1] = pending
    await supervisor._read_loop()

    assert seen == []  # Includes frame/target, binding, dialog and Fetch handlers.
    assert pending.result() == reply  # Runtime replies remain usable.


@pytest.fixture
def wire(monkeypatch):
    """In-memory WebSocket transport with real supervisor threads/loops/registry."""
    import websockets

    connections = []

    class Connection:
        def __init__(self):
            self.trace = CDPTrace([page("unrelated"), page("selected")])
            self.messages = asyncio.Queue()
            self.closed = False

        async def send(self, raw):
            request = json.loads(raw)
            reply = await self.trace(request["method"], request.get("params"),
                                     session_id=request.get("sessionId"))
            await self.messages.put(json.dumps({"id": request["id"], **reply}))

        async def __aiter__(self):
            while (message := await self.messages.get()) is not None:
                yield message

        async def close(self):
            self.closed = True
            await self.messages.put(None)

    async def connect(*args, **kwargs):
        connection = Connection()
        connections.append(connection)
        return connection

    monkeypatch.setattr(websockets, "connect", connect)
    yield connections
    bs.SUPERVISOR_REGISTRY.stop_all()


def test_public_scoped_connection_is_visible_to_native_vault_without_replacing_browser(wire):
    from tools.browser_vault_tool import _ensure_supervisor

    native = bs.SUPERVISOR_REGISTRY.get_or_start("browser-task", "ws://test")
    scoped = bs.get_or_start_supervisor("browser-task:vault", "ws://test",
                                         target_id="selected", instrument_dialogs=False)
    try:
        assert _ensure_supervisor("browser-task:vault") is scoped
        assert scoped._page_session_id == "session-selected"
        assert bs.get_or_start_supervisor("browser-task:vault", "ws://test",
                                           target_id="selected", instrument_dialogs=False) is scoped
        assert {method for method, _, _ in wire[1].trace.calls} == {
            "Target.getTargets", "Target.attachToTarget",
        }
    finally:
        bs.stop_supervisor("browser-task:vault")

    assert bs.SUPERVISOR_REGISTRY.get("browser-task:vault") is None
    assert wire[1].closed
    assert not scoped._thread.is_alive()
    assert bs.SUPERVISOR_REGISTRY.get("browser-task") is native
    assert native.snapshot().active
    assert not wire[0].closed


@pytest.mark.parametrize("original,requested,url", [
    ({}, {"target_id": "selected", "instrument_dialogs": False}, "ws://test"),
    ({"target_id": "selected", "instrument_dialogs": False},
     {"target_id": "unrelated", "instrument_dialogs": False}, "ws://test"),
    ({"target_id": "selected", "instrument_dialogs": False},
     {"target_id": "selected", "instrument_dialogs": True}, "ws://test"),
    ({"target_id": "selected", "instrument_dialogs": False},
     {"target_id": "selected", "instrument_dialogs": False}, "ws://different"),
])
def test_conflicting_scope_does_not_reuse_or_stop_existing_connection(wire, original, requested, url):
    original_supervisor = bs.get_or_start_supervisor("same-task", "ws://test", **original)

    with pytest.raises(ValueError, match="binding"):
        bs.get_or_start_supervisor("same-task", url, **requested)

    assert bs.SUPERVISOR_REGISTRY.get("same-task") is original_supervisor
    assert original_supervisor.snapshot().active
    assert not wire[0].closed
    assert len(wire) == 1


def test_start_timeout_closes_connection_and_stops_thread(monkeypatch, wire):
    import time

    async def stalled_attach(self):
        await asyncio.Event().wait()

    monkeypatch.setattr(bs.CDPSupervisor, "_attach_initial_page", stalled_attach)
    supervisor = bs.CDPSupervisor("stalled-vault", "ws://test", target_id="selected",
                                  instrument_dialogs=False)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        supervisor.start(timeout=0.1)

    assert time.monotonic() - started < 3.0
    assert wire[0].closed
    assert not supervisor._thread.is_alive()
    assert not supervisor.snapshot().active


@pytest.mark.parametrize("second_target", ["selected", "unrelated"])
def test_concurrent_scope_registration_never_returns_the_wrong_binding(monkeypatch, wire, second_target):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    barrier = threading.Barrier(2)
    attach = bs.CDPSupervisor._attach_initial_page

    async def synchronized_attach(self):
        await asyncio.to_thread(barrier.wait, timeout=5)
        await attach(self)

    monkeypatch.setattr(bs.CDPSupervisor, "_attach_initial_page", synchronized_attach)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(bs.get_or_start_supervisor, "concurrent-task", "ws://test",
                               target_id=target, instrument_dialogs=False)
                   for target in ("selected", second_target)]
        errors = [future.exception(timeout=10) for future in futures]

    if second_target == "selected":
        assert errors == [None, None]
        assert futures[0].result() is futures[1].result()
    else:
        assert sum(isinstance(error, ValueError) for error in errors) == 1
        winner = next(future.result() for future in futures if future.exception() is None)
        assert bs.SUPERVISOR_REGISTRY.get("concurrent-task") is winner
    assert sum(connection.closed for connection in wire) == 1


def test_failed_bound_start_closes_transport_without_registering(wire):
    with pytest.raises(RuntimeError, match="target"):
        bs.get_or_start_supervisor("missing-task", "ws://test",
                                   target_id="missing", instrument_dialogs=False)
    assert bs.SUPERVISOR_REGISTRY.get("missing-task") is None
    assert wire[0].closed
    assert [method for method, _, _ in wire[0].trace.calls] == ["Target.getTargets"]


@pytest.mark.asyncio
@pytest.mark.parametrize("targets,expected", [([page("first"), page("second")], "first"), ([], "created")])
async def test_unbound_defaults_still_choose_or_create_page_and_instrument(monkeypatch, targets, expected):
    trace = CDPTrace(targets)
    supervisor = bs.CDPSupervisor("native-task", "ws://test")
    monkeypatch.setattr(supervisor, "_cdp", trace)
    await supervisor._attach_initial_page()

    assert supervisor._page_session_id == "session-" + expected
    assert {"Page.enable", "Runtime.enable", "Target.setAutoAttach", "Fetch.enable",
            "Page.addScriptToEvaluateOnNewDocument", "Runtime.evaluate"} <= {
        method for method, _, _ in trace.calls
    }


def test_unbound_registry_keeps_endpoint_replacement_behavior(wire):
    old = bs.get_or_start_supervisor("native-task", "ws://old")
    replacement = bs.get_or_start_supervisor("native-task", "ws://new")
    assert replacement is not old
    assert wire[0].closed
    assert replacement.snapshot().active


@pytest.fixture
def disposable_chrome(tmp_path):
    """Own headless Chrome on blank pages; never attach to the user's browser."""
    import os
    from pathlib import Path
    import shutil
    import subprocess
    import time
    from websockets.sync.client import connect

    if os.environ.get("HERMES_E2E_BROWSER") != "1":
        pytest.skip("set HERMES_E2E_BROWSER=1 to launch disposable Chrome")
    chrome = next((path for name in ("chromium", "google-chrome", "chromium-browser")
                   if (path := shutil.which(name))), None)
    mac_chrome = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if chrome is None and mac_chrome.is_file():
        chrome = str(mac_chrome)
    if chrome is None:
        pytest.skip("Chrome/Chromium not installed")

    profile = tmp_path / "chrome-profile"
    proc = subprocess.Popen([
        chrome, "--headless=new", "--remote-debugging-port=0", f"--user-data-dir={profile}",
        "--no-first-run", "--no-default-browser-check", "--disable-gpu", "--no-sandbox",
        "--disable-background-networking", "--disable-component-update", "--disable-sync",
        "--disable-extensions", "--disable-default-apps", "--disable-breakpad",
        "--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE localhost, EXCLUDE 127.0.0.1", "about:blank",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
       env={**os.environ, "HOME": str(tmp_path), "TMPDIR": str(tmp_path)})
    try:
        endpoint_file = profile / "DevToolsActivePort"
        deadline = time.monotonic() + 15
        while not endpoint_file.exists():
            assert proc.poll() is None, "disposable Chrome exited before exposing CDP"
            if time.monotonic() > deadline:
                pytest.fail("disposable Chrome did not expose CDP within 15s")
            time.sleep(0.05)
        port, browser_path = endpoint_file.read_text().splitlines()[:2]
        browser_ws = f"ws://127.0.0.1:{port}{browser_path}"
        with connect(browser_ws, open_timeout=5, close_timeout=1) as control:
            next_id = 0

            def call(method, params=None, session_id=None):
                nonlocal next_id
                next_id += 1
                request = {"id": next_id, "method": method, "params": params or {}}
                if session_id:
                    request["sessionId"] = session_id
                control.send(json.dumps(request))
                while True:
                    response = json.loads(control.recv(timeout=5))
                    if response.get("id") == next_id:
                        assert "error" not in response, response
                        return response["result"]

            yield browser_ws, port, call
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)



@pytest.mark.integration
@pytest.mark.parametrize("endpoint", ["browser", "page"])
def test_real_bound_connection_is_noninstrumenting_and_cannot_wander(disposable_chrome, endpoint):
    browser_ws, port, call = disposable_chrome
    targets = []
    for marker in ("unrelated", "selected"):
        target = call("Target.createTarget", {"url": "about:blank"})["targetId"]
        sid = call("Target.attachToTarget", {"targetId": target, "flatten": True})["sessionId"]
        call("Runtime.evaluate", {"expression": f"window.fixtureMarker={json.dumps(marker)}; window.originalAlert=alert"}, sid)
        targets.append(target)
    selected = targets[1]
    ws_url = browser_ws if endpoint == "browser" else f"ws://127.0.0.1:{port}/devtools/page/{selected}"
    supervisor = bs.get_or_start_supervisor("real-vault", ws_url, target_id=selected,
                                           instrument_dialogs=False, start_timeout=5)
    try:
        assert supervisor.evaluate_runtime("window.fixtureMarker")["result"] == "selected"
        assert supervisor.evaluate_runtime("alert === window.originalAlert")["result"] is True
        assert supervisor.evaluate_runtime("!!window.__hermesDialogBridgeInstalled")["result"] is False
        assert supervisor.snapshot().frame_tree["top"] is None
        # Blank pages cannot satisfy a vault origin; refusal leaves the binding intact.
        assert supervisor.focus_page("", accept="window.fixtureMarker === 'unrelated'")["ok"] is False
        assert supervisor.evaluate_runtime("window.fixtureMarker")["result"] == "selected"
        call("Target.closeTarget", {"targetId": selected})
        assert supervisor.focus_page("")["ok"] is False
        assert supervisor.evaluate_runtime("window.fixtureMarker")["ok"] is False
    finally:
        bs.stop_supervisor("real-vault")

    assert not supervisor._thread.is_alive()
    assert targets[0] in {target["targetId"] for target in call("Target.getTargets")["targetInfos"]}
