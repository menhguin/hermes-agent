"""Bounded regressions for the HTTPX2 boundary used by MCP 2.

Use the real SDK HTTP module and Hermes' transport wrapper. Payloads stay below
4 MiB; this tests decoder/parser bounds, not machine exhaustion or timing.
"""
import zlib

import pytest

from tools.mcp_tool import sdk_httpx
from tools.mcp_tool_errors import _make_mcp_body_cap_transport


@pytest.mark.asyncio
async def test_compressed_response_has_bounded_intermediate_chunks():
    http = sdk_httpx()
    assert http is not None
    payload = b"A" * (4 * 1024 * 1024)
    compressed = zlib.compress(payload, wbits=31)

    class Wire(http.AsyncByteStream):
        async def __aiter__(self):
            yield compressed

    def respond(request):
        return http.Response(200, headers={"content-encoding": "gzip"}, stream=Wire())

    # Native cap counts the wire, not decoded data. The fixed SDK must bound
    # decoder allocations without rejecting a legitimate large JSON response.
    transport = _make_mcp_body_cap_transport(http, http.MockTransport(respond), limit=8192)
    async with http.AsyncClient(transport=transport) as client:
        async with client.stream("GET", "https://fixture.invalid") as response:
            chunks = [chunk async for chunk in response.aiter_bytes()]
    assert b"".join(chunks) == payload
    assert max(map(len, chunks)) <= 1024 * 1024


@pytest.mark.asyncio
async def test_sse_rejects_unterminated_event_over_sdk_bound():
    http = sdk_httpx()
    assert http is not None

    class Wire(http.AsyncByteStream):
        async def __aiter__(self):
            yield b"data: "
            for _ in range(33):
                yield b"A" * 32768
            yield b"\n\n"

    def respond(request):
        return http.Response(200, headers={"content-type": "text/event-stream"}, stream=Wire())

    transport = _make_mcp_body_cap_transport(http, http.MockTransport(respond))
    async with http.AsyncClient(transport=transport) as client:
        with pytest.raises(http.SSEError):
            async with client.sse("https://fixture.invalid") as events:
                async for _ in events:
                    pass


@pytest.mark.asyncio
async def test_sse_many_valid_events_can_exceed_one_mib_in_total():
    http = sdk_httpx()
    assert http is not None
    event = "A" * 32768

    class Wire(http.AsyncByteStream):
        async def __aiter__(self):
            for _ in range(40):
                yield ("data: " + event + "\n\n").encode()

    def respond(request):
        return http.Response(200, headers={"content-type": "text/event-stream"}, stream=Wire())

    transport = _make_mcp_body_cap_transport(http, http.MockTransport(respond))
    async with http.AsyncClient(transport=transport) as client:
        async with client.sse("https://fixture.invalid") as events:
            values = [item.data async for item in events]
    assert values == [event] * 40
