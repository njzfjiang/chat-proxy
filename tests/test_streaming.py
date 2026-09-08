import asyncio
import logging

import pytest

from chat_proxy.streaming import ObservedStreamingResponse


@pytest.mark.anyio
@pytest.mark.parametrize("fail_send", [False, True])
async def test_delivery_records_only_successful_sends(caplog, fail_send):
    async def body():
        yield b"data: hello\n\n"

    async def receive():
        await asyncio.Event().wait()

    async def send(message):
        if fail_send and message["type"] == "http.response.body":
            raise OSError("closed")

    response = ObservedStreamingResponse(body(), request_id="req_test")
    scope = {"type": "http", "asgi": {"spec_version": "2.4"}}
    with caplog.at_level(logging.INFO, logger="uvicorn.error.chat_proxy.stream"):
        if fail_send:
            with pytest.raises(Exception):
                await response(scope, receive, send)
        else:
            await response(scope, receive, send)
    assert "stream_delivery request_id=req_test" in caplog.text
    assert f"asgi_final_sent={not fail_send}" in caplog.text
    assert ("bytes=0" if fail_send else "bytes=13") in caplog.text


@pytest.mark.anyio
async def test_generator_failure_does_not_report_final_send(caplog):
    async def body():
        yield b"first"
        raise RuntimeError("persistence failed")

    async def receive():
        await asyncio.Event().wait()

    async def send(message):
        pass

    response = ObservedStreamingResponse(body(), request_id="req_error")
    with caplog.at_level(logging.INFO, logger="uvicorn.error.chat_proxy.stream"):
        with pytest.raises(RuntimeError):
            await response({"type": "http", "asgi": {"spec_version": "2.4"}}, receive, send)
    assert "asgi_final_sent=False" in caplog.text
    assert "bytes=5" in caplog.text
    assert "error_type=RuntimeError" in caplog.text
