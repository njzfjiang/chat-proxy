"""ASGI delivery diagnostics; send completion is not a client receipt."""

import hashlib
import logging
import time

from starlette.responses import StreamingResponse


logger = logging.getLogger("uvicorn.error.chat_proxy.stream")


class ObservedStreamingResponse(StreamingResponse):
    def __init__(self, *args, request_id: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.request_id = request_id

    async def __call__(self, scope, receive, send):
        started = time.monotonic()
        chunks = byte_count = 0
        final_sent = disconnected = False
        error_type = None
        digest = hashlib.sha256()

        async def observed_receive():
            nonlocal disconnected
            message = await receive()
            if message["type"] == "http.disconnect":
                disconnected = True
            return message

        async def observed_send(message):
            nonlocal chunks, byte_count, final_sent
            await send(message)
            if message["type"] == "http.response.body":
                data = message.get("body", b"")
                if data:
                    chunks += 1
                    byte_count += len(data)
                    digest.update(data)
                if not message.get("more_body", False):
                    final_sent = True

        try:
            await super().__call__(scope, observed_receive, observed_send)
        except BaseException as exc:
            error_type = type(exc).__name__
            raise
        finally:
            logger.info(
                "stream_delivery request_id=%s asgi_final_sent=%s "
                "disconnect_received=%s chunks=%s bytes=%s sha256=%s "
                "duration_ms=%s error_type=%s",
                self.request_id, final_sent, disconnected, chunks, byte_count,
                digest.hexdigest(), round((time.monotonic() - started) * 1000),
                error_type,
            )
