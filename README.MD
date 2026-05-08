# chat-proxy

OpenAI-compatible Base URL proxy for Kelivo. It forwards chat completion
requests to an upstream provider and writes a searchable projection into the
existing KMLog SQLite FTS database.

## Run

```powershell
$env:CHAT_PROXY_UPSTREAM_BASE="https://api.openai.com/v1"
$env:CHAT_PROXY_DB="C:\Users\ellat\Desktop\K_Space\kmlog-search\chat_data\chat_search.db"
python -m chat_proxy.main
```

Kelivo provider settings:

- Base URL: `http://127.0.0.1:8787`
- Chat Path: `/chat/completions`

The proxy also accepts `/v1/chat/completions`.

## Optional Metadata

Preferred headers:

```text
X-Kelivo-Conversation-Id: kai-main
X-Kelivo-Client-Id: mei-phone
X-Kelivo-Assistant-Key: kai
X-Kelivo-Provider-Key: openai
```

If headers are unavailable, the proxy can read a system prompt block:

```text
[kelivo_meta]
client=mei-phone
assistant=kai
provider=openai
conversation=kai-main
[/kelivo_meta]
```

Without metadata, the proxy assigns a deterministic `proxy_assigned_*`
conversation id from client/model hints.

## Kelivo Dev Metadata

When Kelivo sends analysis metadata, the proxy automatically keeps the full
request in SQLite but strips `_kelivo_analysis_meta` before forwarding upstream.
It reads conversation/provider hints from:

```text
X-Kelivo-Analysis-Version: 1
```

```json
{
  "_kelivo_analysis_meta": {
    "conversation_id": "chat-id",
    "conversation_title": "Chat title",
    "assistant_id": "assistant-id",
    "provider_key": "openai",
    "model_id": "gpt-4.1"
  }
}
```
