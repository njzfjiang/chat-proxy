# chat-proxy

OpenAI-compatible Base URL proxy for Kelivo. It forwards chat completion
requests to an upstream provider and writes a searchable projection into the
existing KMLog SQLite FTS database.

## Run

```powershell
$env:CHAT_PROXY_UPSTREAM_BASE="https://api.openai.com/v1"
$env:CHAT_PROXY_UPSTREAM_API_KEY="..."
$env:CHAT_PROXY_CHAT_MODEL="gpt-4.1-mini"
$env:CHAT_PROXY_CHAT_RECENT_K="20"
$env:CHAT_PROXY_PROVIDER_KEY="openai"
$env:CHAT_PROXY_WORLDBOOK_PATHS="C:\Users\ellat\Documents\KM-backup\Kai-Mei-Memory-Vault\7-WorldBook\AI_triggered_expansion_pack_v1.json;C:\Users\ellat\Documents\KM-backup\Kai-Mei-Memory-Vault\7-WorldBook\长期记忆系统.json"
$env:CHAT_PROXY_DB="C:\Users\ellat\Desktop\K_Space\kmlog-search\chat_data\chat_search.db"
python -m chat_proxy.main
```

The proxy automatically loads `.env` from the current working directory before
reading config. Set `CHAT_PROXY_ENV_FILE` to use a different env file.

To serve the web UI from the same FastAPI process, build it first:

```powershell
cd web
npm install
npm run build
cd ..
python -m chat_proxy.main
```

Then open `http://127.0.0.1:8787`. During frontend development, run
`npm run dev` in `web/` and use `http://127.0.0.1:5173`; Vite proxies API calls
to `127.0.0.1:8787`.

If the UI must live under a subpath such as `/chat/`, build it with a matching
asset base and API prefix:

```bash
cd web
VITE_BASE_PATH=/chat/ VITE_API_BASE=/chat/api npm run build
```

For nginx, serve `/chat/` from `web/dist` and proxy `/chat/api/` to the FastAPI
root path.

Kelivo provider settings:

- Base URL: `http://127.0.0.1:8787`
- Chat Path: `/chat/completions`

The proxy also accepts `/v1/chat/completions`.

## Webapp API

For browser/mobile clients, use the proxy as the only backend. The client should
persist `client_id` and `conversation_id`, generate a fresh `request_id` per
send attempt, and let the proxy handle storage, summaries, candidates, and
future retrieval.

```http
POST /chat
```

```json
{
  "client_id": "phone-uuid",
  "conversation_id": "conversation-uuid",
  "request_id": "request-uuid",
  "assistant_key": "kai",
  "system_prompt": "Optional per-request instruction.",
  "user_text": "hello"
}
```

`POST /chat` also accepts OpenAI-compatible `messages` instead of `user_text`.
The model, provider API key, and default provider key belong on the backend via
`CHAT_PROXY_CHAT_MODEL`, `CHAT_PROXY_UPSTREAM_API_KEY`, and
`CHAT_PROXY_PROVIDER_KEY`. Browser/mobile clients may pass `model` or
`provider_key` as explicit overrides, but they should never store provider API
keys. `system_prompt` is inserted only into the current upstream request.

When clients send `user_text`, the proxy builds the upstream short-context
window from SQLite instead of trusting the browser's current view. It prepends up
to `CHAT_PROXY_CHAT_RECENT_K` recent `chat` messages from the same
`conversation_id`, then appends the new user message. This lets mobile/web/dev
clients share canonical context even if one client has not loaded every recent
turn.

If `CHAT_PROXY_WORLDBOOK_PATHS` or legacy `CHAT_PROXY_WORLDBOOK_PATH` is set,
`/chat` also scans the current request and recent context for matching
worldbook keywords. Separate multiple paths with semicolons. Matching entries
are inserted as a system context block before recent turns. Tune this with
`CHAT_PROXY_WORLDBOOK_ENABLED`, `CHAT_PROXY_WORLDBOOK_MAX_ITEMS`, and
`CHAT_PROXY_WORLDBOOK_CHARS_TOTAL`.

Explicit `request_id` values are idempotent: a completed duplicate replays the
stored response, while an unfinished duplicate returns `409`.

Conversation and summary reads:

```text
POST /conversations
GET /conversations?limit=50
GET /conversations/{conversation_id}/messages?limit=50&before_id=123
GET /conversations/{conversation_id}/messages?after_id=123
DELETE /conversations/{conversation_id}/messages/{message_id}
GET /conversations/{conversation_id}/rolling-short
GET /conversations/{conversation_id}/rolling-short/versions
POST /conversations/{conversation_id}/rolling-short/rollback
GET /daily-summaries?date_key=2026-05-16
GET /daily-summaries?days_ago=1
GET /daily-summaries?limit=30
POST /daily-summaries/run
```

Debug the final context composition for recent requests:

```text
GET /admin/requests?limit=20
GET /admin/requests?conversation_id=conversation-uuid
GET /admin/requests?request_id=request-uuid
```

Look at `metadata.injected_context_snapshot` to see which components were
injected, including rolling summary, worldbook snippets, recent turn ids, and
character counts. Full request/response payloads and raw metadata are hidden by
default; add `include_payloads=true` only when you need to inspect them.
In the web UI, tap the `Debug` button under a stored message to open the same
trace in the right-side inspector.

Message reads include `token_usage` when the upstream provider returns it. The
web UI falls back to a local estimate when true usage is unavailable. For
streaming requests, the web UI sends `stream_options.include_usage=true`; support
depends on the selected OpenAI-compatible provider.

The OpenAI-compatible endpoints still support the same identity via headers:

```text
X-Kelivo-Client-Id
X-Kelivo-Conversation-Id
X-Kelivo-Request-Id
X-Kelivo-Assistant-Key
X-Kelivo-Provider-Key
```

## Rolling Summary

Rolling summaries are off by default. When enabled, the proxy stores one short
summary per `conversation_id`, injects it into the next upstream request as a
system message, and updates it asynchronously after assistant messages are
written to SQLite.

```powershell
$env:CHAT_PROXY_SUMMARY_ENABLED="true"
$env:CHAT_PROXY_SUMMARY_UPSTREAM_BASE="https://api.deepseek.com/v1"
$env:CHAT_PROXY_SUMMARY_API_KEY="..."
$env:CHAT_PROXY_SUMMARY_MODEL="deepseek-v4-flash"
$env:CHAT_PROXY_SUMMARY_RECENT_K="30"
```

The original `request_json` stored in SQLite remains unchanged; only the
forwarded upstream body receives the summary system message.

## Daily Summary

Daily summaries are also off by default. When enabled, the proxy keeps an
audit-friendly summary for the local day and stores proposed memory candidates
using the Kai mapping categories. Candidates remain in SQLite with
`status='candidate'`; they are not written to mem0 or the memory vault. Each
candidate also receives a deterministic `target_layer` suggestion:
`anchor_table`, `wb`, `vault`, `mem0`, or `ignore`.

```powershell
$env:CHAT_PROXY_DAILY_SUMMARY_ENABLED="true"
$env:CHAT_PROXY_DAILY_SUMMARY_UPSTREAM_BASE="https://api.deepseek.com/v1"
$env:CHAT_PROXY_DAILY_SUMMARY_API_KEY="..."
$env:CHAT_PROXY_DAILY_SUMMARY_MODEL="deepseek-v4-flash"
$env:CHAT_PROXY_DAILY_SUMMARY_RECENT_K="200"
$env:CHAT_PROXY_DAILY_SUMMARY_TIMEZONE="America/Toronto"
```

If the daily-specific upstream/model/API key variables are omitted, the proxy
reuses the rolling-summary settings. Review today's generated draft with:

```text
GET /admin/daily-summary
GET /admin/daily-summary/2026-05-16
```

To manually test an older local day without waiting for the scheduled update:

```powershell
python -m chat_proxy.manual_daily_summary --date 2026-05-15
```

Or run it from HTTP / the web UI:

```json
POST /daily-summaries/run
{
  "days_ago": 1,
  "force": false
}
```

Use `"date_key": "2026-05-15"` instead of `days_ago` when you need an exact
local date. `force: true` deletes that day's generated summary first and then
rebuilds it.

The job is incremental. To rerun the whole day from existing messages during
testing, reset that day's generated summary first:

```powershell
python -m chat_proxy.manual_daily_summary --date 2026-05-15 --force
```

Manual backfills scan up to 100000 messages by default so older dates are not
missed by the normal scheduled-job recent-message window. Use `--scan-limit` if
you need to change that.

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
