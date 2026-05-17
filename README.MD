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

The proxy automatically loads `.env` from the current working directory before
reading config. Set `CHAT_PROXY_ENV_FILE` to use a different env file.

Kelivo provider settings:

- Base URL: `http://127.0.0.1:8787`
- Chat Path: `/chat/completions`

The proxy also accepts `/v1/chat/completions`.

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
