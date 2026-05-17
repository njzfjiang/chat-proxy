import { FormEvent, ReactNode, useEffect, useMemo, useRef, useState } from "react";
import {
  CalendarDays,
  CheckCircle2,
  Edit3,
  Copy,
  GitBranch,
  Loader2,
  Menu,
  MessageSquarePlus,
  RefreshCcw,
  Archive,
  History,
  RotateCcw,
  Send,
  Settings2,
  Trash2,
  X
} from "lucide-react";

type Conversation = {
  conversation_id: string;
  title: string | null;
  assistant_key: string | null;
  client_id: string | null;
  message_count: number;
  last_message_at: string | null;
  last_message_preview: string;
  rolling_summary: string | null;
  rolling_summary_status: string | null;
  archived_at: string | null;
};

type ChatMessage = {
  id: number;
  timestamp: string;
  role: string;
  content: string;
  conversation_id: string;
  kind: string | null;
  token_usage?: TokenUsage | null;
};

type TokenUsage = {
  prompt_tokens?: number;
  completion_tokens?: number;
  total_tokens?: number;
  cached_tokens?: number;
  reasoning_tokens?: number;
};

type RollingShort = {
  conversation_id: string;
  summary: {
    summary: string;
    status: string | null;
    updated_at: string;
    version: number;
  } | null;
};

type DailySummary = {
  date_key: string;
  summary: {
    summary: string;
    status: string | null;
    updated_at: string;
    version: number;
  } | null;
  memory_candidates: Array<{
    id: number;
    label: string;
    domain: string;
    target_layer: string | null;
    confidence: string | null;
  }>;
};

type RollingVersion = {
  version: number;
  summary: string;
  created_at: string;
};

const CLIENT_KEY = "chatProxyWeb.clientId";
const CONVERSATION_KEY = "chatProxyWeb.conversationId";
const MODEL_KEY = "chatProxyWeb.model";
const ASSISTANT_KEY = "chatProxyWeb.assistantKey";
const PROVIDER_KEY = "chatProxyWeb.providerKey";
const SYSTEM_PROMPT_KEY = "chatProxyWeb.systemPrompt";
const API_BASE = normalizeApiBase(import.meta.env.VITE_API_BASE);

export function App() {
  const [clientId] = useState(() => persistedId(CLIENT_KEY, "client"));
  const [activeConversationId, setActiveConversationId] = useState(() =>
    localStorage.getItem(CONVERSATION_KEY) || ""
  );
  const [model, setModel] = useState(() => localStorage.getItem(MODEL_KEY) || "");
  const [assistantKey, setAssistantKey] = useState(
    () => localStorage.getItem(ASSISTANT_KEY) || "kai"
  );
  const [providerKey, setProviderKey] = useState(() => localStorage.getItem(PROVIDER_KEY) || "");
  const [systemPrompt, setSystemPrompt] = useState(
    () => localStorage.getItem(SYSTEM_PROMPT_KEY) || ""
  );
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [rolling, setRolling] = useState<RollingShort | null>(null);
  const [daily, setDaily] = useState<DailySummary | null>(null);
  const [draft, setDraft] = useState("");
  const [conversationFilter, setConversationFilter] = useState("");
  const [showArchived, setShowArchived] = useState(false);
  const [loading, setLoading] = useState(false);
  const [sending, setSending] = useState(false);
  const [dailyRunning, setDailyRunning] = useState(false);
  const [dailyLoading, setDailyLoading] = useState(false);
  const [dailyDaysAgo, setDailyDaysAgo] = useState(0);
  const [dailyForce, setDailyForce] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [copiedMessageId, setCopiedMessageId] = useState<number | null>(null);
  const [failedSend, setFailedSend] = useState<{
    conversationId: string;
    text: string;
  } | null>(null);
  const messageEndRef = useRef<HTMLDivElement | null>(null);
  const composerRef = useRef<HTMLTextAreaElement | null>(null);

  const activeConversation = useMemo(
    () =>
      conversations.find(
        (item) => item.conversation_id === activeConversationId
      ) || null,
    [activeConversationId, conversations]
  );
  const filteredConversations = useMemo(() => {
    const query = conversationFilter.trim().toLowerCase();
    if (!query) {
      return conversations;
    }
    return conversations.filter((conversation) =>
      [
        conversation.title,
        conversation.assistant_key,
        conversation.conversation_id,
        conversation.last_message_preview
      ]
        .filter(Boolean)
        .some((value) => String(value).toLowerCase().includes(query))
    );
  }, [conversationFilter, conversations]);
  const todayKey = useMemo(() => localDateKey(new Date()), []);

  useEffect(() => {
    localStorage.setItem(MODEL_KEY, model);
  }, [model]);

  useEffect(() => {
    localStorage.setItem(ASSISTANT_KEY, assistantKey);
  }, [assistantKey]);

  useEffect(() => {
    localStorage.setItem(PROVIDER_KEY, providerKey);
  }, [providerKey]);

  useEffect(() => {
    localStorage.setItem(SYSTEM_PROMPT_KEY, systemPrompt);
  }, [systemPrompt]);

  useEffect(() => {
    void refreshConversations();
    void loadDaily(todayKey);
  }, [showArchived, todayKey]);

  useEffect(() => {
    if (!activeConversationId) {
      setMessages([]);
      setRolling(null);
      return;
    }
    localStorage.setItem(CONVERSATION_KEY, activeConversationId);
    void loadConversation(activeConversationId);
  }, [activeConversationId]);

  useEffect(() => {
    messageEndRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, sending]);

  async function refreshConversations() {
    setLoading(true);
    setError(null);
    try {
      const data = await requestJson<{ conversations: Conversation[] }>(
        `/conversations?limit=100&include_archived=${showArchived ? "true" : "false"}`
      );
      setConversations(data.conversations);
      if (!activeConversationId && data.conversations[0]) {
        setActiveConversationId(data.conversations[0].conversation_id);
      }
    } catch (err) {
      setError(errorText(err));
    } finally {
      setLoading(false);
    }
  }

  async function loadConversation(conversationId: string) {
    setLoading(true);
    setError(null);
    try {
      const [messageData, rollingData] = await Promise.all([
        requestJson<{ messages: ChatMessage[] }>(
          `/conversations/${encodeURIComponent(conversationId)}/messages?limit=80`
        ),
        requestJson<RollingShort>(
          `/conversations/${encodeURIComponent(conversationId)}/rolling-short`
        )
      ]);
      setMessages(messageData.messages);
      setRolling(rollingData);
    } catch (err) {
      setError(errorText(err));
    } finally {
      setLoading(false);
    }
  }

  async function loadDaily(dateKey: string) {
    try {
      setDaily(
        await requestJson<DailySummary>(
          `/daily-summaries?date_key=${encodeURIComponent(dateKey)}`
        )
      );
    } catch {
      setDaily(null);
    }
  }

  async function loadDailyByDaysAgo() {
    if (dailyLoading) {
      return;
    }
    setDailyLoading(true);
    setError(null);
    try {
      setDaily(
        await requestJson<DailySummary>(
          `/daily-summaries?days_ago=${encodeURIComponent(String(dailyDaysAgo))}`
        )
      );
    } catch (err) {
      setError(errorText(err));
    } finally {
      setDailyLoading(false);
    }
  }

  async function runDailySummary() {
    if (dailyRunning) {
      return;
    }
    setDailyRunning(true);
    setError(null);
    try {
      const data = await requestJson<DailySummary>("/daily-summaries/run", {
        method: "POST",
        body: JSON.stringify({
          days_ago: dailyDaysAgo,
          force: dailyForce
        })
      });
      setDaily(data);
    } catch (err) {
      setError(errorText(err));
    } finally {
      setDailyRunning(false);
    }
  }

  async function newConversation() {
    setError(null);
    try {
      const data = await requestJson<{
        conversation_id: string;
      }>("/conversations", {
        method: "POST",
        body: JSON.stringify({
          client_id: clientId,
          assistant_key: assistantKey,
          ...(providerKey.trim() ? { provider_key: providerKey.trim() } : {}),
          title: assistantKey
        })
      });
      setActiveConversationId(data.conversation_id);
      setMessages([]);
      setRolling(null);
      setSidebarOpen(false);
      await refreshConversations();
    } catch (err) {
      setError(errorText(err));
    }
  }

  async function renameConversation(conversation: Conversation) {
    const current = conversation.title || conversation.assistant_key || "";
    const next = window.prompt("Rename conversation", current);
    if (next === null) {
      return;
    }
    const title = next.trim();
    if (!title) {
      return;
    }
    try {
      await requestJson(`/conversations/${encodeURIComponent(conversation.conversation_id)}`, {
        method: "PATCH",
        body: JSON.stringify({ title })
      });
      await refreshConversations();
    } catch (err) {
      setError(errorText(err));
    }
  }

  async function archiveConversation(conversation: Conversation) {
    try {
      await requestJson(`/conversations/${encodeURIComponent(conversation.conversation_id)}`, {
        method: "PATCH",
        body: JSON.stringify({ archived: true })
      });
      if (conversation.conversation_id === activeConversationId) {
        setActiveConversationId("");
      }
      await refreshConversations();
    } catch (err) {
      setError(errorText(err));
    }
  }

  async function sendMessage(event: FormEvent) {
    event.preventDefault();
    const text = draft.trim();
    if (!text || sending) {
      return;
    }
    setDraft("");
    await submitText(text);
  }

  async function submitText(text: string) {
    const conversationId =
      activeConversationId || `conv_${crypto.randomUUID().replaceAll("-", "")}`;
    if (!activeConversationId) {
      setActiveConversationId(conversationId);
    }
    setSending(true);
    setError(null);
    setFailedSend(null);
    const userMessageId = Date.now();
    const assistantMessageId = userMessageId + 1;
    setMessages((current) => [
      ...current,
      {
        id: userMessageId,
        timestamp: new Date().toISOString(),
        role: "user",
        content: text,
        conversation_id: conversationId,
        kind: "chat"
      },
      {
        id: assistantMessageId,
        timestamp: new Date().toISOString(),
        role: "assistant",
        content: "",
        conversation_id: conversationId,
        kind: "chat",
        token_usage: null
      }
    ]);
    try {
      await streamChat({
        payload: chatPayload(conversationId, text),
        onText: (chunk) => {
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantMessageId
                ? { ...message, content: message.content + chunk }
                : message
            )
          );
        }
      });
      await Promise.all([
        loadConversation(conversationId),
        refreshConversations(),
        loadDaily(todayKey)
      ]);
    } catch (err) {
      setError(errorText(err));
      setFailedSend({ conversationId, text });
      setMessages((current) =>
        current.filter((message) => message.id !== assistantMessageId)
      );
    } finally {
      setSending(false);
    }
  }

  async function retryFailedSend() {
    if (!failedSend || sending) {
      return;
    }
    setActiveConversationId(failedSend.conversationId);
    const text = failedSend.text;
    setFailedSend(null);
    await submitText(text);
  }

  function chooseConversation(conversationId: string) {
    setActiveConversationId(conversationId);
    setSidebarOpen(false);
  }

  function editMessage(message: ChatMessage) {
    setDraft(message.content);
    composerRef.current?.focus();
  }

  async function branchFromMessage(message: ChatMessage) {
    setError(null);
    try {
      const data = await requestJson<{
        conversation_id: string;
      }>(
        `/conversations/${encodeURIComponent(message.conversation_id)}/branches`,
        {
          method: "POST",
          body: JSON.stringify({
            source_message_id: message.id,
            title: `${activeConversation?.title || "Conversation"} branch`
          })
        }
      );
      setActiveConversationId(data.conversation_id);
      await refreshConversations();
      await loadConversation(data.conversation_id);
    } catch (err) {
      setError(errorText(err));
    }
  }

  async function deleteMessage(message: ChatMessage) {
    const confirmed = window.confirm("Delete this message?");
    if (!confirmed) {
      return;
    }
    setError(null);
    try {
      await requestJson(
        `/conversations/${encodeURIComponent(message.conversation_id)}/messages/${message.id}`,
        { method: "DELETE" }
      );
      setMessages((current) => current.filter((item) => item.id !== message.id));
      await Promise.all([
        refreshConversations(),
        loadConversation(message.conversation_id)
      ]);
    } catch (err) {
      setError(errorText(err));
    }
  }

  async function rollbackRollingSummary() {
    if (!activeConversationId) {
      return;
    }
    setError(null);
    try {
      const data = await requestJson<{ versions: RollingVersion[] }>(
        `/conversations/${encodeURIComponent(activeConversationId)}/rolling-short/versions?limit=2`
      );
      const currentVersion = rolling?.summary?.version;
      const previous = data.versions.find(
        (version) => version.version !== currentVersion
      );
      if (!previous) {
        throw new Error("No previous rolling summary version.");
      }
      await requestJson(
        `/conversations/${encodeURIComponent(activeConversationId)}/rolling-short/rollback`,
        {
          method: "POST",
          body: JSON.stringify({ version: previous.version })
        }
      );
      await loadConversation(activeConversationId);
      await refreshConversations();
    } catch (err) {
      setError(errorText(err));
    }
  }

  function chatPayload(conversationId: string, text: string) {
    return {
      client_id: clientId,
      conversation_id: conversationId,
      request_id: crypto.randomUUID(),
      assistant_key: assistantKey,
      ...(providerKey.trim() ? { provider_key: providerKey.trim() } : {}),
      ...(model.trim() ? { model: model.trim() } : {}),
      ...(systemPrompt.trim() ? { system_prompt: systemPrompt.trim() } : {}),
      stream: true,
      stream_options: { include_usage: true },
      user_text: text
    };
  }

  async function copyMessage(message: ChatMessage) {
    await navigator.clipboard.writeText(message.content);
    setCopiedMessageId(message.id);
    window.setTimeout(() => setCopiedMessageId(null), 1200);
  }

  return (
    <div className="app-shell">
      <aside className={`sidebar ${sidebarOpen ? "open" : ""}`}>
        <div className="sidebar-head">
          <div>
            <p className="eyebrow">Kelivo</p>
            <h1>Conversations</h1>
          </div>
          <button className="icon-button mobile-only" onClick={() => setSidebarOpen(false)} aria-label="Close conversations">
            <X size={18} />
          </button>
        </div>
        <div className="sidebar-actions">
          <button className="primary-action" onClick={newConversation}>
            <MessageSquarePlus size={17} />
            <span>New</span>
          </button>
          <button className="icon-button" onClick={refreshConversations} aria-label="Refresh conversations">
            {loading ? <Loader2 size={17} className="spin" /> : <RefreshCcw size={17} />}
          </button>
        </div>
        <div className="conversation-filter">
          <input
            value={conversationFilter}
            onChange={(event) => setConversationFilter(event.target.value)}
            placeholder="Search conversations"
          />
          <label>
            <input
              checked={showArchived}
              onChange={(event) => setShowArchived(event.target.checked)}
              type="checkbox"
            />
            <span>Archived</span>
          </label>
        </div>
        <div className="conversation-list">
          {filteredConversations.map((conversation) => (
            <div
              className={`conversation-row ${
                conversation.conversation_id === activeConversationId ? "active" : ""
              }`}
              key={conversation.conversation_id}
            >
              <button
                className="conversation-main"
                onClick={() => chooseConversation(conversation.conversation_id)}
              >
                <span className="conversation-title">
                  {conversation.title || conversation.assistant_key || "Untitled"}
                </span>
                <span className="conversation-preview">
                  {conversation.last_message_preview || conversation.conversation_id}
                </span>
                <span className="conversation-meta">
                  {conversation.message_count} messages
                </span>
              </button>
              <div className="conversation-tools">
                <button aria-label="Rename conversation" onClick={() => void renameConversation(conversation)}>
                  <Edit3 size={14} />
                </button>
                <button aria-label="Archive conversation" onClick={() => void archiveConversation(conversation)}>
                  <Archive size={14} />
                </button>
              </div>
            </div>
          ))}
        </div>
      </aside>

      <main className="main-pane">
        <header className="topbar">
          <button className="icon-button mobile-only" onClick={() => setSidebarOpen(true)} aria-label="Open conversations">
            <Menu size={19} />
          </button>
          <div className="title-block">
            <p className="eyebrow">{activeConversationId || "No conversation selected"}</p>
            <h2>{activeConversation?.title || activeConversation?.assistant_key || "Kelivo Web"}</h2>
          </div>
          <div className="status-pill">
            {sending ? <Loader2 size={15} className="spin" /> : <CheckCircle2 size={15} />}
            <span>{sending ? "Sending" : "Ready"}</span>
          </div>
          <button className="icon-button mobile-only" onClick={() => setInspectorOpen(true)} aria-label="Open route and summaries">
            <Settings2 size={18} />
          </button>
        </header>

        {error && (
          <div className="error-strip">
            <span>{error}</span>
            {failedSend && (
              <button onClick={retryFailedSend} disabled={sending}>
                Retry
              </button>
            )}
          </div>
        )}

        <section className="chat-grid">
          <div className="message-pane">
            <div className="messages">
              {messages.length === 0 ? (
                <div className="empty-state">
                  <MessageSquarePlus size={24} />
                  <span>Start a conversation</span>
                </div>
              ) : (
                messages.map((message) => (
                  <article className={`message ${message.role}`} key={`${message.id}-${message.role}`}>
                    <div className="message-header">
                      <span>{message.role}</span>
                      <time>
                        {copiedMessageId === message.id
                          ? "Copied"
                          : formatTime(message.timestamp)}
                      </time>
                    </div>
                    <div className="message-content">
                      {message.content ? (
                        <MarkdownText text={message.content} />
                      ) : (
                        <span className="stream-placeholder">Receiving...</span>
                      )}
                    </div>
                    <div className="message-token-line">
                      {tokenLabel(message)}
                    </div>
                    <div className="message-tools">
                      {message.role === "user" && (
                        <button
                          aria-label="Edit message"
                          onClick={() => editMessage(message)}
                          type="button"
                        >
                          <Edit3 size={14} />
                          <span>Edit</span>
                        </button>
                      )}
                      {message.role === "user" && (
                        <button
                          aria-label="Retry message"
                          onClick={() => void submitText(message.content)}
                          type="button"
                        >
                          <RotateCcw size={14} />
                          <span>Retry</span>
                        </button>
                      )}
                      <button
                        aria-label="Branch from message"
                        onClick={() => void branchFromMessage(message)}
                        type="button"
                      >
                        <GitBranch size={14} />
                        <span>Branch</span>
                      </button>
                      <button
                        aria-label="Copy message"
                        onClick={() => void copyMessage(message)}
                        type="button"
                      >
                        <Copy size={14} />
                        <span>Copy</span>
                      </button>
                      <button
                        aria-label="Delete message"
                        className="danger-tool"
                        onClick={() => void deleteMessage(message)}
                        type="button"
                      >
                        <Trash2 size={14} />
                        <span>Delete</span>
                      </button>
                    </div>
                  </article>
                ))
              )}
              <div ref={messageEndRef} />
            </div>

            <form className="composer" onSubmit={sendMessage}>
              <textarea
                ref={composerRef}
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
                    event.preventDefault();
                    void sendMessage(event);
                  }
                }}
                placeholder="Message"
                rows={3}
              />
              <button className="send-button" disabled={!draft.trim() || sending} type="submit" aria-label="Send message">
                {sending ? <Loader2 size={19} className="spin" /> : <Send size={19} />}
              </button>
            </form>
          </div>

          <aside className={`inspector ${inspectorOpen ? "open" : ""}`}>
            <section className="settings-panel">
              <div className="panel-head">
                <Settings2 size={17} />
                <h3>Route</h3>
                <button className="icon-button mobile-only panel-close" onClick={() => setInspectorOpen(false)} aria-label="Close route and summaries">
                  <X size={17} />
                </button>
              </div>
              <label>
                <span>Model override</span>
                <input value={model} onChange={(event) => setModel(event.target.value)} placeholder="Backend default" />
              </label>
              <label>
                <span>Assistant</span>
                <input value={assistantKey} onChange={(event) => setAssistantKey(event.target.value)} />
              </label>
              <label>
                <span>Provider override</span>
                <input value={providerKey} onChange={(event) => setProviderKey(event.target.value)} placeholder="Backend default" />
              </label>
              <label>
                <span>Client</span>
                <input value={clientId} readOnly />
              </label>
              <label>
                <span>System prompt</span>
                <textarea
                  className="system-prompt"
                  value={systemPrompt}
                  onChange={(event) => setSystemPrompt(event.target.value)}
                  placeholder="Optional per-request instruction"
                  rows={4}
                />
              </label>
            </section>

            <section className="summary-panel">
              <div className="panel-head">
                <RefreshCcw size={17} />
                <h3>Rolling</h3>
                <button
                  className="panel-tool"
                  disabled={!rolling?.summary}
                  onClick={() => void rollbackRollingSummary()}
                  type="button"
                  aria-label="Rollback rolling summary"
                >
                  <History size={14} />
                </button>
              </div>
              <div className="summary-text">
                <MarkdownText text={rolling?.summary?.summary || "No rolling summary yet."} />
              </div>
            </section>

            <section className="summary-panel">
              <div className="panel-head">
                <CalendarDays size={17} />
                <h3>{daily?.date_key || todayKey}</h3>
              </div>
              <div className="daily-runner">
                <label>
                  <span>Days ago</span>
                  <input
                    min={0}
                    max={365}
                    type="number"
                    value={dailyDaysAgo}
                    onChange={(event) =>
                      setDailyDaysAgo(Math.max(0, Number(event.target.value) || 0))
                    }
                  />
                </label>
                <label className="inline-check">
                  <input
                    checked={dailyForce}
                    onChange={(event) => setDailyForce(event.target.checked)}
                    type="checkbox"
                  />
                  <span>Force</span>
                </label>
                <button
                  className="panel-action"
                  disabled={dailyLoading}
                  onClick={() => void loadDailyByDaysAgo()}
                  type="button"
                >
                  {dailyLoading ? <Loader2 size={14} className="spin" /> : <CalendarDays size={14} />}
                  <span>Load</span>
                </button>
                <button
                  className="panel-action"
                  disabled={dailyRunning}
                  onClick={() => void runDailySummary()}
                  type="button"
                >
                  {dailyRunning ? <Loader2 size={14} className="spin" /> : <RefreshCcw size={14} />}
                  <span>Run</span>
                </button>
              </div>
              <div className="summary-text">
                <MarkdownText text={daily?.summary?.summary || "No daily summary yet."} />
              </div>
              {daily?.memory_candidates?.length ? (
                <div className="candidate-list">
                  {daily.memory_candidates.slice(0, 6).map((candidate) => (
                    <div className="candidate" key={candidate.id}>
                      <span>{candidate.label}</span>
                      <small>{candidate.target_layer || candidate.domain}</small>
                    </div>
                  ))}
                </div>
              ) : null}
            </section>
          </aside>
        </section>
      </main>
    </div>
  );
}

async function requestJson<T = unknown>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(apiPath(path), {
    ...init,
    headers: {
      "content-type": "application/json",
      ...(init?.headers || {})
    }
  });
  const text = await response.text();
  const data = text ? JSON.parse(text) : null;
  if (!response.ok) {
    throw new Error(apiErrorMessage(data, response.statusText));
  }
  return data as T;
}

async function streamChat({
  payload,
  onText
}: {
  payload: Record<string, unknown>;
  onText: (text: string) => void;
}) {
  const response = await fetch(apiPath("/chat"), {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload)
  });
  if (!response.ok || !response.body) {
    const text = await response.text();
    const data = text ? JSON.parse(text) : null;
    throw new Error(apiErrorMessage(data, response.statusText));
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split(/\r?\n/);
    buffer = lines.pop() || "";
    for (const line of lines) {
      const chunk = parseSseLine(line);
      if (chunk) {
        onText(chunk);
      }
    }
  }
  if (buffer) {
    const chunk = parseSseLine(buffer);
    if (chunk) {
      onText(chunk);
    }
  }
}

function apiPath(path: string) {
  if (/^https?:\/\//.test(path)) {
    return path;
  }
  const normalizedPath = path.startsWith("/") ? path : `/${path}`;
  return `${API_BASE}${normalizedPath}`;
}

function normalizeApiBase(value: unknown) {
  const raw = typeof value === "string" ? value.trim() : "";
  if (!raw || raw === "/") {
    return "";
  }
  return raw.endsWith("/") ? raw.slice(0, -1) : raw;
}

function parseSseLine(line: string) {
  if (!line.startsWith("data:")) {
    return "";
  }
  const payload = line.slice(5).trim();
  if (!payload || payload === "[DONE]") {
    return "";
  }
  try {
    const data = JSON.parse(payload);
    return extractStreamText(data);
  } catch {
    return "";
  }
}

function extractStreamText(data: unknown): string {
  if (typeof data !== "object" || data === null) {
    return "";
  }
  const payload = data as {
    choices?: Array<{
      delta?: { content?: unknown };
      message?: { content?: unknown };
      text?: unknown;
    }>;
    output_text?: unknown;
  };
  const first = payload.choices?.[0];
  const value =
    first?.delta?.content ??
    first?.message?.content ??
    first?.text ??
    payload.output_text;
  return typeof value === "string" ? value : "";
}

function apiErrorMessage(data: unknown, fallback: string) {
  if (typeof data !== "object" || data === null) {
    return fallback;
  }
  const payload = data as { error?: unknown };
  if (typeof payload.error === "string") {
    return payload.error;
  }
  if (typeof payload.error === "object" && payload.error !== null) {
    const error = payload.error as { message?: unknown };
    if (typeof error.message === "string" && error.message.trim()) {
      return error.message;
    }
  }
  return fallback;
}

function persistedId(key: string, prefix: string) {
  const existing = localStorage.getItem(key);
  if (existing) {
    return existing;
  }
  const next = `${prefix}_${crypto.randomUUID().replaceAll("-", "")}`;
  localStorage.setItem(key, next);
  return next;
}

function localDateKey(date: Date) {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function formatTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "";
  }
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit"
  }).format(date);
}

function tokenLabel(message: ChatMessage) {
  const usage = message.token_usage;
  if (usage?.total_tokens) {
    const parts = [`${usage.total_tokens} tokens`];
    if (usage.prompt_tokens || usage.completion_tokens) {
      parts.push(
        `in ${usage.prompt_tokens ?? "?"} / out ${usage.completion_tokens ?? "?"}`
      );
    }
    if (usage.reasoning_tokens) {
      parts.push(`reasoning ${usage.reasoning_tokens}`);
    }
    if (usage.cached_tokens) {
      parts.push(`cached ${usage.cached_tokens}`);
    }
    return parts.join(" · ");
  }
  const estimated = estimateTokens(message.content);
  return estimated ? `≈${estimated} tokens` : "≈0 tokens";
}

function estimateTokens(text: string) {
  const trimmed = text.trim();
  if (!trimmed) {
    return 0;
  }
  let asciiWordChars = 0;
  let nonAsciiChars = 0;
  for (const char of trimmed) {
    if (/\s/.test(char)) {
      continue;
    }
    if (char.charCodeAt(0) < 128) {
      asciiWordChars += 1;
    } else {
      nonAsciiChars += 1;
    }
  }
  return Math.max(1, Math.ceil(asciiWordChars / 4 + nonAsciiChars * 0.75));
}

function MarkdownText({ text }: { text: string }) {
  const blocks = splitMarkdownBlocks(text);
  return (
    <div className="markdown-body">
      {blocks.map((block, index) => {
        if (block.type === "code") {
          return (
            <pre key={index}>
              <code>{block.content}</code>
            </pre>
          );
        }
        return renderMarkdownLines(block.content, index);
      })}
    </div>
  );
}

function splitMarkdownBlocks(text: string) {
  const blocks: Array<{ type: "text" | "code"; content: string }> = [];
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  let buffer: string[] = [];
  let code: string[] | null = null;
  for (const line of lines) {
    if (line.trim().startsWith("```")) {
      if (code) {
        blocks.push({ type: "code", content: code.join("\n") });
        code = null;
      } else {
        if (buffer.length) {
          blocks.push({ type: "text", content: buffer.join("\n") });
          buffer = [];
        }
        code = [];
      }
      continue;
    }
    if (code) {
      code.push(line);
    } else {
      buffer.push(line);
    }
  }
  if (code) {
    blocks.push({ type: "code", content: code.join("\n") });
  }
  if (buffer.length) {
    blocks.push({ type: "text", content: buffer.join("\n") });
  }
  return blocks;
}

function renderMarkdownLines(text: string, keyPrefix: number) {
  const nodes: ReactNode[] = [];
  const lines = text.split("\n");
  let paragraph: string[] = [];
  let list: string[] = [];

  function flushParagraph() {
    if (!paragraph.length) {
      return;
    }
    nodes.push(
      <p key={`${keyPrefix}-p-${nodes.length}`}>
        {renderInlineMarkdown(paragraph.join(" "))}
      </p>
    );
    paragraph = [];
  }

  function flushList() {
    if (!list.length) {
      return;
    }
    nodes.push(
      <ul key={`${keyPrefix}-ul-${nodes.length}`}>
        {list.map((item, index) => (
          <li key={index}>{renderInlineMarkdown(item)}</li>
        ))}
      </ul>
    );
    list = [];
  }

  for (const rawLine of lines) {
    const line = rawLine.trimEnd();
    const heading = /^(#{1,3})\s+(.+)$/.exec(line);
    const bullet = /^[-*]\s+(.+)$/.exec(line);
    const quote = /^>\s?(.+)$/.exec(line);
    if (!line.trim()) {
      flushParagraph();
      flushList();
      continue;
    }
    if (heading) {
      flushParagraph();
      flushList();
      const level = heading[1].length;
      const Tag = level === 1 ? "h4" : level === 2 ? "h5" : "h6";
      nodes.push(
        <Tag key={`${keyPrefix}-h-${nodes.length}`}>
          {renderInlineMarkdown(heading[2])}
        </Tag>
      );
      continue;
    }
    if (bullet) {
      flushParagraph();
      list.push(bullet[1]);
      continue;
    }
    if (quote) {
      flushParagraph();
      flushList();
      nodes.push(
        <blockquote key={`${keyPrefix}-q-${nodes.length}`}>
          {renderInlineMarkdown(quote[1])}
        </blockquote>
      );
      continue;
    }
    flushList();
    paragraph.push(line);
  }
  flushParagraph();
  flushList();
  return nodes.length ? nodes : <p key={`${keyPrefix}-empty`} />;
}

function renderInlineMarkdown(text: string): ReactNode[] {
  const nodes: ReactNode[] = [];
  const pattern = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)/g;
  let cursor = 0;
  for (const match of text.matchAll(pattern)) {
    if (match.index > cursor) {
      nodes.push(text.slice(cursor, match.index));
    }
    const token = match[0];
    if (token.startsWith("`")) {
      nodes.push(<code key={nodes.length}>{token.slice(1, -1)}</code>);
    } else if (token.startsWith("**")) {
      nodes.push(<strong key={nodes.length}>{token.slice(2, -2)}</strong>);
    } else {
      nodes.push(<em key={nodes.length}>{token.slice(1, -1)}</em>);
    }
    cursor = match.index + token.length;
  }
  if (cursor < text.length) {
    nodes.push(text.slice(cursor));
  }
  return nodes;
}

function errorText(error: unknown) {
  return error instanceof Error ? error.message : "Something went wrong.";
}
