import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  CalendarDays,
  CheckCircle2,
  Loader2,
  Menu,
  MessageSquarePlus,
  RefreshCcw,
  Send,
  Settings2,
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
};

type ChatMessage = {
  id: number;
  timestamp: string;
  role: string;
  content: string;
  conversation_id: string;
  kind: string | null;
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

const CLIENT_KEY = "chatProxyWeb.clientId";
const CONVERSATION_KEY = "chatProxyWeb.conversationId";
const MODEL_KEY = "chatProxyWeb.model";
const ASSISTANT_KEY = "chatProxyWeb.assistantKey";
const PROVIDER_KEY = "chatProxyWeb.providerKey";
const SYSTEM_PROMPT_KEY = "chatProxyWeb.systemPrompt";

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
  const [loading, setLoading] = useState(false);
  const [sending, setSending] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messageEndRef = useRef<HTMLDivElement | null>(null);

  const activeConversation = useMemo(
    () =>
      conversations.find(
        (item) => item.conversation_id === activeConversationId
      ) || null,
    [activeConversationId, conversations]
  );
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
  }, [todayKey]);

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
        "/conversations?limit=100"
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

  async function sendMessage(event: FormEvent) {
    event.preventDefault();
    const text = draft.trim();
    if (!text || sending) {
      return;
    }
    const conversationId =
      activeConversationId || `conv_${crypto.randomUUID().replaceAll("-", "")}`;
    if (!activeConversationId) {
      setActiveConversationId(conversationId);
    }
    setDraft("");
    setSending(true);
    setError(null);
    setMessages((current) => [
      ...current,
      {
        id: Date.now(),
        timestamp: new Date().toISOString(),
        role: "user",
        content: text,
        conversation_id: conversationId,
        kind: "chat"
      }
    ]);
    try {
      await requestJson("/chat", {
        method: "POST",
        body: JSON.stringify({
          client_id: clientId,
          conversation_id: conversationId,
          request_id: crypto.randomUUID(),
          assistant_key: assistantKey,
          ...(providerKey.trim() ? { provider_key: providerKey.trim() } : {}),
          ...(model.trim() ? { model: model.trim() } : {}),
          ...(systemPrompt.trim() ? { system_prompt: systemPrompt.trim() } : {}),
          user_text: text
        })
      });
      await Promise.all([
        loadConversation(conversationId),
        refreshConversations(),
        loadDaily(todayKey)
      ]);
    } catch (err) {
      setError(errorText(err));
    } finally {
      setSending(false);
    }
  }

  function chooseConversation(conversationId: string) {
    setActiveConversationId(conversationId);
    setSidebarOpen(false);
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
        <div className="conversation-list">
          {conversations.map((conversation) => (
            <button
              className={`conversation-row ${
                conversation.conversation_id === activeConversationId ? "active" : ""
              }`}
              key={conversation.conversation_id}
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
        </header>

        {error && <div className="error-strip">{error}</div>}

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
                      <time>{formatTime(message.timestamp)}</time>
                    </div>
                    <div className="message-content">{message.content}</div>
                  </article>
                ))
              )}
              {sending && (
                <article className="message assistant pending">
                  <div className="message-header">
                    <span>assistant</span>
                    <Loader2 size={14} className="spin" />
                  </div>
                </article>
              )}
              <div ref={messageEndRef} />
            </div>

            <form className="composer" onSubmit={sendMessage}>
              <textarea
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

          <aside className="inspector">
            <section className="settings-panel">
              <div className="panel-head">
                <Settings2 size={17} />
                <h3>Route</h3>
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
              </div>
              <div className="summary-text">
                {rolling?.summary?.summary || "No rolling summary yet."}
              </div>
            </section>

            <section className="summary-panel">
              <div className="panel-head">
                <CalendarDays size={17} />
                <h3>{todayKey}</h3>
              </div>
              <div className="summary-text">
                {daily?.summary?.summary || "No daily summary yet."}
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
  const response = await fetch(path, {
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

function errorText(error: unknown) {
  return error instanceof Error ? error.message : "Something went wrong.";
}
