from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


TOOL_GROUPS = {
    "context_read",
    "memory_write",
    "external_action",
    "code",
    "web",
}


@dataclass(frozen=True)
class ToolDef:
    name: str
    group: str
    mutates_state: bool
    requires_confirmation: bool
    callable_by: tuple[str, ...]
    description: str


TOOLS: tuple[ToolDef, ...] = (
    ToolDef(
        name="get_core_anchors",
        group="context_read",
        mutates_state=False,
        requires_confirmation=False,
        callable_by=("context_builder", "agent"),
        description="Read active core anchors by function or key.",
    ),
    ToolDef(
        name="get_rolling_summary",
        group="context_read",
        mutates_state=False,
        requires_confirmation=False,
        callable_by=("context_builder", "agent"),
        description="Read the rolling summary for a conversation.",
    ),
    ToolDef(
        name="get_recent_turns",
        group="context_read",
        mutates_state=False,
        requires_confirmation=False,
        callable_by=("context_builder", "agent"),
        description="Read recent chat turns for a conversation.",
    ),
    ToolDef(
        name="search_chatlog_history",
        group="context_read",
        mutates_state=False,
        requires_confirmation=False,
        callable_by=("context_builder", "agent"),
        description="Search chat log history through the deployed kmlog API.",
    ),
    ToolDef(
        name="get_worldbook_entries",
        group="context_read",
        mutates_state=False,
        requires_confirmation=False,
        callable_by=("context_builder", "agent"),
        description="Read matching local WorldBook entries.",
    ),
    ToolDef(
        name="get_daily_summary",
        group="context_read",
        mutates_state=False,
        requires_confirmation=False,
        callable_by=("context_builder", "agent"),
        description="Read a daily summary by date.",
    ),
    ToolDef(
        name="get_daily_memory_candidates",
        group="context_read",
        mutates_state=False,
        requires_confirmation=False,
        callable_by=("context_builder", "agent"),
        description="Read auditable daily memory candidates.",
    ),
    ToolDef(
        name="get_health_snapshot",
        group="context_read",
        mutates_state=False,
        requires_confirmation=False,
        callable_by=("context_builder", "agent"),
        description="Read a health snapshot when a provider is configured.",
    ),
    ToolDef(
        name="save_raw_turns",
        group="memory_write",
        mutates_state=True,
        requires_confirmation=False,
        callable_by=("agent", "frontend"),
        description="Persist raw chat turns through the chat proxy.",
    ),
    ToolDef(
        name="update_wishes",
        group="memory_write",
        mutates_state=True,
        requires_confirmation=True,
        callable_by=("agent", "frontend"),
        description="Create or update wishes.",
    ),
    ToolDef(
        name="promote_memory_candidate",
        group="memory_write",
        mutates_state=True,
        requires_confirmation=True,
        callable_by=("agent",),
        description="Promote a reviewed memory candidate into long-term memory.",
    ),
    ToolDef(
        name="write_daily_summary",
        group="memory_write",
        mutates_state=True,
        requires_confirmation=True,
        callable_by=("agent",),
        description="Write or refresh a daily summary.",
    ),
    ToolDef(
        name="update_core_anchor",
        group="memory_write",
        mutates_state=True,
        requires_confirmation=True,
        callable_by=("agent",),
        description="Create or update a curated core anchor.",
    ),
    ToolDef(
        name="write_notion_page",
        group="external_action",
        mutates_state=True,
        requires_confirmation=True,
        callable_by=("agent",),
        description="Write a Notion page through an external integration.",
    ),
    ToolDef(
        name="create_task",
        group="external_action",
        mutates_state=True,
        requires_confirmation=True,
        callable_by=("agent",),
        description="Create an external task.",
    ),
    ToolDef(
        name="calculator",
        group="code",
        mutates_state=False,
        requires_confirmation=False,
        callable_by=("agent",),
        description="Perform deterministic calculations.",
    ),
    ToolDef(
        name="code_runner",
        group="code",
        mutates_state=False,
        requires_confirmation=True,
        callable_by=("agent",),
        description="Run code in a controlled runtime.",
    ),
    ToolDef(
        name="web_search",
        group="web",
        mutates_state=False,
        requires_confirmation=False,
        callable_by=("agent",),
        description="Search the web when current external facts are needed.",
    ),
    ToolDef(
        name="file_reader",
        group="external_action",
        mutates_state=False,
        requires_confirmation=False,
        callable_by=("agent",),
        description="Read user-approved files.",
    ),
    ToolDef(
        name="notion_search",
        group="external_action",
        mutates_state=False,
        requires_confirmation=False,
        callable_by=("agent",),
        description="Search Notion workspace content.",
    ),
    ToolDef(
        name="github_tool",
        group="external_action",
        mutates_state=True,
        requires_confirmation=True,
        callable_by=("agent",),
        description="Read or mutate GitHub state through an integration.",
    ),
    ToolDef(
        name="artifact_tool",
        group="external_action",
        mutates_state=True,
        requires_confirmation=True,
        callable_by=("agent",),
        description="Create or update user-visible artifacts.",
    ),
)


def resolve_tools_policy(policy: Mapping[str, Any] | None) -> dict[str, Any]:
    policy = policy if isinstance(policy, Mapping) else {}
    tool_mode = str(policy.get("tool_mode") or "read_only").strip() or "read_only"
    if tool_mode not in {"none", "read_only", "agent_managed"}:
        tool_mode = "read_only"
    expose_tools = bool(policy.get("expose_tools"))
    write_requires_confirmation = policy.get("write_requires_confirmation")
    if not isinstance(write_requires_confirmation, bool):
        write_requires_confirmation = True

    requested_groups = _requested_groups(policy.get("allowed_tool_groups"))
    if not requested_groups:
        if tool_mode == "read_only":
            requested_groups = ["context_read"]
        elif tool_mode == "agent_managed":
            requested_groups = sorted(TOOL_GROUPS)
        else:
            requested_groups = []

    available_tools = []
    for tool in TOOLS:
        if tool.group not in requested_groups:
            continue
        if tool_mode == "none":
            continue
        if tool_mode == "read_only" and tool.mutates_state:
            continue
        available_tools.append(tool)

    notes = [
        "Context builder declares tool policy but does not execute action tools.",
    ]
    if tool_mode == "none":
        notes.append("No runtime tools are exposed for this turn.")
    elif tool_mode == "read_only":
        notes.append("Only non-mutating tools are available.")
    else:
        notes.append("Agent may manage tools, but writes still require policy checks.")
    if write_requires_confirmation:
        notes.append("State-changing tools require confirmation.")

    groups = sorted({tool.group for tool in available_tools})
    return {
        "policy": {
            "expose_tools": expose_tools,
            "allowed_tool_groups": requested_groups,
            "write_requires_confirmation": write_requires_confirmation,
            "tool_mode": tool_mode,
        },
        "available_tool_groups": groups,
        "available_tools": [_tool_payload(tool) for tool in available_tools]
        if expose_tools
        else [],
        "policy_notes": notes,
    }


def _requested_groups(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out = []
    seen = set()
    for raw_item in value:
        item = str(raw_item or "").strip()
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _tool_payload(tool: ToolDef) -> dict[str, Any]:
    payload = asdict(tool)
    payload["callable_by"] = list(tool.callable_by)
    return payload
