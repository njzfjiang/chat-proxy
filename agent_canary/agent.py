from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models import Model

from .context_client import BuildContextClient, ContextEvidence, ContextToolTrace


class CanaryAnswer(BaseModel):
    answer: str = Field(description="Concise answer grounded only in tool evidence")
    evidence_status: Literal["supported", "insufficient"]
    claim_type: Literal[
        "canon",
        "design_suggestion",
        "personal_statement",
        "unknown",
    ]


@dataclass
class AgentDeps:
    context_client: BuildContextClient
    prompt: str
    conversation_id: str
    as_of_timestamp: str | None = None
    exclude_conversation_ids: list[str] = field(default_factory=list)
    tool_traces: list[ContextToolTrace] = field(default_factory=list)
    prefetched_context: tuple[ContextEvidence, ContextToolTrace] | None = None


INSTRUCTIONS = """
You are a read-only memory canary. Call retrieve_context exactly once before
answering. Use only facts visible in that tool result. Do not fill gaps from
general knowledge or assumptions. If the evidence is incomplete, say so and
set evidence_status to insufficient. Distinguish established canon from prior
design suggestions and personal statements; when the evidence does not
establish one of these categories, use unknown.
Keep the answer concise and in the user's language.
""".strip()


def build_agent(model: Model | str) -> Agent[AgentDeps, CanaryAnswer]:
    agent = Agent(
        model,
        deps_type=AgentDeps,
        output_type=CanaryAnswer,
        instructions=INSTRUCTIONS,
        name="retrieval_canary",
        retries=1,
    )

    @agent.tool
    async def retrieve_context(ctx: RunContext[AgentDeps]) -> ContextEvidence:
        """Retrieve the proxy-rendered context for the current user question."""
        if ctx.deps.prefetched_context is not None:
            evidence, trace = ctx.deps.prefetched_context
            ctx.deps.prefetched_context = None
        else:
            evidence, trace = await ctx.deps.context_client.retrieve(
                prompt=ctx.deps.prompt,
                conversation_id=ctx.deps.conversation_id,
                as_of_timestamp=ctx.deps.as_of_timestamp,
                exclude_conversation_ids=ctx.deps.exclude_conversation_ids,
            )
        ctx.deps.tool_traces.append(trace)
        return evidence

    return agent
