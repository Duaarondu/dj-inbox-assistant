# dj_inbox/agent/runtime.py
"""Run an ADK workflow for one question and collect what the eval and the traces need.

company_id goes into the SESSION STATE, never into the question or a tool argument. Tools read it with
tool_context.state["company_id"], and ADK hides tool_context from the model, so a model (or an email
pretending to give instructions) can't ask for another company's data.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from google.adk import Runner, Workflow
from google.adk.sessions import InMemorySessionService
from google.genai import types


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]
    response: Any = None


@dataclass
class RunResult:
    output: Any                                   # what the workflow's top node returned
    tool_calls: list[ToolCall] = field(default_factory=list)
    model_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0


async def run_workflow_async(root: Workflow, question: str, company_id: str) -> RunResult:
    """One question, one fresh in-memory session, so no answer can leak into the next question."""
    sessions = InMemorySessionService()
    session_id = uuid.uuid4().hex
    await sessions.create_session(app_name=root.name, user_id="eval", session_id=session_id,
                                  state={"company_id": company_id})
    runner = Runner(node=root, session_service=sessions)
    message = types.Content(role="user", parts=[types.Part(text=question)])

    result = RunResult(output=None)
    by_id: dict[str, ToolCall] = {}
    start = time.perf_counter()
    async for event in runner.run_async(user_id="eval", session_id=session_id, new_message=message):
        if event.content:
            for call in event.get_function_calls():
                tool_call = ToolCall(name=call.name, args=dict(call.args or {}))
                by_id[call.id or f"{call.name}-{len(by_id)}"] = tool_call
                result.tool_calls.append(tool_call)
            for response in event.get_function_responses():
                matched = by_id.get(response.id or "")
                if matched is None:                # no id to match on: attach to the last call with that name
                    matched = next((c for c in reversed(result.tool_calls) if c.name == response.name), None)
                if matched is not None:
                    matched.response = response.response
        if event.usage_metadata:                   # one per model call
            result.model_calls += 1
            result.tokens_in += event.usage_metadata.prompt_token_count or 0
            result.tokens_out += (event.usage_metadata.candidates_token_count or 0) + (
                event.usage_metadata.thoughts_token_count or 0)   # thinking tokens are billed as output
        if event.author == root.name and getattr(event, "output", None) is not None:
            result.output = event.output           # the last output from the top node is the answer
    result.latency_s = time.perf_counter() - start
    return result


def run_workflow(root: Workflow, question: str, company_id: str) -> RunResult:
    """Blocking version for scripts and eval runners."""
    return asyncio.run(run_workflow_async(root, question, company_id))
