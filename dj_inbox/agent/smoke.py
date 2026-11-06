# dj_inbox/agent/smoke.py
"""Live smoke test: one real Gemini call through ADK, with one tool that reads company_id from the session.

    python -m dj_inbox.agent.smoke                      (default: gemini-3.5-flash-lite)
    python -m dj_inbox.agent.smoke gemini-3.8-flash

Passes when the model calls the tool exactly once and the answer names the company. Costs a fraction of a cent.
"""
from __future__ import annotations

import sys

from google.adk import Agent, Context, Workflow
from google.adk.tools.tool_context import ToolContext
from google.adk.workflow import node

from dj_inbox.agent.llm import adk_gemini
from dj_inbox.agent.runtime import run_workflow

COMPANY_ID = "northshore"
COMPANY_NAMES = {"northshore": "Northshore Sound Co."}


def get_company(tool_context: ToolContext) -> dict:
    """Return the name of the DJ company this assistant works for."""
    company_id = tool_context.state["company_id"]
    return {"company_id": company_id, "company_name": COMPANY_NAMES[company_id]}


def build(model_name: str) -> Workflow:
    answerer = Agent(
        name="answerer",
        model=adk_gemini(model_name),
        instruction="Call get_company, then answer the user's question in one short sentence.",
        tools=[get_company],
    )

    @node(rerun_on_resume=True)
    async def answer(ctx: Context, node_input: str) -> str:
        return await ctx.run_node(answerer, node_input)

    return Workflow(name="smoke", edges=[("START", answer)])


def main() -> None:
    model_name = sys.argv[1] if len(sys.argv) > 1 else "gemini-3.5-flash-lite"
    result = run_workflow(build(model_name), "Which company do you work for?", COMPANY_ID)

    print(f"model:       {model_name}")
    print(f"answer:      {result.output}")
    for call in result.tool_calls:
        print(f"tool call:   {call.name}({call.args}) -> {call.response}")
    print(f"model calls: {result.model_calls}   tokens in/out: {result.tokens_in}/{result.tokens_out}")
    print(f"latency:     {result.latency_s:.2f}s")

    ok = [c.name for c in result.tool_calls] == ["get_company"] and "Northshore" in str(result.output)
    print("PASS" if ok else "FAIL: expected one get_company call and 'Northshore' in the answer")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
