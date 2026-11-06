# tests/test_agent_runtime.py
"""The ADK plumbing, tested with a scripted model: no credentials, no cost, runs in CI."""
from __future__ import annotations

import pytest
from google.adk import Agent, Context, Workflow
from google.adk.tools.tool_context import ToolContext
from google.adk.workflow import node

from dj_inbox.agent.runtime import run_workflow
from dj_inbox.agent.testing import FakeLlm, fake_text, fake_tool_call


def get_company(tool_context: ToolContext) -> dict:
    """Return the company this assistant works for."""
    return {"company_id": tool_context.state["company_id"]}


def count_weddings(year: int, tool_context: ToolContext) -> dict:
    """Count booked weddings in a year."""
    return {"count": 33, "year": year, "company_id": tool_context.state["company_id"]}


def one_agent_workflow(model: FakeLlm, tools: list) -> Workflow:
    answerer = Agent(name="answerer", model=model, instruction="Answer the question.", tools=tools)

    @node(rerun_on_resume=True)
    async def answer(ctx: Context, node_input: str) -> str:
        return await ctx.run_node(answerer, node_input)

    return Workflow(name="test_app", edges=[("START", answer)])


def test_answer_tool_calls_and_tokens_are_collected():
    model = FakeLlm(script=[fake_tool_call("count_weddings", year=2026), fake_text("We booked 33 weddings.")])
    result = run_workflow(one_agent_workflow(model, [count_weddings]), "How many weddings in 2026?", "northshore")

    assert result.output == "We booked 33 weddings."
    assert [(c.name, c.args) for c in result.tool_calls] == [("count_weddings", {"year": 2026})]
    assert result.tool_calls[0].response == {"count": 33, "year": 2026, "company_id": "northshore"}
    assert result.model_calls == 2
    assert (result.tokens_in, result.tokens_out) == (200, 40)


def test_company_id_comes_from_the_session_not_the_model():
    model = FakeLlm(script=[fake_tool_call("get_company"), fake_text("Northshore.")])
    result = run_workflow(one_agent_workflow(model, [get_company]), "Who are you?", "company-b")

    assert result.tool_calls[0].response == {"company_id": "company-b"}
    assert model.tool_names_shown() == ["get_company"]
    assert model.tool_params_shown("get_company") == set()       # the model can't pass a company_id


def test_tool_context_is_hidden_but_real_arguments_are_shown():
    model = FakeLlm(script=[fake_text("Not sure.")])
    run_workflow(one_agent_workflow(model, [count_weddings]), "How many?", "northshore")

    assert model.tool_params_shown("count_weddings") == {"year"}


def test_each_question_gets_a_fresh_session():
    model = FakeLlm(script=[fake_text("first"), fake_text("second")])
    workflow = one_agent_workflow(model, [])
    run_workflow(workflow, "question one", "northshore")
    run_workflow(workflow, "question two", "northshore")

    second_request_text = str(model.requests[1].contents)
    assert "question one" not in second_request_text   # nothing from question one leaks into question two


def test_fake_llm_fails_loudly_when_called_more_than_scripted():
    model = FakeLlm(script=[fake_tool_call("get_company")])    # no text turn after the tool result
    with pytest.raises(Exception, match="only scripted 1 turns"):
        run_workflow(one_agent_workflow(model, [get_company]), "Who are you?", "northshore")
