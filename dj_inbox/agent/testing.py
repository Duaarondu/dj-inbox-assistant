# dj_inbox/agent/testing.py
"""A scripted stand-in for Gemini, so agent tests run in CI with no credentials and no cost.

Give it the model turns in order: a turn can call a tool (fake_tool_call) or answer with text (fake_text).
It records every request, so a test can check which tools and instructions the model was shown.
"""
from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Any

from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types
from pydantic import Field


def fake_tool_call(name: str, **args: Any) -> types.Content:
    return types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))])


def fake_text(text: str) -> types.Content:
    return types.Content(role="model", parts=[types.Part(text=text)])


class FakeLlm(BaseLlm):
    model: str = "fake-llm"
    script: list[types.Content] = Field(default_factory=list)
    requests: list[LlmRequest] = Field(default_factory=list)
    tokens_in: int = 100
    tokens_out: int = 20

    async def generate_content_async(self, llm_request: LlmRequest, stream: bool = False
                                     ) -> AsyncGenerator[LlmResponse, None]:
        self.requests.append(llm_request)
        turn = len(self.requests) - 1
        if turn >= len(self.script):
            raise AssertionError(f"FakeLlm was called {turn + 1} times but only scripted {len(self.script)} turns")
        usage = types.GenerateContentResponseUsageMetadata(prompt_token_count=self.tokens_in,
                                                           candidates_token_count=self.tokens_out)
        yield LlmResponse(content=self.script[turn], usage_metadata=usage)

    def tool_names_shown(self, request_index: int = 0) -> list[str]:
        """Names of the tools the model was offered on a given call."""
        config = self.requests[request_index].config
        return [d.name for tool in (config.tools or []) for d in (tool.function_declarations or [])]

    def tool_params_shown(self, name: str, request_index: int = 0) -> set[str]:
        """Parameter names the model can fill in for one tool (tool_context must never be one of them)."""
        config = self.requests[request_index].config
        for tool in config.tools or []:
            for d in tool.function_declarations or []:
                if d.name == name:
                    schema = d.parameters_json_schema or (d.parameters.model_dump() if d.parameters else {})
                    return set((schema or {}).get("properties", {}))
        raise KeyError(name)
