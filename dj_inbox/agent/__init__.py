"""The ADK agent: answer a question by calling read-only tools, then verify every number against the sources.

    llm.py      Gemini on Google Cloud for ADK agents (same project, timeout and retries as dj_inbox/gcp.py)
    runtime.py  run a workflow for one question; collect the answer, tool calls, tokens and latency
    testing.py  FakeLlm: a scripted model so tests run without credentials or cost
    smoke.py    one live call to prove the setup works:  python -m dj_inbox.agent.smoke
"""
