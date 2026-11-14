"""The Analysis Agent's Gemini path, with a fake google-genai client.

Offline.  Covers what differs from the Claude path (tests/test_analysis_agent_harness.py):
the request shape, reading non-thought text parts, mapping finish reasons onto
the shared flags, and not retrying a free-tier daily quota.
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from google.genai import errors as genai_errors

from src import config
from src.agents import analysis_agent


def _response(text: str = "", *, finish: str = "STOP", block: str | None = None, thought: str | None = None):
    parts = []
    if thought is not None:
        parts.append(SimpleNamespace(text=thought, thought=True))
    parts.append(SimpleNamespace(text=text, thought=None))
    return SimpleNamespace(
        candidates=[SimpleNamespace(content=SimpleNamespace(parts=parts), finish_reason=SimpleNamespace(name=finish))],
        prompt_feedback=SimpleNamespace(block_reason=SimpleNamespace(name=block)) if block else None,
        usage_metadata=SimpleNamespace(prompt_token_count=900, candidates_token_count=200, thoughts_token_count=50),
        model_version="gemini-3.8-flash-001",
    )


class _FakeModels:
    def __init__(self, responder) -> None:
        self._responder = responder
        self.calls: list[dict] = []

    def generate_content(self, **request):
        self.calls.append(request)
        result = self._responder(request)
        if isinstance(result, Exception):
            raise result
        return _response(result) if isinstance(result, str) else result


@contextlib.contextmanager
def _patched(responder, *, max_retries: int = 1):
    models = _FakeModels(responder)
    saved = (analysis_agent._get_gemini_client, config.LLM_MODEL, config.LLM_MAX_RETRIES, config.LLM_RETRY_BASE_SECONDS)
    analysis_agent._get_gemini_client = lambda: SimpleNamespace(models=models)
    config.LLM_MODEL, config.LLM_MAX_RETRIES, config.LLM_RETRY_BASE_SECONDS = "gemini-3.8-flash", max_retries, 0.0
    try:
        yield models
    finally:
        analysis_agent._get_gemini_client, config.LLM_MODEL, config.LLM_MAX_RETRIES, config.LLM_RETRY_BASE_SECONDS = saved


def _dimension(**overrides) -> dict:
    return {
        "dimension_id": "test_dimension",
        "name": "Test",
        "prompt_template": {"system": "sys", "user_template": "Evaluate {company_name}.\n{retrieved_chunks}"},
        **overrides,
    }


def _analyze(dim=None):
    return analysis_agent.analyze_dimension(
        dimension=dim or _dimension(),
        company_name="TestCo",
        retrieved_chunks=[{"id": "x_chunk_1", "text": "net loss of $1.2B", "headings": ["Item 7"], "source_page": 5, "score": 0.8}],
        retrieval_score=0.8,
        retrieval_attempts=1,
    )


def _payload(verdict: str = "HIGH", **extra) -> str:
    return json.dumps({"verdict": verdict, "reasoning": "r", "evidence": [], "flags": [], **extra})


def _error(code: int, message: str = "error") -> genai_errors.APIError:
    cls = genai_errors.ClientError if code < 500 else genai_errors.ServerError
    return cls(code, {"error": {"code": code, "message": message, "status": "X"}})


def test_request_carries_schema_system_and_temperature() -> None:
    schema = {"type": "object", "properties": {"verdict": {"type": "string"}}}
    with _patched(lambda r: _payload()) as models:
        _analyze(_dimension(output_schema=schema))
    request = models.calls[0]
    assert request["model"] == "gemini-3.8-flash"
    assert request["config"]["system_instruction"] == "sys"
    assert request["config"]["temperature"] == config.LLM_TEMPERATURE
    assert request["config"]["response_mime_type"] == "application/json"
    assert request["config"]["response_json_schema"] == schema
    assert "<evidence id=\"E1\"" in request["contents"]


def test_answer_skips_thought_parts_and_records_usage() -> None:
    reply = _response(_payload("LOW"), thought='{"verdict": "HIGH"}')
    with _patched(lambda r: reply):
        verdict = _analyze()
    assert verdict["rating"] == config.RATING_LOW
    assert verdict["served_model"] == "gemini-3.8-flash-001"
    assert verdict["usage"] == {"input_tokens": 900, "output_tokens": 200, "thinking_tokens": 50}
    assert "model_fallback" not in verdict["flags"]


def test_max_tokens_is_truncated() -> None:
    with _patched(lambda r: _response('{"verdict": "HI', finish="MAX_TOKENS")):
        assert _analyze()["flags"] == ["truncated"]


def test_recitation_block_is_a_refusal_naming_the_cause() -> None:
    with _patched(lambda r: _response("", finish="RECITATION")):
        verdict = _analyze()
    assert verdict["flags"] == ["refusal"]
    assert "RECITATION" in verdict["reasoning"]


def test_blocked_prompt_is_a_refusal() -> None:
    with _patched(lambda r: _response("", finish="STOP", block="SAFETY")):
        assert _analyze()["flags"] == ["refusal"]


def test_per_minute_rate_limit_is_retried() -> None:
    calls = {"n": 0}

    def flaky(request):
        calls["n"] += 1
        return _error(429, "GenerateRequestsPerMinutePerProjectPerModel-FreeTier") if calls["n"] == 1 else _payload()

    with _patched(flaky, max_retries=3) as models:
        verdict = _analyze()
    assert len(models.calls) == 2
    assert verdict["rating"] == config.RATING_HIGH


def test_retry_waits_as_long_as_the_server_asks(monkeypatch) -> None:
    waits: list[float] = []
    monkeypatch.setattr(analysis_agent.time, "sleep", waits.append)
    calls = {"n": 0}

    def flaky(request):
        calls["n"] += 1
        return _error(429, "per minute limit. Please retry in 25.8s.") if calls["n"] == 1 else _payload()

    with _patched(flaky, max_retries=3):
        _analyze()
    assert waits == [26.8]


def test_daily_quota_is_not_retried() -> None:
    with _patched(lambda r: _error(429, "GenerateRequestsPerDayPerProjectPerModel-FreeTier"), max_retries=3) as models:
        verdict = _analyze()
    assert len(models.calls) == 1, "A daily quota will not clear within the backoff"
    assert "analysis_error" in verdict["flags"]
    assert "PerDay" in verdict["reasoning"]


def test_server_error_retried_client_error_not() -> None:
    with _patched(lambda r: _error(503), max_retries=2) as models:
        _analyze()
    assert len(models.calls) == 2
    with _patched(lambda r: _error(400), max_retries=3) as models:
        _analyze()
    assert len(models.calls) == 1


def test_model_name_selects_the_provider() -> None:
    saved = config.LLM_MODEL
    try:
        config.LLM_MODEL = "gemini-3.8-flash"
        assert config.llm_provider() == "google"
        config.LLM_MODEL = "claude-sonnet-5-5"
        assert config.llm_provider() == "anthropic"
        config.LLM_MODEL = "gpt-9"
        assert config.llm_provider() == ""
        assert any("gemini-* or claude-*" in e for e in config.validate_llm_config())
    finally:
        config.LLM_MODEL = saved
