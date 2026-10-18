"""Harness tests for the Analysis Agent using a mocked LLM.

Runnable as a script or via pytest.  Covers the two paths through
`analyze_dimension()` and the no-evidence fallback in particular:

  1. no_evidence_uses_explicit_rating  — fallback comes from `no_evidence_rating`
  2. no_evidence_ignores_prose         — prose naming a forbidden rating is not scanned
  3. no_evidence_skips_llm             — the no-evidence path makes no model call
  4. no_evidence_honours_high          — an explicit HIGH fallback is respected
  5. real_dimensions_never_fall_to_low — regression guard over the shipped configs
  6. valid_json_response               — happy path parses rating and citations
  7. parse_error_flagged               — unparseable response is flagged, not silent
  8. analysis_error_flagged            — model exception is flagged, not silent
  9. loader_rejects_bad_rating         — unrecognized config value is dropped with a warning
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.agents import analysis_agent, document_agent
from src.agents.document_agent import load_dimensions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeLLM:
    """Records invocations so tests can assert whether the model was called."""

    def __init__(self, responder) -> None:
        self._responder = responder
        self.calls: list[list[dict]] = []

    def invoke(self, messages):
        self.calls.append(messages)
        result = self._responder(messages)
        if isinstance(result, Exception):
            raise result
        return _FakeResponse(result)


@contextlib.contextmanager
def patched_llm(responder, *, max_retries: int = 1):
    """Swap `analysis_agent._get_llm` for a fake, restoring it on exit.

    Restoring matters for the same reason it does in the retrieval harness:
    `analysis_agent` is a cached module, so a fake left installed would leak
    into any later test in the session.

    Retry backoff is zeroed and retries default to 1 so failure-path tests do
    not spend real seconds sleeping; pass `max_retries` to exercise the retry
    loop itself.
    """
    fake = _FakeLLM(responder)
    original = analysis_agent._get_llm
    orig_retries = config.LLM_MAX_RETRIES
    orig_delay = config.LLM_RETRY_BASE_SECONDS
    analysis_agent._get_llm = lambda: fake
    config.LLM_MAX_RETRIES = max_retries
    config.LLM_RETRY_BASE_SECONDS = 0.0
    try:
        yield fake
    finally:
        analysis_agent._get_llm = original
        config.LLM_MAX_RETRIES = orig_retries
        config.LLM_RETRY_BASE_SECONDS = orig_delay


def _dimension(**overrides) -> dict:
    """Minimal dimension spec; overrides patch individual fields."""
    dim = {
        "dimension_id": "test_dimension",
        "name": "Test Dimension",
        "retrieval_queries": ["q"],
        "alternate_queries": [],
        "rubric": {},
        "prompt_template": {"system": "sys", "user_template": "Evaluate {company_name}.\n{retrieved_chunks}"},
        "target_sections": [],
    }
    dim.update(overrides)
    return dim


def _chunk(text: str = "some evidence", score: float = 0.8) -> dict:
    return {"text": text, "score": score, "headings": ["Item 7"], "source_page": 12}


def _analyze(dim, chunks=(), score: float = 0.0):
    return analysis_agent.analyze_dimension(
        dimension=dim,
        company_name="TestCo",
        retrieved_chunks=list(chunks),
        retrieval_score=score,
        retrieval_attempts=1,
    )


# ---------------------------------------------------------------------------
# Test 1: explicit no_evidence_rating drives the fallback
# ---------------------------------------------------------------------------

def test_no_evidence_uses_explicit_rating() -> None:
    dim = _dimension(no_evidence_rating="INCONCLUSIVE")
    verdict = _analyze(dim)
    # INCONCLUSIVE normalizes to MEDIUM via _coerce_rating.
    assert verdict["rating"] == config.RATING_MEDIUM
    assert "no_evidence" in verdict["flags"]


# ---------------------------------------------------------------------------
# Test 2: prose is not scanned for the rating
# ---------------------------------------------------------------------------

def test_no_evidence_ignores_prose() -> None:
    """The old implementation scanned `no_evidence_handling` for the first of
    (HIGH, LOW, MEDIUM, INCONCLUSIVE).  Because configs state intent by naming
    the rating they forbid, that returned the forbidden value.  With no explicit
    field set the fallback must be MEDIUM, never LOW.
    """
    dim = _dimension(
        no_evidence_handling="Flag INCONCLUSIVE - do not default to LOW under any circumstances.",
    )
    verdict = _analyze(dim)
    assert verdict["rating"] == config.RATING_MEDIUM, (
        f"Prose mentioning LOW must not produce a LOW verdict; got {verdict['rating']}"
    )


def test_no_evidence_ignores_incidental_substrings() -> None:
    """'LOW' is a substring of 'FLOW' — guidance mentioning cash flow must not
    be read as a LOW rating.
    """
    dim = _dimension(no_evidence_handling="Check operating cash flow before concluding.")
    assert _analyze(dim)["rating"] == config.RATING_MEDIUM


# ---------------------------------------------------------------------------
# Test 3: no-evidence path must not call the model
# ---------------------------------------------------------------------------

def test_no_evidence_skips_llm() -> None:
    dim = _dimension(no_evidence_rating="INCONCLUSIVE")
    with patched_llm(lambda m: '{"verdict": "HIGH"}') as fake:
        verdict = _analyze(dim)
    assert fake.calls == [], "No-evidence path must not invoke the model"
    assert "no_evidence" in verdict["flags"]


# ---------------------------------------------------------------------------
# Test 4: an explicit non-default fallback is honoured
# ---------------------------------------------------------------------------

def test_no_evidence_honours_high() -> None:
    dim = _dimension(no_evidence_rating="HIGH")
    assert _analyze(dim)["rating"] == config.RATING_HIGH


# ---------------------------------------------------------------------------
# Test 5: regression guard over the shipped dimension configs
# ---------------------------------------------------------------------------

def test_real_dimensions_never_fall_to_low() -> None:
    """D2 (going concern) and D4 previously returned LOW when evidence was
    missing — reporting 'no evidence found' as 'low risk'.  No shipped
    dimension may fall back to LOW.
    """
    for dim in load_dimensions():
        verdict = _analyze(dim)
        assert verdict["rating"] != config.RATING_LOW, (
            f"{dim['dimension_id']} falls back to LOW when no evidence is retrieved"
        )
        assert "no_evidence" in verdict["flags"]
        assert "no model analysis" in verdict["reasoning"].lower(), (
            f"{dim['dimension_id']} reasoning must state that no analysis was performed"
        )


# ---------------------------------------------------------------------------
# Test 6: happy path
# ---------------------------------------------------------------------------

def test_valid_json_response() -> None:
    payload = json.dumps({
        "verdict": "HIGH",
        "reasoning": "Recurring losses and negative operating cash flow.",
        "evidence": [{"source_passage": "net loss of $1.2B", "source_page": 53}],
        "flags": ["negative_operating_cash_flow"],
    })
    dim = _dimension()
    with patched_llm(lambda m: f"```json\n{payload}\n```") as fake:
        verdict = _analyze(dim, chunks=[_chunk()], score=0.72)

    assert len(fake.calls) == 1, "Evidence present: model should be called exactly once"
    assert verdict["rating"] == config.RATING_HIGH
    assert verdict["flags"] == ["negative_operating_cash_flow"]
    assert verdict["evidence_citations"] == [{"text": "net loss of $1.2B", "source_page": 53}]
    # confidence carries the retrieval score, not anything the model reported
    assert verdict["confidence"] == 0.72


# ---------------------------------------------------------------------------
# Test 7 & 8: failures are flagged rather than silently rated MEDIUM
# ---------------------------------------------------------------------------

def test_parse_error_flagged() -> None:
    dim = _dimension()
    with patched_llm(lambda m: "I could not produce JSON for this filing."):
        verdict = _analyze(dim, chunks=[_chunk()])
    assert verdict["rating"] == config.RATING_MEDIUM
    assert "parse_error" in verdict["flags"], "Unparseable output must be flagged"


def test_analysis_error_flagged() -> None:
    dim = _dimension()
    with patched_llm(lambda m: RuntimeError("503 model overloaded")):
        verdict = _analyze(dim, chunks=[_chunk()])
    assert verdict["rating"] == config.RATING_MEDIUM
    assert "analysis_error" in verdict["flags"], "Model failure must be flagged"


# ---------------------------------------------------------------------------
# A model answer that is not a rating is flagged, not silently coerced
# ---------------------------------------------------------------------------

def _answer(verdict: str):
    payload = json.dumps({"verdict": verdict, "reasoning": "r", "evidence": [], "flags": []})
    with patched_llm(lambda m: payload):
        return _analyze(_dimension(), chunks=[_chunk()])


def test_inconclusive_answer_flagged() -> None:
    """The rubrics ask for INCONCLUSIVE when required evidence is missing.
    It still coerces to MEDIUM, so without a flag it would read as a finding.
    """
    from src.graph.build import summarize_flags

    verdict = _answer("INCONCLUSIVE")
    assert verdict["rating"] == config.RATING_MEDIUM
    assert "inconclusive" in verdict["flags"]
    _, degraded = summarize_flags([verdict])
    assert degraded == {"test_dimension": ["inconclusive"]}


def test_unrecognized_answer_flagged() -> None:
    verdict = _answer("N/A")
    assert verdict["rating"] == config.RATING_MEDIUM
    assert "invalid_rating" in verdict["flags"]


def test_rating_synonym_not_flagged() -> None:
    verdict = _answer("moderate")
    assert verdict["rating"] == config.RATING_MEDIUM
    assert verdict["flags"] == []


def test_no_evidence_not_double_flagged() -> None:
    verdict = _analyze(_dimension(no_evidence_rating="INCONCLUSIVE"))
    assert verdict["flags"] == ["no_evidence"]


# ---------------------------------------------------------------------------
# Retry: transient failures must not become verdicts
# ---------------------------------------------------------------------------

def test_retries_then_succeeds() -> None:
    """A transient failure (rate limit) must be retried, not converted into a
    MEDIUM verdict indistinguishable from real analysis.
    """
    calls = {"n": 0}

    def flaky(messages):
        calls["n"] += 1
        if calls["n"] < 3:
            return RuntimeError("429 rate limit exceeded")
        return json.dumps({"verdict": "HIGH", "reasoning": "ok", "evidence": [], "flags": []})

    with patched_llm(flaky, max_retries=3) as fake:
        verdict = _analyze(_dimension(), chunks=[_chunk()])

    assert len(fake.calls) == 3, f"Expected 3 attempts, got {len(fake.calls)}"
    assert verdict["rating"] == config.RATING_HIGH
    assert "analysis_error" not in verdict["flags"], "Recovered call must not be flagged"


def test_gives_up_after_max_retries() -> None:
    """Exhausted retries still fail closed to a flagged verdict, never a silent one."""
    with patched_llm(lambda m: RuntimeError("429 rate limit"), max_retries=2) as fake:
        verdict = _analyze(_dimension(), chunks=[_chunk()])

    assert len(fake.calls) == 2, "Should stop at max_retries"
    assert "analysis_error" in verdict["flags"]


# ---------------------------------------------------------------------------
# Test 9: loader validation of the new field
# ---------------------------------------------------------------------------

def test_loader_rejects_bad_rating() -> None:
    spec = {
        "dimension_id": "x",
        "name": "X",
        "retrieval_queries": ["q"],
        "prompt_template": {"user_template": "t"},
        "target_sections": [],
        "rubric": {},
        "no_evidence_rating": "PROBABLY_FINE",
    }
    dim = document_agent._validate_dimension(spec, Path("x.json"))
    assert "no_evidence_rating" not in dim, "Unrecognized value must be dropped, not passed through"
    # ...and the analysis agent's own default then applies.
    assert _analyze(dim)["rating"] == config.RATING_MEDIUM


def test_loader_normalizes_case() -> None:
    spec = {
        "dimension_id": "x",
        "name": "X",
        "retrieval_queries": ["q"],
        "prompt_template": {"user_template": "t"},
        "target_sections": [],
        "rubric": {},
        "no_evidence_rating": "  high  ",
    }
    dim = document_agent._validate_dimension(spec, Path("x.json"))
    assert dim["no_evidence_rating"] == "HIGH"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> int:
    tests = [
        test_no_evidence_uses_explicit_rating,
        test_no_evidence_ignores_prose,
        test_no_evidence_ignores_incidental_substrings,
        test_no_evidence_skips_llm,
        test_no_evidence_honours_high,
        test_real_dimensions_never_fall_to_low,
        test_valid_json_response,
        test_parse_error_flagged,
        test_analysis_error_flagged,
        test_inconclusive_answer_flagged,
        test_unrecognized_answer_flagged,
        test_rating_synonym_not_flagged,
        test_no_evidence_not_double_flagged,
        test_retries_then_succeeds,
        test_gives_up_after_max_retries,
        test_loader_rejects_bad_rating,
        test_loader_normalizes_case,
    ]
    print("=== Analysis Agent Harness ===")
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✓ {t.__name__}")
        except Exception as e:
            print(f"  ✗ {t.__name__}: {e}")
            failed += 1
    print(f"\n{'All passed' if not failed else f'{failed} failed'}")
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
