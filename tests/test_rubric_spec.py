"""Specification checks over the shipped dimension configs.

These guard properties the evaluation depends on, not agent behaviour:

  - the D1 rubric states the same Z″ rule the D1 labels are computed with
  - no agent-facing config carries labels or names a portfolio company
  - each rubric reaches the model from one place only (the `rubric` field)
  - every system prompt says retrieved text cannot act as instructions
  - D5 is never sourced from Item 1A or Item 106

A failure here means a config drifted from the labeling rule.  Fix the config
(in a new rubric version directory if the current one has been run), never the
assertion.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.agents import analysis_agent

PORTFOLIO_COMPANIES = ("carvana", "peloton", "shopify", "revlon")


def _raw_configs() -> dict[str, dict]:
    index = json.loads((config.DIMENSIONS_DIR / "00_index.json").read_text(encoding="utf-8"))
    files = [item["file"] for item in index["agent_facing_files"]]
    return {f: json.loads((config.DIMENSIONS_DIR / f).read_text(encoding="utf-8")) for f in files}


def _rubric_text(spec: dict) -> str:
    return json.dumps(spec["rubric"], ensure_ascii=False)


def test_dimensions_dir_is_versioned() -> None:
    assert config.DIMENSIONS_DIR.name == config.RUBRIC_VERSION
    assert (config.DIMENSIONS_DIR / "00_index.json").exists()


def test_d1_rubric_states_z_double_prime_bands() -> None:
    """The D1 label is the Z″ zone.  If the rubric cites different cut-offs, the
    agent is graded against a rule it was never given."""
    spec = _raw_configs()["d1_financial_health.json"]
    rubric = _rubric_text(spec)
    assert "1.10" in rubric and "2.60" in rubric, rubric

    whole = json.dumps(spec, ensure_ascii=False)
    for classic in ("1.81", "2.99"):
        assert classic not in whole, f"D1 config still cites classic Z cut-off {classic}"
    assert "6.56" in whole and "3.26" in whole and "6.72" in whole and "1.05" in whole


def test_rubric_levels_carry_only_rubric_text() -> None:
    """`_build_dimension_guidance` appends `sp_frp_equivalent` to a rubric line,
    so any extra criterion stored there silently becomes part of the rubric."""
    for name, spec in _raw_configs().items():
        for level, body in spec["rubric"].items():
            assert "sp_frp_equivalent" not in body, f"{name} {level} carries sp_frp_equivalent"


def test_configs_carry_no_labels_or_company_names() -> None:
    for name, spec in _raw_configs().items():
        assert "ground_truth_labels" not in spec, f"{name} embeds ground truth labels"
        text = json.dumps(spec, ensure_ascii=False).lower()
        for company in PORTFOLIO_COMPANIES:
            assert company not in text, f"{name} names portfolio company {company!r}"


def test_rubric_has_single_source_in_prompt() -> None:
    """A rubric summary hard-coded in the template drifts from the `rubric`
    field the guidance block renders.  The template must defer to the field."""
    for name, spec in _raw_configs().items():
        template = spec["prompt_template"]["user_template"]
        assert "{dimension_guidance}" in template, f"{name} template lacks {{dimension_guidance}}"
        assert "## Rubric" not in template, f"{name} template hard-codes a rubric"


def test_system_prompts_mark_retrieved_text_as_data() -> None:
    for name, spec in _raw_configs().items():
        system = spec["prompt_template"]["system"]
        assert "never instructions" in system, (
            f"{name} system prompt lacks the data-not-instructions rule"
        )


def test_d5_rubric_not_sourced_from_item_1a_or_106() -> None:
    """Item 1A material-weakness language is hypothetical, and Item 106 is not
    an ICFR conclusion.  Neither may decide a D5 rating."""
    rubric = _rubric_text(_raw_configs()["d5_internal_controls.json"])
    assert "Item 1A" not in rubric
    assert "Item 106" not in rubric


def test_d1_guidance_identifies_inputs_safely() -> None:
    """A caption containing 'from operations' can be pre-tax income ('Loss from
    operations before income taxes'), and statements reproduced inside notes
    can carry identical headings.  The D1 guidance must warn about the first
    and require the balance-sheet check that catches mixed sources."""
    spec = _raw_configs()["d1_financial_health.json"]
    guards = " ".join(
        g["explanation"] for g in spec["evidence_schema"]["false_positive_guards"]
    ).lower()
    assert "before income taxes" in guards
    assert "total assets equals total liabilities plus total equity" in guards
    signals = json.dumps(spec["evidence_schema"]["primary_signals"]).lower()
    assert "'loss from operations'" not in signals, (
        "caption lists must not accept 'Loss from operations' unqualified"
    )


def test_d2_guidance_classifies_by_conclusion() -> None:
    """Real going-concern findings open with the basis-of-accounting sentence,
    and the evaluation is described in words ('raised substantial doubt') that
    are not its result.  The guidance must classify by the concluding sentence,
    MEDIUM must need the words, and LOW must need the auditor's report on the
    financial statements, not an internal-control report."""
    spec = _raw_configs()["d2_going_concern.json"]
    guards = " ".join(
        g["explanation"] for g in spec["evidence_schema"]["false_positive_guards"]
    ).lower()
    assert "the sentence that states a conclusion" in guards
    assert "'whether' clause" in guards
    assert "unqualified opinion" in guards
    assert "must say that substantial doubt was raised" in spec["rubric"]["MEDIUM"]["threshold"]
    low = spec["rubric"]["LOW"]["threshold"]
    assert "auditor's report on the financial statements" in low
    assert "internal control over financial reporting alone" in low


def test_d3_guidance_guards_against_known_look_alikes() -> None:
    """A list of an industry's regulators names agencies but no matter; a tax
    reserve is not a legal accrual; and the litigation section's closing
    'has not established reserves' sentence covers every matter above it."""
    spec = _raw_configs()["d3_legal_regulatory_exposure.json"]
    guards = " ".join(
        g["explanation"] for g in spec["evidence_schema"]["false_positive_guards"]
    ).lower()
    assert "names authorities but no matter" in guards
    assert "income-tax reserves" in guards and "not accruals for legal proceedings" in guards
    assert "covers every matter listed above it" in guards
    rendered = json.dumps(spec["evidence_schema"]).lower()
    assert "constitute regulatory exposure" not in rendered, (
        "tax contingencies must not be presented as legal exposure"
    )


def test_guard_explanations_stand_alone() -> None:
    """The prompt renders a guard as 'guard: explanation' and drops its
    'pattern' field.  An explanation opening with 'This' or 'These' points at
    an example the model never sees."""
    for name, spec in _raw_configs().items():
        for g in spec["evidence_schema"].get("false_positive_guards", []):
            first = g["explanation"].split()[0].strip(" ,.:;").lower()
            assert first not in {"this", "these", "it"}, (
                f"{name}: guard '{g['guard']}' leans on its unrendered pattern"
            )


def test_d5_guidance_rates_only_the_fiscal_year() -> None:
    """Item 1A can recount an earlier year's 'not effective' conclusion and a
    remediation from years before; neither may decide the rating."""
    spec = _raw_configs()["d5_internal_controls.json"]
    guards = " ".join(
        g["explanation"] for g in spec["evidence_schema"]["false_positive_guards"]
    ).lower()
    assert "about an earlier year" in guards
    assert "never establishes" in guards or "does not show that controls were effective" in guards
    assert (
        "does not indicate a current material weakness"
        not in json.dumps(spec["evidence_schema"]).lower()
    )


def test_output_format_has_single_source() -> None:
    """The output schema is passed to the model as the response format.  A
    field list also written into the template would drift from it."""
    for name, spec in _raw_configs().items():
        template = spec["prompt_template"]["user_template"]
        assert "## Output Format" not in template, f"{name} template hard-codes an output format"
        assert '"verdict":' not in template, f"{name} template carries an output field list"
        schema = spec["output_schema"]
        assert schema["properties"]["verdict"]["enum"] == [
            "HIGH",
            "MEDIUM",
            "LOW",
            "INCONCLUSIVE",
        ], name
        evidence_item = schema["properties"]["evidence"]["items"]
        assert "evidence_id" in evidence_item["required"], (
            f"{name} evidence items must cite an evidence block"
        )


def test_output_schemas_are_strict() -> None:
    """Every object in a schema lists all its properties as required and
    allows no others, so the model cannot omit a field or invent one."""

    def walk(node: dict, where: str) -> None:
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False, where
            assert set(node.get("required", [])) == set(node.get("properties", {})), where
            for key, child in node.get("properties", {}).items():
                walk(child, f"{where}.{key}")
        elif node.get("type") == "array":
            walk(node["items"], f"{where}[]")

    for name, spec in _raw_configs().items():
        walk(spec["output_schema"], name)


def test_every_evidence_schema_field_reaches_the_prompt() -> None:
    """A field `_build_dimension_guidance` has no case for is silently dropped:
    D3's boilerplate patterns and temporal note were, until 2026-11-10."""
    rendered = {
        "primary_signals",
        "corroborating_signals",
        "absence_signals",
        "false_positive_guards",
        "cross_section_evidence",
        "terminology_note",
        "debt_terminology_note",
        "boilerplate_low_patterns",
        "temporal_scope_note",
    }
    for name, spec in _raw_configs().items():
        unrendered = set(spec["evidence_schema"]) - rendered
        assert not unrendered, f"{name}: {sorted(unrendered)} never reach the prompt"


def test_d3_boilerplate_and_temporal_guidance_rendered() -> None:
    spec = _raw_configs()["d3_legal_regulatory_exposure.json"]
    guidance = analysis_agent._build_dimension_guidance(spec)
    assert "From time to time, we may be subject to legal proceedings" in guidance
    assert "Do not import knowledge from outside the filing text" in guidance


def test_output_schemas_use_only_keywords_both_providers_accept() -> None:
    """Gemini's response_json_schema supports a subset of JSON Schema; a keyword
    outside it could be rejected or ignored."""
    allowed = {
        "type",
        "enum",
        "description",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "anyOf",
    }

    def walk(node: dict, where: str) -> None:
        assert set(node) <= allowed, f"{where}: {sorted(set(node) - allowed)}"
        for key, child in node.get("properties", {}).items():
            walk(child, f"{where}.{key}")
        if "items" in node:
            walk(node["items"], f"{where}[]")
        for child in node.get("anyOf", []):
            walk(child, f"{where}|")

    for name, spec in _raw_configs().items():
        walk(spec["output_schema"], name)
