"""
Document Agent for the M&A Due Diligence Analyzer.

Responsibilities:
  1. Initialize DDState for an analysis run
  2. Load dimension specs from data/dimensions in configured order
  3. Validate minimum schema for each dimension config
  4. Confirm the filing is indexed: its Pinecone namespace (the filing stem)
     must hold vectors, so a missing index fails here instead of producing
     one zero-results verdict per dimension
"""

from __future__ import annotations
 
import json
import logging
from pathlib import Path
from typing import Any

from src import config
from src.indexing.pinecone_ops import namespace_vector_count
from src.state import DDState, DimensionSpec

_log = logging.getLogger(__name__)


_REQUIRED_DIMENSION_KEYS = {
	"dimension_id",
	"name",
	"retrieval_queries",
	"prompt_template",
	"target_sections",
	"rubric",
}

# Accepted values for the optional `no_evidence_rating` field.  INCONCLUSIVE is
# normalized to MEDIUM downstream by the analysis agent's rating coercion.
_VALID_NO_EVIDENCE_RATINGS = {"HIGH", "MEDIUM", "LOW", "INCONCLUSIVE"}


def _read_json(path: Path) -> dict[str, Any]:
	with path.open("r", encoding="utf-8") as f:
		data = json.load(f)
	if not isinstance(data, dict):
		raise ValueError(f"Expected JSON object in {path}, got {type(data).__name__}")
	return data


def _extract_dimension_files(index_data: dict[str, Any]) -> list[str]:
	"""Extract ordered dimension file list from 00_index.json (agent_facing_files[].file)."""
	agent_files = index_data.get("agent_facing_files")
	if not isinstance(agent_files, list):
		return []
	return [
		item["file"]
		for item in agent_files
		if isinstance(item, dict) and isinstance(item.get("file"), str)
	]


def _validate_dimension(spec: dict[str, Any], source_path: Path) -> DimensionSpec:
	missing = sorted(_REQUIRED_DIMENSION_KEYS.difference(spec.keys()))
	if missing:
		raise ValueError(f"{source_path.name} missing required keys: {missing}")

	retrieval_queries = spec.get("retrieval_queries")
	if not isinstance(retrieval_queries, list) or not retrieval_queries:
		raise ValueError(f"{source_path.name} must define non-empty retrieval_queries")

	prompt_template = spec.get("prompt_template", {})
	if not isinstance(prompt_template, dict):
		raise ValueError(f"{source_path.name} prompt_template must be an object")
	if "user_template" not in prompt_template:
		raise ValueError(f"{source_path.name} prompt_template.user_template is required")

	target_sections = spec.get("target_sections", [])
	if not isinstance(target_sections, list):
		raise ValueError(f"{source_path.name} target_sections must be a list")

	alternate_queries = spec.get("alternate_queries", [])
	if not isinstance(alternate_queries, list):
		_log.warning(
			"%s: alternate_queries is not a list; retry will be disabled for this dimension",
			source_path.name,
		)
		alternate_queries = []
	elif not alternate_queries:
		_log.warning(
			"%s: alternate_queries is empty; confidence-based retry will be disabled",
			source_path.name,
		)

	dim: DimensionSpec = {
		"dimension_id": str(spec["dimension_id"]),
		"name": str(spec["name"]),
		"retrieval_queries": [str(q) for q in retrieval_queries],
		"alternate_queries": [str(q) for q in alternate_queries],
		"rubric": spec.get("rubric", {}),
		"prompt_template": prompt_template,
		"target_sections": [str(s) for s in target_sections],
	}

	mandatory_items = spec.get("mandatory_10k_items")
	if isinstance(mandatory_items, list):
		dim["mandatory_10k_items"] = mandatory_items

	evidence_schema = spec.get("evidence_schema")
	if isinstance(evidence_schema, dict):
		dim["evidence_schema"] = evidence_schema

	no_evidence_handling = spec.get("no_evidence_handling")
	if isinstance(no_evidence_handling, str) and no_evidence_handling.strip():
		dim["no_evidence_handling"] = no_evidence_handling.strip()

	# Explicit fallback rating for the no-evidence path.  Kept as its own field
	# rather than inferred from `no_evidence_handling` prose, which states intent
	# by naming the rating it forbids ("do not default to LOW").
	no_evidence_rating = spec.get("no_evidence_rating")
	if isinstance(no_evidence_rating, str) and no_evidence_rating.strip():
		normalized = no_evidence_rating.strip().upper()
		if normalized in _VALID_NO_EVIDENCE_RATINGS:
			dim["no_evidence_rating"] = normalized
		else:
			_log.warning(
				"%s: unrecognized no_evidence_rating %r; expected one of %s. "
				"Falling back to MEDIUM at analysis time.",
				source_path.name,
				no_evidence_rating,
				sorted(_VALID_NO_EVIDENCE_RATINGS),
			)

	rubric_sources = spec.get("rubric_sources")
	if isinstance(rubric_sources, list):
		dim["rubric_sources"] = [str(s) for s in rubric_sources if isinstance(s, str)]

	confidence_floor = spec.get("confidence_floor")
	if isinstance(confidence_floor, (int, float)):
		dim["confidence_floor"] = float(confidence_floor)

	# JSON schema the Analysis Agent passes to the model as the output format.
	# Its verdict enum must allow INCONCLUSIVE, which the rubrics ask for when
	# the required evidence was not retrieved.
	output_schema = spec.get("output_schema")
	if output_schema is not None:
		verdict_enum = (
			output_schema.get("properties", {}).get("verdict", {}).get("enum")
			if isinstance(output_schema, dict) else None
		)
		if not isinstance(verdict_enum, list) or "INCONCLUSIVE" not in verdict_enum:
			raise ValueError(f"{source_path.name} output_schema must define a verdict enum including INCONCLUSIVE")
		dim["output_schema"] = output_schema

	return dim


def load_dimensions(dimensions_dir: Path | None = None) -> list[DimensionSpec]:
	"""Load dimension specs in canonical order from 00_index.json."""
	if dimensions_dir is None:
		dimensions_dir = config.DIMENSIONS_DIR

	index_path = dimensions_dir / "00_index.json"
	if not index_path.exists():
		raise FileNotFoundError(f"Dimension index not found: {index_path}")

	index_data = _read_json(index_path)
	dimension_files = _extract_dimension_files(index_data)
	if not dimension_files:
		raise ValueError(
			"No dimension files found in 00_index.json. Expected agent_facing_files[].file"
		)

	loaded: list[DimensionSpec] = []
	for filename in dimension_files:
		path = dimensions_dir / filename
		if not path.exists():
			raise FileNotFoundError(f"Dimension file listed in index but missing: {path}")
		spec = _read_json(path)
		loaded.append(_validate_dimension(spec, path))

	return loaded


def _check_index(namespace: str) -> int:
	"""Return the namespace's vector count, failing if the filing is not indexed.

	A count that differs from the filing's processed-chunk file is logged, not
	fatal: retrieval still works, but chunk IDs no longer map to that file.
	"""
	count = namespace_vector_count(namespace)
	if count == 0:
		raise RuntimeError(
			f"Pinecone namespace '{namespace}' is empty or missing. Index the filing first: "
			f"python -m src.indexing.run_pipeline data/filings/{namespace}.pdf"
		)
	processed = config.PARSED_DIR / f"{namespace}_processed.json"
	if processed.exists():
		with processed.open("r", encoding="utf-8") as f:
			expected = len(json.load(f))
		if expected != count:
			_log.warning(
				"Namespace '%s' holds %s vectors but %s lists %s chunks; re-index the filing",
				namespace, count, processed.name, expected,
			)
	else:
		_log.warning("No processed-chunk file for '%s'; chunk IDs cannot be checked", namespace)
	return count


def run(state: DDState) -> DDState:
	"""
	Document Agent node function.

	Initializes dimension loop state and resets accumulated outputs.
	"""
	company = state.get("company_name", "")
	filing_path = state.get("filing_path", "")
	if not company:
		raise ValueError("DDState.company_name is required")
	if not filing_path:
		raise ValueError("DDState.filing_path is required")

	dimensions = load_dimensions()
	namespace = Path(filing_path).stem
	indexed_chunk_count = _check_index(namespace)

	state["namespace"] = namespace
	state["indexed_chunk_count"] = indexed_chunk_count
	state["dimensions"] = dimensions
	state["current_dim_index"] = 0
	state["verdicts"] = []
	state["retrieval_attempts"] = []
	state["confidence_scores"] = []
	state["filing_chunks"] = []
	state["report"] = {}

	# Scratch fields consumed by downstream nodes.
	state["_retrieved_chunks"] = []
	state["_retrieval_score"] = 0.0
	state["_retrieval_attempts"] = 1
	state["_warnings"] = []

	_log.info(
		"Document Agent initialized run for %s with %s dimensions (namespace '%s', %s vectors)",
		company,
		len(dimensions),
		namespace,
		indexed_chunk_count,
	)
	return state
