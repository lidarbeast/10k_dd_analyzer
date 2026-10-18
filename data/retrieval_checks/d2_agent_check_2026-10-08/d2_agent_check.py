"""D2 agent check on the held-out filing.  Uses the agent's own prompt
builder, model call and parser.

A and B1 ran on gemini-3.8-flash on 2026-10-08 (d2_check_A.json,
d2_check_B1.json; the content-block extraction was done here then).  Since
2026-10-18 the agent calls Claude through `_call_model`, and results are
written to d2_check_<set>.<model>.json so the Gemini results are kept."""
import json, random, sys, logging
logging.disable(logging.WARNING)
from src import config
from src.agents import analysis_agent as aa
from src.agents import retrieval_agent as ra
from src.agents.document_agent import load_dimensions

SP = sys.argv[2]
spec = next(d for d in load_dimensions() if d["dimension_id"] == "d2_going_concern")
store = json.load(open(f"{SP}/rev_chunks.json"))
CUT = "At December 31, 2022, the Company had a liquidity position"

def chunk(i, cut=False):
    md = store[f"rev-20221231_chunk_{i}"]
    text = md["text"]
    if cut:
        import re
        text = text[: re.search(r"\s+".join(CUT.split()), text).start()].rstrip()
    return {"id": f"rev-20221231_chunk_{i}", "text": text, "headings": md.get("headings"), "score": 0.0}

def controlled(ids, seed=7):
    out = [chunk(i, cut=(i == "344c")) if i != "344c" else chunk(344, cut=True) for i in ids]
    random.Random(seed).shuffle(out)
    return out

SETS = {
    "B1": [279, 281, 344, 345],                 # conclusion only in a running-header chunk
    "B2": [279, 281, 282, 283, "344c", 245],    # traps only; expected LOW
    "B3": ["344c", 245, 284],                   # no FS-report passage; expected INCONCLUSIVE
}

name = sys.argv[1]
if name == "A":
    state = {"current_dim_index": 0, "dimensions": [spec], "company_name": "Revlon, Inc.",
             "filing_chunks": [], "filing_path": "data/filings/rev-20221231.pdf"}
    state = ra.run(state)
    chunks = state["_retrieved_chunks"]
    meta = {"retrieval_score": state["_retrieval_score"], "attempts": state["_retrieval_attempts"]}
else:
    chunks = controlled(SETS[name])
    meta = {}

system, user = aa._build_prompt(spec, "Revlon, Inc.", chunks)
reply = aa._call_model(system, user, spec.get("output_schema"), spec["dimension_id"])
text = reply.text
parsed = aa._parse_verdict(text, spec["dimension_id"])
record = {"set": name, "model": config.LLM_MODEL, "served_model": reply.model, "stop_reason": reply.stop_reason,
          "usage": reply.usage, "chunk_order": [c["id"] + (" (cut)" if c["id"].endswith("344") and name != "A" and "344c" in SETS.get(name, []) else "") for c in chunks],
          "meta": meta, "prompt_chars": len(system) + len(user), "parsed": parsed, "raw": text}
json.dump(record, open(f"{SP}/d2_check_{name}.{config.LLM_MODEL}.json", "w"), ensure_ascii=False, indent=1)
print(name, "verdict:", parsed.get("verdict"), "| flags:", parsed.get("flags"), "| order:", record["chunk_order"])
print("criteria:", parsed.get("rubric_criteria_met"), "| aud_retrieved:", parsed.get("auditor_report_retrieved"),
      "| as2415:", parsed.get("as2415_paragraph_found"), "| mgmt:", parsed.get("management_disclosure"), "| ch11:", parsed.get("chapter_11_debtor_in_possession"))
print("reasoning:", parsed.get("reasoning"))
