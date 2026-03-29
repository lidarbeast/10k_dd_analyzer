# Dimension Config Quality Checklist

**Use this checklist to validate that each of d1-d5 configs applies well to your test portfolio.**

---

## Quick Validation Steps

For **each dimension** (d1-d5), run this assessment:

### Step 1: Verify Config Structure ✅

File: `data/dimensions/{dimension_id}.json`

Required fields:
- [ ] `dimension_id` (e.g., "d1_financial_health")
- [ ] `version` (semantic versioning)
- [ ] `name` (human-readable label)
- [ ] `description` (what it measures)
- [ ] `retrieval_queries` (array of 8-10 primary search queries)
- [ ] `alternate_queries` (array of 3-4 fallback queries for low-confidence retry)
- [ ] `rubric` with `HIGH`, `MEDIUM`, `LOW` thresholds
- [ ] `prompt_template` with `system` and `user_template`
- [ ] `target_sections` (["item_7", "item_8", ...])
- [ ] `ground_truth_labels` (test expectations for your portfolio)

**Pass Criteria:** All fields present and non-empty.

---

### Step 2: Validate Retrieval Queries Against Filings

For each company (Shopify, Carvana, Peloton, Revlon):

```bash
# Count query keyword hits in each filing (example for d1)
python3 << 'EOF'
import subprocess
import re

dimension = "d1"  # Change for d2, d3, etc.
queries = [
    "net loss operating losses revenue decline",  # From d1_financial_health.json
    "total debt long-term debt outstanding principal",
    "cash flow from operations working capital",
    # ... add all retrieval_queries from your dimension config
]

pdfs = [
    "shop-20251231.pdf",
    "cvna-20221231.pdf",
    "pton-20230630.pdf",
    "rev-20221231.pdf"
]

for pdf in pdfs:
    result = subprocess.run(["pdftotext", f"data/filings/{pdf}", "-"], 
                          capture_output=True, text=True, timeout=5)
    text = result.stdout.lower()
    
    print(f"\n{pdf}:")
    total_hits = 0
    for query in queries:
        hits = len(re.findall(query, text, re.IGNORECASE))
        total_hits += hits
        print(f"  {query[:40]:.<40} {hits:>3}")
    print(f"  {'TOTAL':.<40} {total_hits:>3}")
EOF
```

**Pass Criteria:**
- ✅ Each filing has > 10 total hits across all queries
- ✅ No company returns ALL zeros on queries (indicates mismatch)
- ✅ Distressed companies (Revlon) have higher hit counts than healthy (Shopify)

---

### Step 3: Validate Rubric Thresholds Against Expected Verdicts

1. **Load the config:**
   ```bash
   python3 -c "import json; cfg = json.load(open('data/dimensions/d1_financial_health.json')); print('Rubric:', cfg['rubric'])"
   ```

2. **Check ground_truth_labels:**
   ```json
   {
     "company": "Shopify FY2025",
     "label": "LOW"
   },
   {
     "company": "Revlon FY2022",
     "label": "HIGH"
   }
   ```

3. **Verify rubric can distinguish cases:**
   - [ ] Healthy company (Shopify) → LOW threshold met? (e.g., Z > 2.99)
   - [ ] Recovering company (Carvana) → MEDIUM threshold met? (e.g., Z 1.81-2.99)
   - [ ] Challenged company (Peloton) → MEDIUM/HIGH threshold met?
   - [ ] Bankrupt company (Revlon) → HIGH threshold met? (e.g., Z < 1.81)

**Pass Criteria:** Rubric thresholds can reasonably classify all 4 companies into expected buckets.

---

### Step 4: Validate Prompt Template

Check that `prompt_template.user_template` includes:

- [ ] Clear instructions on how to apply the rubric
- [ ] Reference to the retrieved chunks placeholder (`{retrieved_chunks}`)
- [ ] Output schema (JSON format expected from LLM)
- [ ] Scoring criteria tied to the rubric
- [ ] Any dimension-specific guidance (e.g., "flag going concern language for d1")

**Pass Criteria:** Template is clear enough for LLM to understand, and output schema matches what downstream agents expect.

---

### Step 5: Run Phase 2 Validation (Live Agent Test)

Once all configs pass Steps 1-4:

```bash
# Test d1 against all companies
for company in "Shopify" "Carvana" "Peloton" "Revlon"; do
    filing=$(python3 -c "
        companies = {
            'Shopify': 'data/filings/shop-20251231.pdf',
            'Carvana': 'data/filings/cvna-20221231.pdf',
            'Peloton': 'data/filings/pton-20230630.pdf',
            'Revlon': 'data/filings/rev-20221231.pdf'
        }
        print(companies['$company'])
    ")
    
    python -m src.agents.local_runner \
        --company "$company" \
        --filing "$filing" \
        --max-dimensions 1 \
        --out "results/${company}_d1.json"
done
```

**Pass Criteria:**
- ✅ All agents complete without errors
- ✅ Verdicts match expected ground_truth_labels
- ✅ Confidence scores > 0.75 (higher = better signal clarity)

---

## Test Portfolio Quick Reference

### Financial Health Metrics (for D1 validation)

| Company | Expected D1 Verdict | Key Signals | Query Hit Ratio |
|---------|---|---|---|
| **Shopify** | **LOW** | Strong cash, no debt, no distress | ✅ High (financial health keywords) |
| **Carvana** | **MEDIUM-HIGH** | Recovered equity, elevated debt, some GC | ✅ Very high (debt + restructuring keywords) |
| **Peloton** | **MEDIUM** | Eroded equity, moderate debt, covenant stress | ✅ Very high (covenant + restructuring) |
| **Revlon** | **HIGH** | Negative equity, extreme debt, GC language | ✅ Extreme (debt 684x, GC 32x) |

### Query Density Interpretation

- **< 5 total hits:** Config may not apply to this company
- **5-20 hits:** Config applies, but signals weak (healthy company, e.g., Shopify)
- **20-50 hits:** Config applies well, signals clear (recovery, e.g., Carvana)
- **50+ hits:** Config applies strongly, signals abundant (distressed, e.g., Revlon)

---

## Common Config Issues & Fixes

### Issue 1: Queries too generic
**Symptom:** All companies return equally high hit counts  
**Fix:** Refine queries to be more specific (e.g., "net operating loss in operating activities" vs. "loss")

### Issue 2: Queries too narrow
**Symptom:** Only distressed companies return hits, healthy companies zero  
**Fix:** Broaden queries or add alternate queries that surface positives (e.g., "no material debt" or "strong liquidity")

### Issue 3: Rubric thresholds don't distinguish cases
**Symptom:** Shopify and Revlon both classify as MEDIUM  
**Fix:** Adjust threshold boundaries (e.g., Z-Score cutoffs for Altman) or add explicit signal checks (e.g., "going concern flag")

### Issue 4: LLM output format unexpected
**Symptom:** Agent fails on JSON parsing  
**Fix:** Match output schema in prompt_template to what analysis_agent.py expects

---

## Dimension-Specific Guidance

### D1: Financial Health
- **Focus:** Altman Z-Score, debt/equity ratios, cash flow, **going concern language**
- **Quick validation:**
  - [ ] Queries cover: net loss, total debt, cash flow, equity
  - [ ] Rubric includes going concern as explicit signal
  - [ ] Alternate queries surface covenant/restructuring language
  - [ ] Shopify → LOW, Revlon → HIGH

### D2: Going Concern / Liquidity
- **Focus:** Item 7 MD&A Liquidity discussion, 12-month sufficiency, funding sources
- **Quick validation:**
  - [ ] Queries surface: liquidity sources, capital commitments, debt maturities
  - [ ] Target sections: item_7, debt footnotes
  - [ ] Alternate queries: refinancing risk, credit facility availability

### D3: Legal / Regulatory Exposure
- **Focus:** Legal proceedings, regulatory penalties, compliance failures
- **Quick validation:**
  - [ ] Queries surface: litigation, regulatory warnings, settlements
  - [ ] Target sections: item_1a, item_3, legal proceedings
  - [ ] Alternate queries: license suspensions, consent decrees

### D4: Operational / Third-Party Dependency
- **Focus:** Supply chain concentration, key customer/vendor dependence, operational resilience
- **Quick validation:**
  - [ ] Queries surface: customer concentration, supplier reliance, supply chain disruption
  - [ ] Target sections: item_1, item_1a (risk factors)
  - [ ] Alternate queries: pandemic impacts, logistics constraints

### D5: Internal Controls / Governance
- **Focus:** SOX compliance, audit opinions, control deficiencies, management changes
- **Quick validation:**
  - [ ] Queries surface: material weakness, management changes, audit opinions
  - [ ] Target sections: item_9a, audit committee correspondence
  - [ ] Alternate queries: restatements, investigation disclosures

---

## Validation Scorecard

Use this after running Phase 2:

| Check | D1 | D2 | D3 | D4 | D5 | Notes |
|---|---|---|---|---|---|---|
| Config structure valid | ✅ | ⏳ | ⏳ | ⏳ | ⏳ | All required fields present |
| Queries hit all companies | ✅ | ⏳ | ⏳ | ⏳ | ⏳ | No zero-hit companies |
| Rubric distinguishes distress | ✅ | ⏳ | ⏳ | ⏳ | ⏳ | Shopify ≠ Revlon |
| Ground truth labels present | ✅ | ⏳ | ⏳ | ⏳ | ⏳ | Test expectations defined |
| Agents run without error | ✅ | ⏳ | ⏳ | ⏳ | ⏳ | No parsing/type errors |
| Verdicts match expectations | ✅ | ⏳ | ⏳ | ⏳ | ⏳ | All 4 companies correct |

**Legend:**
- ✅ = Passed, ready for production
- ⏳ = In progress, needs validation
- ❌ = Failed, requires config adjustment

---

## Next Steps

1. **For D1:** ✅ Ready to test (enhanced 3/29, assessment complete)
2. **For D2-D5:** 
   - Run Steps 1-4 validation on each config
   - Identify any gaps in retrieval queries or rubric thresholds
   - Adjust configs based on test results
   - Repeat Phase 2 validation until all pass

3. **After Phase 2 Success:**
   - Commit final configs and assessment results
   - Deploy agents to production
   - Monitor verdict accuracy against ground truth labels

