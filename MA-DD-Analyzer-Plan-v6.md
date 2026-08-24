# M&A DD Analyzer — Implementation Plan v6

**Supersedes v5.** Two changes: **4 dimensions instead of 5**, and a **front-loaded schedule** using the Bangladesh weeks.

Carry over from v4: dimension rubrics, Docling ingestion pipeline, DDState schema, per-company research notes.

**New target: working, evaluated, and deployed by mid-December 2026** — three months earlier than the previous plan.

---

## Dropping D4 (Customer/Supplier Dependency)

Of the five, D4 was the weakest: its label came from reading Item 1A dependency language and making a judgment call, with no external source to check against. D3 (Item 3 litigation self-disclosure) has the same issue but is at least anchored to a mandatory disclosure item with defined materiality thresholds.

**Remaining four:**

| | Dimension | Ground truth source | Objective? |
|---|---|---|---|
| D1 | Financial Health | Altman Z-score computed from XBRL | ✅ |
| D2 | Going Concern | AS 2415 "substantial doubt" in auditor's report, cross-checked vs. PCAOB | ✅ |
| D3 | Legal / Litigation | Item 3 disclosure against published materiality rubric | ⚠️ documented judgment |
| D5 | Internal Controls | Item 9A binary management + auditor conclusions | ✅ |

**This improves your headline claim from 3-of-5 to 3-of-4 objectively grounded.** 75% instead of 60%, and the one judgment-based dimension is the stronger of the two you had.

**Dataset: 3 companies × 4 dimensions = 12 labeled examples.**

### A useful side effect

v4's internal inconsistencies now resolve themselves. The document said "12 dimension-company pairs" in several places, `validate_gt.py` was specified to check for 12 entries, and §5/§7 referred to "all 4 verdict cards." Those weren't errors — they're artifacts of the doc originally being written for four dimensions and only partly updated to five. Reverting to four makes the whole thing self-consistent again.

**Still to fix:** the Shopify fiscal year mismatch (FY2025 in the company table vs. FY2022 in the ground-truth header — pick one, verify on EDGAR), the model versions (replace Gemini 1.5 with current models), and the ground-truth claim wording:

> Three of four dimensions use objective, independently verifiable ground truth: D1 from Altman Z-scores computed on XBRL data, D2 from the presence or absence of AS 2415 "substantial doubt" language in the auditor's report cross-checked against PCAOB records, and D5 from the binary management and auditor conclusions in Item 9A. D3 uses rubric-based labeling against published materiality standards, with sources documented per label.

---

## Revised hour budget

Dropping a dimension saves ~8 hrs, mostly in the Analysis Agent (one fewer rubric to tune) and ground truth.

### REQUIRED — ~85 hrs

| Block | Hrs |
|---|---|
| Ground truth: hand-verify D1/D2/D5, reframe D3, finalize `labels.json` + `validate_gt.py` (12 entries) | 7–9 |
| Retrieval Agent: Pinecone hybrid search, section filter, top-k with scores | 6–8 |
| **Reranker layer** | 3–4 |
| Analysis Agent: 4 rubrics, structured JSON verdicts, confidence threshold, retries | 10–22 |
| StateGraph: DDState, conditional loop across 4 dimensions | 4–5 |
| First end-to-end run + debugging | 7–9 |
| Eval harness: verdict vs. label, per-dimension accuracy | 4–5 |
| MLflow: 2 experiments | 6–8 |
| FastAPI + async streaming | 6–8 |
| Docker + compose | 4–5 |
| Azure Container Apps deploy | 6–8 |
| Minimal CI (build + test) | 4–5 |
| README, architecture diagram, demo recording | 6–8 |

### BONUS — ~9 hrs

| Block | Hrs |
|---|---|
| Cost instrumentation + model routing | 5 |
| Prompt-injection defense | 4 |

---

## Schedule

I'm reading "15–20 hours" as **per week**. Over roughly 4 weeks that's **60–80 hours** — which changes the shape of this project substantially.

### Bangladesh (now → Sept 17): 60–80 hrs

This is enough to finish the **entire core system**, not just the groundwork.

**Priority order — job search items stay ahead of the project:**

1. Book and sit the language test
2. Resume + LinkedIn finished, new-grad applications flowing (time-sensitive — the window is open now)
3. Email ECE grad office + course instructor re: the project course
4. Project, with everything left over

**Project sequence:**

| Order | Block | Hrs |
|---|---|---|
| 1 | Ground truth (do first — makes everything else measurable) | 7–9 |
| 2 | Repo corrections + README ground-truth section | 2 |
| 3 | Retrieval Agent | 6–8 |
| 4 | Reranker layer | 3–4 |
| 5 | Analysis Agent ← the risk item | 10–22 |
| 6 | StateGraph + conditional loop | 4–5 |
| 7 | End-to-end run + debugging | 7–9 |
| 8 | Eval harness | 4–5 |
| 9 | MLflow experiments 1 & 2 | 6–8 |

**Total: 49–72 hrs.** Fits inside 60–80 with room for the Analysis Agent to overrun.

**Sept 17 milestone: a working, evaluated agentic pipeline with logged experiments.** No deployment yet, but the substance is done.

### Fall term (Sept 18 → mid-Dec): ~30 hrs at 2.5–3/week

Only the deployment and presentation layer remains.

- FastAPI + async streaming (6–8)
- Docker + compose (4–5)
- Azure Container Apps (6–8)
- Minimal CI (4–5)
- README, architecture diagram, demo recording (6–8)

**Mid-December: project complete and demoable.** Linkable in applications from January.

Everything else in fall goes to **interview prep and networking** — which was always the real bottleneck.

### Winter (Jan → Apr): optional

If the MEng project course accepts it, pitch it as an **extension** rather than a build: additional companies, the remaining MLflow experiments, an ablation study, a written findings report. That's a stronger pitch than a proposal — you'd be arriving with a working system and a research question.

If the course doesn't accept it, the project is already done and winter belongs entirely to interviews.

Bonuses land here, or in fall if you're ahead. **Cost instrumentation first** — it's the only item in the plan that generates a defensible quantitative claim you can put on your resume from your own measurement.

---

## Why front-loading is the right call

The earlier plan had you finishing in March, which meant carrying the project through two terms alongside coursework, interview prep, and networking. Compressing it into the Bangladesh weeks means:

- **The project stops competing for term-time hours.** Fall goes to interview prep, which is what actually converts your existing experience into offers.
- **A linkable repo exists before winter applications.** The second new-grad wave (Jan–March) and the experienced-hire push (March–June) both get a finished artifact rather than a half-built one.
- **The Analysis Agent risk gets absorbed now.** If rubric adherence takes 22 hours instead of 10, you have the slack. In a Waterloo term you wouldn't.
- **It resolves the regret directly.** You wanted uninterrupted time for this. You have four weeks of it.

**One caution:** don't let the project crowd out the job search. The new-grad window is open now and closes in October, and a language test booking has been outstanding since this planning started. Project hours are the surplus after those, not before.

---

## Positioning

Rename the repo. "M&A Due Diligence Analyzer" reads as a finance tool and buries what's scarce.

> **Agentic RAG with externally-grounded evaluation over SEC filings**

Open the README with the evaluation methodology, not the use case: three of four dimensions labeled from XBRL-computed Altman Z-scores, PCAOB going-concern records, and binary Item 9A conclusions — no human or model annotation in the loop for those. That paragraph is the reason this outranks the hundreds of RAG demos it'll be compared against.

The agent is the implementation. The eval set is the contribution.
