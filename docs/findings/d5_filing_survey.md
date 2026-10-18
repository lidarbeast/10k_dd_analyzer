# D5 filing survey: how internal control over financial reporting can be evaluated from 10-K filings

**Date:** 2026-10-08

**Purpose:** Inform how the D5 guidance should be written so that it works across filings, not just the ones in the portfolio.
- This is a survey of the documents. It is not a set of labels and not a change to the guidance.
- There are no D5 worksheets yet.

**Filings surveyed**

| Filing | Role | Fiscal year-end | Filer status | Auditor's ICFR report |
|---|---|---|---|---|
| `cvna-20221231` Carvana | portfolio | Dec 31, 2022 | large accelerated | separate report, Item 8 (p. 82), Grant Thornton |
| `pton-20230630` Peloton | portfolio | Jun 30, 2023 | large accelerated | separate report, Item 8 (p. 124), Ernst & Young |
| `shop-20251231` Shopify | portfolio | Dec 31, 2025 | large accelerated | **combined** with the financial-statement report, in **Item 15** (p. 81), PwC |
| `rev-20221231` Revlon | **held-out**; two registrants | Dec 31, 2022 | **non-accelerated** | **none**: no attestation required |

**Method**

1. **Read the PDFs.**
   - Read Item 9A of each filing in full, from its heading to Item 9B.
   - Read each auditor's ICFR opinion.
   - Searched every page for "material weakness", the two conclusion sentences ("was / was not effective as of", "maintained / has not maintained … effective internal control"), cross-references between the two auditor's opinions, and attestation wording.
2. **Look at what the agent would see.** Ran the project's parser on Carvana and Peloton (outputs now kept in `data/parsed/`) and used the stored Shopify output. Revlon's chunks were read from Pinecone.
3. **Retrieval.** Revlon only, with the current `v1` queries and no model call. The record is `data/retrieval_checks/d5_2026-10-08_rev-20221231.md`.

---

## 1. Per-filing findings

### 1.1 The two conclusions, and remediation

| | Carvana | Peloton | Shopify | Revlon (held-out) |
|---|---|---|---|---|
| Management's ICFR conclusion (Item 9A) | "management has **concluded** that its internal control over financial reporting **was effective** as of December 31, 2022" (p. 132) | "our management concluded that our internal control over financial reporting **was not effective** as of June 30, 2023 because of the material weaknesses described below" (p. 182) | management "have **determined** that the Company's internal control over financial reporting **was effective** as of December 31, 2025" (p. 77) | management "**determined** that the Company's internal control over financial reporting **was effective** as of December 31, 2022" (p. 118) |
| Auditor's ICFR opinion | "the Company **maintained**, in all material respects, effective internal control…" | "because of the effect of the material weaknesses described below…, Peloton … **has not maintained** effective internal control…" (p. 124) | "the Company **maintained**, in all material respects, effective internal control…" (p. 81, in the same paragraph as the financial-statement opinion) | **none.** The financial-statement auditor says: "The Company is not required to have, nor were we engaged to perform, an audit of its internal control over financial reporting" (pp. 135, 138). The cover-page 404(b) box reads "No ☒" for both registrants (p. 2). Item 9A says nothing about attestation |
| Disclosure-controls conclusion (Item 307) | effective | **not effective**, "solely as a result of these material weaknesses" | effective | effective |
| Material weaknesses at year-end | none | two: inventory (continuing since FY2021, "not yet fully remediated") and Precor IT access controls (new) | none | none |
| Remediation completed during the year | none | **yes**: the FY2022 impairment-controls weakness, "as of June 30, 2023, successfully remediated" (p. 184). It sits under a HIGH | none | none in FY2022. **Item 1A** (pp. 26, 54) recounts a FY2018 weakness remediated in **2019** |
| Changes in ICFR | none in Q4 | "other than the completed and ongoing remediation efforts" | none | none in Q4 |

Management and the auditor agree in every filing that has both. AS 2201 has the auditor evaluate management's assessment, so disagreement is rare, and this sample cannot test it.

### 1.2 Where "material weakness" appears, and what each kind of sentence means

"Material weakness" appears on 2 pages in Carvana, 8 in Peloton, 1 in Shopify and 3 in Revlon. Only the first kind below decides a rating; the second kind matters only for MEDIUM.

| Kind | Example | Where |
|---|---|---|
| ① **Year-end conclusion** (management or auditor) | "was not effective as of June 30, 2023"; "has not maintained effective internal control" | Peloton 9A, Peloton auditor |
| ② **Remediation completed during the rated year** | "as of June 30, 2023, successfully remediated this material weakness" | Peloton 9A (under a HIGH) |
| ③ Remediation **in progress** | "additional time is required to complete the material weakness remediation work … we have not yet fully remediated" | Peloton 9A |
| ④ Conclusion or remediation from an **earlier year** | "concluded that its internal control over financial reporting was not effective for the fiscal year ended December 31, 2018 … During 2019 … effectively remediated" | Revlon **Item 1A** (p. 54) |
| ⑤ Undated remediation | "The Company previously identified a material weakness … which has now been remediated." | Revlon Item 1A summary (p. 26) |
| ⑥ Item 1A **hypothetical** | "If we fail to remediate material weaknesses…" | Carvana p. 53, Revlon p. 54 |
| ⑦ Item 1A describing the **actual** weaknesses | "In the course of preparing our financial statements for fiscal 2023, we identified a material weakness … related to Precor ITGCs" | Peloton pp. 45–46 |
| ⑧ **Definition** sentence | "A material weakness is a deficiency, or a combination of deficiencies…" | Peloton 9A, Peloton auditor, Revlon 9A |
| ⑨ Auditor's **method** wording | "assessing the risk that a material weakness exists" | every ICFR audit report (Carvana p. 82, Shopify p. 81) |
| ⑩ **Cross-reference to the other opinion** | Financial-statement report: "…our report dated August 23, 2023 expressed an **adverse** opinion thereon" (about internal control). ICFR report: "…does not affect our report dated August 23, 2023, which expressed an **unqualified** opinion thereon" (about the financial statements) | Peloton pp. 122, 124; Carvana pp. 80, 82 |
| ⑪ Financial statements "present fairly" inside Item 9A | "management concluded that the consolidated financial statements … present fairly, in all material respects…" | Peloton 9A (p. 184). This is about the financial statements, not ICFR |

### 1.3 What the parser produces

- **Item 9A is split at its sub-headings.** Management's conclusion arrives in a chunk headed "Management's (Annual) Report on Internal Control over Financial Reporting" (Carvana 421, Peloton 349, Shopify 220). In Revlon, whose sub-headings are lettered (a)–(c) inline, it is in one chunk headed "Item 9A". Every conclusion chunk is self-identifying: it states the conclusion, the date and the COSO framework.
- **Peloton's Item 9A spans seven chunks over four pages.** The year-end "not effective" conclusion (chunk 349) and the completed-remediation passage (chunk 352, headed "Remediation of Previously Disclosed Material Weakness") are **separate chunks**.
  - If only chunk 352 were retrieved, a model would see "successfully remediated" with no year-end conclusion. That reads like MEDIUM, though the filing is HIGH.
- **The auditor's ICFR report is split like the financial-statement report:** title, opinion, basis, definition and limitations, signature.
  - The opinion chunk is self-contained (Carvana 232, Peloton 233). Peloton's ends with the cross-reference "…which expressed an unqualified opinion thereon" (kind ⑩), which is about the financial statements.
  - Shopify's combined report states both opinions in one chunk.
- **Item 9A carries a pointer that has only a report title.** Peloton chunk 355 is headed "Report of Independent Registered Public Accounting Firm", but only points to Item 8. Like the bare title chunks in D2, it carries no opinion.
- **Item 1A chunks keep their risk-factor heading.** Revlon's chunk 106 is headed by an unrelated risk factor ("The Company's success depends, in part, on the quality, efficacy and safety of its products."), not by anything about controls.

### 1.4 Retrieval on Revlon (current `v1` queries)

The full record is in `data/retrieval_checks/d5_2026-10-08_rev-20221231.md`. Confidence was 0.784, so only the main queries ran.

| Needed | Retrieved |
|---|---|
| Management's conclusion (Item 9A, chunk 242) | ✅ rank 1 |
| The auditor's "not required … audit of its internal control" statement (Basis for Opinion, chunks 281 and 288) | ✅ ranks 4–5, via the ICFR-report query |
| Trap: the Item 1A 2018 "not effective" conclusion and 2019 remediation (chunk 106) | retrieved, rank 3 |

**Retrieval is adequate here. The risk is interpretation:** the trap passage is retrieved, and only its dates show it is not about FY2022.

---

## 2. Cross-filing patterns

**P1. The two conclusions are formulaic and self-identifying.**
- Management's conclusion: "internal control over financial reporting **was [not] effective as of** [year-end]". The verb varies ("concluded", "determined").
- The auditor's opinion: "**maintained** / **has not maintained** … effective internal control over financial reporting as of [year-end]".
- Both survive chunking intact in all four filings. The negation and the date are what matter.

**P2. "Material weakness" appears far more often outside the conclusions than in them** (kinds ③–⑨).
- Item 1A can be hypothetical (Carvana, Revlon) or can describe the real weaknesses (Peloton). The current guard says Item 1A "does NOT indicate a current material weakness", which is wrong for Peloton.
- The correct rule is that Item 1A never *decides* the rating, either way.

**P3. Only conclusions about the rated fiscal year count.**
- Revlon's Item 1A contains a "not effective" conclusion (FY2018) and a completed remediation (2019), both years earlier. Without their dates, they would be read as HIGH or MEDIUM. The correct rating is LOW.
- Date-checking applies to both levels:
  - **HIGH** needs a "not effective" conclusion *as of the rated year-end*.
  - **MEDIUM** needs a remediation *completed during the rated year*.

**P4. Remediation takes three forms, and only one counts.**
- **Counts:** remediation completed during the year, which is MEDIUM, and only if the year-end conclusions are effective.
- **Doesn't count:** remediation in progress (Peloton's inventory weakness: "not yet fully remediated"), or remediation completed in an earlier year (Revlon).
- **HIGH outranks a completed remediation** when another weakness remains, as in Peloton.

**P5. Each auditor's report names the other opinion's outcome.**
- An adverse ICFR report contains "unqualified" (about the financial statements). A financial-statement report contains "adverse" (about internal control).
- Read correctly, the financial-statement report's sentence is the auditor's own statement of the ICFR outcome. Read carelessly, either sentence flips the classification.

**P6. Where the auditor's ICFR opinion sits, if anywhere, varies.**
- It can be a separate report in Item 8 (2 filings), combined with the financial-statement report in Item 15 (1 filing), or absent (1 filing).
- When absent, the evidence is the financial-statement auditor's Basis-for-Opinion sentence and the cover-page 404(b) box, not Item 9A, which is what the current guidance says to expect.

**P7. Disclosure controls follow ICFR in this sample.** Peloton's disclosure-controls conclusion is "not effective" expressly because of the ICFR weaknesses, which the existing guard already handles. No filing has an Item 307 failure without an ICFR weakness.

**P8. What this sample cannot test:**
- **MEDIUM:** no filing is effective at year-end with a remediation completed in the year. Peloton's completed remediation sits under a HIGH, and Revlon's is from 2019.
- **Disagreement** between management and auditor.
- **An auditor-only finding:** a filing where only the auditor's opinion is negative.
- **A weakness found after year-end** but before filing.

---

## 3. Implications for the D5 guidance

**Status (2026-10-08):**
- **Recommendations 1–9 are applied** to `data/dimensions/v1/d5_internal_controls.json`, with open questions 1–2 settled as recommended. They are listed in that file's `rubric_clarifications`.
- **The rubric thresholds are unchanged.**
- **Rendered guidance:** 6,088 → 6,003 characters. The new date, remediation and cross-reference rules replaced repetition, so the length barely moved.
- **Recommendation 10:** no retrieval change.
- **New spec test:** `test_guard_explanations_stand_alone` fails if any dimension's guard explanation opens with "This"/"These"/"It", that is, leans on its unrendered `pattern`. It would have caught all three old D5 guards.
- Section 4 describes the guidance **as it stood before** these changes.

1. **Rewrite the Item 1A guard.** Item 1A never decides the rating, whether it describes hypothetical or real weaknesses. The rating comes from the year-end conclusions, plus a remediation statement for MEDIUM.
   - This also fixes the "This is…" explanations, which point at examples the model never sees, because the prompt does not render a guard's `pattern`. Three D5 guards have this gap.
2. **Add a date rule, the most important change.** Only a conclusion as of the rated fiscal year-end, and a remediation completed during the rated fiscal year, count. A conclusion or remediation from an earlier year, or an undated "has now been remediated", does not.
3. **Separate the three remediation forms.**
   - Completed during the year: MEDIUM, if the year-end conclusions are effective.
   - In progress: does not count.
   - Completed earlier: does not count.
   - A completed remediation never shows effectiveness at year-end. MEDIUM needs the effective conclusion itself; with only a remediation passage, the no-evidence rule (INCONCLUSIVE) applies.
4. **Add a guard for cross-references between the opinions.** Inside an ICFR report, "unqualified opinion" refers to the financial statements; inside a financial-statement report, "adverse opinion thereon" refers to internal control.
   - The financial-statement report's sentence ("…audited … the Company's internal control over financial reporting … and our report dated X expressed an adverse opinion") is the auditor's own statement of the ICFR outcome. I recommend accepting it as evidence of the auditor's opinion (open question 2).
5. **Say where "no attestation" evidence is.**
   - The financial-statement auditor's sentence "not required to have, nor were we engaged to perform, an audit of its internal control over financial reporting", or the cover-page 404(b) box.
   - Drop "the filing usually says so in Item 9A". Instruction 2's `not_required` option should point to the same evidence.
6. **Say where the auditor's ICFR report can be:** Item 8, Item 15, or combined with the financial-statement report. It is not always "usually in Item 8".
7. **Add the definition sentence (⑧) and the "present fairly" sentence (⑪) as look-alikes**, alongside the existing method-wording guard (⑨). Neither is a conclusion.
8. **Note the chunking risk.** A year-end conclusion and a remediation passage can arrive as separate passages. Check the conclusion before reading the remediation.
9. **Trim as for D2 and D3,** so each rule has one home. The current D5 guidance repeats the unqualified/adverse distinction in the system prompt, a primary signal and a guard.
10. **Retrieval: no change needed.** On the held-out filing the conclusion is at rank 1, and the ICFR-report query finds the "not required" sentence.

## 4. How the current D5 guidance measures up

| Current guidance item | Assessment |
|---|---|
| Rubric (HIGH / MEDIUM / LOW), sources | Correct, and decidable from formal conclusions. **No date requirement is stated**, though MEDIUM says "during the fiscal year" (recommendation 2). |
| Guard: Item 1A forward-looking risk factor | **Wrong as written.** It says Item 1A "does NOT indicate a current material weakness", but Peloton's Item 1A describes its actual weaknesses. Its explanation also refers to an unrendered pattern (recommendation 1). |
| Guard: unqualified vs adverse | Correct. Explanation starts "This is…" (pattern not rendered). |
| Guard: auditor method wording | Correct. Explanation starts "This is…" (pattern not rendered). |
| Guard: disclosure controls vs ICFR | Correct, and Peloton exercises it. |
| Guard: significant deficiencies | Correct. |
| Remediation signal | Correct, but does not separate completed, in-progress and earlier-year remediation (recommendation 3). |
| Cross-section: "usually in Item 8" | Incomplete: Shopify's report is in Item 15 (recommendation 6). |
| Cross-section: no-attestation filers "usually say so in Item 9A" | **Not true for the held-out filing**, whose evidence is the auditor's Basis for Opinion and the cover page (recommendation 5). |
| No guard on cross-references between the opinions, the definition sentence, or "present fairly" in Item 9A | **Missing** (recommendations 4, 7). |
| `no_evidence_handling` | Correct. |
| Name `d5_internal_controls_governance` | Overstates the scope: nothing about governance is rated (open question 1). |

## 5. Open questions for the owner

**Status (2026-10-08):** questions 1–2 were settled as recommended (renamed to `d5_internal_controls`; the cross-reference sentence is accepted). Question 3 is on hold.

1. **Rename `d5_internal_controls_governance` → `d5_internal_controls`?**
   - Same reasoning as D2: nothing outside ICFR is rated, and `v1` has not been run.
   - I recommend yes.
2. **Accept the auditor's ICFR outcome from the financial-statement report's cross-reference sentence?**
   - That sentence is the auditor's own statement, and it is self-identifying.
   - I recommend yes, but only for the sentence about internal control in the financial-statement report, not "unqualified" inside an ICFR report.
3. **MEDIUM is untestable on this portfolio,** and there is no held-out MEDIUM case either. Revlon tests only the MEDIUM look-alike. Adding a held-out filing with a completed in-year remediation would need EDGAR search. I suggest holding this, as with D2.

**Note:** D5 labels come straight from formal conclusions, so this dimension needs no rubric clarification on any portfolio filing. Every issue above is about how the *agent* reads the passages.
