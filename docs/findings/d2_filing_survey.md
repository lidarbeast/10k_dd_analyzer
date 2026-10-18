# D2 filing survey: how going concern can actually be evaluated from 10-K filings

**Date:** 2026-10-01

**Purpose:** Inform how the D2 guidance should be written so that it works across filings, not just the ones in the portfolio.
- This is a survey of the documents. It is not a set of labels and not a change to the guidance.
- The owner's D2 worksheets were **not** read for this survey.

**Filings surveyed**

| Filing | Role | Fiscal year-end | Auditor | Auditor's report dated |
|---|---|---|---|---|
| `cvna-20221231` Carvana | portfolio | Dec 31, 2022 | Grant Thornton | Feb 23, 2023 |
| `pton-20230630` Peloton | portfolio | Jun 30, 2023 | Ernst & Young | Aug 23, 2023 |
| `shop-20251231` Shopify | portfolio | Dec 31, 2025 | PwC (Canada) | Feb 11, 2026 |
| `rev-20221231` Revlon | **held-out**; in Chapter 11; two registrants in one 10-K | Dec 31, 2022 | KPMG | Mar 16, 2023 |

**Method**

1. **Read the PDFs.** Searched every page of each PDF (`pypdfium2`) for:
   - going-concern wording: "going concern", "substantial doubt", "ability to continue";
   - Chapter 11 and bankruptcy wording;
   - liquidity sufficiency statements.

   Then read each auditor's report in full. The reports were located with the worksheet extractor (`find_auditor_reports`).
2. **Look at what the agent would see.** Ran the project's parser locally on Carvana, Peloton and Revlon, and used the stored Shopify output (`data/parsed/shop-20251231_processed.json`). Then inspected the chunks. Nothing was uploaded.
3. **Retrieval.** Revlon only, with the current `v1` queries, and no model call. The record is `data/retrieval_checks/d2_2026-10-01_rev-20221231.md`.
4. **EDGAR.** Not used.

---

## 1. Per-filing findings

### 1.1 The auditor's report on the financial statements

| | Carvana | Peloton | Shopify | Revlon (held-out) |
|---|---|---|---|---|
| Form | separate reports: financial statements (FS), and internal control (ICFR) | separate FS and ICFR reports | **one combined report** covering both opinions | one FS report per registrant (2); no ICFR attestation (non-accelerated filer) |
| FS report pages | 80–81 | 122–123 | 81–84 | 135–137 and 138–140 |
| Paragraph headings | Opinion · Basis · Critical audit matters | Opinion · Basis · Critical audit matter | Opinions · Basis · Definition and limitations of ICFR · Critical audit matters | Opinion · **Going Concern** · Basis · Critical audit matter |
| Going-concern paragraph | none | none | none | **yes, in both reports**, directly after the opinion |
| Emphasis-of-matter paragraph | none | none | none | none |
| Critical audit matters | finance-receivable derecognition; acquisition fair value | Tread+ returns reserve | revenue (principal vs agent) | goodwill (Fragrances unit) |
| What the FS report says about the *other* opinion | ICFR opinion "unqualified" | ICFR opinion "**adverse** opinion thereon" | — (combined) | — |

None of the critical audit matters concerns liquidity or going concern.

Revlon's paragraph is identical in both reports:

> **Going Concern** — The accompanying consolidated financial statements have been prepared assuming that the Company will continue as a going concern. As discussed in Note 1 to the consolidated financial statements, the Company has filed a Joint Plan of Reorganization of Revlon, Inc. and Its Debtor Affiliates Pursuant to Chapter 11 of the Bankruptcy Code, which raises substantial doubt about its ability to continue as a going concern. Management's plans in regard to these matters are also described in Note 1. The consolidated financial statements do not include any adjustments that might result from the outcome of this uncertainty.

Revlon's opinion itself is **unqualified** ("present fairly, in all material respects"). The going-concern paragraph sits beside an unqualified opinion; it does not replace it.

### 1.2 Going-concern wording anywhere in the filing

| | Carvana | Peloton | Shopify | Revlon |
|---|---|---|---|---|
| "going concern" | **once**: the basis-of-accounting sentence in Note 2 "Liquidity" (p. 88) | **never** | **never** | throughout |
| "substantial doubt" | never | never | never | 8 sentences (below) |
| Management's going-concern disclosure | Note 2 "Liquidity" (pp. 88–89): basis sentence, losses, financing raised, then a sufficiency conclusion. **No "substantial doubt".** See 1.3 | none | none | Note 1 "Liquidity and Ability to Continue as a Going Concern" (pp. 160–162) states substantial doubt **exists**. Repeated word for word twice in MD&A (pp. 83, 101–102) |
| Liquidity sufficiency statement | MD&A p. 72 ("at least the next 12 months"); Note 2 p. 89 ("at least one year from the financial statement issuance date") | MD&A p. 107 ("at least the next 12 months and beyond") | none found | — |
| Chapter 11 | none. "Bankruptcy-remote subsidiaries" (a financing term); a supplier's bankruptcy as a risk | none. Hypothetical bankruptcy in debt risk factors; Lehman Brothers as a counterparty example | none | petition June 15, 2022; operating as debtors-in-possession; every statement and notes page headed "(DEBTOR-IN-POSSESSION)"; confirmation hearing scheduled April 3, 2023, after the auditor's report and signatures (March 16, 2023) |

**Revlon uses "substantial doubt" and "going concern" in eight kinds of sentence. Only the first two are findings.**

| Kind | Example (Revlon) | Where |
|---|---|---|
| ① Auditor's conclusion | "…which raises substantial doubt about its ability to continue as a going concern." | pp. 135, 138 |
| ② Management's conclusion (ASC 205-40-50-13) | "…substantial doubt exists regarding our ability to continue as a going concern within one year after the date that the financial statements are issued." | pp. 83, 102, 162 |
| ③ Description of the evaluation | "we conducted an evaluation as to whether there were conditions and events, considered in the aggregate, that **raised substantial doubt**…" | pp. 83, 101–102, 160 |
| ④ Horizon statement | "Each reporting period, the Company assesses its ability to continue as a going concern for one year from the date the financial statements are issued." | same |
| ⑤ Dependency statement | "the Company's ability to continue as a going concern is contingent upon… (i) implement the Plan…" | pp. 83, 102, 162 |
| ⑥ Basis-of-accounting sentence | "…prepared assuming that [we] will continue as a going concern" | **opens both ① and ②** |
| ⑦ No-adjustments sentence | "…do not include any adjustments… should we be unable to continue as a going concern" | closes both ① and ② |
| ⑧ Risk-factor and valuation wording | "we may cease to continue as a going concern" (p. 39, conditional); "selling… the Debtors' businesses as a going concern" (p. 37, a valuation sense) | Item 1A |

**Where ③ sits matters.** ③ contains the exact words of the MEDIUM signal ("conditions… raised substantial doubt"), but inside a "whether" clause. It describes the evaluation, not its result.

### 1.3 Carvana's "Liquidity" paragraph, the one ambiguous passage in the sample

This is Note 2, pp. 88–89, quoted in order:

> The accompanying consolidated financial statements of the Company have been prepared in conformity with U.S. GAAP, which contemplate continuation of the Company as a going concern. Since inception, the Company has incurred losses, and expects to incur additional losses in the future… [equity offering of $1.2 billion; $3.275 billion of senior notes; forward-flow and floor-plan facilities with unused capacity]… Management believes that current working capital, results of operations, and existing financing arrangements are sufficient to fund operations for at least one year from the financial statement issuance date.

The paragraph has the **shape** of an ASC 205-40-50-12 "alleviated" disclosure:
- it states conditions (losses);
- it states plans (the financing raised);
- its conclusion uses ASC 205-40's own horizon ("one year from the … issuance date").

But it **never says substantial doubt was raised.**
- Read as written, it is management's no-doubt conclusion, expressed as a sufficiency statement.
- Read by its shape, it could be taken as a 50-12 disclosure.

The current rubric text requires the notes to "disclose that conditions or events raised substantial doubt", which points to the first reading. **This is the one place in the sample where a careful reader could reasonably land on a different level** (open question 1).

### 1.4 What the parser produces (what the agent actually receives)

Parsed chunk counts: Carvana 441, Peloton 374, Shopify 412 (stored March parse), Revlon 612.

**The auditor's report is split at its paragraph headings, one chunk each.** In order:
- the title (100–135 characters, holding only the addressee);
- the opinion;
- the going-concern paragraph, when there is one;
- the basis for opinion;
- the introduction to the critical audit matters, then one chunk per critical audit matter;
- the signature.

Each chunk's heading is its own paragraph heading. The report title is not carried onto any other chunk.

What follows from that:

- **The going-concern paragraph is a chunk of its own.** Revlon's is chunks 280 and 287, 632 characters, headed only "Going Concern".
  - Nothing in it says it comes from the auditor's report. Only the phrasing shows the auditor's voice ("As discussed in Note 1 to the consolidated financial statements…").
- **So the opinion chunk never shows whether a going-concern paragraph follows it.** Retrieving the opinion proves the report was reached, not that it has no going-concern paragraph.
- **The FS and ICFR reports produce chunks with identical titles and identical "Basis for opinion" headings.** Each one's opinion chunk also mentions the other opinion:
  - Peloton's FS opinion chunk ends "…expressed an **adverse** opinion thereon", referring to internal control;
  - Carvana's ICFR opinion chunk ends "…expressed an unqualified opinion on those financial statements".
- **Notes subsections split at page breaks, and the continuation loses the heading.**
  - **Revlon Note 1.** Chunk 344 carries the heading "Liquidity and Ability to Continue as a Going Concern". It holds ⑥ ③ ④ and the liquidity position, but **not** the conclusion. The conclusion ② is in chunk 345, whose heading is the running page header "COMBINED NOTES TO CONSOLIDATED FINANCIAL STATEMENTS (DEBTOR-IN-POSSESSION)". So the chunk that carries the going-concern heading contains "raised substantial doubt" (③) and no conclusion.
  - **Carvana Note 2.** Same pattern. Chunk 258 ("Liquidity") holds the basis sentence, the losses and the financing. Chunk 259 holds the sufficiency conclusion, under the running header "…NOTES TO CONSOLIDATED FINANCIAL STATEMENTS (Continued)".
- **MD&A copies are indistinguishable from the notes.** Revlon's MD&A chunks 153 and 201 carry the **same heading** as the note ("Liquidity and Ability to Continue as a Going Concern"), and each holds the full conclusion in a single chunk. Nothing in a chunk shows whether it came from MD&A or the notes.
- **The model sees only a heading and a score per passage.** Evidence blocks are rendered as `### Evidence i [first heading] (score)` ([analysis_agent.py:271-277](../../src/agents/analysis_agent.py#L271-L277)). There is no page number and no chunk ID, so the model cannot tell that two passages are adjacent.

### 1.5 Retrieval on Revlon (current `v1` queries)

The full record is in `data/retrieval_checks/d2_2026-10-01_rev-20221231.md`.

**Retrieved by the main queries:**
- the auditor's going-concern paragraph (rank 6);
- management's conclusion, three times: two MD&A copies and the note continuation (ranks 2, 5 and 8);
- the Note 1 heading chunk, without its conclusion (rank 1).

**Not retrieved:**
- **The FS opinion paragraph.** Query 1 returned the two "Basis for Opinion" chunks instead (ranks 3–4).
- **The debtor-in-possession status sentence.** It appears only as the "(DEBTOR-IN-POSSESSION)" page header on rank 8's heading.

**The alternate queries did not run.** Confidence was 0.665, above the 0.60 trigger. They would have retrieved:
- the Chapter 11 status chunks;
- the second registrant's opinion paragraph.

Revlon's level does not depend on either gap. **A filing without going-concern language would depend on the first one**: LOW requires the opinion.

---

## 2. Cross-filing patterns

**P1. Most filings never use the words.**
- Two of the three portfolio filings contain neither "going concern" nor "substantial doubt" anywhere; the third uses "going concern" once, in the basis-of-accounting sentence.
- For a typical filer, D2 is therefore decided by **absence**. Two consequences:
  - The going-concern queries will match nothing relevant, and will return liquidity or risk-factor text instead.
  - The verdict rests entirely on whether the auditor's opinion on the financial statements was retrieved. The three portfolio filings are likely to exercise the LOW / INCONCLUSIVE path, which the held-out filing cannot test.

**P2. Every finding opens with the boilerplate sentence.**
- Both of Revlon's findings (① the auditor's paragraph and ② the note) begin with the basis-of-accounting sentence ⑥ and end with the no-adjustments sentence ⑦.
- A guard that says "the basis sentence is not a finding" is correct only if the passage is judged by its **concluding** sentence. A model that pattern-matches the first sentence would dismiss a real finding.

**P3. "Substantial doubt" appears in non-finding sentences right next to the finding.**
- The non-finding kinds are ③ the evaluation description, ④ the horizon statement and ⑤ the dependency statement. They sit in the same paragraph as ②.
- After chunking, they can sit in a **different chunk** from it.
- ③ uses the MEDIUM signal's own words.

**P4. A liquidity paragraph can have the 50-12 shape without the 50-12 words (Carvana, 1.3).**
- Losses, then financing, then "sufficient for one year from issuance" is how a company with recurring losses explains why there is **no** doubt.
- It is also how a 50-12 disclosure is built. Only an explicit statement that substantial doubt was raised tells them apart.

**P5. Where a passage sits in the filing does not survive chunking.** Three distinctions disappear:
- **MD&A or notes:** copies carry identical headings;
- **auditor or management:** the auditor's paragraph has no auditor marker;
- **FS report or ICFR report:** both have the same title and the same "Basis for opinion" heading.

Any rubric condition phrased as "in the notes" or "in the auditor's report" is decided by **wording** in practice, not by location.

**P6. The LOW evidence condition is a proxy.**
- The going-concern paragraph is never in the opinion chunk, so seeing the opinion does not show the paragraph is absent.
- What makes absence credible is that an existing going-concern paragraph is a strong match for the going-concern queries; in Revlon it was retrieved.
- The opinion chunk shows the right document was reached. It cannot show more than that unless the model can see what follows it.

**P7. Opinion words mislead in both directions.**
- **An unqualified opinion can carry a going-concern paragraph** (Revlon), so "unqualified" says nothing about D2.
- **An adverse ICFR opinion is not a going-concern conclusion**, yet it is quoted inside Peloton's FS report.

**P8. Chapter 11 status is all-or-nothing in this sample, and the timing must be read against the 10-K date.**
- **Revlon carries status everywhere:**
  - present-tense status sentences ("have operated their businesses as 'debtors-in-possession'");
  - "(DEBTOR-IN-POSSESSION)" on every statement page;
  - a future confirmation hearing.
- **The portfolio filings mention bankruptcy only hypothetically, for third parties, or as a financing term** ("bankruptcy-remote subsidiaries").
- **Not in the sample:** a company that emerged from Chapter 11 before filing its 10-K.

**P9. Combined filings carry one report per registrant.**
- Revlon's two reports are identical, so the choice did not matter here.
- In general it could: a parent and its subsidiary can have different conclusions.

**P10. What this sample cannot test:**
- the MEDIUM level (no 50-12 disclosure in any filing);
- an emphasis-of-matter paragraph;
- auditor and management reaching different conclusions;
- a company that emerged from Chapter 11 before filing;
- retrieval on the LOW path in a held-out filing.

---

## 3. Implications for the D2 guidance

**Status (2026-10-01):**
- **Recommendations 1–9 are applied** to `data/dimensions/v1/d2_going_concern.json`, with open questions 1–4 settled as recorded in section 5. They are listed in that file's `rubric_clarifications`.
- **From recommendation 10, the query 1 rewording is applied** and checked on the held-out filing (`data/retrieval_checks/d2_2026-10-01_rev-20221231.md`). The Chapter 11 query stays an alternate (question 4).
- **The rest of recommendation 10** (making absence observable, heading ancestry) is deferred with the other parser and retrieval work.
- Section 4 describes the guidance **as it stood before** these changes.

1. **Judge a passage by its concluding sentence.**
   - Rewrite the basis-of-accounting guard to say that the basis sentence (⑥) and the no-adjustments sentence (⑦) also open and close every real finding.
   - So the passage must be classified by the sentence that states a conclusion, never by its first sentence.
2. **Name the non-finding "substantial doubt" sentences.**
   - The kinds: ③ evaluation description (especially "whether there were conditions and events… that raised substantial doubt"), ④ horizon statement, ⑤ dependency statement, ⑧ conditional risk-factor and valuation wording.
   - When a passage contains only these, look for the conclusion in another passage. If none is present, the passage is not a finding.
3. **Make MEDIUM's wording requirement explicit.**
   - A 50-12 disclosure must say that substantial doubt was raised (or existed before management's plans) **and** that the plans alleviated it.
   - A liquidity paragraph that describes losses, financing and a sufficiency conclusion without saying substantial doubt was raised is the **no-doubt** outcome, not 50-12.
   - The current rubric text already requires this ("disclose that conditions or events raised substantial doubt"); the guidance should say it outright. *Owner decision: open question 1.*
4. **Decide management's statement by its wording, wherever it appears.**
   - Chunks cannot show MD&A from the notes (P5), and MD&A repeats the notes.
   - Criterion HIGH (b) and MEDIUM would then read "management states (in the notes, or in MD&A repeating them)…".
   - This changes the rubric text, so it is a `v1` edit before freezing. *Owner decision: open question 2.*
5. **Define "the auditor's opinion on the financial statements" positively, for the LOW condition.**
   - **What it is:** the paragraph that says "We have audited the accompanying consolidated balance sheets…" and "In our opinion, the consolidated financial statements present fairly…". In a combined report, that paragraph covers both opinions.
   - **What does not count:** the report title alone; an ICFR-only report; a mention of the FS opinion inside the ICFR report.
   - *Owner decision, open question 3:* whether the "Basis for Opinion" and critical-audit-matter passages of the FS report also count.
6. **Add two guards.**
   - (a) An unqualified opinion does not exclude a going-concern paragraph.
   - (b) An ICFR opinion (adverse or unqualified) or a material weakness is not a going-concern conclusion, even when quoted inside the FS report.
7. **Give an attribution hint.** It is minor and changes no verdict, but it keeps the output fields (`as2415_paragraph_found`, `management_disclosure`) right.
   - A "Going Concern" passage that refers to "Note X to the consolidated financial statements" and to "management's plans" is the auditor's.
   - One written in the first person ("we", "our") is management's.
8. **Say how to judge Chapter 11 status at the 10-K date.**
   - **What establishes it:** present-tense status ("operating as debtors-in-possession"), and statements headed "Debtor-in-Possession".
   - **What does not end it:** a confirmation hearing or expected emergence **after** the 10-K date.
   - **What does end it:** emergence **before** the 10-K date ("emerged", plan effective, Successor/Predecessor columns).
   - **What does not count:** hypothetical bankruptcy risk, bankruptcy-remote entities, a counterparty's bankruptcy.
9. **Combined filings.** Use the report addressed to the registrant named in the task. It is one sentence, and the same issue was already handled in D1.
10. **Retrieval and pipeline.** Recorded here; the owner has deferred retrieval and parser work.
    - **Query 1.** Reword it toward the opinion paragraph ("We have audited the accompanying consolidated balance sheets… In our opinion… present fairly in all material respects") and drop "basis for opinion", which is what pulled the Basis chunks. Verify on the held-out filing.
    - **Chapter 11 query.** A Chapter 11 query sits only in the alternates. Moving one into the main set would guarantee status evidence for a Chapter 11 filer whose auditor did not add a paragraph. The cost is risk-factor noise in every other filing; the guards in item 8 would have to absorb it.
    - **Make absence observable.** Either:
      - show page and chunk ID in each evidence header, so the model can see that the opinion chunk is followed directly by "Basis for opinion"; or
      - fetch the chunk after the opinion chunk (neighbour expansion).
    - **Heading ancestry in the parser** (the same fix as D1 recommendation 6) would restore MD&A-versus-notes and auditor-versus-management location.

## 4. How the current D2 guidance measures up

| Current guidance item | Assessment |
|---|---|
| Rubric levels, sources, Chapter 11 project rule | Correct. The three levels map to ASC 205-40's three outcomes. |
| Guard: basis-of-accounting sentence is not a qualification | **Correct but risky as written.** The same sentence opens both of Revlon's findings (P2). Needs the "judge by the concluding sentence" wording (recommendation 1). |
| Guard: management's no-doubt conclusion | Covers only the explicit "concluded there are no conditions … that raise substantial doubt" form. **Misses the sufficiency-statement form** (Carvana), which is how the no-doubt outcome appeared in this sample. Widen it (recommendation 3). |
| Guard: alleviated vs. not alleviated | Correct. **Missing the evaluation-description trap** (③), which uses the MEDIUM wording (recommendation 2). |
| Guard: emphasis-of-matter paragraph | Correct. Not exercised by the sample. |
| Guard: conditional and risk-factor language | Correct. Revlon p. 39 is a live example. |
| Guard: liquidity and distress not rated here | Correct. Carvana and Peloton sufficiency statements fall under it. |
| Mandatory item: auditor's report, opinion on the financial statements | Correct, but should exclude ICFR-only reports and name the combined-report case (recommendation 5). |
| Criteria tied to "the financial statement notes" | **Not observable from chunks** (P5). Revlon's MD&A copies are retrieved ahead of the note. See recommendation 4. |
| Chapter 11 signal, "as of the filing date" | Correct, but gives no way to judge timing (recommendation 8). |
| No guard on "unqualified opinion" or on ICFR opinions | **Missing** (recommendation 6). |
| `no_evidence_handling`: INCONCLUSIVE when the opinion is not retrieved | Correct. Its weak point is retrieval (1.5), not the rule. |
| Output schema | Fine. The `management_disclosure` and `as2415_paragraph_found` fields depend on attribution (recommendation 7). |

## 5. Open questions for the owner

**Status (2026-10-01):** questions 1–4 were settled by the owner adopting the recommendations below. Question 5 is on hold.

1. **Carvana-type liquidity paragraphs** (1.3). Should a paragraph with conditions, plans and a one-year sufficiency conclusion, but no statement that substantial doubt was raised, count as 50-12 (MEDIUM)?
   - **My recommendation is no**, which matches the current rubric text.
   - If your worksheet reached a different level for Carvana, this is the place to settle it **before** `v1` is frozen.
   - **Decided: no.** MEDIUM requires the words.
2. **Management's statement in MD&A.** Should HIGH (b) and MEDIUM accept the statement when it appears in MD&A, since chunks cannot show MD&A from the notes (recommendation 4)? I recommend yes.
   - **Decided: yes** (project rule).
3. **What satisfies LOW's evidence condition.** The opinion paragraph only, or any part of the FS auditor's report (Basis for Opinion, critical audit matters)?
   - Either is a proxy (P6).
   - "Opinion paragraph only" is stricter and depends on the query fix in recommendation 10.
   - **Decided: any part of the FS report** (opinion, basis for opinion, critical audit matters). An ICFR-only report or a bare title does not count.
   - **Why:** the going-concern paragraph is always a separate chunk, so no section of the report shows its absence better than another. Requiring the opinion paragraph alone would add retrieval failures (INCONCLUSIVE where LOW is right) without lowering the risk of a false LOW.
4. **Should a Chapter 11 query move into the main set** (recommendation 10)?
   - **Decided: no, it stays an alternate.**
   - A Chapter 11 debtor's going-concern disclosures discuss the case (Revlon's auditor's paragraph and note both do), so status evidence arrives with the main going-concern queries. The case where (c) alone decides (management says the plan alleviates the doubt) is one such disclosure.
   - A fourth main query would cut every query's quota from 3 slots to 2, including the auditor's-report query that the LOW path depends on.
5. **Held-out coverage.** Revlon cannot test the LOW path, which is where the portfolio filings will land, or the MEDIUM path. Should we add held-out development filings? Two candidates:
   - **one healthy filer:** retrieval only, no label needed;
   - **one filing with a 50-12 "alleviated" disclosure.** Finding one needs EDGAR full-text search, so it needs `SEC_USER_AGENT` or a browser.

**A note on the EDGAR cross-check (worksheet Q4).**
- In the main 10-K documents surveyed here, "substantial doubt" appears **only** in Revlon.
- EDGAR full-text search also indexes a filing's exhibits (credit agreements, for example, often mention going-concern qualifications). So a hit for Carvana, Peloton or Shopify most likely comes from an exhibit, or from a different filing, not from the 10-K text itself. Check the document type on each hit.

---

## 6. Scope decision (2026-10-01)

**Question:** is D2 a legitimate dimension, and can it be evaluated on this portfolio?

**Decision (owner):** keep D2 as defined: formal going-concern conclusions only, labelled from the filing and cross-checked on EDGAR.

**Changes made:**
1. **Renamed** `d2_going_concern_liquidity` → `d2_going_concern`, together with the config file name and the output schema's `dimension` value.
   - The rubric already excluded liquidity, so the old name misdescribed what is rated.
   - `v1` had not been run, so this is an edit before freezing.
2. **Reporting rule.** Every per-dimension accuracy is reported beside its majority-class baseline.
   - No portfolio filing has a HIGH condition (section 1.2), so for now D2 results test false-alarm resistance and evidence discipline.
   - **Detection is not yet tested, and is meant to be tested later** (owner, 2026-10-01). That needs a scored filing with a going-concern conclusion.
   - Recorded in CLAUDE.md and in `evaluation_config` in `data/dimensions/v1/00_index.json`. The eval harness, when built, must implement it.

**Why a D1 HIGH can sit beside a D2 LOW.**
- In both distressed portfolio filings, the Z″ score is driven by losses and the accumulated deficit; the working-capital term is positive.
- The large debts fall due beyond the one-year going-concern horizon.

**Not chosen:**
- **Redefining D2 as a liquidity score.** There is no published threshold, so the cut-offs would be invented, and it would largely duplicate D1.
- **Adding a portfolio company with a going-concern paragraph.** Deferred, not rejected; it is the way to test detection later. It would take the dataset from 12 to 16 examples.

**Open questions in section 5:** questions 1–4 were settled the same day (see section 5); question 5 is on hold.

---

## 7. Agent check on the held-out filing (2026-10-08)

Revlon only; no portfolio filing was run.

**Setup:**
- **Model:** `gemini-3.8-flash`.
- **Method:** the agent's own `_build_prompt`, `_invoke_with_retry` and `_parse_verdict`. Only the content-block text extraction is done in the harness, because of the deferred `str(response.content)` bug.
- **Single samples:** each set was run once.
- **Files:** script, evidence and raw outputs are in `data/retrieval_checks/d2_agent_check_2026-10-08/`.

**The guidance was trimmed before this check** (no rule changed; listed in the config's `rubric_clarifications`):
- **Size:** the rendered guidance went from 8,957 to 6,338 characters, because each rule now has one home.
- **Fix:** guard explanations now name their own examples. The prompt renders a guard's name and explanation but **not** its `pattern` field, so explanations such as "Either form…" pointed at examples the model never saw.
- **Same gap in D5:** three D5 guards begin "This is…" and have the same problem. Not fixed yet.

| Set | Evidence | Tests | Expected | Result |
|---|---|---|---|---|
| **A** real retrieval | the 8 chunks retrieval returned (confidence 0.715, main queries only): both opinion paragraphs, auditor's going-concern paragraph, two MD&A conclusions, Note 1 heading chunk, two unrelated | detection end to end | HIGH | ✅ **HIGH**, all three criteria. Quoted the auditor's paragraph and management's conclusion verbatim, each attributed to the right source. Cited Chapter 11 status from the "(DEBTOR-IN-POSSESSION)" running header on an unrelated note chunk, which the guidance allows. |
| **B1** controlled | opinion (279), basis (281), Note 1 heading chunk (344), note conclusion under a running-header heading (345), shuffled | finding a conclusion whose chunk has lost its heading; not mistaking 344's "raised substantial doubt" for the finding | HIGH via (b) and (c) | ✅ **HIGH**, criteria (b) and (c), quoting the conclusion sentence from 345. Correctly reported no auditor's going-concern paragraph, since 280 was not in the set. |
| **B2** controlled | opinion, basis, both critical-audit-matter chunks, Note 1 heading chunk **cut before its "DIP ABL Facility" sentence**, forward-looking statements | false-alarm traps only: the basis sentence, the "whether … raised substantial doubt" clause, the horizon statement | LOW | ⏳ not run: daily quota exhausted |
| **B3** controlled | the cut Note 1 chunk, forward-looking statements, the auditor's signature chunk | evidence discipline: no opinion, basis or critical-audit-matter passage | INCONCLUSIVE | ⏳ not run: daily quota exhausted |

**Why chunk 344 is cut in B2 and B3.**
- Its full text mentions the "DIP ABL Facility". Debtor-in-possession financing exists only in Chapter 11, so a model inferring Chapter 11 from it would be right, and the set would no longer test false alarms.
- The cut keeps the sentences under test and drops the liquidity-position sentence. The harness records the cut.

**Rerun when the quota resets.** Each command uses one model call:

```bash
LLM_MAX_RETRIES=1 python data/retrieval_checks/d2_agent_check_2026-10-08/d2_agent_check.py B2 data/retrieval_checks/d2_agent_check_2026-10-08
LLM_MAX_RETRIES=1 python data/retrieval_checks/d2_agent_check_2026-10-08/d2_agent_check.py B3 data/retrieval_checks/d2_agent_check_2026-10-08
```

**Reading so far:**
- Detection works end to end on the held-out filing.
- The guidance handles a conclusion separated from its heading.
- The two tests that matter most for the portfolio (false alarms and evidence discipline) are still pending.
