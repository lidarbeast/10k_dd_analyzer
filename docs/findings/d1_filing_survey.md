# D1 filing survey: how Z″ can actually be evaluated from 10-K filings

**Date:** 2026-09-29
**Purpose:** Inform how the D1 guidance should be written so it works across filings, not just the ones in the portfolio. This is a survey of the documents, not a set of labels and not a change to the guidance.

**Filings surveyed**

| Filing | Role | Fiscal year-end |
|---|---|---|
| `cvna-20221231` Carvana | portfolio | Dec 31, 2022 |
| `pton-20230630` Peloton | portfolio | Jun 30, 2023 |
| `shop-20251231` Shopify | portfolio | Dec 31, 2025 |
| `rev-20221231` Revlon | **held-out**, in Chapter 11, two registrants in one 10-K | Dec 31, 2022 |

**Method**

1. **Read the statements from the PDF.** For each filing, read the primary financial statements from the PDF text (`pypdfium2`). Then search the whole filing for every other page where the same captions appear.
2. **Look at what the agent would see.** Ran the project's own parser (Docling parse, chunk and post-process) locally on Carvana, Peloton and Revlon, and used the existing parsed output for Shopify (`data/parsed/shop-20251231_processed.json`). Then inspected the chunks. No Pinecone upload was done for this survey.
3. **Retrieval.** Checked retrieval behaviour only on Revlon (see `data/retrieval_checks/`).
4. **XBRL.** Not surveyed, because `SEC_USER_AGENT` is not set yet.

---

## 1. Per-filing findings

### 1.1 Where the seven inputs are, and how they are presented

| Input | Carvana | Peloton | Shopify | Revlon (held-out) |
|---|---|---|---|---|
| Balance sheet | p. 83, classified | p. 126, classified | p. 85, classified | p. 141 (parent registrant), classified |
| Current assets total | labelled "Total current assets" | labelled | **no caption**: a bare subtotal row | labelled |
| Current liabilities total | labelled | labelled | **no caption** | labelled |
| Total assets | labelled | labelled | labelled | labelled |
| Retained earnings line | "Accumulated deficit" (parent's share only; see 1.2) | "Accumulated deficit" | "Accumulated surplus" | "Accumulated deficit" |
| Operating subtotal (EBIT) | **absent**: expenses, including interest, run straight to "Net loss before income taxes" | "Loss from operations" | "Income (loss) from operations" | "Operating income" |
| Interest expense | inside the expense block, positive "486" | below the operating line, shown negative "(97.1)" | **no line at all** | "Interest expense, net" (net of interest income), positive |
| Total equity | "Total stockholders' equity (deficit)", **below** a noncontrolling-interests line; a separate "…attributable to Carvana Co." total sits above it | "Total stockholders' (deficit) equity" | "Total shareholders' equity" | "Total stockholders' deficiency" |
| Total liabilities | labelled | labelled | **no line**; bare subtotals for the current and long-term groups only | **no line**; includes "Liabilities subject to compromise" |
| Units | In millions | in millions, 1 decimal | In US $ millions | dollars in millions, 1 decimal |
| Column order | current year first | current year first | current year first | current year first |

Z″ from the primary statements, for reference:
- The three portfolio values are in `data/model_readings/d1/`.
- Revlon: X1 −0.0195, X2 −1.4108, X3 0.0321, X4 −0.5168, **Z″ −5.05**. Total liabilities was derived as 2,489.8 − (−2,662.7) = 5,152.5.

### 1.2 Structural features that affect the inputs

- **Carvana (Up-C structure).**
  - Two equity totals: one attributable to the parent (−518) and one including noncontrolling interests (NCI) (−1,053).
  - The "Accumulated deficit" line holds only the parent's share of losses. The NCI's share sits inside the NCI line. So X2 and X4 are drawn on different perimeters: X2 excludes the NCI's share, while X4 includes NCI through total equity. The label path (XBRL `RetainedEarningsAccumulatedDeficit`) uses the same line, so label and verdict stay consistent. Worth stating in the README.
- **Revlon (Chapter 11).**
  - "Liabilities subject to compromise" (3,711.2) sits outside both current and long-term liabilities. It belongs in total liabilities, which has no line of its own.
  - Equity is a "deficiency".
- **Revlon (combined 10-K).** It holds **two registrants' full statements**: Revlon, Inc. (pp. 141–143) and Revlon Consumer Products Corporation (pp. 147–149). The figures differ; for example, current assets are 1,211.5 against 1,414.5.

### 1.3 Places outside the primary statements where the same captions appear

These are where an agent or a retriever can pick up the wrong figure.

| Filing | Location | What it contains | Risk |
|---|---|---|---|
| Revlon | Note 23, pp. 253–255 | Condensed balance sheet and statement of operations of the **debtor entities only**, under headings identical to the primary statements'. The note's balance sheet **has a labelled "Total liabilities" line (5,379.7) that the primary balance sheet lacks**. Its statement of operations has another pre-tax "(Loss) income from operations before income taxes" line. | Different perimeter (total assets 3,491.5 against 2,489.8). An agent looking for a labelled "Total liabilities" line is drawn to the wrong statement. The 2026-09-28 retrieval check actually returned this chunk. |
| Revlon | pp. 142, 148 | "Loss from operations **before income taxes**" (642.2), which is **pre-tax** income | A label list that accepts "Loss from operations" picks the wrong line. With it as EBIT, Z″ goes from −5.05 to −7.00. |
| Peloton | tax note, p. 174 | "Loss from operations before income taxes" (1,258.0) | Same trap as Revlon, in a second filing. |
| Peloton | MD&A, p. 94 | Results table repeating "Loss from operations" and "Interest expense" | Same values as the statement. Harmless here, but not guaranteed. |
| Shopify | MD&A, p. 69 | "Key Balance Sheet Information": **"Total liabilities 1,716"**, a line the balance sheet itself lacks | Agrees with the balance-sheet identity here. MD&A figures are not guaranteed to match GAAP definitions in general. |
| Shopify | MD&A, p. 66 | Quarterly "Income from operations" (8 quarter columns) | A quarterly figure could be taken for the annual one. |
| Shopify | MD&A, pp. 74–75 | Constant-currency tables: "Income from operations 1,468 (36) 1,432" | Non-GAAP adjusted figures sit next to the GAAP figure. |
| Revlon | segment note, pp. 237–238 | Segment "operating income" | Segment figures, not consolidated. The 2026-09-28 retrieval check returned this chunk. |
| Carvana | MD&A, pp. 65, 69 | Narrative on interest expense | Harmless. |

### 1.4 What the parser produces (what the agent actually receives)

Tables are serialized row by row as `Row label, Column header = value.`. Multi-line column headers are joined with dots, for example `Years Ended December 31,.2022` or `December 31,.2025.(in US $ millions)`.

| | Carvana | Peloton | Shopify |
|---|---|---|---|
| Statement title kept as the chunk heading? | **No**: the balance-sheet heading is the units line "(In millions, except…)"; the income statement keeps its title | Yes, title plus units | Title yes, **units line lost** |
| Balance-sheet chunks | 3 (the last is 236 characters, a split tail) | 2 | 3 |
| Income-statement chunks | 2+ | 4 | 2+ |
| Chunks D1 needs, minimum | **4**: balance sheet assets, balance sheet liabilities and equity, plus both income-statement chunks, because 2022 interest expense and pre-tax loss fall in different chunks | 3 | 3–4 |
| Rows cut at chunk boundaries | Yes: "Total liabilities & stockholders' equity… = $" / "8,698."; interest expense 2022 and 2021 in different chunks | Yes: "Net loss… =" / "(1,261.7)." | Yes |
| Unlabeled subtotal rendered as | n/a | n/a | `, December 31, 2025 = 8,296.` (empty row label) |

**Revlon (held-out).**
- **Primary statements.** The balance sheet is 2 chunks (#292–293), headed "(DEBTOR-IN-POSSESSION) CONSOLIDATED BALANCE SHEETS (dollars in millions…)". The statement of operations is 2 chunks (#294–295), headed "REVLON, INC. AND SUBSIDIARIES (DEBTOR-IN-POSSESSION) CONSOLIDATED STATEMENTS OF OPERATIONS".
- **Note 23's debtor-only tables.** The balance sheet becomes chunks #584–585, headed "CONSOLIDATED BALANCE SHEETS (dollars in millions…)". The statement of operations becomes chunks #587–588, **headed identically to the primary statement**.
- **The note's own title is a separate text chunk** (#583: "23. CONDENSED CONSOLIDATION DEBTOR-IN-POSSESSION FINANCIAL INFORMATION"). Table chunks carry only their nearest heading, not the note they sit in. **Nothing in the chunk tells the agent the table is inside a note.**

---

## 2. Cross-filing patterns

These recur in two or more of the four filings, or follow from how statements are built in general. They are what the guidance should be written around.

**P1. Captions are unreliable; position is reliable.** The same input carries different captions:
- the operating subtotal: operating income / income (loss) from operations / loss from operations;
- retained earnings: accumulated deficit / accumulated surplus;
- equity: stockholders' / shareholders', deficit / deficiency.

Worse, **a caption that sounds like operating income can be pre-tax income** ("Loss from operations before income taxes", seen in 2 of 4 filings). What does not vary is each input's place in the statement:
- **EBIT** is the subtotal after all operating expenses and before non-operating items (interest, other income or expense, reorganization items, taxes).
- **Retained earnings** is the equity component that accumulates net income less dividends. It is not paid-in capital, treasury stock or accumulated other comprehensive income.
- **Total equity** is the figure that, added to total liabilities, gives total liabilities and equity.

**P2. Totals may be uncaptioned or absent.**
- Uncaptioned subtotals: 1 of 4 filings.
- No "Total liabilities" line: 2 of 4.
- No operating subtotal: 1 of 4.

In every case the missing figure is fixed by an identity the statement itself guarantees:
- total liabilities = total liabilities and equity − total equity;
- a section subtotal = the sum of the section's lines;
- EBIT = pre-tax income + interest expense.

Each derivation can be checked against the statement. A guessed figure cannot.

**P3. Interest expense varies in position, sign and definition.** It can sit above the pre-tax line inside expenses, or below the operating line. It can be printed positive or negative, reported net of interest income, or not reported at all. Any EBIT fallback that adds interest back must use the magnitude, and must say what to do when the line is absent or netted.

**P4. The primary statements are not the only tables with these captions.** In this sample the same captions also appear in:
- MD&A summary tables, quarterly tables and constant-currency (non-GAAP) tables;
- segment notes and tax notes;
- condensed financial statements of a subset of entities placed inside a note, under a heading identical to the primary statement's;
- a second registrant's complete statements.

A rule that says "use the consolidated statements" is not enough: the Revlon Note 23 table is literally headed "CONSOLIDATED BALANCE SHEETS". The primary statements can be recognised positively by:
- their location (Item 8, after the auditor's report, before the notes);
- their footer ("See accompanying notes…");
- being the complete statement, so total assets equals total liabilities and equity, rather than a note table;
- the registrant's name.

**P5. The consolidation perimeter must be consistent across inputs.** NCI (Up-C structures), a second registrant, and debtor-only statements each give a different perimeter. Mixing them corrupts the ratios silently. All seven inputs should come from one statement set, and the balance-sheet identity (assets = liabilities + equity) is a cheap check that the balance-sheet inputs belong together.

**P6. The parser fragments the statements, and drops the context that identifies them.**
- Table chunks carry only their nearest heading, not its ancestry. So a statement reproduced inside a note is indistinguishable by heading from the primary statement (Revlon Note 23).
- A balance sheet spans 2–3 chunks and an income statement 2–4.
- Rows are cut at chunk boundaries, so a caption can end one chunk and its value start the next.
- The statement title and the units line are each lost from the chunk heading in some filings.
- D1 needs 3–4 specific chunks out of the 8 retrieved, and every one has to arrive.

This is a pipeline property, not a filing property, and it will affect every filing.

**P7. The traps did not flip a zone in this sample, but they can in general.** Every Z″ in the sample is far from 1.10 and 2.60 (−6.7 to +12.5). Wrong-line errors of the size seen here (Revlon −5.05 → −7.00; Peloton EBIT −1,197.1 against the tax-note −1,258.0) flip a zone only for filings near a band edge. **This sample cannot test near-boundary behaviour.**

---

## 3. Implications for the D1 guidance

**Status (2026-09-29):**
- **Recommendations 1–5 are applied** to `data/dimensions/v1/d1_financial_health.json`, along with the unclassified-balance-sheet case from section 4. They are recorded in that file's `rubric_clarifications`.
- **Recommendations 6 and 8** (parser and retrieval) are deferred by the owner.
- **Recommendation 7** (arithmetic in code) is open.
- Section 4 describes the guidance **as it stood before** these changes.

1. **Define each input by its structural role, and give caption examples only as illustrations.** In particular, define EBIT as "the subtotal after operating expenses and before interest, other non-operating items and taxes". Say explicitly that a caption containing "before income taxes" is pre-tax income, whatever else it says.
2. **Define the source positively.** Use the registrant's primary financial statements in Item 8, the complete statements that balance. Name the look-alikes to reject: condensed or combined statements inside notes, a second registrant's statements, MD&A, quarterly, segment and non-GAAP tables.
3. **Allow derivations only by identities the statement guarantees, and require a check.** The permitted identities:
   - total liabilities = total liabilities and equity − total equity;
   - an uncaptioned subtotal equals the sum of its section;
   - EBIT = pre-tax income + |interest expense| when no operating subtotal exists.

   Say what to do when the interest line is absent or netted (use pre-tax income, and record it). Nothing else may be constructed.
4. **Require the perimeter check.** Total assets = total liabilities + total equity using the extracted figures. If it fails, the inputs came from mixed sets: return INCONCLUSIVE.
5. **Tell the agent how its evidence is formatted.** Explain the `Row label, Column = value` serialization, empty-label subtotal rows, and rows cut across passages (a caption at the end of one passage may have its value at the start of another from the same page). Say that the chunk heading may be the units line rather than the statement title.
6. **The source problem cannot be fully solved in the guidance.** When a note reproduces a statement under an identical heading, the chunk carries no sign that it is in a note. The balance-sheet identity also holds inside either statement set, so recommendation 4 does not catch a balance sheet taken wholly from the note. The reliable fix is in the parser: carry the heading ancestry (for example "23. CONDENSED CONSOLIDATION…") into each table chunk's headings. The same gap is already recorded in `00_index.json` as the reason the section boost does nothing.
7. **Consider moving the arithmetic out of the model.** Have the model extract the seven figures with their passages, and let code compute Z″, run the identity checks, and flag failures as degraded. Every check in points 3–4 is deterministic.
8. **Retrieval and the pipeline.** D1 needs 3–4 specific chunks out of 8.
   - Consider a per-dimension `top_k`, or a minimum set of statement chunks for D1.
   - At the parser level, consider keeping the statement title and units line on every table chunk, and not splitting a row across chunks. Both would help D1, D2 and D5.

## 4. How the current D1 guidance (as edited 2026-09-29) measures up

| Current guidance item | Assessment |
|---|---|
| Label list for the operating line, including **"Loss from operations"** | **Harmful as written.** It matches "Loss from operations before income taxes", which is pre-tax income in 2 of 4 filings. Replace with the structural definition (recommendation 1). |
| Label list for retained earnings ("Accumulated surplus", etc.) | Harmless but brittle. A structural definition covers every variant. |
| Unlabeled-subtotal rule, with a sum check | General, not Shopify-specific. It matches the parser's output format. Keep. |
| "Consolidated statements only" warning | **Insufficient.** It does not catch condensed statements inside notes under an identical heading. Replace with the positive source definition (recommendation 2), and fix heading ancestry in the parser (recommendation 6), since the chunk alone cannot show it is in a note. |
| "More than one registrant" warning | Correct, but it should come with the perimeter check (recommendation 4) so it can be enforced. |
| EBIT fallbacks, with the interest sign rule | Correct. Add the netted-interest case (Revlon's "Interest expense, net"). |
| "Liabilities subject to compromise" included in total liabilities | Correct and general to Chapter 11 filers. |
| Nothing on rows cut across passages, or on titles and units lost from headings | **Missing** (recommendation 5). |
| Nothing on unclassified balance sheets (banks, insurers) | Not seen in this sample. X1 is undefined there, so the existing missing-input rule already returns INCONCLUSIVE. Worth one explicit sentence. |

## 5. Open questions for the owner

1. Should the model compute Z″ at all, or only extract the figures, with code computing and checking (recommendation 7)?
2. Should the parser changes (heading ancestry, units line, rows kept whole; recommendations 6 and 8) come before any D1 run? They affect D2 and D5 too.
3. Should D1 get its own retrieval budget (recommendation 8)?
4. The sample has no filing near a band edge. Should a held-out near-boundary filing be added before D1 results are reported?

---

## 6. Agent check on the held-out filing (2026-09-29)

Revlon only; no portfolio filing was run. The model was run with a one-off `LLM_MODEL=gemini-3.8-flash` override, because the configured `gemini-2.0-flash` has been retired (404). `.env` and the config are unchanged. Each configuration was run once, so these are single samples.

**The D1 guidance was de-duplicated before this check.** Each rule now appears once. The rendered guidance and instructions went from 10,190 to 6,613 characters, with no rule removed.

### A. Real retrieval path

Retrieval returned these 8 chunks:
- Note 23's debtor-only balance sheet, twice (p. 253);
- only the assets chunk of the primary balance sheet (p. 141);
- the second registrant's statement of operations and equity statement (pp. 148–149);
- three unrelated chunks.

It did **not** return the primary statement of operations (p. 142) or the primary balance sheet's equity chunk.

| Prompt | Verdict | What happened |
|---|---|---|
| Before de-duplication | INCONCLUSIVE | Treated the Note 23 table as the primary balance sheet. Took EBIT (87.6) from the **second registrant's** statement of operations. Stopped because the note table shows no accumulated-deficit line. |
| After de-duplication | INCONCLUSIVE | Also treated the Note 23 table as the primary balance sheet, and stopped for the same reason. Extracted no other inputs. |

**INCONCLUSIVE is the correct verdict for this evidence, but both prompts reached it for the wrong reason.** The look-alike guard did not stop the model using a note table headed "CONSOLIDATED BALANCE SHEETS", as section 1.4 predicted. This confirms that recommendations 6 and 8 (heading ancestry in the parser, and retrieval) are needed. The guidance cannot tell these tables apart on its own.

### B. Controlled evidence (guidance isolated from retrieval)

The evidence was 10 chunks in shuffled order:
- the primary balance sheet (#292–293) and statement of operations (#294–295);
- the Note 23 look-alikes (#584, 585, 587);
- three chunks of the second registrant's statements.

| Prompt | Verdict | Result |
|---|---|---|
| Before de-duplication | **HIGH, Z″ −5.054** | Chose the primary statements. All seven inputs matched the hand reading (CA 1,211.5; CL 1,260.0; TA 2,489.8; RE −3,512.5; EBIT 79.9; equity −2,662.7). It derived total liabilities (5,152.5) by the allowed identity, and the balance check passed. |
| After de-duplication (run 2026-09-30, same model, evidence rebuilt from Pinecone by chunk ID and verified identical, including the shuffled order) | **HIGH, Z″ −5.054** | Identical to the pre-de-duplication run: it chose the primary statements over every look-alike, gave the same seven inputs and the same derived total liabilities (5,152.5), passed the balance check, and computed X1 −0.0195, X2 −1.4108, X3 0.0321, X4 −0.5168. |

**Reading:** when the primary statements are present, both prompts let the model choose them over look-alikes, with identical output (one sample each). De-duplication cut the prompt by 35% with no change in behaviour on this test. When they are absent, it cannot stop the model from using the look-alikes.

### Blockers found for any agent run

1. **`LLM_MODEL=gemini-2.0-flash` is retired** (HTTP 404). Choosing its replacement is an owner decision (see CLAUDE.md, "Model versions").
2. **The newer model returns its content as a list of blocks** (`[{'type': 'text', 'text': ...}]`), not a string. `analysis_agent` calls `str(response.content)`, which would turn every verdict into a flagged `parse_error`. This needs a small code fix before the model is switched.
3. **Free-tier quota: 20 requests per day per model.** One full run is 3 companies × 4 dimensions = 12 calls, before retries, and the retry loop spends quota on failed attempts.
