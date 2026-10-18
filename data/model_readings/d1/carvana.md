# D1 — Carvana FY2022 — MODEL READING (not a label)

> **MODEL READING — NOT A LABEL.** Claude's application of the v1 D1 rubric and guidance to the filing, for comparison with the human label. It must never be copied into `labels.json`; the D1 label comes from XBRL computation. A disagreement is settled by a human checking the filing and XBRL, never by deferring to this file.

- **Filing:** `data/filings/cvna-20221231.pdf` — fiscal year ended December 31, 2022
- **Rubric version:** `v1` — D1 guidance as settled on 2026-09-29 (third pass). Re-checked against it on 2026-10-01: all inputs and the verdict unchanged.
- **Statements:** primary balance sheet PDF p. 83; primary statement of operations PDF p. 84 (single registrant; no look-alike tables used)
- **Units:** In millions (US$) · **Columns:** 2022 | 2021 — first column is the rated year
- **Method:** inputs read by Claude from the statements; a script checked every quoted line appears verbatim on its page, checked the balance sheet balances, and did the arithmetic.

## D1 rubric (v1)

Z″ = 6.56·X1 + 3.26·X2 + 6.72·X3 + 1.05·X4, where X1 = (current assets − current liabilities) / total assets, X2 = retained earnings / total assets, X3 = EBIT / total assets, X4 = total stockholders' equity (incl. NCI) / total liabilities.

**HIGH:** Altman Z″ < 1.10 (distress zone).

**MEDIUM:** 1.10 ≤ Altman Z″ ≤ 2.60 (grey zone).

**LOW:** Altman Z″ > 2.60 (safe zone).

## Inputs

| Input | Value | Source line(s), verbatim | PDF p. | How derived |
|---|---|---|---|---|
| Total current assets | 4,594 | `Total current assets 4,594 4,892` | 83 | captioned line |
| Total current liabilities | 2,592 | `Total current liabilities 2,592 2,890` | 83 | captioned line |
| Total assets | 8,698 | `Total assets $ 8,698 $ 7,015` | 83 | captioned line |
| Retained earnings (accumulated deficit) | -2,076 | `Accumulated deficit (2,076) (489)` | 83 | equity component accumulating net loss (Carvana Co.'s share; NCI's share sits in the NCI line) |
| EBIT | -2,407 | `Net loss before income taxes (2,893) (286) (462)`<br>`Interest expense, including $0, $0, and $1, respectively, to related parties 486 176 131` | 84 | **Fallback 1:** the statement has no operating subtotal, so EBIT = pre-tax loss (−2,893) + interest expense (486; shown positive, not net of interest income) |
| Total stockholders' equity incl. NCI | -1,053 | `Total stockholders' equity (deficit) (1,053) 525` | 83 | total below the noncontrolling-interests line — includes NCI of (535) |
| Total liabilities | 9,751 | `Total liabilities 9,751 6,490` | 83 | captioned line |

Balance-sheet check: total assets (8,698) = total liabilities (9,751) + total equity (−1,053) ✓

## Computation

| Ratio | Formula | Value |
|---|---|---|
| X1 | (4,594 − 2,592) / 8,698 | 0.2302 |
| X2 | -2,076 / 8,698 | -0.2387 |
| X3 | -2,407 / 8,698 | -0.2767 |
| X4 | -1,053 / 9,751 | -0.1080 |

Z″ = 6.56 × 0.2302 + 3.26 × -0.2387 + 6.72 × -0.2767 + 1.05 × -0.1080 = 1.510 + -0.778 + -1.860 + -0.113 = **-1.241**

## Verdict

- **Z″:** -1.241
- **Zone:** distress
- **Rating:** HIGH
- **Fallbacks used:** EBIT Fallback 1 — the statement of operations has no operating subtotal (it runs from gross profit and expenses, including interest expense, straight to "Net loss before income taxes"), so EBIT = pre-tax loss + interest expense, as the guidance prescribes.
- **Ambiguities — places the rubric did not give a clear answer:**
  - None under the current guidance: it prescribes Fallback 1 for a statement with no operating subtotal, and defines retained earnings by role, excluding noncontrolling interests.
  - Note, not an ambiguity: Fallback 1 keeps "Other expense, net" (70) inside EBIT; an operating loss built from the expense lines would be −2,337 (Z″ −1.187; see Sensitivity). The guidance does not allow that construction, and the zone is the same either way.
  - Note, not an ambiguity: under the Up-C structure the accumulated-deficit line is Carvana Co.'s share only, while total equity includes NCI. The XBRL label path uses the same line, so the reading and the label stay comparable.
- **Read by / date:** Claude (model reading — not a label) / 2026-09-29

## Sensitivity — alternative input choices (not the rubric's)

For comparison with a label computed differently (for example from XBRL tags). The verdict above uses the rubric's choices.

| Alternative | Z″ | Zone |
|---|---|---|
| Rubric choices (above) | -1.241 | distress |
| EBIT = gross profit − SG&A − goodwill impairment (1,246 − 2,736 − 847), i.e. excluding other expense of 70 | -1.187 | distress |
| Equity excluding NCI (attributable to Carvana Co.: (518)) | -1.184 | distress |
