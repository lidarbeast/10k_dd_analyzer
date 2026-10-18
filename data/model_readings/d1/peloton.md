# D1 — Peloton FY2023 — MODEL READING (not a label)

> **MODEL READING — NOT A LABEL.** Claude's application of the v1 D1 rubric and guidance to the filing, for comparison with the human label. It must never be copied into `labels.json`; the D1 label comes from XBRL computation. A disagreement is settled by a human checking the filing and XBRL, never by deferring to this file.

- **Filing:** `data/filings/pton-20230630.pdf` — fiscal year ended June 30, 2023
- **Rubric version:** `v1` — D1 guidance as settled on 2026-09-29 (third pass). Re-checked against it on 2026-10-01: all inputs and the verdict unchanged.
- **Statements:** primary balance sheet PDF p. 126; primary statement of operations PDF p. 127 (no look-alike tables used)
- **Units:** in millions (US$) · **Columns:** June 30, 2023 | June 30, 2022 — first column is the rated year
- **Method:** inputs read by Claude from the statements; a script checked every quoted line appears verbatim on its page, checked the balance sheet balances, and did the arithmetic.

## D1 rubric (v1)

Z″ = 6.56·X1 + 3.26·X2 + 6.72·X3 + 1.05·X4, where X1 = (current assets − current liabilities) / total assets, X2 = retained earnings / total assets, X3 = EBIT / total assets, X4 = total stockholders' equity (incl. NCI) / total liabilities.

**HIGH:** Altman Z″ < 1.10 (distress zone).

**MEDIUM:** 1.10 ≤ Altman Z″ ≤ 2.60 (grey zone).

**LOW:** Altman Z″ > 2.60 (safe zone).

## Inputs

| Input | Value | Source line(s), verbatim | PDF p. | How derived |
|---|---|---|---|---|
| Total current assets | 1,639.1 | `Total current assets 1,639.1 2,634.6` | 126 | captioned line |
| Total current liabilities | 761.4 | `Total current liabilities 761.4 1,105.5` | 126 | captioned line |
| Total assets | 2,769.1 | `Total assets $ 2,769.1 $ 4,028.5` | 126 | captioned line |
| Retained earnings (accumulated deficit) | -4,931.8 | `Accumulated deficit (4,931.8) (3,710.6)` | 126 | captioned line |
| EBIT | -1,197.1 | `Loss from operations (1,197.1) (2,734.0) (187.8)` | 127 | operating subtotal by position: after total operating expenses, before other expense (interest). Not the tax note's pre-tax "Loss from operations before income taxes" (1,258.0, p. 174) |
| Total stockholders' equity incl. NCI | -295.1 | `Total stockholders’ (deficit) equity (295.1) 592.9` | 126 | captioned line — no NCI |
| Total liabilities | 3,064.2 | `Total liabilities 3,064.2 3,435.6` | 126 | captioned line |

Balance-sheet check: total assets (2,769.1) = total liabilities (3,064.2) + total equity (−295.1) ✓

## Computation

| Ratio | Formula | Value |
|---|---|---|
| X1 | (1,639.1 − 761.4) / 2,769.1 | 0.3170 |
| X2 | -4,931.8 / 2,769.1 | -1.7810 |
| X3 | -1,197.1 / 2,769.1 | -0.4323 |
| X4 | -295.1 / 3,064.2 | -0.0963 |

Z″ = 6.56 × 0.3170 + 3.26 × -1.7810 + 6.72 × -0.4323 + 1.05 × -0.0963 = 2.079 + -5.806 + -2.905 + -0.101 = **-6.733**

## Verdict

- **Z″:** -6.733
- **Zone:** distress
- **Rating:** HIGH
- **Fallbacks used:** none
- **Ambiguities — places the rubric did not give a clear answer:**
  - None. Every input is a captioned line in the primary statements, there are no noncontrolling interests, and no derivation was needed.
- **Read by / date:** Claude (model reading — not a label) / 2026-09-29

## Sensitivity — alternative input choices (not the rubric's)

For comparison with a label computed differently (for example from XBRL tags). The verdict above uses the rubric's choices.

| Alternative | Z″ | Zone |
|---|---|---|
| Rubric choices (above) | -6.733 | distress |
| EBIT = pre-tax loss + interest expense (−1,258.0 + 97.1) | -6.645 | distress |
