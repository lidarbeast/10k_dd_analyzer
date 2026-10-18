# D1 — Shopify FY2025 — MODEL READING (not a label)

> **MODEL READING — NOT A LABEL.** Claude's application of the v1 D1 rubric and guidance to the filing, for comparison with the human label. It must never be copied into `labels.json`; the D1 label comes from XBRL computation. A disagreement is settled by a human checking the filing and XBRL, never by deferring to this file.

- **Filing:** `data/filings/shop-20251231.pdf` — fiscal year ended December 31, 2025
- **Rubric version:** `v1` — D1 guidance as settled on 2026-09-29 (third pass). Re-checked against it on 2026-10-01: all inputs and the verdict unchanged.
- **Statements:** primary balance sheet PDF p. 85; primary statement of operations PDF p. 86 (no look-alike tables used; the MD&A "Total liabilities 1,716" on p. 69 was not used, though it agrees)
- **Units:** In US $ millions · **Columns:** December 31, 2025 | December 31, 2024 — first column is the rated year
- **Method:** inputs read by Claude from the statements; a script checked every quoted line appears verbatim on its page, checked the balance sheet balances, and did the arithmetic.

## D1 rubric (v1)

Z″ = 6.56·X1 + 3.26·X2 + 6.72·X3 + 1.05·X4, where X1 = (current assets − current liabilities) / total assets, X2 = retained earnings / total assets, X3 = EBIT / total assets, X4 = total stockholders' equity (incl. NCI) / total liabilities.

**HIGH:** Altman Z″ < 1.10 (distress zone).

**MEDIUM:** 1.10 ≤ Altman Z″ ≤ 2.60 (grey zone).

**LOW:** Altman Z″ > 2.60 (safe zone).

## Inputs

| Input | Value | Source line(s), verbatim | PDF p. | How derived |
|---|---|---|---|---|
| Total current assets | 8,296 | `Current assets`<br>`8,296 7,254` | 85 | **uncaptioned subtotal** closing the current-assets section; checked: equals 1,545 + 4,233 + 500 + 1,784 + 234 |
| Total current liabilities | 1,392 | `Current liabilities`<br>`1,392 1,956` | 85 | **uncaptioned subtotal** closing the current-liabilities section; checked: equals 1,075 + 300 + 17 + 0 |
| Total assets | 15,189 | `Total assets 15,189 13,924` | 85 | captioned line |
| Retained earnings (accumulated deficit) | 2,860 | `Accumulated surplus 2,860 1,629` | 85 | equity component accumulating net income (captioned "Accumulated surplus") |
| EBIT | 1,468 | `Income (loss) from operations 1,468 1,075 (1,418)` | 86 | operating subtotal by position: after total operating expenses, before other income, net |
| Total stockholders' equity incl. NCI | 13,473 | `Total shareholders’ equity 13,473 11,558` | 85 | captioned line — no NCI |
| Total liabilities | 1,716 | `Total liabilities and shareholders’ equity 15,189 13,924`<br>`Total shareholders’ equity 13,473 11,558` | 85 | **derived by identity:** no total-liabilities line, so total liabilities and shareholders' equity (15,189) − total shareholders' equity (13,473); equals the two uncaptioned liability subtotals 1,392 + 324 |

Balance-sheet check (total liabilities was derived, so the guidance's alternative form): total assets (15,189) = total liabilities and shareholders' equity (15,189) ✓

## Computation

| Ratio | Formula | Value |
|---|---|---|
| X1 | (8,296 − 1,392) / 15,189 | 0.4545 |
| X2 | 2,860 / 15,189 | 0.1883 |
| X3 | 1,468 / 15,189 | 0.0966 |
| X4 | 13,473 / 1,716 | 7.8514 |

Z″ = 6.56 × 0.4545 + 3.26 × 0.1883 + 6.72 × 0.0966 + 1.05 × 7.8514 = 2.982 + 0.614 + 0.649 + 8.244 = **12.489**

## Verdict

- **Z″:** 12.489
- **Zone:** safe
- **Rating:** LOW
- **Fallbacks used:** Total liabilities derived by identity (total liabilities and shareholders' equity − total shareholders' equity); current assets and current liabilities read from uncaptioned subtotals, each checked against its section.
- **Ambiguities — places the rubric did not give a clear answer:**
  - None under the current guidance: it covers uncaptioned subtotals (usable once checked against their section) and identifies retained earnings by role, whatever its caption. (The first version of this reading, written before those guidance changes, listed both as ambiguities.)
- **Read by / date:** Claude (model reading — not a label) / 2026-09-29

## Sensitivity — alternative input choices (not the rubric's)

For comparison with a label computed differently (for example from XBRL tags). The verdict above uses the rubric's choices.

| Alternative | Z″ | Zone |
|---|---|---|
| Rubric choices (above) | 12.489 | safe |
| EBIT = income before income taxes (1,509); no interest expense line is presented | 12.507 | safe |
