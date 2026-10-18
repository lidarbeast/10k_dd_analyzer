# D5 reading — MODEL READING — Shopify FY2025

> **MODEL READING — NOT A LABEL.** Claude's application of the v1 D5 rubric (with the 2026-10-08 guidance), for comparison with the human labels only. It must never be copied into `labels.json`; a disagreement is settled by a human re-reading the filing.

- **Filing:** `data/filings/shop-20251231.pdf` — FY2025 (year ended December 31, 2025)
- **Filing SHA-256:** `6368a3ddaa7fb47d26fbc68fe77ec62c293276fe4c1fdd2ffd8aaf8ac12a6176`
- **Filer status (cover page):** large accelerated filer; Section 404(b) attestation box checked (PDF p. 3)
- **Rubric version:** `v1`
- **Sources A–C:** copied verbatim from the PDF text by deterministic extraction (no model). Checklist and verdict: Claude.

## D5 rubric (v1)

**HIGH:** A material weakness exists at fiscal year-end: management concludes in Item 9A that ICFR was not effective as of fiscal year-end, OR the auditor's AS 2201 opinion is adverse (the company did not maintain effective ICFR). Either conclusion alone is sufficient.

**MEDIUM:** No material weakness at fiscal year-end (management concludes ICFR was effective, and the auditor's AS 2201 opinion, if present, is unqualified), BUT the filing states that a previously reported material weakness was remediated during the fiscal year.

**LOW:** No material weakness at fiscal year-end (management concludes ICFR was effective, or the auditor's AS 2201 opinion is unqualified, and neither conclusion is negative) AND the filing does not report remediation of a previously reported material weakness during the fiscal year.

## Checklist (filled in by Claude)

- **Q1 — Management's ICFR conclusion as of the rated year-end** (S-K Item 308(a)): **EFFECTIVE**
  - Quote and PDF page: "Based on this assessment, management, including the Chief Executive Officer and Chief Financial Officer, have determined that the Company's internal control over financial reporting was effective as of December 31, 2025." (PDF p. 77)
- **Q2 — Auditor's AS 2201 opinion on ICFR:** **UNQUALIFIED**
  - Auditor, report date, location: PricewaterhouseCoopers LLP, February 11, 2026 — one report combining the opinions on the financial statements and on internal control; Item 9A says it appears in Item 15 (PDF pp. 81–84)
  - Quote and PDF page: "Also in our opinion, the Company maintained, in all material respects, effective internal control over financial reporting as of December 31, 2025, based on criteria established in Internal Control ‒ Integrated Framework (2013) issued by the COSO." (PDF p. 81)
  - Cross-reference in the report on the financial statements: Not applicable — the combined report states both opinions in one paragraph.
- **Q3 — Material weaknesses at year-end:** None. "Material weakness" appears once in the filing: the auditor's method wording (PDF p. 81).
- **Q4 — Remediation of a previously reported material weakness completed during the rated year:** **NO**
  - Evidence: "There were no changes in our internal control over financial reporting … that occurred during our most recently completed fiscal quarter that have materially affected, or are reasonably likely to materially affect, our internal control over financial reporting." (PDF p. 77). No remediation is reported anywhere in the filing.
- **Q5 — Disclosure controls and procedures (Item 307; context, not rated):** Effective — "have concluded that the Company's disclosure controls and procedures were effective as of December 31, 2025" (PDF p. 77).
- **Look-alikes checked:** Auditor, p. 81: "assessing the risk that a material weakness exists" — method wording. The combined paragraph's first opinion ("the consolidated financial statements referred to above present fairly") is the financial-statement opinion, not the ICFR opinion.

## Verdict (filled in by Claude)

- **Rating:** **LOW**
- **Deciding criterion:** LOW — management's conclusion is effective and the auditor's opinion is unqualified, and no remediation during FY2025 is reported.
- **Deciding passage (verbatim):** "Based on this assessment, management, including the Chief Executive Officer and Chief Financial Officer, have determined that the Company's internal control over financial reporting was effective as of December 31, 2025."
- **PDF page:** 77 (management); 81 (auditor)
- **Ambiguities — places the rubric did not give a clear answer:** None. Item 9A places the auditor's report in Item 15 rather than Item 8 — a location difference, not a rating issue.
- **Read by / date:** Claude (model reading — not a label) / 2026-10-08

## Source A: Item 9A — Controls and Procedures (verbatim)

```text
Item 9A: Controls and Procedures
All control systems, no matter how well designed, have inherent limitations. Accordingly, even disclosure controls and
procedures and internal controls over financial reporting determined to be effective can only provide reasonable assurance
of achieving their control objectives with respect to financial statement preparation and presentation.
Evaluation of disclosure controls and procedures: Management of the Company, under the supervision of the Chief
Executive Officer and Chief Financial Officer, is responsible for establishing and maintaining
66
[PDF p. 77]
Table of Contents
disclosure controls and procedures (as defined by the SEC in Rule 13a-15(e) under the Exchange Act) for the Company to
ensure that material information relating to the Company, including its consolidated subsidiaries, that is required to be
made known to the Chief Executive Officer and Chief Financial Officer by others within the Company and disclosed by the
Company in reports filed or submitted by it under the Exchange Act is (i) recorded, processed, summarized and reported
within the time periods specified in the SEC's rules and forms; and (ii) accumulated and communicated to the Company's
management, including its Chief Executive Officer and Chief Financial Officer, to allow timely decisions regarding required
disclosure.
Management, including the Chief Executive Officer and Chief Financial Officer, have evaluated the effectiveness of the
Company's disclosure controls and procedures as of December  31, 2025 and have concluded that the Company's
disclosure controls and procedures were effective as of December 31, 2025.
Management’s Annual Report on Internal Control over Financial Reporting
Management of the Company, under the supervision of the Chief Executive Officer and the Chief Financial Officer, is
responsible for establishing and maintaining adequate internal control over the Company's financial reporting. Internal
control over financial reporting is designed to provide reasonable assurance regarding the reliability of financial reporting
and the preparation of financial statements in accordance with United States generally accepted accounting principles.
Management, including the Chief Executive Officer and Chief Financial Officer, have assessed the effectiveness of the
Company's internal control over financial reporting in accordance with Internal Control – Integrated Framework (2013)
issued by the Committee of Sponsoring Organizations of the Treadway Commission. Based on this assessment,
management, including the Chief Executive Officer and Chief Financial Officer, have determined that the Company's
internal control over financial reporting was effective as of December 31, 2025.
The effectiveness of the Company's internal control over financial reporting as of December 31, 2025 has been audited by
PricewaterhouseCoopers LLP, an independent registered public accounting firm, as stated in their report which appears in
Item 15 of this Annual Report on Form 10-K.
Changes in internal control over financial reporting: There were no changes in our internal control over financial
reporting (as defined in Exchange Act Rule 13a15(f)) identified in connection with the evaluation required by Exchange Act
Rules 13a-15(d) or 15d-15(d) that occurred during our most recently completed fiscal quarter that have materially affected,
or are reasonably likely to materially affect, our internal control over financial reporting.
```

## Source B: auditor's opinion on internal control over financial reporting (verbatim, opinion paragraph only)

```text
Opinions on the Financial Statements and Internal Control over Financial Reporting
We have audited the accompanying consolidated balance sheets of Shopify Inc. and its subsidiaries (the Company) as of
December  31, 2025 and 2024, and the related consolidated statements of operations and comprehensive income, of
changes in shareholders’ equity and of cash flows for each of the three years in the period ended December 31, 2025,
including the related notes (collectively referred to as the consolidated financial statements). We also have audited the
Company’s internal control over financial reporting as of December  31, 2025, based on criteria established in Internal
Control ‒ Integrated Framework (2013) issued by the Committee of Sponsoring Organizations of the Treadway
Commission (COSO).
In our opinion, the consolidated financial statements referred to above present fairly, in all material respects, the financial
position of the Company as of December 31, 2025 and 2024, and the results of its operations and its cash flows for each of
the three years in the period ended December 31, 2025 in conformity with accounting principles generally accepted in the
United States of America. Also in our opinion, the Company maintained, in all material respects, effective internal control
over financial reporting as of December 31, 2025, based on criteria established in Internal Control ‒ Integrated Framework
(2013) issued by the COSO.
```

## Source C: every sentence mentioning "material weakness" in the filing (pointer only)

Sentences on a page of Source A or B are marked by page, so a marked sentence may still sit outside the conclusion (for example the auditor's method wording). Everything else is outside the conclusions — Item 1A risk factors, the auditor's method wording, definitions — and decides nothing on its own.

- p. 81 *(page of the auditor's ICFR opinion)* — Our audit of internal control over financial reporting included obtaining an understanding of internal control over financial reporting, assessing the risk that a material weakness exists, and testing and evaluating the design and operating effectiveness of internal control based on the assessed risk.
