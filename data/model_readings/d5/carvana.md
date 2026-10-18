# D5 reading — MODEL READING — Carvana FY2022

> **MODEL READING — NOT A LABEL.** Claude's application of the v1 D5 rubric (with the 2026-10-08 guidance), for comparison with the human labels only. It must never be copied into `labels.json`; a disagreement is settled by a human re-reading the filing.

- **Filing:** `data/filings/cvna-20221231.pdf` — FY2022 (year ended December 31, 2022)
- **Filing SHA-256:** `ab35659411bad5ebfdd19b76420a3af5871299bb46054a32de2279617dbd0982`
- **Filer status (cover page):** large accelerated filer; Section 404(b) attestation box checked (PDF p. 2)
- **Rubric version:** `v1`
- **Sources A–C:** copied verbatim from the PDF text by deterministic extraction (no model). Checklist and verdict: Claude.

## D5 rubric (v1)

**HIGH:** A material weakness exists at fiscal year-end: management concludes in Item 9A that ICFR was not effective as of fiscal year-end, OR the auditor's AS 2201 opinion is adverse (the company did not maintain effective ICFR). Either conclusion alone is sufficient.

**MEDIUM:** No material weakness at fiscal year-end (management concludes ICFR was effective, and the auditor's AS 2201 opinion, if present, is unqualified), BUT the filing states that a previously reported material weakness was remediated during the fiscal year.

**LOW:** No material weakness at fiscal year-end (management concludes ICFR was effective, or the auditor's AS 2201 opinion is unqualified, and neither conclusion is negative) AND the filing does not report remediation of a previously reported material weakness during the fiscal year.

## Checklist (filled in by Claude)

- **Q1 — Management's ICFR conclusion as of the rated year-end** (S-K Item 308(a)): **EFFECTIVE**
  - Quote and PDF page: "Based on the assessment, management has concluded that its internal control over financial reporting was effective as of December 31, 2022 to provide reasonable assurance regarding the reliability of financial reporting and the preparation of financial statements in accordance with U.S. GAAP." (PDF p. 132)
- **Q2 — Auditor's AS 2201 opinion on ICFR:** **UNQUALIFIED**
  - Auditor, report date, location: Grant Thornton LLP, February 23, 2023 — a separate report on internal control, in Item 8 (PDF p. 82)
  - Quote and PDF page: "In our opinion, the Company maintained, in all material respects, effective internal control over financial reporting as of December 31, 2022, based on criteria established in the 2013 Internal Control—Integrated Framework issued by COSO." (PDF p. 82)
  - Cross-reference in the report on the financial statements: "…our report dated February 23, 2023, expressed an unqualified opinion." (PDF p. 80, about internal control) — consistent with Q2.
- **Q3 — Material weaknesses at year-end:** None. "Material weakness" appears on two pages only: p. 53 (Item 1A, hypothetical) and p. 82 (the auditor's method wording).
- **Q4 — Remediation of a previously reported material weakness completed during the rated year:** **NO**
  - Evidence: "There was no change in our internal control over financial reporting identified in management's evaluation during the fourth quarter of 2022 that materially affected, or is reasonably likely to materially affect, our internal control over financial reporting." (PDF p. 132). No remediation is reported anywhere in the filing.
- **Q5 — Disclosure controls and procedures (Item 307; context, not rated):** Effective — "our chief executive officer and chief financial officer concluded that our disclosure controls and procedures were effective as of such date" (PDF p. 132).
- **Look-alikes checked:** Item 1A, p. 53: "If we fail to remediate material weaknesses in our internal control over financial reporting…" — hypothetical, decides nothing. Auditor, p. 82: "assessing the risk that a material weakness exists" — method wording, not a finding.

## Verdict (filled in by Claude)

- **Rating:** **LOW**
- **Deciding criterion:** LOW — management's conclusion is effective and the auditor's opinion is unqualified, and no remediation during FY2022 is reported.
- **Deciding passage (verbatim):** "Based on the assessment, management has concluded that its internal control over financial reporting was effective as of December 31, 2022 to provide reasonable assurance regarding the reliability of financial reporting and the preparation of financial statements in accordance with U.S. GAAP."
- **PDF page:** 132 (management); 82 (auditor)
- **Ambiguities — places the rubric did not give a clear answer:** None.
- **Read by / date:** Claude (model reading — not a label) / 2026-10-08

## Source A: Item 9A — Controls and Procedures (verbatim)

```text
ITEM 9A. CONTROLS AND PROCEDURES.
Evaluation of Disclosure Controls and Procedures
Under the supervision and with the participation of our management, including the chief executive officer and chief financial officer, we
conducted an evaluation of the effectiveness of our disclosure controls and procedures (as such term is defined in Rules 13a-15(e) and 15d-15(e)
under the Exchange Act) as of the end of the period covered by this report. Based on this evaluation, our chief executive officer and chief
financial officer concluded that our disclosure controls and procedures were effective as of such date. Our disclosure controls and procedures are
designed to ensure that information required to be disclosed in the reports we file or submit under the Exchange Act is recorded, processed,
summarized and reported within the time periods specified in the SEC’s rules and forms and that such information is accumulated and
communicated to management, including the chief executive officer and chief financial officer, to allow timely decisions regarding required
disclosure.
Management's Report on Internal Controls over Financial Reporting
Our management is responsible for establishing and maintaining adequate internal control over financial reporting (as defined in Rules 13a￾15(f) and 15d-15(f) under the Exchange Act). Management conducted an evaluation of the effectiveness of our internal control over financial
reporting based on the framework in Internal Control — Integrated Framework (2013) issued by the Committee of Sponsoring Organizations of
the Treadway Commission. Based on the assessment, management has concluded that its internal control over financial reporting was effective
as of December 31, 2022 to provide reasonable assurance regarding the reliability of financial reporting and the preparation of financial
statements in accordance with U.S. GAAP. Our independent registered public accounting firm, Grant Thornton LLP, has issued an audit report
with respect to our internal control over financial reporting, which appears in Part II, Item 8 of this Annual Report on Form 10-K.
Changes in Internal Controls Over Financial Reporting
There was no change in our internal control over financial reporting identified in management's evaluation during the fourth quarter of 2022
that materially affected, or is reasonably likely to materially affect, our internal control over financial reporting.
Limitations of Effectiveness of Controls and Procedures and Internal Control over Financial Reporting
In designing and evaluating the disclosure controls and procedures and internal control over financial reporting, management recognizes
that any controls and procedures, no matter how well designed and operated, can provide only reasonable assurance of achieving the desired
control objectives. In addition, the design of disclosure controls and procedures and internal control over financial reporting must reflect the fact
that there are resource constraints and that management is required to apply judgment in evaluating the benefits of possible controls and
procedures relative to their costs.
```

## Source B: auditor's opinion on internal control over financial reporting (verbatim, opinion paragraph only)

```text
Opinion on internal control over financial reporting
We have audited the internal control over financial reporting of Carvana Co. (a Delaware corporation) and subsidiaries (the “Company”) as of
December 31, 2022, based on criteria established in the 2013 Internal Control—Integrated Framework issued by the Committee of Sponsoring
Organizations of the Treadway Commission (“COSO”). In our opinion, the Company maintained, in all material respects, effective internal
control over financial reporting as of December 31, 2022, based on criteria established in the 2013 Internal Control—Integrated Framework
issued by COSO.
We also have audited, in accordance with the standards of the Public Company Accounting Oversight Board (United States) (“PCAOB”), the
consolidated financial statements of the Company as of and for the year ended December 31, 2022, and our report dated February 23, 2023
expressed an unqualified opinion on those financial statements.
```

## Source C: every sentence mentioning "material weakness" in the filing (pointer only)

Sentences on a page of Source A or B are marked by page, so a marked sentence may still sit outside the conclusion (for example the auditor's method wording). Everything else is outside the conclusions — Item 1A risk factors, the auditor's method wording, definitions — and decides nothing on its own.

- p. 53 — If we fail to remediate material weaknesses in our internal control over financial reporting or otherwise establish and maintain effective internal control over financial reporting and disclosure controls and procedures, we may not be able to accurately report our financial results, or report them in a timely manner.
- p. 53 — If our senior management is unable to conclude that we have effective internal control over financial reporting or to certify the effectiveness of such controls; if our independent registered public accounting firm cannot render an unqualified opinion on management's assessment and the effectiveness of our internal control over financial reporting; or if material weaknesses in our internal control over financial reporting is identified, we could be subject to regulatory scrutiny, a loss of public and investor confidence, and to litigation from investors and stockholders, which could have a material adverse effect on our business and our stock price.
- p. 82 *(page of the auditor's ICFR opinion)* — Our audit included obtaining an understanding of internal control over financial reporting, assessing the risk that a material weakness exists, testing and evaluating the design and operating effectiveness of internal control based on the assessed risk, and performing such other procedures as we considered necessary in the circumstances.
