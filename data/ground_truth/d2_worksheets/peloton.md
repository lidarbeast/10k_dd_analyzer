# D2 labeling worksheet — Peloton FY2023

- **Filing:** `data/filings/pton-20230630.pdf` — FY2023 (year ended June 30, 2023)
- **Filing SHA-256:** `312efc2529e02cfe2fa486b1e9d5eae325e0766367c59ab7903443dbc668cdd3`
- **SEC CIK:** `1639825`
- **10-K signature date:** August 23, 2023 (PDF p. 192)
- **Rubric version:** `v1`
- **Built by:** `python -m src.evaluation.d2_worksheets` — deterministic extraction; no model was used

## How to use this worksheet

1. Read **Source A** (the auditor's report on the financial statements) in full. Look for a paragraph after the opinion that states substantial doubt exists — usually titled "Going Concern" or "Substantial Doubt…". The heading map under each report helps you find it, but read the text.
2. Read **Source C** (notes subsections) and the entries in **Source B**, which include MD&A. Decide whether management's disclosure — in the notes or MD&A — says doubt exists (not alleviated), says in words that doubt was raised and alleviated, or neither. Judge each passage by its concluding sentence (see the glossary). If Source C is empty, read Note 1 in the PDF.
3. Use **Source D** to decide whether the company was a Chapter 11 debtor-in-possession on the 10-K's signature date.
4. Run the **EDGAR full-text search** cross-check (Q4) and record what it finds.
5. Apply the rubric. Rate only what the filing discloses — not news, later events or anything you know from elsewhere.
6. If the rubric does not give a clear answer, **stop and write it under Ambiguities.** Rubric fixes happen before v1 is frozen, and never after seeing agent output. Do this before looking at any agent output for D2.

## D2 rubric (v1)

**HIGH:** One or more of: (a) the auditor's report on the financial statements contains an AS 2415 going-concern explanatory paragraph stating that substantial doubt about the company's ability to continue as a going concern exists; (b) management states, in the financial statement notes or in MD&A, that there is substantial doubt about the company's ability to continue as a going concern (ASC 205-40-50-13, doubt not alleviated); (c) the company is operating as a debtor-in-possession under Chapter 11 as of the filing date.

**MEDIUM:** No HIGH condition, AND management discloses, in the financial statement notes or in MD&A, that conditions or events raised substantial doubt and that management's plans alleviated it (ASC 205-40-50-12). The disclosure must say that substantial doubt was raised; a description of losses, financing and a conclusion that resources are sufficient, without that statement, is not this disclosure.

**LOW:** No HIGH or MEDIUM condition, AND the retrieved passages include part of the auditor's report on the financial statements: its opinion, basis for opinion or critical audit matters (in a combined report, the same sections). A report on internal control over financial reporting alone, or the report title alone, does not satisfy this.

## Glossary

- **Substantial doubt** (ASC 205-40): conditions and events that make it probable the company will be unable to meet its obligations as they fall due within one year after the financial statements are issued.
- **Auditor's going-concern paragraph** (PCAOB AS 2415.12): a paragraph the auditor adds after the opinion when substantial doubt remains after considering management's plans. It is usually titled "Going Concern" or "Substantial Doubt About the Company's Ability to Continue as a Going Concern" and states that conditions "raise substantial doubt about its ability to continue as a going concern". The opinion itself stays unqualified — the paragraph is the finding, so an unqualified opinion does not mean there is no paragraph.
- **Not alleviated** (ASC 205-40-50-13): management states that there IS substantial doubt ("substantial doubt exists…", "these conditions raise substantial doubt…"), in the notes or in MD&A. → HIGH (b). Plans described as "intended to mitigate" the conditions are part of this disclosure and do not make it MEDIUM.
- **Alleviated** (ASC 205-40-50-12): management says in words that conditions raised substantial doubt and that its plans alleviated it, in the notes or in MD&A. → MEDIUM. A paragraph describing losses, financing and a conclusion that funds are sufficient, without saying substantial doubt was raised, is NOT this — it is the no-doubt outcome.
- **Judge the concluding sentence**: the basis-of-accounting sentence ("prepared assuming the Company will continue as a going concern") and the no-adjustments sentence appear in almost every filing, but they also open and close every real finding. Classify a passage by the sentence that states its conclusion.
- **Sentences about the evaluation**: "we evaluated whether there were conditions and events … that raised substantial doubt", "we assess our ability … for one year from issuance", "our ability to continue … is contingent upon…". These describe the evaluation, not its result.
- **Management's no-doubt outcome**: "management … concluded there are no conditions or events that raise substantial doubt", or "cash and financing are sufficient to fund operations for at least one year from issuance / the next 12 months". Consistent with LOW.
- **Emphasis-of-matter paragraph** (AS 3101.19): an optional auditor paragraph drawing attention to a disclosure (for example liquidity). Not an AS 2415 paragraph unless it states substantial doubt exists.
- **Critical audit matter**: a matter the auditor found especially challenging. Not a going-concern paragraph, even if it concerns liquidity.
- **Internal-control opinion** (AS 2201): the auditor's opinion on internal control over financial reporting, sometimes mentioned inside the financial-statement report ("…expressed an adverse opinion thereon"). Not a going-concern conclusion, adverse or not.
- **Conditional wording**: "if we are unable to raise capital, there may be substantial doubt…" (typically Item 1A). Not a conclusion; AS 2415 forbids conditional wording in an actual one.
- **Debtor-in-possession / Chapter 11**: a company operating under Chapter 11 bankruptcy protection. HIGH (c) needs it to be the case **at the date the 10-K was filed** — a case that ended (plan effective, company emerged) before then does not count; a hearing or emergence scheduled after it does not end it. Hypothetical bankruptcy, "bankruptcy-remote" entities and other parties' bankruptcies do not count.
- **Horizons**: management looks one year from the issuance date; the auditor up to one year beyond the balance-sheet date. Either HIGH source alone is enough.

## Checklist (you fill this in)

- **Q1 — Auditor's going-concern paragraph** (AS 2415): YES / NO
  - Auditor and report date: 
  - If YES, quote and PDF page: 
- **Q2 — Management's disclosure, notes or MD&A** (ASC 205-40): NONE / ALLEVIATED (50-12) / NOT ALLEVIATED (50-13)
  - Quote and PDF page: 
- **Q3 — Chapter 11 debtor-in-possession on the signature date:** YES / NO
  - Evidence and PDF page (petition date; emergence date if any): 
- **Q4 — EDGAR full-text search cross-check:**
  - Link (search for "substantial doubt" in this company's 10-K filings made within six months after fiscal year-end 2023-06-30): <https://www.sec.gov/edgar/search/#/q=%22substantial%20doubt%22&dateRange=custom&ciks=0001639825&startdt=2023-06-30&enddt=2023-12-30&forms=10-K>
  - If the link does not pre-fill: go to https://www.sec.gov/edgar/search/, search `"substantial doubt"`, set the company to CIK 1639825, form 10-K, filed between 2023-06-30 and 2023-12-30.
  - A hit only means the words appear somewhere in the filing — risk factors match too, and EDGAR also searches the filing's exhibits, so check each hit's document type. Record what the hits are, and whether they agree with Q1–Q2: 

## Verdict (you fill this in)

- **Rating:** HIGH / MEDIUM / LOW
- **Deciding criterion:** HIGH (a) auditor paragraph / HIGH (b) not alleviated / HIGH (c) Chapter 11 / MEDIUM alleviated / LOW
- **Deciding passage (verbatim):** 
- **PDF page:** 
- **Ambiguities — places the rubric did not give a clear answer:** 
- **Labeled by / date:** 

## Source A: auditor's report(s) on the financial statements (verbatim)

**Report at PDF pp. 122–123** — opinion(s) on the financial statements. Signed: `/s/ Ernst & Young LLP`; report date: `August 23, 2023`.

Paragraph headings found in this report (a map only, not a finding): `Opinion on the Financial Statements`, `Basis for Opinion`, `Critical Audit Matter`

```text
Report of Independent Registered Public Accounting Firm
To the Stockholders and the Board of Directors of Peloton Interactive, Inc.
Opinion on the Financial Statements
We have audited the accompanying consolidated balance sheets of Peloton Interactive, Inc. (the Company) as of June 30, 2023 and
2022, the related consolidated statements of operations and comprehensive loss, stockholders' equity (deficit) and cash flows for each
of the three years in the period ended June 30, 2023, and the related notes (collectively referred to as the “consolidated financial
statements”). In our opinion, the consolidated financial statements present fairly, in all material respects, the financial position of the
Company at June 30, 2023 and 2022, and the results of its operations and its cash flows for each of the three years in the period ended
June 30, 2023, in conformity with U.S. generally accepted accounting principles.
We also have audited, in accordance with the standards of the Public Company Accounting Oversight Board (United States) (PCAOB),
the Company's internal control over financial reporting as of June 30, 2023, based on criteria established in Internal Control-Integrated
Framework issued by the Committee of Sponsoring Organizations of the Treadway Commission (2013 framework) and our report dated
August 23, 2023 expressed an adverse opinion thereon.
Basis for Opinion
These financial statements are the responsibility of the Company's management. Our responsibility is to express an opinion on the
Company’s financial statements based on our audits. We are a public accounting firm registered with the PCAOB and are required to be
independent with respect to the Company in accordance with the U.S. federal securities laws and the applicable rules and regulations of
the Securities and Exchange Commission and the PCAOB.
We conducted our audits in accordance with the standards of the PCAOB. Those standards require that we plan and perform the audit
to obtain reasonable assurance about whether the financial statements are free of material misstatement, whether due to error or fraud.
Our audits included performing procedures to assess the risks of material misstatement of the financial statements, whether due to error
or fraud, and performing procedures that respond to those risks. Such procedures included examining, on a test basis, evidence
regarding the amounts and disclosures in the financial statements. Our audits also included evaluating the accounting principles used
and significant estimates made by management, as well as evaluating the overall presentation of the financial statements. We believe
that our audits provide a reasonable basis for our opinion.
Critical Audit Matter
The critical audit matter communicated below is a matter arising from the current period audit of the financial statements that was
communicated or required to be communicated to the audit committee and that: (1) relates to accounts or disclosures that are material
to the financial statements and (2) involved our especially challenging, subjective or complex judgments. The communication of the
critical audit matter does not alter in any way our opinion on the consolidated financial statements, taken as a whole, and we are not, by
communicating the critical audit matter below, providing a separate opinion on the critical audit matter or on the account or disclosure to
which it relates.
68
[PDF p. 123]
Tread+ Returns Reserve
Description of the Matter As described in Note 13 of the consolidated financial statements, the Company announced on May
5, 2021 a voluntary recall of its Tread+ product. Return reserves related to the impacts of the
Tread+ recall of $24.4 million were included within Accounts payable and accrued expenses in the
Consolidated Balance Sheets as of June 30, 2023. Accruals for costs associated with the Tread+
repair of $10.0 million were included within Accounts payable and accrued expenses in the
Consolidated Balance Sheets as of June 30, 2023.
Auditing management’s estimate of the reserve related to the Tread+ product recall was especially
challenging due to the high degree of estimation in determining the expected rate of returns or
claims, which can fluctuate based on factors such as timing of relaunch of the products, customer
behavior and other market conditions with uncertain future outcomes.
How We Addressed the Matter in
Our Audit
We obtained an understanding, evaluated the design, and tested the operating effectiveness of the
controls over the process for estimating the reserve. For example, we tested management’s review
control over the estimation process to calculate the returns reserve and the development of the
underlying assumptions utilized.
To test the reserve and cost estimates, our audit procedures included, among others, validating that
the Company’s estimation methodology agrees with the terms and conditions of the announced
Tread+ recall and examining the Company’s key assumptions and judgements. We also performed
sensitivity analyses on the key assumptions to evaluate the effect of such changes on the estimated
reserve, validated the reliability of data used in the calculation, and recalculated the estimated
reserve balance based on the Company’s methodology.
/s/ Ernst & Young LLP
We have served as the Company’s auditor since 2017.
New York, New York
August 23, 2023
```

Not included: reports on internal control only (not D2 evidence): PDF p. 124; report headings with no report under them (an index entry, or a reference to a report printed elsewhere): PDF p. 185.

## Source B: every "substantial doubt" / "going concern" sentence in the filing (pointer only)

Includes boilerplate (the basis-of-accounting sentence) and conditional risk-factor wording. Use it to find pages, then read them in the PDF.

**None found.**

## Source C: notes subsections on basis of presentation / liquidity / going concern (verbatim)

Copied from each matching heading to the next heading-like line (at most 60 lines). The end is found by a rule of thumb — check the PDF page.

**`1. Description of Business and Basis of Presentation`** — PDF p. 131

```text
1. Description of Business and Basis of Presentation
Description and Organization
Peloton Interactive, Inc. (“Peloton” or the “Company”) is the largest interactive fitness platform in the world with a loyal community of
Members, which we define as any individual who has a Peloton account through a paid Connected Fitness Subscription or a paid
Peloton App Membership. The Company pioneered connected, technology-enabled fitness with the creation of its interactive fitness
equipment (“Connected Fitness Products”) and the streaming of immersive, instructor-led boutique classes to its Members anytime,
anywhere. The Company makes fitness entertaining, approachable, effective, and convenient while fostering social connections that
encourage Members to be the best versions of themselves.
```

**`Basis of Presentation and Consolidation`** — PDF pp. 131–132

```text
Basis of Presentation and Consolidation
The accompanying consolidated financial statements have been prepared in accordance with accounting principles generally accepted
in the United States (“GAAP”) and applicable rules  and regulations of the U.S. Securities and Exchange Commission (“SEC”). The
consolidated financial statements include the accounts of Peloton Interactive, Inc. and its subsidiaries in which the Company has a
controlling financial interest. All significant intercompany balances and transactions have been eliminated.
Certain monetary amounts, percentages, and other figures included elsewhere in these financial statements have been subject to
rounding adjustments. Accordingly, figures shown as totals in certain tables may not be the arithmetic aggregation of the figures that
precede them, and figures expressed as percentages in the text may not total 100% or, as applicable, when aggregated may not be the
arithmetic aggregation of the percentages that precede them.
Certain immaterial amounts from prior years have been reclassified to conform with current-year presentation.
76
[PDF p. 132]
```

## Source D: Chapter 11 / bankruptcy mentions (pointer only)

Sentences mentioning Chapter 11, debtor-in-possession, the Bankruptcy Code or a voluntary petition. Hypothetical risk-factor wording matches too.

**None found.**
