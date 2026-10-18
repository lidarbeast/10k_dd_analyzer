# D2 labeling worksheet — Shopify FY2025

- **Filing:** `data/filings/shop-20251231.pdf` — FY2025 (year ended December 31, 2025)
- **Filing SHA-256:** `6368a3ddaa7fb47d26fbc68fe77ec62c293276fe4c1fdd2ffd8aaf8ac12a6176`
- **SEC CIK:** `1594805`
- **10-K signature date:** February 11, 2026 (PDF p. 132)
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
  - Link (search for "substantial doubt" in this company's 10-K filings made within six months after fiscal year-end 2025-12-31): <https://www.sec.gov/edgar/search/#/q=%22substantial%20doubt%22&dateRange=custom&ciks=0001594805&startdt=2025-12-31&enddt=2026-07-02&forms=10-K>
  - If the link does not pre-fill: go to https://www.sec.gov/edgar/search/, search `"substantial doubt"`, set the company to CIK 1594805, form 10-K, filed between 2025-12-31 and 2026-07-02.
  - A hit only means the words appear somewhere in the filing — risk factors match too, and EDGAR also searches the filing's exhibits, so check each hit's document type. Record what the hits are, and whether they agree with Q1–Q2: 

## Verdict (you fill this in)

- **Rating:** HIGH / MEDIUM / LOW
- **Deciding criterion:** HIGH (a) auditor paragraph / HIGH (b) not alleviated / HIGH (c) Chapter 11 / MEDIUM alleviated / LOW
- **Deciding passage (verbatim):** 
- **PDF page:** 
- **Ambiguities — places the rubric did not give a clear answer:** 
- **Labeled by / date:** 

## Source A: auditor's report(s) on the financial statements (verbatim)

**Report at PDF pp. 81–84** — opinion(s) on the financial statements and internal control over financial reporting. Signed: `/s/ PricewaterhouseCoopers LLP`; report date: `February 11, 2026`.

Paragraph headings found in this report (a map only, not a finding): `Opinions on the Financial Statements and Internal Control over Financial Reporting`, `Basis for Opinions`, `Definition and Limitations of Internal Control over Financial Reporting`, `Critical Audit Matters`

```text
Report of Independent Registered Public Accounting Firm
To the Board of Directors and Shareholders of Shopify Inc.
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
Basis for Opinions
The Company’s management is responsible for these consolidated financial statements, for maintaining effective internal
control over financial reporting, and for its assessment of the effectiveness of internal control over financial reporting,
included in Management’s Annual Report on Internal Control over Financial Reporting appearing under Item 9A. Our
responsibility is to express opinions on the Company’s consolidated financial statements and on the Company’s internal
control over financial reporting based on our audits. We are a public accounting firm registered with the Public Company
Accounting Oversight Board (United States) (PCAOB) and are required to be independent with respect to the Company in
accordance with the U.S. federal securities laws and the applicable rules and regulations of the Securities and Exchange
Commission and the PCAOB.
We conducted our audits in accordance with the standards of the PCAOB. Those standards require that we plan and
perform the audits to obtain reasonable assurance about whether the consolidated financial statements are free of material
misstatement, whether due to error or fraud, and whether effective internal control over financial reporting was maintained
in all material respects.
Our audits of the consolidated financial statements included performing procedures to assess the risks of material
misstatement of the consolidated financial statements, whether due to error or fraud, and performing procedures that
respond to those risks. Such procedures included examining, on a test basis, evidence regarding the amounts and
disclosures in the consolidated financial statements. Our audits also included evaluating the accounting principles used
and significant estimates made by management, as well as evaluating the overall presentation of the consolidated financial
statements. Our audit of internal control over financial reporting included obtaining an understanding of internal control
over financial reporting, assessing the risk that a material weakness exists, and testing and evaluating the design and
operating effectiveness of internal control based on the assessed risk. Our audits also included performing such other
procedures as we considered necessary in the circumstances. We believe that our audits provide a reasonable basis for
our opinions.
Definition and Limitations of Internal Control over Financial Reporting
A company’s internal control over financial reporting is a process designed to provide reasonable assurance regarding the
reliability of financial reporting and the preparation of financial statements for
[PDF p. 82]
71
[PDF p. 83]
external purposes in accordance with generally accepted accounting principles. A company’s internal control over financial
reporting includes those policies and procedures that (i) pertain to the maintenance of records that, in reasonable detail,
accurately and fairly reflect the transactions and dispositions of the assets of the company; (ii) provide reasonable
assurance that transactions are recorded as necessary to permit preparation of financial statements in accordance with
generally accepted accounting principles, and that receipts and expenditures of the company are being made only in
accordance with authorizations of management and directors of the company; and (iii) provide reasonable assurance
regarding prevention or timely detection of unauthorized acquisition, use, or disposition of the company’s assets that could
have a material effect on the financial statements.
Because of its inherent limitations, internal control over financial reporting may not prevent or detect misstatements. Also,
projections of any evaluation of effectiveness to future periods are subject to the risk that controls may become inadequate
because of changes in conditions, or that the degree of compliance with the policies or procedures may deteriorate.
Critical Audit Matters
The critical audit matter communicated below is a matter arising from the current period audit of the consolidated financial
statements that was communicated or required to be communicated to the audit committee and that (i) relates to accounts
or disclosures that are material to the consolidated financial statements and (ii) involved our especially challenging,
subjective, or complex judgments. The communication of critical audit matters does not alter in any way our opinion on the
consolidated financial statements, taken as a whole, and we are not, by communicating the critical audit matter below,
providing a separate opinion on the critical audit matter or on the accounts or disclosures to which it relates.
Revenue Recognition — Principal versus Agent Considerations
As described in Note 3 to the consolidated financial statements, management follows the guidance provided in ASC 606,
Revenue from Contracts with Customers, for determining whether the Company is the principal or an agent in
arrangements with customers that involve another party that contributes to providing a specified service to a customer. In
these instances, management determines whether the Company has promised to provide the service itself (as principal) or
to arrange for the specified service to be provided by another party (as an agent). As disclosed by management, this
determination is a matter of significant judgment that depends on the facts and circumstances of each arrangement. The
Company recognizes revenue from the sale of shipping labels, the sale of apps, the sale of themes, card services and
installments services on a net basis as the Company is the agent in the arrangement with customers. All other revenue is
reported on a gross basis, as management has determined it is the principal in the respective arrangements. Revenue
reported on a gross basis makes up a significant portion of total revenues of $11,556 million.
The principal considerations for our determination that performing procedures relating to Revenue Recognition – Principal
versus Agent Considerations is a critical audit matter are (i) that there was significant judgment applied by management, in
some instances, in assessing whether the Company (a) was primarily responsible for the fulfillment of the specified
service, (b) had control of the specified service before it was transferred to the customer, and (c) had full discretion in
establishing the price for the specified service; and (ii) a high degree of auditor judgment, subjectivity and effort in
performing audit procedures and evaluating the results of those procedures.
Addressing the matter involved performing procedures and evaluating audit evidence in connection with forming our overall
opinion on the consolidated financial statements. These procedures included testing the effectiveness of controls relating
to management’s determination as to whether the Company had promised to provide the service as principal or as an
agent. These procedures also included, among others, testing the reasonableness of management’s determination as to
whether the Company had promised to provide the service as principal or as an agent, which included assessing whether
the Company was primarily responsible for the fulfillment of the specified service, had control of the
72
[PDF p. 84]
specified service before it was transferred to the customer, and had full discretion in establishing the price for the specified
service by considering the contractual terms with customers, on a sample basis, and agreements with service providers,
where applicable, and considering whether these conclusions were consistent with evidence obtained in other areas of the
audit.
/s/ PricewaterhouseCoopers LLP
Chartered Professional Accountants, Licensed Public Accountants
Ottawa, Canada
February 11, 2026
We have served as the Company’s auditor since 2011.
```

## Source B: every "substantial doubt" / "going concern" sentence in the filing (pointer only)

Includes boilerplate (the basis-of-accounting sentence) and conditional risk-factor wording. Use it to find pages, then read them in the PDF.

**None found.**

## Source C: notes subsections on basis of presentation / liquidity / going concern (verbatim)

Copied from each matching heading to the next heading-like line (at most 60 lines). The end is found by a rule of thumb — check the PDF page.

**`2. Basis of Presentation and Consolidation`** — PDF p. 89

```text
2. Basis of Presentation and Consolidation
These consolidated financial statements include the accounts of the Company and its directly and indirectly held wholly
owned subsidiaries including, but not limited to: Shopify LLC and Shopify Holdings (USA) 2 Inc., both incorporated in the
state of Delaware in the United States; Shopify International Limited, incorporated in Ireland; Shopify Commerce Singapore
Pte. Ltd., incorporated in Singapore. All intercompany accounts and transactions have been eliminated upon consolidation.
These consolidated financial statements of the Company have been presented in United States dollars ("USD") and have
been prepared in accordance with accounting principles generally accepted in the United States of America ("U.S. GAAP"),
including the applicable rules and regulations of the Securities and Exchange Commission ("SEC") regarding financial
reporting.
```

## Source D: Chapter 11 / bankruptcy mentions (pointer only)

Sentences mentioning Chapter 11, debtor-in-possession, the Bankruptcy Code or a voluntary petition. Hypothetical risk-factor wording matches too.

**None found.**
