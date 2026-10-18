# D2 worksheet — MODEL READING — Carvana FY2022

> **MODEL READING — NOT A LABEL.** Claude's application of the v1 D2 rubric (with the 2026-10-01 clarifications), filled in on a freshly generated copy of the worksheet for comparison with the human worksheet. It must never be copied into `labels.json`; a disagreement is settled by a human re-reading the filing.


- **Filing:** `data/filings/cvna-20221231.pdf` — FY2022 (year ended December 31, 2022)
- **Filing SHA-256:** `ab35659411bad5ebfdd19b76420a3af5871299bb46054a32de2279617dbd0982`
- **SEC CIK:** `1690820`
- **10-K signature date:** February 23, 2023 (PDF p. 140)
- **Rubric version:** `v1`
- **Built by:** `python -m src.evaluation.d2_worksheets` — deterministic extraction; no model was used

## How to use this worksheet

1. Read **Source A** (the auditor's report on the financial statements) in full. Look for a paragraph after the opinion that states substantial doubt exists — usually titled "Going Concern" or "Substantial Doubt…". The heading map under each report helps you find it, but read the text.
2. Read **Source C** (notes subsections) and the notes entries in **Source B**. Decide whether management's disclosure says doubt exists (not alleviated), was raised and alleviated, or neither. If Source C is empty, read Note 1 in the PDF.
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
- **Auditor's going-concern paragraph** (PCAOB AS 2415.12): a paragraph the auditor adds after the opinion when substantial doubt remains after considering management's plans. It is usually titled "Going Concern" or "Substantial Doubt About the Company's Ability to Continue as a Going Concern" and states that conditions "raise substantial doubt about its ability to continue as a going concern". The opinion itself stays unqualified — the paragraph is the finding.
- **Not alleviated** (ASC 205-40-50-13): management states in the notes that there IS substantial doubt ("these conditions raise substantial doubt…"). → HIGH (b).
- **Alleviated** (ASC 205-40-50-12): management discloses that conditions raised substantial doubt but that its plans alleviate it; the notes then describe the conditions and the plans, without saying doubt exists. → MEDIUM.
- **Basis-of-accounting sentence**: "prepared assuming the Company will continue as a going concern" or "contemplate continuation as a going concern". Standard wording in almost every filing — not a finding.
- **Management's no-doubt conclusion**: "management evaluated … and concluded there are no conditions or events that raise substantial doubt". A LOW signal, not a going-concern disclosure.
- **Emphasis-of-matter paragraph** (AS 3101.19): an optional auditor paragraph drawing attention to a disclosure (for example liquidity). Not an AS 2415 paragraph unless it states substantial doubt exists.
- **Critical audit matter**: a matter the auditor found especially challenging. Not a going-concern paragraph, even if it concerns liquidity.
- **Conditional wording**: "if we are unable to raise capital, there may be substantial doubt…" (typically Item 1A). Not a conclusion; AS 2415 forbids conditional wording in an actual one.
- **Debtor-in-possession / Chapter 11**: a company operating under Chapter 11 bankruptcy protection. HIGH (c) needs it to be the case **at the date the 10-K was filed** — a case that ended (plan effective, company emerged) before then does not count.
- **Horizons**: management looks one year from the issuance date; the auditor up to one year beyond the balance-sheet date. Either HIGH source alone is enough.

## Checklist (filled in by Claude)

- **Q1 — Auditor's going-concern paragraph** (AS 2415): **NO**
  - Auditor and report date: Grant Thornton LLP (Southfield, Michigan), February 23, 2023 — report on the financial statements, PDF pp. 80–81. Headings: Opinion on the financial statements · Basis for opinion · Critical audit matters (derecognition of transferred finance receivables; ADESA acquisition fair value). No going-concern or emphasis-of-matter paragraph; neither critical audit matter concerns liquidity.
  - If YES, quote and PDF page: —
- **Q2 — Management's disclosure in the notes** (ASC 205-40): **NONE**
  - Quote and PDF page: Note 2 'Liquidity', PDF pp. 88–89: "The accompanying consolidated financial statements of the Company have been prepared in conformity with U.S. GAAP, which contemplate continuation of the Company as a going concern. Since inception, the Company has incurred losses, and expects to incur additional losses in the future …" [equity offering, 2030 Notes, forward-flow and floor-plan facilities] "… Management believes that current working capital, results of operations, and existing financing arrangements are sufficient to fund operations for at least one year from the financial statement issuance date." Basis-of-accounting sentence plus a sufficiency conclusion; it never says substantial doubt was raised, so it is the no-doubt outcome, not 50-12 (rubric MEDIUM, clarified 2026-10-01). No MD&A going-concern statement either (MD&A p. 72 is a 12-month sufficiency statement).
- **Q3 — Chapter 11 debtor-in-possession on the signature date:** **NO**
  - Evidence and PDF page (petition date; emergence date if any): No mention of the company in bankruptcy anywhere in the filing. The only bankruptcy wording is 'bankruptcy remote subsidiaries' (p. 16, a financing term), a web-hosting provider's possible bankruptcy (p. 35, risk factor) and transfer-accounting language (p. 93).
- **Q4 — EDGAR full-text search cross-check:**
  - Link (search for "substantial doubt" in this company's 10-K filings made within six months after fiscal year-end 2022-12-31): <https://www.sec.gov/edgar/search/#/q=%22substantial%20doubt%22&dateRange=custom&ciks=0001690820&startdt=2022-12-31&enddt=2023-07-02&forms=10-K>
  - If the link does not pre-fill: go to https://www.sec.gov/edgar/search/, search `"substantial doubt"`, set the company to CIK 1690820, form 10-K, filed between 2022-12-31 and 2023-07-02.
  - A hit only means the words appear somewhere in the filing (risk factors match too). Record what the hits are, and whether they agree with Q1–Q2: not run: Claude has no EDGAR access in this session (no `SEC_USER_AGENT`, no browser). Expectation from reading the PDF: the main 10-K text contains no "substantial doubt" anywhere, so any hit should come from an exhibit or another filing — check the document type on each hit.

## Verdict (filled in by Claude)

- **Rating:** **LOW**
- **Deciding criterion:** LOW — no HIGH or MEDIUM condition, and the auditor's report on the financial statements was read in full
- **Deciding passage (verbatim):** "In our opinion, the financial statements present fairly, in all material respects, the financial position of the Company as of December 31, 2022 and 2021, and the results of its operations and its cash flows for each of the three years in the period ended December 31, 2022, in conformity with accounting principles generally accepted in the United States of America." — followed directly by 'Basis for opinion'; no going-concern paragraph in the report.
- **PDF page:** 80 (opinion); 88–89 (Liquidity paragraph, the no-doubt outcome)
- **Ambiguities — places the rubric did not give a clear answer:** The Note 2 'Liquidity' paragraph has the shape of a 50-12 disclosure (conditions → plans → one-year sufficiency on ASC 205-40's own horizon) without its words. Under the original v1 wording this was the one place a careful reader could land on MEDIUM; the 2026-10-01 clarification (MEDIUM requires the words 'substantial doubt was raised … alleviated') decides it as LOW. Context, not a rating input: Z″ is deep in distress (D1), but the large debt (2030 Notes) falls outside the one-year horizon.
- **Labeled by / date:** Claude (model reading, not a label), 2026-10-08

## Source A: auditor's report(s) on the financial statements (verbatim)

**Report at PDF pp. 80–81** — opinion(s) on the financial statements. Signed: `/s/ GRANT THORNTON LLP`; report date: `February 23, 2023`.

Paragraph headings found in this report (a map only, not a finding): `Opinion on the financial statements`, `Basis for opinion`, `Critical audit matters`

```text
REPORT OF INDEPENDENT REGISTERED PUBLIC ACCOUNTING FIRM
Board of Directors and Shareholders
Carvana Co.
Opinion on the financial statements
We have audited the accompanying consolidated balance sheets of Carvana Co. (a Delaware corporation) and subsidiaries (the “Company”) as of
December 31, 2022 and 2021, the related consolidated statements of operations, changes in stockholders’ equity (deficit), and cash flows for
each of the three years in the period ended December 31, 2022, and the related notes and financial statement schedules included under Item 15(a)
(collectively referred to as the “financial statements”). In our opinion, the financial statements present fairly, in all material respects, the financial
position of the Company as of December 31, 2022 and 2021, and the results of its operations and its cash flows for each of the three years in the
period ended December 31, 2022, in conformity with accounting principles generally accepted in the United States of America.
We also have audited, in accordance with the standards of the Public Company Accounting Oversight Board (United States) (“PCAOB”), the
Company’s internal control over financial reporting as of December 31, 2022, based on criteria established in the 2013 Internal Control—
Integrated Framework issued by the Committee of Sponsoring Organizations of the Treadway Commission (“COSO”), and our report dated
February 23, 2023, expressed an unqualified opinion.
Basis for opinion
These financial statements are the responsibility of the Company’s management. Our responsibility is to express an opinion on the Company’s
financial statements based on our audits. We are a public accounting firm registered with the PCAOB and are required to be independent with
respect to the Company in accordance with the U.S. federal securities laws and the applicable rules and regulations of the Securities and
Exchange Commission and the PCAOB.
We conducted our audits in accordance with the standards of the PCAOB. Those standards require that we plan and perform the audit to obtain
reasonable assurance about whether the financial statements are free of material misstatement, whether due to error or fraud. Our audits included
performing procedures to assess the risks of material misstatement of the financial statements, whether due to error or fraud, and performing
procedures that respond to those risks. Such procedures included examining, on a test basis, evidence regarding the amounts and disclosures in
the financial statements. Our audits also included evaluating the accounting principles used and significant estimates made by management, as
well as evaluating the overall presentation of the financial statements. We believe that our audits provide a reasonable basis for our opinion.
Critical audit matters
The critical audit matters communicated below are matters arising from the current period audit of the financial statements that were
communicated or required to be communicated to the audit committee and that: (1) relate to accounts or disclosures that are material to the
financial statements and (2) involved our especially challenging, subjective, or complex judgments. The communication of critical audit matters
does not alter in any way our opinion on the financial statements, taken as a whole, and we are not, by communicating the critical audit matters
below, providing separate opinions on the critical audit matters or on the accounts or disclosures to which they relate.
Derecognition of Transferred Finance Receivables
As described further in Notes 2, 8, and 9 to the financial statements, the Company is party to various transfer agreements pursuant to which it
sells finance receivables meeting specified underwriting criteria to financing partners. The Company also transfers its finance receivables to
securitization trusts as asset backed securitization transactions. In accordance with Accounting Standards Codification (ASC) 860, Transfers and
Servicing (“ASC 860”), the Company's transfer of finance receivables to financing partners and asset backed securitizations are considered sales
of financial assets, in which the Company derecognizes the finance receivables upon the completion of the transfer, as presented within the
statements of cash flows. We identified the determination that transfers of finance receivables meet the derecognition criteria of ASC 860 as a
critical audit matter.
The principal considerations for our determination that transfers of finance receivables meet the derecognition criteria of ASC 860 is a critical
audit matter is because the transfers of material finance receivables from the Company to both its financing partners and securitization trusts
requires complex auditor judgments to determine that these transactions meet the derecognition criteria of ASC 860, specifically regarding the
legal isolation of the transferred finance receivables from the Company.
76
[PDF p. 81]
Our audit procedures related to the determination that transfers of finance receivables meet the derecognition criteria of ASC 860 included the
following, among others:
• We tested the design and operating effectiveness of management's review control over the accounting determination that the transfers of
finance receivables meet the derecognition criteria set forth in ASC 860.
• We read the various transfer and sale agreements between the Company and its financing partners and securitization trusts, assessed the
true sale and non-consolidation legal opinions, and evaluated the information with respect to management's analysis of the criteria set
forth in ASC 860 to permit the derecognition of the finance receivables.
Fair value of property and equipment acquired in the ADESA US Auction, LLC acquisition
As described further in Note 3 to the financial statements, on May 9, 2022, the Company completed the acquisition of 100% of the equity
interests in the United States physical auction business of ADESA US Auction, LLC for approximately $2.2 billion. The assets acquired and
liabilities assumed were recorded at their respective fair value at the acquisition date. The acquisition included auction sites throughout the
United States. The Company utilized a third-party valuation firm to determine the fair value of the acquired property and equipment, which
included the land and buildings of the auction sites. The acquisition date fair value of the acquired land and buildings comprised $1.3 billion of
the allocation of the fair value of the consideration. We identified the estimation of the acquisition date fair value of the acquired land and
buildings included in property and equipment as a critical audit matter.
The principal consideration for our determination that the acquisition date fair value of the acquired land and buildings included in property and
equipment is a critical audit matter is that auditing the determination of the acquisition date fair values required the use of those with specialized
skill and knowledge to evaluate the inputs and assumptions in order to conclude on the appropriateness of management's acquisition date fair
value of the acquired land and buildings.
Our audit procedures related to the acquisition date fair value of the acquired land and buildings included in property and equipment included the
following, among others:
• We tested the design and operating effectiveness of management's review control evaluating the assumptions applied by the third-party
valuation firm, and the determination of fair value of the acquired land and buildings.
• We utilized a valuation specialist to test the reasonableness of the specialist's conclusions for a selection of properties as of the
acquisition date. This included evaluating:
◦ the estimated fair values for a selection of acquired land, which were valued based on comparable market data. The valuation
specialist performed independent market research to test the reasonableness of the specialist's fair value conclusions of the
selected acquired land;
◦ the estimated fair values for a selection of acquired buildings and site improvements, which were valued based on the
estimated replacement cost less physical depreciation. The valuation specialist performed independent market research to test
the base costs used in the appraisal report for the buildings and associated site improvements at the selected acquired buildings.
/s/ GRANT THORNTON LLP
We have served as the Company’s auditor since 2015.
Southfield, Michigan
February 23, 2023
```

Not included: reports on internal control only (not D2 evidence): PDF p. 82.

## Source B: every "substantial doubt" / "going concern" sentence in the filing (pointer only)

Includes boilerplate (the basis-of-accounting sentence) and conditional risk-factor wording. Use it to find pages, then read them in the PDF.

- p. 88 — GAAP, which contemplate continuation of the Company as a going concern.

## Source C: notes subsections on basis of presentation / liquidity / going concern (verbatim)

Copied from each matching heading to the next heading-like line (at most 60 lines). The end is found by a rule of thumb — check the PDF page.

**`Basis of Presentation`** — PDF p. 88

```text
Basis of Presentation
The accompanying consolidated financial statements of the Company have been prepared in accordance with accounting principles
generally accepted in the United States of America ("U.S. GAAP"). As discussed in Note 1 — Business Organization, Carvana Group is
considered a VIE and Carvana Co. consolidates its financial results due to the determination that it is the primary beneficiary. All intercompany
balances and transactions have been eliminated.
```

**`Liquidity`** — PDF pp. 88–89

```text
Liquidity
The accompanying consolidated financial statements of the Company have been prepared in conformity with U.S. GAAP, which
contemplate continuation of the Company as a going concern. Since inception, the Company has incurred losses, and expects to incur additional
losses in the future as it continues to build inspection and reconditioning centers ("IRCs") and vending machines, serve more of the U.S.
population, and enhance technology and software. In the second quarter of 2022, the Company completed an equity offering of 15.6 million
shares of Class A common stock for net proceeds of $1.2 billion and issued a total of $3.275 billion in aggregate principal amount of 10.25%
senior unsecured notes due 2030 (the "2030 Notes"). The Company used a portion of the net proceeds from the Class A common stock offering
for general corporate purposes and to pay any costs, fees, and expenses incurred by it in connection with the offering. The Company used the net
proceeds from the issuance and sale of the 2030 Notes (a) to finance the $2.2 billion acquisition of the U.S. physical auction business of ADESA
U.S. Auction, LLC ("ADESA") and other ancillary transactions in connection therewith, and to pay related fees and expenses in
84
[PDF p. 89]
CARVANA CO. AND SUBSIDIARIES
NOTES TO CONSOLIDATED FINANCIAL STATEMENTS
(Continued)
connection therewith, and (b) for working capital, capital expenditures, and other general corporate purposes. In March 2022, the Company's
forward flow partner committed to purchase a total of $5.0 billion of the Company's finance receivables through March 2023, and such facility
had approximately $1.2 billion of unused capacity as of December 31, 2022. In January 2023, the Company and its forward flow partner
amended the commitment to purchase a total of $4.0 billion of the Company's finance receivables through January 2024. In addition, the
Company has a $2.2 billion floor plan facility through September 22, 2023, and $2.0 billion thereafter through March 22, 2024, and such facility
had approximately $1.6 billion of unused capacity as of December 31, 2022. Management believes that current working capital, results of
operations, and existing financing arrangements are sufficient to fund operations for at least one year from the financial statement issuance date.
```

## Source D: Chapter 11 / bankruptcy mentions (pointer only)

Sentences mentioning Chapter 11, debtor-in-possession, the Bankruptcy Code or a voluntary petition. Hypothetical risk-factor wording matches too.

**None found.**
