# D5 reading — MODEL READING — Peloton FY2023

> **MODEL READING — NOT A LABEL.** Claude's application of the v1 D5 rubric (with the 2026-10-08 guidance), for comparison with the human labels only. It must never be copied into `labels.json`; a disagreement is settled by a human re-reading the filing.

- **Filing:** `data/filings/pton-20230630.pdf` — FY2023 (year ended June 30, 2023)
- **Filing SHA-256:** `312efc2529e02cfe2fa486b1e9d5eae325e0766367c59ab7903443dbc668cdd3`
- **Filer status (cover page):** large accelerated filer; Section 404(b) attestation box checked (PDF p. 3)
- **Rubric version:** `v1`
- **Sources A–C:** copied verbatim from the PDF text by deterministic extraction (no model). Checklist and verdict: Claude.

## D5 rubric (v1)

**HIGH:** A material weakness exists at fiscal year-end: management concludes in Item 9A that ICFR was not effective as of fiscal year-end, OR the auditor's AS 2201 opinion is adverse (the company did not maintain effective ICFR). Either conclusion alone is sufficient.

**MEDIUM:** No material weakness at fiscal year-end (management concludes ICFR was effective, and the auditor's AS 2201 opinion, if present, is unqualified), BUT the filing states that a previously reported material weakness was remediated during the fiscal year.

**LOW:** No material weakness at fiscal year-end (management concludes ICFR was effective, or the auditor's AS 2201 opinion is unqualified, and neither conclusion is negative) AND the filing does not report remediation of a previously reported material weakness during the fiscal year.

## Checklist (filled in by Claude)

- **Q1 — Management's ICFR conclusion as of the rated year-end** (S-K Item 308(a)): **NOT EFFECTIVE**
  - Quote and PDF page: "Based on that evaluation, our management concluded that our internal control over financial reporting was not effective as of June 30, 2023 because of the material weaknesses described below." (PDF p. 182)
- **Q2 — Auditor's AS 2201 opinion on ICFR:** **ADVERSE**
  - Auditor, report date, location: Ernst & Young LLP, August 23, 2023 — a separate report on internal control, in Item 8 (PDF p. 124)
  - Quote and PDF page: "In our opinion, because of the effect of the material weaknesses described below on the achievement of the objectives of the control criteria, Peloton Interactive, Inc. (the Company) has not maintained effective internal control over financial reporting as of June 30, 2023, based on the COSO criteria." (PDF p. 124)
  - Cross-reference in the report on the financial statements: "…our report dated August 23, 2023 expressed an adverse opinion thereon." (PDF p. 122, about internal control) — consistent with Q2. The internal-control report's own closing "…which expressed an unqualified opinion thereon" (PDF p. 124) refers to the financial statements.
- **Q3 — Material weaknesses at year-end:** Two. (1) Inventory, first reported for FY2021 and still open: "we have not yet fully remediated the material weakness relating to controls around the existence, completeness, and valuation of inventory" (PDF p. 182). (2) New in FY2023: "we identified a material weakness related to ITGCs in the area of user access over a certain information technology system specific to Precor" (PDF p. 183).
- **Q4 — Remediation of a previously reported material weakness completed during the rated year:** **YES — but it does not decide the rating**
  - Evidence: The FY2022 weakness over impairment-related controls: "We have completed execution of our remediation plan for this material weakness and, as of June 30, 2023, successfully remediated this material weakness" (PDF p. 184). MEDIUM needs effective year-end conclusions; both are negative, so HIGH stands.
- **Q5 — Disclosure controls and procedures (Item 307; context, not rated):** Not effective — "Solely as a result of these material weaknesses, our Chief Executive Officer and Chief Financial Officer concluded that our disclosure controls and procedures were not effective as of June 30, 2023." (PDF p. 182). Rests expressly on the ICFR weaknesses.
- **Look-alikes checked:** Item 1A (pp. 45–46) describes the real weaknesses ("In the course of preparing our financial statements for fiscal 2023, we identified a material weakness in our internal control over financial reporting related to Precor ITGCs") — consistent with Item 9A but not used to decide. Item 9A's statement that the financial statements "present fairly" (PDF p. 184) is about the financial statements, not ICFR. The definition ("A material weakness is a deficiency…", PDF pp. 182 and 124) is not a conclusion. The internal-control report's "unqualified opinion thereon" refers to the financial statements.

## Verdict (filled in by Claude)

- **Rating:** **HIGH**
- **Deciding criterion:** HIGH — management concludes ICFR was not effective as of June 30, 2023, and the auditor's AS 2201 opinion is adverse. Either alone suffices.
- **Deciding passage (verbatim):** "Based on that evaluation, our management concluded that our internal control over financial reporting was not effective as of June 30, 2023 because of the material weaknesses described below."
- **PDF page:** 182 (management); 124 (auditor)
- **Ambiguities — places the rubric did not give a clear answer:** None on the rubric. Precedence worth recording: a remediation completed during the year (Q4) sits beside two open weaknesses; HIGH outranks it, because MEDIUM requires effective year-end conclusions.
- **Read by / date:** Claude (model reading — not a label) / 2026-10-08

## Source A: Item 9A — Controls and Procedures (verbatim)

```text
Item 9A. Controls and Procedures
Evaluation of Disclosure Controls and Procedures
Under the supervision of our Chief Executive Officer and Chief Financial Officer, we evaluated the effectiveness of our disclosure
controls and procedures (as defined in Rules 13a-15(e) and 15d-15(e) under the Exchange Act) as of June 30, 2023.
Disclosure controls and procedures are designed to ensure that information required to be disclosed in our reports filed or submitted
under the Exchange Act is recorded, processed, summarized, and reported within the time periods specified in the SEC’s rules and
forms. Disclosure controls and procedures include controls and procedures designed to ensure that information required to be disclosed
in our reports filed or submitted under the Exchange Act is accumulated and communicated to management, including our Chief
Executive Officer and Chief Financial Officer, to allow timely decisions regarding required disclosure. As described below, we have
identified a material weakness in our internal control over financial reporting related to controls around the existence, completeness, and
valuation of inventory and have identified a new material weakness related to information technology general controls (“ITGCs”) in the
area of user access over a certain information technology system specific to Precor. Solely as a result of these material weaknesses, our
Chief Executive Officer and Chief Financial Officer concluded that our disclosure controls and procedures were not effective as of
June 30, 2023.
Management’s Report on Internal Control over Financial Reporting
Management is responsible for establishing and maintaining adequate internal control over financial reporting for Peloton. With the
participation of the Chief Executive Officer and the Chief Financial Officer, management conducted an evaluation of the effectiveness of
our internal control over financial reporting as of June 30, 2023 based on the guidelines established in the Internal Control—Integrated
Framework (2013 framework) issued by the Committee of Sponsoring Organizations of the Treadway Commission (COSO). Based on
that evaluation, our management concluded that our internal control over financial reporting was not effective as of June 30, 2023
because of the material weaknesses described below.
Inventory and ITGCs Material Weaknesses and Remediation Plans
A material weakness is a deficiency, or a combination of deficiencies, in internal control over financial reporting such that there is a
reasonable possibility that a material misstatement of a company’s annual or interim consolidated financial statements will not be
prevented or detected on a timely basis.
As previously reported in Part II, Item 9A. “Controls and Procedures” of our Annual Reports on Form 10-K for the fiscal years ended
June 30, 2021 and June 30, 2022, we have identified a material weakness in our internal control over financial reporting related to
controls around the existence, completeness, and valuation of inventory.
During the fiscal year ended June 30, 2023, management has continued its enhancements to the Company’s inventory management
process related to the existence, completeness and valuation of inventory and has made further progress in the implementation of the
remediation efforts, including:
• Increased frequency of the periodic inventory physical count process at our distribution centers and final mile locations;
• Increased accuracy of periodic inventory count at all third-party logistics service providers through increased communication,
oversight of their inventory management policies and procedures, and higher partner accountability when dealing with errors;
• Increased operational accuracy of inventory cycle count processes;
• Improved the timeliness and accuracy of transactional processing between Peloton and third-party service providers and
increased the accuracy of inventory data across Peloton internal systems, Peloton warehouses, and third-party providers;
• Consolidation of our inventory network and reduced our inventory exposure to locations with historically high physical count
inaccuracy; and
• Enhancements to training of standard operating procedures and internal controls to key stakeholders within the supply chain,
logistics, and inventory processes.
While management has improved the periodic inventory physical count accuracy across locations and has made progress throughout
the fiscal year ended June 30, 2023 towards the remediation of the material weakness, including the implementation of a number of
operational enhancements to our inventory management process as outlined above, additional time is required to complete the material
weakness remediation work, as management needs to ensure the sustainability of the design and operational effectiveness of recently
implemented controls. In addition, as part of remediation effort, we may deem it necessary to implement additional controls, such as
enhancements to cycle counts controls and systematic reconciliations between warehouse management and Peloton enterprise
resource planning systems. We believe the steps discussed above will be effective in the remediation of the material weakness, but
these remediation steps currently remain ongoing. As a result, we have not yet fully remediated the material weakness relating to
controls around the existence, completeness, and valuation of inventory. We concluded this material weakness did not result in any
material misstatements in our financial statements or disclosures in any of the fiscal years ended June 30, 2021, 2022 or 2023.
[PDF p. 183]
In addition, in connection with our assessment of the effectiveness of internal control over financial reporting as of June 30, 2023, we
identified a material weakness related to ITGCs in the area of user access over a certain information technology system specific to
Precor. Specifically, the Company did not design and maintain sufficient user access controls to ensure appropriate segregation of duties
and adequately restrict user and privileged access to a financial application, programs and data to appropriate Company personnel.
Automated and manual business process controls that are dependent on the affected ITGCs were also deemed ineffective because they
could have been adversely affected to the extent that they rely upon information and configurations from the affected IT system. In order
to remediate this material weakness, we are implementing the following measures:
106
[PDF p. 184]
• rationalization of access privileges for all system users and critical transactions based
• on job responsibilities considering segregation of duties;
• limiting excess rights and access for all system users;
• implementation of controls that require periodic re-evaluation of user access privileges, including administrative access;
• engaging an accounting advisory firm to assist with the documentation, evaluation, remediation and testing of our internal
control over financial reporting based on the criteria established in Internal Control - Integrated Framework (2013) issued by the
Committee of Sponsoring Organizations (COSO) of the Treadway Commission; and
• training of relevant personnel on the design and operation of our internal controls over financial reporting.
These steps are subject to ongoing senior management review, as well as oversight by the audit committee of our Board of Directors.
Additional or modified measures may also be required to remediate the material weaknesses. We will not be able to conclude that we
have completely remediated the material weaknesses until the applicable controls are fully implemented and operate for a sufficient
period of time and management has concluded, through formal testing, that the remediated controls are operating effectively. We will
continue to monitor the design and effectiveness of these and other processes, procedures, and controls and make any further changes
management deems appropriate.
Management considered and documented the procedures performed to gain comfort over the completeness and accuracy of the
financial information relied upon at Precor and to ensure material errors do not exist within the Precor information consolidated into
Peloton’s financial statements.
We concluded with respect to each of the inventory and ITGCs-related material weaknesses described above that these material
weaknesses did not result in any material misstatements in our financial statements or disclosures in any of the fiscal years ended June
30, 2021, 2022 or 2023. Based on additional procedures and post-closing review, management concluded that the consolidated financial
statements included in this Annual Report on Form 10-K present fairly, in all material respects, our financial position, results of
operations, and cash flows for the periods presented in conformity with accounting principles generally accepted in the United States.
Remediation of Previously Disclosed Material Weakness
As previously reported in Part II, Item 9A. “Controls and Procedures” of our Annual Report on Form 10-K for the fiscal year ended June
30, 2022, in connection with our assessment of the effectiveness of internal control over financial reporting as of June 30, 2022, we
identified control deficiencies relating to (i) the design of our controls associated with the application of fair value measurements
pertaining to goodwill and long-lived asset impairment analyses and (ii) evidence of the review supporting the validation of the inputs and
assumptions used in our goodwill and long-lived asset impairment testing and restructuring assessment, which deficiencies in the
aggregate, represented a material weakness in our internal control over financial reporting. We have completed execution of our
remediation plan for this material weakness and, as of June 30, 2023, successfully remediated this material weakness by implementing
the following measures:
• Enhanced the design of our controls and implemented guidelines setting forth specific requirements for documenting our
procedures for validating the data used in our impairment analysis and restructuring assessment.
• Implemented additional review and analysis procedures to validate compliance with our guidelines and our policies outlining the
application fair value in accounting processes when required, including steps to improve the operation and monitoring of control
activities and procedures associated with our impairment assessments; and
• Hiring additional resources to effectively implement additional review and analysis procedures over the assumptions, inputs,
and methodologies.
Changes in Internal Control over Financial Reporting
Other than the completed and ongoing remediation efforts described above, there were no changes in our internal control over financial
reporting (as defined in Rules 13a-15(f) and 15d-15(f) under the Exchange Act) during the three months ended June 30, 2023 that have
materially affected, or are reasonably likely to materially affect, our internal control over financial reporting.
Limitations on the Effectiveness of Disclosure Controls and Procedures
Our management, including our Chief Executive Officer and Chief Financial Officer, does not expect that our disclosure controls and
procedures or internal control over financial reporting will prevent all errors and all fraud. A control system, no matter how well designed
and implemented, can provide only reasonable, not absolute, assurance that the control system’s objectives will be met. Further, the
design of a control system must reflect the fact that there are resource constraints and the benefits of controls must be considered
relative to their costs. Because of the inherent limitations in all control systems, no evaluation of controls can provide absolute assurance
that all control issues within a company are detected. The inherent limitations include the realities that judgments in decision-making can
be faulty and that breakdowns can occur because of simple errors or mistakes. Controls can also be circumvented by the individual acts
of some persons, by collusion of two or more people, or by management override of the controls. Also, projections of any evaluation of
effectiveness to future periods are subject to the risk that controls may become inadequate because of changes in conditions or that the
degree of compliance with the policies or procedures may deteriorate. Because of the inherent limitations in a cost-effective control
system, misstatements due to error or fraud may occur and may not be detected.
[PDF p. 185]
Report of Independent Registered Public Accounting Firm
Ernst & Young LLP, our independent registered public accounting firm that audited the consolidated financial statements, has issued an
audit report on our internal control over financial reporting as of June 30, 2023, which is included in Item 8 of this Annual Report on Form
10-K.
107
[PDF p. 186]
```

## Source B: auditor's opinion on internal control over financial reporting (verbatim, opinion paragraph only)

```text
Opinion on Internal Control Over Financial Reporting
We have audited Peloton Interactive, Inc.’s internal control over financial reporting as of June 30, 2023, based on criteria established in
Internal Control—Integrated Framework issued by the Committee of Sponsoring Organizations of the Treadway Commission (2013
framework) (the COSO criteria). In our opinion, because of the effect of the material weaknesses described below on the achievement of
the objectives of the control criteria, Peloton Interactive, Inc. (the Company) has not maintained effective internal control over financial
reporting as of June 30, 2023, based on the COSO criteria.
A material weakness is a deficiency, or combination of deficiencies, in internal control over financial reporting, such that there is a
reasonable possibility that a material misstatement of the company’s annual or interim financial statements will not be prevented or
detected on a timely basis. The following material weaknesses have been identified and included in management’s assessment.
Management has identified a material weakness in the Company’s inventory process. Management has also identified a material
weakness associated with ineffective user access controls over an information technology system that supports the Company’s financial
reporting process. Consequently, automated and manual business process controls that rely upon information from the affected system
were also deemed ineffective.
We also have audited, in accordance with the standards of the Public Company Accounting Oversight Board (United States) (PCAOB),
the consolidated balance sheets of the Company as of June 30, 2023 and 2022, the related consolidated statements of operations and
comprehensive loss, stockholders’ equity (deficit) and cash flows for each of the three years in the period ended June 30, 2023, and the
related notes. These material weaknesses were considered in determining the nature, timing and extent of audit tests applied in our
audit of the 2023 consolidated financial statements, and this report does not affect our report dated August 23, 2023, which expressed
an unqualified opinion thereon.
```

## Source C: every sentence mentioning "material weakness" in the filing (pointer only)

Sentences on a page of Source A or B are marked by page, so a marked sentence may still sit outside the conclusion (for example the auditor's method wording). Everything else is outside the conclusions — Item 1A risk factors, the auditor's method wording, definitions — and decides nothing on its own.

- p. 6 — • risks related to acquisitions or dispositions, including the acquisition of Precor; • risks related to expansion into international markets; • disruptions or failures of information technology systems, websites; • risks related to payment processing, cybersecurity or data privacy; • risks related to the Peloton App and its ability to work with a range of mobile and streaming technologies, systems, networks, and standards; • our ability to effectively price and market our Connected Fitness Products and subscriptions and our limited operating history with which to predict the profitability of our subscription model; • any inaccuracies in operational and business metrics or forecasts of market growth; • our ability to maintain effective internal control over financial and management systems and remediate material weaknesses; • impacts from warranty claims or product returns; • our ability to maintain, protect, and enhance our intellectual property; • our ability to stay in compliance with laws and regulations that currently apply or become applicable to our business both in the United States and internationally; • our reliance on third parties for computing, storage, processing and similar services and delivery and installation of our products; • our ability to attract and retain highly skilled personnel and maintain our culture; • risks related to our common stock and indebtedness; and • those risks and uncertainties described in the sections titled “Management’s Discussion and Analysis of Financial Condition and Results of Operations” in Part II, Item 7 and “Risk Factors” in Part I, Item 1A of this Annual Report on Form 10-K, as such factors may be updated in our filings with the Securities and Exchange Commission (the “SEC”).
- p. 45 — We have identified material weaknesses in our internal control over financial reporting, and if our remediation of such material weaknesses is not effective, or if we fail to develop and maintain an effective system of disclosure controls and internal control over financial reporting, our ability to produce timely and accurate financial statements or comply with applicable laws and regulations could be impaired.
- p. 45 — In the course of preparing our financial statements for fiscal 2021, fiscal 2022 and fiscal 2023, we identified material weaknesses in our internal control over financial reporting.
- p. 45 — A material weakness is a deficiency, or combination of deficiencies, in internal control over financial reporting, such that there is a reasonable possibility that a material misstatement of the annual or interim financial statements will not be prevented or detected on a timely basis.
- p. 45 — In the course of preparing our financial statements for fiscal 2023, we identified a material weakness related to information technology general controls (“ITGCs”) in the area of user access over a certain information technology system specific to Precor.
- p. 45 — We have concluded that this material weakness arose because we did not design and maintain sufficient user access controls to ensure appropriate segregation of duties and adequately restrict user and privileged access to a financial application, programs, and data to appropriate Company personnel.
- p. 45 — The material weakness that we identified in fiscal 2021 and 2022 related to reporting involving inventory; an additional material weakness we identified in fiscal 2022 related to controls that validate the inputs and assumptions used in our impairment testing.
- p. 45 — We have concluded that these material weaknesses arose because our controls were not effectively designed, documented and maintained to (i) verify that our physical inventory counts were correctly counted and communicated; and (ii) apply fair value measurements and validate the inputs and assumptions used in our impairment testing for reporting in our financial statements.
- p. 45 — To address our material weaknesses, we have made changes to our program and controls as set forth in Part II, Item 9A “Controls and Procedures.
- p. 45 — ” Unless otherwise described in Part II, Item 9A “Controls and Procedures”, we will not be able to fully remediate these material weaknesses until these steps have been completed and have been operating effectively for a sufficient period of time.
- p. 45 — Furthermore, we cannot assure you that the measures we have taken to date, and actions we may take in the future, will be sufficient to remediate the control deficiencies that led to our material weakness in our internal control over financial reporting or that they will prevent or avoid potential future material weaknesses.
- p. 46 — In the course of preparing our financial statements for fiscal 2023, we identified a material weakness in our internal control over financial reporting related to Precor ITGCs.
- p. 46 — Previously, in fiscal 2021 and 2022, we had identified a material weakness related to controls around the existence, completeness, and valuation of inventory.
- p. 47 — See “— We have identified material weaknesses in our internal control over financial reporting, and if our remediation of such material weaknesses is not effective, or if we fail to develop and maintain an effective system of disclosure controls and internal control over financial reporting, our ability to produce timely and accurate financial statements or comply with applicable laws and regulations could be impaired.
- p. 124 *(page of the auditor's ICFR opinion)* — In our opinion, because of the effect of the material weaknesses described below on the achievement of the objectives of the control criteria, Peloton Interactive, Inc.
- p. 124 *(page of the auditor's ICFR opinion)* — A material weakness is a deficiency, or combination of deficiencies, in internal control over financial reporting, such that there is a reasonable possibility that a material misstatement of the company’s annual or interim financial statements will not be prevented or detected on a timely basis.
- p. 124 *(page of the auditor's ICFR opinion)* — The following material weaknesses have been identified and included in management’s assessment.
- p. 124 *(page of the auditor's ICFR opinion)* — Management has identified a material weakness in the Company’s inventory process.
- p. 124 *(page of the auditor's ICFR opinion)* — Management has also identified a material weakness associated with ineffective user access controls over an information technology system that supports the Company’s financial reporting process.
- p. 124 *(page of the auditor's ICFR opinion)* — These material weaknesses were considered in determining the nature, timing and extent of audit tests applied in our audit of the 2023 consolidated financial statements, and this report does not affect our report dated August 23, 2023, which expressed an unqualified opinion thereon.
- p. 124 *(page of the auditor's ICFR opinion)* — Our audit included obtaining an understanding of internal control over financial reporting, assessing the risk that a material weakness exists, testing and evaluating the design and operating effectiveness of internal control based on the assessed risk, and performing such other procedures as we considered necessary in the circumstances.
- p. 182 *(Item 9A page)* — As described below, we have identified a material weakness in our internal control over financial reporting related to controls around the existence, completeness, and valuation of inventory and have identified a new material weakness related to information technology general controls (“ITGCs”) in the area of user access over a certain information technology system specific to Precor.
- p. 182 *(Item 9A page)* — Solely as a result of these material weaknesses, our Chief Executive Officer and Chief Financial Officer concluded that our disclosure controls and procedures were not effective as of June 30, 2023.
- p. 182 *(Item 9A page)* — Based on that evaluation, our management concluded that our internal control over financial reporting was not effective as of June 30, 2023 because of the material weaknesses described below.
- p. 182 *(Item 9A page)* — Inventory and ITGCs Material Weaknesses and Remediation Plans A material weakness is a deficiency, or a combination of deficiencies, in internal control over financial reporting such that there is a reasonable possibility that a material misstatement of a company’s annual or interim consolidated financial statements will not be prevented or detected on a timely basis.
- p. 182 *(Item 9A page)* — “Controls and Procedures” of our Annual Reports on Form 10-K for the fiscal years ended June 30, 2021 and June 30, 2022, we have identified a material weakness in our internal control over financial reporting related to controls around the existence, completeness, and valuation of inventory.
- p. 182 *(Item 9A page)* — While management has improved the periodic inventory physical count accuracy across locations and has made progress throughout the fiscal year ended June 30, 2023 towards the remediation of the material weakness, including the implementation of a number of operational enhancements to our inventory management process as outlined above, additional time is required to complete the material weakness remediation work, as management needs to ensure the sustainability of the design and operational effectiveness of recently implemented controls.
- p. 182 *(Item 9A page)* — We believe the steps discussed above will be effective in the remediation of the material weakness, but these remediation steps currently remain ongoing.
- p. 182 *(Item 9A page)* — As a result, we have not yet fully remediated the material weakness relating to controls around the existence, completeness, and valuation of inventory.
- p. 182 *(Item 9A page)* — We concluded this material weakness did not result in any material misstatements in our financial statements or disclosures in any of the fiscal years ended June 30, 2021, 2022 or 2023.
- p. 183 *(Item 9A page)* — In addition, in connection with our assessment of the effectiveness of internal control over financial reporting as of June 30, 2023, we identified a material weakness related to ITGCs in the area of user access over a certain information technology system specific to Precor.
- p. 184 *(Item 9A page)* — Additional or modified measures may also be required to remediate the material weaknesses.
- p. 184 *(Item 9A page)* — We will not be able to conclude that we have completely remediated the material weaknesses until the applicable controls are fully implemented and operate for a sufficient period of time and management has concluded, through formal testing, that the remediated controls are operating effectively.
- p. 184 *(Item 9A page)* — We concluded with respect to each of the inventory and ITGCs-related material weaknesses described above that these material weaknesses did not result in any material misstatements in our financial statements or disclosures in any of the fiscal years ended June 30, 2021, 2022 or 2023.
- p. 184 *(Item 9A page)* — Remediation of Previously Disclosed Material Weakness As previously reported in Part II, Item 9A.
- p. 184 *(Item 9A page)* — “Controls and Procedures” of our Annual Report on Form 10-K for the fiscal year ended June 30, 2022, in connection with our assessment of the effectiveness of internal control over financial reporting as of June 30, 2022, we identified control deficiencies relating to (i) the design of our controls associated with the application of fair value measurements pertaining to goodwill and long-lived asset impairment analyses and (ii) evidence of the review supporting the validation of the inputs and assumptions used in our goodwill and long-lived asset impairment testing and restructuring assessment, which deficiencies in the aggregate, represented a material weakness in our internal control over financial reporting.
- p. 184 *(Item 9A page)* — We have completed execution of our remediation plan for this material weakness and, as of June 30, 2023, successfully remediated this material weakness by implementing the following measures: • Enhanced the design of our controls and implemented guidelines setting forth specific requirements for documenting our procedures for validating the data used in our impairment analysis and restructuring assessment.
