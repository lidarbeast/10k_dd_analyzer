"""
Build D2 (Going Concern) labeling worksheets from 10-K PDFs.

The D2 label comes from a human reading the auditor's report and the notes,
cross-checked with SEC EDGAR full-text search — never from a model (CLAUDE.md,
"Never do these").  This module does only the deterministic part of that job:

  1. Locate each auditor's report that covers the financial statements and
     copy it verbatim, from its heading to the auditor's tenure line and
     report date.
  2. List every sentence in the filing containing "substantial doubt" or
     "going concern", with its PDF page — a pointer, not a finding.
  3. Copy the notes subsections headed basis of presentation / liquidity /
     going concern, where management's ASC 205-40 disclosure usually sits.
  4. List Chapter 11 / debtor-in-possession sentences, the 10-K signature
     date, and a pre-filled EDGAR full-text search link.
  5. Lay out the rubric, a glossary and a blank checklist for the reviewer.

No model is called, and nothing here decides whether a going-concern
conclusion exists.  Anything that cannot be located is reported as such.

Usage:
    python -m src.evaluation.d2_worksheets
    python -m src.evaluation.d2_worksheets --company Carvana
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pypdfium2 as pdfium

from src import config
from src.evaluation.d3_worksheets import (
    _SENTENCE_SPLIT,
    PORTFOLIO,
    Section,
    _page_at,
    _section,
    _sha256,
    join_pages,
    read_pages,
)

OUTPUT_DIR = config.GROUND_TRUTH_DIR / "d2_worksheets"

_MONTH = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
_REPORT = re.compile(r"^[ \t]*report\s+of\s+independent\s+registered\s+public\s+accounting\s+firm[ \t]*$", re.I | re.M)
_OPINION = re.compile(r"^[ \t]*opinions?\s+on\s+[^\n]*$", re.I | re.M)
_SERVED = re.compile(r"^[ \t]*we\s+have\s+served\s+as\s+the\s+company\S{0,2}\s+auditor[^\n]*$", re.I | re.M)
_DATE_LINE = re.compile(rf"^[ \t]*{_MONTH}\s+\d{{1,2}},\s+\d{{4}}[ \t]*$", re.M)
_SIGNED = re.compile(r"^[ \t]*/s/[^\n]*$", re.M)
# Paragraph headings an auditor's report can contain.  Listed for the reviewer
# as a map of the report; their presence is not a finding.
_REPORT_HEADINGS = re.compile(
    r"^[ \t]*(opinions?\s+on\s+[^\n]{3,90}|basis\s+for\s+opinions?|critical\s+audit\s+matters?|going\s+concern|"
    r"substantial\s+doubt[^\n]{0,90}|emphasis\s+of\s+(?:a\s+)?matters?[^\n]{0,60}|explanatory\s+paragraph[^\n]{0,60}|"
    r"other\s+matters?|change\s+in\s+accounting\s+principle[^\n]{0,40}|definition\s+and\s+limitations[^\n]{0,90})[ \t]*$",
    re.I | re.M,
)
_MAX_REPORT_PAGES = 6

# \bgoing: "ongoing concern" must not match.
_GC_TERMS = re.compile(r"substantial\s+doubt|\bgoing\s+concern\b", re.I)
_CH11_TERMS = re.compile(r"chapter\s+11|debtor[\s-]+in[\s-]+possession|bankruptcy\s+code|voluntary\s+petitions?", re.I)
_NOTES_HEADING = re.compile(
    r"^[ \t]*((?:note\s+)?(?:\d{1,2}\s*[.:—–-]+\s*)?[A-Za-z ,&'’()-]{0,40}"
    r"(?:liquidity|going\s+concern|basis\s+of\s+presentation)[A-Za-z ,&'’()-]{0,60})[ \t]*$",
    re.I | re.M,
)
_MAX_SUBSECTION_LINES = 60
_EDGAR_CIK = re.compile(r"edgar/data/(\d+)/")
_SIGNATURE_DATE = re.compile(rf"^\s*Dated?:?\s*({_MONTH}\s+\d{{1,2}},\s+\d{{4}})", re.M)


@dataclass
class AuditorReport:
    section: Section
    covers_financial_statements: bool
    covers_icfr: bool
    headings: list[str] = field(default_factory=list)
    signed: str | None = None
    report_date: str | None = None
    complete: bool = True


def _page_offset(text: str, page: int) -> int | None:
    m = re.search(rf"^\[PDF p\. {page}\]$", text, re.M)
    return m.start() if m else None


def find_auditor_reports(text: str) -> tuple[list[AuditorReport], list[int]]:
    """Every auditor's report in the filing, and the pages of report headings
    that head no report (an index entry, or a sentence referring to a report
    printed elsewhere).

    A report runs from its heading to the "We have served as the Company's
    auditor since …" line, extended to the report date if that follows within
    three lines.  Without a tenure line it runs to the next report heading or
    a page cap, and is marked incomplete."""
    starts = list(_REPORT.finditer(text))
    reports: list[AuditorReport] = []
    skipped: list[int] = []
    for n, m in enumerate(starts):
        page = _page_at(text, m.start())
        limit = starts[n + 1].start() if n + 1 < len(starts) else len(text)
        cap = _page_offset(text, page + _MAX_REPORT_PAGES)
        if cap is not None:
            limit = min(limit, cap)
        window = text[m.start():limit]
        opinions = [o.group(0).strip() for o in _OPINION.finditer(window)]
        if not opinions:
            skipped.append(page)
            continue

        served = _SERVED.search(text, m.end(), limit)
        if served:
            end = served.end()
            tail = text[end:limit].split("\n")[1:4]
            offset = end
            for line in tail:
                offset += len(line) + 1
                if _DATE_LINE.match(line):
                    end = offset
                    break
            complete = True
        else:
            end, complete = limit, False

        section = _section(text, m.start(), end, m.group(0).strip())
        body = section.text
        opinion_text = " ".join(opinions).lower()
        signed = _SIGNED.search(body)
        report_date = _DATE_LINE.search(body[-400:])
        reports.append(AuditorReport(
            section=section,
            covers_financial_statements="financial statements" in opinion_text,
            covers_icfr="internal control" in opinion_text,
            headings=[h.group(1).strip() for h in _REPORT_HEADINGS.finditer(body)],
            signed=signed.group(0).strip() if signed else None,
            report_date=report_date.group(0).strip() if report_date else None,
            complete=complete,
        ))
    return reports, skipped


def _sentences(pages: list[str], pattern: re.Pattern) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    seen: set[str] = set()
    for i, page in enumerate(pages, start=1):
        for sentence in _SENTENCE_SPLIT.split(" ".join(page.split())):
            if pattern.search(sentence):
                key = sentence.lower()
                if key not in seen:
                    seen.add(key)
                    hits.append((i, sentence.strip()))
    return hits


def going_concern_sentences(pages: list[str]) -> list[tuple[int, str]]:
    return _sentences(pages, _GC_TERMS)


def chapter11_sentences(pages: list[str]) -> list[tuple[int, str]]:
    return _sentences(pages, _CH11_TERMS)


_NUMBERED_NOTE = re.compile(r"^\s*(?:note\s+)?\d{1,2}\s*[.:—–-]+\s*[A-Z]")
_PAGE_HEADER = re.compile(
    r"\(continued\)|table\s+of\s+contents|notes\s+to\s+(?:the\s+)?(?:consolidated\s+)?financial\s+statements", re.I
)


def _is_page_header(line: str, since_page_break: int) -> bool:
    """Running headers repeated at the top of each page ('COMPANY AND
    SUBSIDIARIES', 'NOTES TO CONSOLIDATED FINANCIAL STATEMENTS', '(Continued)')."""
    s = line.strip()
    return since_page_break <= 4 and bool(s) and (s.isupper() or bool(_PAGE_HEADER.search(s)))


def _looks_like_heading(line: str) -> bool:
    s = line.strip()
    if _NUMBERED_NOTE.match(s):
        return True
    return (0 < len(s) <= 70 and s[0].isupper() and s[-1] not in ".,;:" and len(s.split()) <= 10
            and not re.fullmatch(r"[\d$(),.\s—-]+", s))


def find_notes_subsections(text: str, after: int, exclude: list[tuple[int, int]]) -> list[Section]:
    """Notes subsections headed basis of presentation / liquidity / going
    concern, from their heading to the next heading-like line (page headers
    just after a page break are skipped), capped at a fixed number of lines.

    Only headings after `after` (the first financial-statement report) and
    outside the `exclude` spans (the auditor's reports) are used, so MD&A
    and the report's own "Going Concern" paragraph are not picked up."""
    out: list[Section] = []
    for m in _NOTES_HEADING.finditer(text, after):
        if any(a <= m.start() < b for a, b in exclude):
            continue
        lines = text[m.end():].split("\n")[1:]
        offset = m.end() + 1
        since_page_break = 99
        taken = 0
        for line in lines:
            if line.startswith("[PDF p."):
                since_page_break = 0
            else:
                since_page_break += 1
                if taken > 0 and not _is_page_header(line, since_page_break) and _looks_like_heading(line):
                    break
            offset += len(line) + 1
            taken += 1
            if taken >= _MAX_SUBSECTION_LINES:
                break
        out.append(_section(text, m.start(), offset, m.group(1).strip()))
    return out


def edgar_cik(pdf_path: Path) -> str | None:
    """The filer's CIK from the EDGAR URL the PDF was printed from, if any."""
    pdf = pdfium.PdfDocument(str(pdf_path))
    for i in range(min(3, len(pdf))):
        m = _EDGAR_CIK.search(pdf[i].get_textpage().get_text_range())
        if m:
            return m.group(1)
    return None


def signature_date(pages: list[str]) -> tuple[int, str] | None:
    """The 10-K's signature date: a 'Date:' / 'Dated' line near the end, or, for
    a signature table ('Signature  Title  Date'), the date most of its rows carry."""
    last = range(len(pages), max(0, len(pages) - 12), -1)
    for i in last:
        m = _SIGNATURE_DATE.search(pages[i - 1])
        if m:
            return i, " ".join(m.group(1).split())
    for i in last:
        page = pages[i - 1]
        if re.search(r"signature\s+title\s+date", page, re.I):
            dates = [" ".join(d.split()) for d in re.findall(rf"{_MONTH}\s+\d{{1,2}},\s+\d{{4}}", page)]
            if dates:
                return i, max(set(dates), key=dates.count)
    return None


def _fiscal_year_end(meta: dict) -> dt.date | None:
    m = re.search(rf"({_MONTH})\s+(\d{{1,2}}),\s+(\d{{4}})", meta["fiscal_year"])
    return dt.datetime.strptime(" ".join(m.groups()), "%B %d %Y").date() if m else None


def edgar_search_link(cik: str | None, fye: dt.date | None) -> str | None:
    if not cik or not fye:
        return None
    end = fye + dt.timedelta(days=183)
    return ("https://www.sec.gov/edgar/search/#/q=%22substantial%20doubt%22&dateRange=custom"
            f"&ciks={int(cik):010d}&startdt={fye.isoformat()}&enddt={end.isoformat()}&forms=10-K")


def _load_d2_rubric() -> dict:
    path = config.DIMENSIONS_DIR / "d2_going_concern.json"
    return json.loads(path.read_text(encoding="utf-8"))["rubric"]


_GLOSSARY = """\
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
- **Horizons**: management looks one year from the issuance date; the auditor up to one year beyond the balance-sheet date. Either HIGH source alone is enough."""


def _report_block(r: AuditorReport) -> list[str]:
    s = r.section
    pages = f"PDF p. {s.first_page}" if s.first_page == s.last_page else f"PDF pp. {s.first_page}–{s.last_page}"
    covers = " and ".join(x for x, ok in (("the financial statements", r.covers_financial_statements),
                                           ("internal control over financial reporting", r.covers_icfr)) if ok)
    out = [f"**Report at {pages}** — opinion(s) on {covers}. Signed: `{r.signed or 'not found'}`; "
           f"report date: `{r.report_date or 'not found'}`."]
    if not r.complete:
        out.append("")
        out.append("⚠️ The report's end (auditor-tenure line) was not found; the text below stops at the next report or a page cap. Check the PDF.")
    out.append("")
    out.append("Paragraph headings found in this report (a map only, not a finding): "
               + (", ".join(f"`{h}`" for h in r.headings) if r.headings else "none found"))
    out.append("")
    out.append("```text")
    out.append(s.text)
    out.append("```")
    return out


def _pointer_list(title: str, note: str, hits: list[tuple[int, str]], collapse_over: int = 12) -> list[str]:
    out = [title, "", note, ""]
    if not hits:
        out.append("**None found.**")
        out.append("")
        return out
    collapsed = len(hits) > collapse_over
    if collapsed:
        out.append(f"<details><summary>{len(hits)} sentences</summary>")
        out.append("")
    for page, sentence in hits:
        out.append(f"- p. {page} — {sentence}")
    out.append("")
    if collapsed:
        out.append("</details>")
        out.append("")
    return out


def render_worksheet(company: str, meta: dict, pdf_path: Path, sha256: str, cik: str | None,
                     reports: list[AuditorReport], skipped: list[int], gc_hits: list[tuple[int, str]],
                     notes: list[Section], ch11_hits: list[tuple[int, str]], signed: tuple[int, str] | None,
                     rubric: dict) -> str:
    fs_reports = [r for r in reports if r.covers_financial_statements]
    other = [r for r in reports if not r.covers_financial_statements]
    report_pages = [(r.section.first_page, r.section.last_page) for r in fs_reports]

    def where(page: int) -> str:
        return " *(in the auditor's report)*" if any(a <= page <= b for a, b in report_pages) else ""

    link = edgar_search_link(cik, _fiscal_year_end(meta))
    fye = _fiscal_year_end(meta)
    out: list[str] = []
    out.append(f"# D2 labeling worksheet — {company} {meta['fiscal_year'].split(' ')[0]}")
    out.append("")
    out.append(f"- **Filing:** `data/filings/{pdf_path.name}` — {meta['fiscal_year']}")
    out.append(f"- **Filing SHA-256:** `{sha256}`")
    out.append(f"- **SEC CIK:** `{cik or 'not found'}`")
    out.append("- **10-K signature date:** " + (f"{signed[1]} (PDF p. {signed[0]})" if signed else
                                                 "not found automatically — see the signature page; the auditor's report date is shown in Source A"))
    out.append(f"- **Rubric version:** `{config.RUBRIC_VERSION}`")
    out.append("- **Built by:** `python -m src.evaluation.d2_worksheets` — deterministic extraction; no model was used")
    out.append("")
    out.append("## How to use this worksheet")
    out.append("")
    out.append("1. Read **Source A** (the auditor's report on the financial statements) in full. Look for a paragraph after the opinion that states substantial doubt exists — usually titled \"Going Concern\" or \"Substantial Doubt…\". The heading map under each report helps you find it, but read the text.")
    out.append("2. Read **Source C** (notes subsections) and the entries in **Source B**, which include MD&A. Decide whether management's disclosure — in the notes or MD&A — says doubt exists (not alleviated), says in words that doubt was raised and alleviated, or neither. Judge each passage by its concluding sentence (see the glossary). If Source C is empty, read Note 1 in the PDF.")
    out.append("3. Use **Source D** to decide whether the company was a Chapter 11 debtor-in-possession on the 10-K's signature date.")
    out.append("4. Run the **EDGAR full-text search** cross-check (Q4) and record what it finds.")
    out.append("5. Apply the rubric. Rate only what the filing discloses — not news, later events or anything you know from elsewhere.")
    out.append("6. If the rubric does not give a clear answer, **stop and write it under Ambiguities.** Rubric fixes happen before v1 is frozen, and never after seeing agent output. Do this before looking at any agent output for D2.")
    out.append("")
    out.append(f"## D2 rubric ({config.RUBRIC_VERSION})")
    out.append("")
    for level in ("HIGH", "MEDIUM", "LOW"):
        out.append(f"**{level}:** {rubric[level]['threshold']}")
        out.append("")
    out.append("## Glossary")
    out.append("")
    out.append(_GLOSSARY)
    out.append("")
    out.append("## Checklist (you fill this in)")
    out.append("")
    out.append("- **Q1 — Auditor's going-concern paragraph** (AS 2415): YES / NO")
    out.append("  - Auditor and report date: ")
    out.append("  - If YES, quote and PDF page: ")
    out.append("- **Q2 — Management's disclosure, notes or MD&A** (ASC 205-40): NONE / ALLEVIATED (50-12) / NOT ALLEVIATED (50-13)")
    out.append("  - Quote and PDF page: ")
    out.append("- **Q3 — Chapter 11 debtor-in-possession on the signature date:** YES / NO")
    out.append("  - Evidence and PDF page (petition date; emergence date if any): ")
    out.append("- **Q4 — EDGAR full-text search cross-check:**")
    if link:
        out.append(f"  - Link (search for \"substantial doubt\" in this company's 10-K filings made within six months after fiscal year-end {fye.isoformat()}): <{link}>")
        out.append(f"  - If the link does not pre-fill: go to https://www.sec.gov/edgar/search/, search `\"substantial doubt\"`, set the company to CIK {cik}, form 10-K, filed between {fye.isoformat()} and {(fye + dt.timedelta(days=183)).isoformat()}.")
    else:
        out.append("  - Link could not be built (CIK or fiscal year-end not found). Search https://www.sec.gov/edgar/search/ for `\"substantial doubt\"`, this company, form 10-K.")
    out.append("  - A hit only means the words appear somewhere in the filing — risk factors match too, and EDGAR also searches the filing's exhibits, so check each hit's document type. Record what the hits are, and whether they agree with Q1–Q2: ")
    out.append("")
    out.append("## Verdict (you fill this in)")
    out.append("")
    out.append("- **Rating:** HIGH / MEDIUM / LOW")
    out.append("- **Deciding criterion:** HIGH (a) auditor paragraph / HIGH (b) not alleviated / HIGH (c) Chapter 11 / MEDIUM alleviated / LOW")
    out.append("- **Deciding passage (verbatim):** ")
    out.append("- **PDF page:** ")
    out.append("- **Ambiguities — places the rubric did not give a clear answer:** ")
    out.append("- **Labeled by / date:** ")
    out.append("")

    out.append("## Source A: auditor's report(s) on the financial statements (verbatim)")
    out.append("")
    if not fs_reports:
        out.append("**Not located automatically.** Find the \"Report of Independent Registered Public Accounting Firm\" on the financial statements in Item 8 and read it there.")
        out.append("")
    for r in fs_reports:
        out.extend(_report_block(r))
        out.append("")
    notes_on_skipped = []
    if other:
        notes_on_skipped.append("reports on internal control only (not D2 evidence): "
                                + ", ".join(f"PDF p. {r.section.first_page}" for r in other))
    if skipped:
        notes_on_skipped.append("report headings with no report under them (an index entry, or a reference to a report printed elsewhere): "
                                + ", ".join(f"PDF p. {p}" for p in skipped))
    if notes_on_skipped:
        out.append("Not included: " + "; ".join(notes_on_skipped) + ".")
        out.append("")

    out.extend(_pointer_list(
        "## Source B: every \"substantial doubt\" / \"going concern\" sentence in the filing (pointer only)",
        "Includes boilerplate (the basis-of-accounting sentence) and conditional risk-factor wording. Use it to find pages, then read them in the PDF.",
        [(p, s + where(p)) for p, s in gc_hits],
    ))

    out.append("## Source C: notes subsections on basis of presentation / liquidity / going concern (verbatim)")
    out.append("")
    out.append("Copied from each matching heading to the next heading-like line (at most "
               f"{_MAX_SUBSECTION_LINES} lines). The end is found by a rule of thumb — check the PDF page.")
    out.append("")
    if not notes:
        out.append("**None found.** Read Note 1 (basis of presentation / significant accounting policies) in the PDF.")
        out.append("")
    for n in notes:
        pages = f"PDF p. {n.first_page}" if n.first_page == n.last_page else f"PDF pp. {n.first_page}–{n.last_page}"
        out.append(f"**`{n.heading}`** — {pages}")
        out.append("")
        out.append("```text")
        out.append(n.text)
        out.append("```")
        out.append("")

    out.extend(_pointer_list(
        "## Source D: Chapter 11 / bankruptcy mentions (pointer only)",
        "Sentences mentioning Chapter 11, debtor-in-possession, the Bankruptcy Code or a voluntary petition. Hypothetical risk-factor wording matches too.",
        ch11_hits,
    ))
    return "\n".join(out)


def build_worksheet(company: str, filings_dir: Path = config.FILINGS_DIR, force: bool = False) -> tuple[Path, dict]:
    meta = PORTFOLIO[company]
    pdf_path = filings_dir / meta["filing"]
    pages = read_pages(pdf_path)
    text = join_pages(pages)

    reports, skipped = find_auditor_reports(text)
    fs_reports = [r for r in reports if r.covers_financial_statements]
    first_report = min((text.index(r.section.text) for r in fs_reports), default=0)
    spans = [(text.index(r.section.text), text.index(r.section.text) + len(r.section.text)) for r in reports]
    notes = find_notes_subsections(text, after=first_report, exclude=spans) if fs_reports else []
    gc_hits = going_concern_sentences(pages)
    ch11_hits = chapter11_sentences(pages)

    worksheet = render_worksheet(company, meta, pdf_path, _sha256(pdf_path), edgar_cik(pdf_path), reports, skipped,
                                 gc_hits, notes, ch11_hits, signature_date(pages), _load_d2_rubric())
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{company.lower()}.md"
    if out_path.exists() and not force:
        # A worksheet on disk may hold a human's answers; regenerating would erase them.
        raise FileExistsError(f"{out_path} exists and may hold filled-in answers; pass --force to overwrite it")
    out_path.write_text(worksheet, encoding="utf-8")

    summary = {
        "fs_reports": [f"pp. {r.section.first_page}-{r.section.last_page}{'' if r.complete else ' (INCOMPLETE)'}" for r in fs_reports] or "NOT LOCATED",
        "notes_subsections": [f"{n.heading[:40]} p.{n.first_page}" for n in notes],
        "gc_sentences": len(gc_hits),
        "ch11_sentences": len(ch11_hits),
    }
    return out_path, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build D2 human-labeling worksheets.")
    parser.add_argument("--company", choices=sorted(PORTFOLIO), action="append",
                        help="Company to build (repeatable). Default: all.")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing worksheets, erasing any answers filled in.")
    args = parser.parse_args()
    for company in args.company or sorted(PORTFOLIO):
        path, summary = build_worksheet(company, force=args.force)
        print(f"{company}: {path.relative_to(config.PROJECT_ROOT)}  {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
