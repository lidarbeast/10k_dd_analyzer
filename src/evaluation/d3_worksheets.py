"""
Build D3 (Legal & Regulatory Exposure) labeling worksheets from 10-K PDFs.

D3 is the one dimension labeled by human judgement, so its labels must come
from a person reading the filing — never from a model (CLAUDE.md, "Never do
these").  This module does only the deterministic part of that job:

  1. Locate Item 3 and the Commitments and Contingencies note by their
     headings and copy them verbatim, with PDF page markers.
  2. Copy the balance-sheet "Total current assets" line (for the 10% test).
  3. List sentences elsewhere in the filing that contain investigation or
     enforcement terms — a keyword pointer, not a selection of matters.
  4. Lay out the rubric, a glossary and a blank matter log for the reviewer.

No model is called, and nothing here decides which matters exist or what
they are rated.  A section that cannot be located is reported as such, never
guessed.

Usage:
    python -m src.evaluation.d3_worksheets
    python -m src.evaluation.d3_worksheets --company Carvana
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pypdfium2 as pdfium

from src import config

_log = logging.getLogger(__name__)

OUTPUT_DIR = config.GROUND_TRUTH_DIR / "d3_worksheets"

PORTFOLIO = {
    "Carvana": {"filing": "cvna-20221231.pdf", "fiscal_year": "FY2022 (year ended December 31, 2022)"},
    "Peloton": {"filing": "pton-20230630.pdf", "fiscal_year": "FY2023 (year ended June 30, 2023)"},
    "Shopify": {"filing": "shop-20251231.pdf", "fiscal_year": "FY2025 (year ended December 31, 2025)"},
}

# Browser print artefacts in PDFs saved from EDGAR HTML (timestamp header,
# URL footer).  Stripped so they do not interrupt the verbatim sections.
_ARTEFACT_LINES = [
    re.compile(r"^\s*\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*[AP]M\b.*$"),
    re.compile(r"^\s*https?://\S+(\s+\d+/\d+)?\s*$"),
]

_ITEM3 = re.compile(r"^[ \t]*item\s+3\s*[.:]\s*legal\s+proceedings\.?[ \t]*$", re.I | re.M)
_ITEM4 = re.compile(r"^[ \t]*item\s+4\s*[.:]", re.I | re.M)
_CONTINGENCIES = re.compile(
    r"^[ \t]*(?:note\s+)?(\d{1,2})[ \t]*[.:—–-]+[ \t]*commitments\s+and\s+contingencies[ \t]*$",
    re.I | re.M,
)
_ITEM9 = re.compile(r"^[ \t]*item\s+9\s*[.:]", re.I | re.M)
_PAGE_MARKER = re.compile(r"^\[PDF p\. (\d+)\]$", re.M)

# Terms that mark a government investigation or enforcement step.  A hit is a
# pointer for the reviewer, not evidence of a matter: generic Item 1A risk
# language ("we may become subject to investigations") matches too.
_ENFORCEMENT_TERMS = re.compile(
    r"\b(investigations?|subpoenas?|civil investigative demands?|consent (?:order|decree)s?|"
    r"civil penalt(?:y|ies)|attorneys? general|enforcement actions?|inquir(?:y|ies)|"
    r"department of justice|federal trade commission|consumer product safety commission|"
    r"consumer financial protection bureau|government agenc(?:y|ies)|regulators?|"
    r"suspen(?:d|ded|sion)|revok(?:e|ed)|revocation|fines?|penalt(?:y|ies)|recalls?)\b",
    re.I,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.;])\s+(?=[A-Z“\"(])")


@dataclass
class Section:
    heading: str
    text: str          # verbatim, with [PDF p. N] markers at page boundaries
    first_page: int
    last_page: int


def clean_page(raw: str) -> str:
    """Normalize line endings and drop browser print artefact lines."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(ln for ln in text.split("\n") if not any(p.match(ln) for p in _ARTEFACT_LINES))


def read_pages(pdf_path: Path) -> list[str]:
    """Return one cleaned text string per PDF page."""
    pdf = pdfium.PdfDocument(str(pdf_path))
    return [clean_page(pdf[i].get_textpage().get_text_range()) for i in range(len(pdf))]


def join_pages(pages: list[str]) -> str:
    return "\n".join(f"[PDF p. {i}]\n{text}" for i, text in enumerate(pages, start=1))


def _page_at(text: str, pos: int) -> int:
    """PDF page number in effect at character offset `pos` of joined text."""
    page = 1
    for m in _PAGE_MARKER.finditer(text, 0, pos + 1):
        page = int(m.group(1))
    return page


def _section(text: str, start: int, end: int, heading: str) -> Section:
    body = text[start:end].rstrip()
    first = _page_at(text, start)
    return Section(heading=heading, text=body, first_page=first, last_page=_page_at(text, end - 1))


def find_item3(text: str) -> Section | None:
    """Item 3 body: the last "Item 3. Legal Proceedings" heading (the table of
    contents comes first) up to the next "Item 4" heading."""
    starts = list(_ITEM3.finditer(text))
    if not starts:
        return None
    start = starts[-1]
    end = _ITEM4.search(text, start.end())
    if end is None:
        return None
    return _section(text, start.start(), end.start(), start.group(0).strip())


def find_contingencies_note(text: str) -> Section | None:
    """Commitments and Contingencies note: its numbered heading up to the next
    note's heading (number + 1), or Item 9 if it is the last note."""
    starts = list(_CONTINGENCIES.finditer(text))
    if not starts:
        return None
    start = starts[-1]
    number = int(start.group(1))
    next_note = re.compile(
        rf"^[ \t]*(?:note\s+)?{number + 1}[ \t]*[.:—–-]+[ \t]*\S", re.I | re.M
    )
    end = next_note.search(text, start.end()) or _ITEM9.search(text, start.end())
    if end is None:
        return None
    return _section(text, start.start(), end.start(), start.group(0).strip())


_UNITS = re.compile(r"\([^()]*\bin\b[^()]*\b(?:thousands|millions|billions)\b[^()]*\)", re.I)
_NUMBERS_ONLY = re.compile(r"^[\s$\u00a0(),.\d\u2014-]+$")


def find_current_assets(pages: list[str]) -> dict | None:
    """The balance sheet's current-assets total, copied verbatim.

    Prefers a "Total current assets" line.  Some balance sheets print the
    subtotal without a label; then the whole current-assets block is copied,
    down to the first numbers-only line, and marked for the reviewer to confirm
    that it is the subtotal.  The reviewer also confirms which column is the
    current year."""
    for i, page in enumerate(pages, start=1):
        if not re.search(r"balance\s+sheets?", page, re.I):
            continue
        lines = page.split("\n")
        units = _UNITS.search(page)
        units_text = units.group(0) if units else None

        total = next((ln.strip() for ln in lines if re.match(r"\s*total\s+current\s+assets\b", ln, re.I)), None)
        if total is not None:
            return {"page": i, "line": total, "units": units_text, "unlabeled_subtotal": False}

        heading = next((k for k, ln in enumerate(lines) if re.fullmatch(r"\s*current\s+assets:?\s*", ln, re.I)), None)
        if heading is None or not any(re.match(r"\s*total\s+assets\b", ln, re.I) for ln in lines):
            continue
        for k in range(heading + 1, len(lines)):
            if _NUMBERS_ONLY.match(lines[k]) and re.search(r"\d", lines[k]):
                block = "\n".join(ln.strip() for ln in lines[heading:k + 1])
                return {"page": i, "line": block, "units": units_text, "unlabeled_subtotal": True}
    return None


def enforcement_keyword_hits(pages: list[str]) -> list[tuple[int, str]]:
    """Sentences containing investigation/enforcement terms, with PDF page."""
    hits: list[tuple[int, str]] = []
    seen: set[str] = set()
    for i, page in enumerate(pages, start=1):
        flat = " ".join(page.split())
        for sentence in _SENTENCE_SPLIT.split(flat):
            if _ENFORCEMENT_TERMS.search(sentence):
                key = sentence.lower()
                if key not in seen:
                    seen.add(key)
                    hits.append((i, sentence.strip()))
    return hits


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _load_d3_rubric() -> dict:
    path = config.DIMENSIONS_DIR / "d3_legal_regulatory_exposure.json"
    return json.loads(path.read_text(encoding="utf-8"))["rubric"]


_GLOSSARY = """\
- **Accrual / "accrued"** (ASC 450-20): the company has recorded an expected loss as a liability because it judges the loss *probable* and can estimate it. Look for "we accrued", "recorded a liability", "an accrual of $X", "included in accrued liabilities".
- **Probable / reasonably possible / remote** (ASC 450-20): the three likelihood tiers. *Probable* = likely → accrue. *Reasonably possible* = more than slight but less than likely → disclose, no accrual. Look for "a loss is reasonably possible", "estimate of the possible loss or range of loss", "cannot reasonably estimate". *Remote* = slight → no disclosure required.
- **Governmental authority**: a federal, state or foreign agency or official acting as a party — e.g. SEC, Department of Justice, FTC, CPSC, CFPB, a state attorney general or DMV, a foreign regulator.
- **Monetary sanction**: a fine, civil penalty, disgorgement or settlement amount payable to a government party.
- **Formal action vs. informal inquiry**: formal = a filed complaint, subpoena, civil investigative demand (CID), consent order or other compulsory step; informal = a voluntary request for information or an "inquiry" with no compulsory step.
- **Class action / derivative suit**: a private suit on behalf of a group of shareholders or customers (class action), or by a shareholder on the company's behalf against its officers (derivative). Private, not governmental.
- **Consolidated current assets**: the balance sheet's "Total current assets". Item 103(b)(2) uses 10% of it as the line below which a damages claim need not be disclosed; the D3 rubric uses it for criterion M-c. Claims raising largely the same issues are added together.
- **Item 105 company-specific**: a risk factor that names the actual agency and matter ("the CPSC is investigating our Tread+ treadmill"), as opposed to generic language ("we may become subject to investigations")."""

_MATTER_LEGEND = """\
| Column | Allowed values |
|---|---|
| Counterparty | `GOV: <which authority>` · `PRIVATE` · `BOTH` |
| Amount stated | `SANCTION $x` · `SETTLEMENT $x` · `ACCRUED $x` · `CLAIMED $x` · `RANGE $x–$y` · `NONE` |
| ASC 450-20 | `ACCRUED` · `REASONABLY POSSIBLE` · `REMOTE` · `NOT STATED` |
| Stage | `FORMAL` · `INFORMAL` · `SETTLED` · `DISMISSED` · `UNCLEAR` |
| Criterion | `H-a` · `H-b` · `H-c` · `M-a` · `M-b` · `M-c` · `none` — letters match the rubric's (a)/(b)/(c) |"""


def render_worksheet(
    company: str,
    meta: dict,
    pdf_path: Path,
    sha256: str,
    item3: Section | None,
    note: Section | None,
    current_assets: dict | None,
    hits: list[tuple[int, str]],
    rubric: dict,
) -> str:
    out: list[str] = []
    out.append(f"# D3 labeling worksheet — {company} {meta['fiscal_year'].split(' ')[0]}")
    out.append("")
    out.append(f"- **Filing:** `data/filings/{pdf_path.name}` — {meta['fiscal_year']}")
    out.append(f"- **Filing SHA-256:** `{sha256}`")
    out.append(f"- **Rubric version:** `{config.RUBRIC_VERSION}`")
    out.append("- **Built by:** `python -m src.evaluation.d3_worksheets` — deterministic extraction; no model was used")
    out.append("")
    out.append("## How to use this worksheet")
    out.append("")
    out.append("1. Read sections A and B below in full. Use section C only as a pointer to other pages worth checking.")
    out.append("2. Log every legal or regulatory matter you find in the matter log, with its PDF page.")
    out.append("3. Apply the rubric literally to each matter, then give the overall rating: the highest level any matter reaches.")
    out.append("4. Rate only what the filing discloses — not news, later events or anything you know from elsewhere.")
    out.append("5. If the rubric does not give a clear answer, **stop and write it under Ambiguities.** Rubric fixes happen before v1 is frozen, and never after seeing agent output.")
    out.append("6. Do this before looking at any agent output for D3.")
    out.append("")
    out.append("## Reference: 10% of consolidated current assets (criterion M-c)")
    out.append("")
    if current_assets:
        out.append(f"Balance sheet, PDF p. {current_assets['page']} — units: {current_assets['units'] or '**not found — check the statement header**'}")
        out.append("")
        out.append("```text")
        out.append(current_assets["line"])
        out.append("```")
        out.append("")
        if current_assets.get("unlabeled_subtotal"):
            out.append("**This balance sheet prints the current-assets subtotal without a label** — it is the last line above. "
                       "Confirm it equals the sum of the lines above it.")
            out.append("")
        out.append("Confirm which column is the fiscal year-end being rated, then fill in:")
        out.append("")
        out.append("- Total current assets (current year): `________`")
        out.append("- 10% line: `________`")
    else:
        out.append("**Not located automatically.** Find \"Total current assets\" on the consolidated balance sheet and fill in:")
        out.append("")
        out.append("- Total current assets (current year, PDF p. ___): `________`")
        out.append("- 10% line: `________`")
    out.append("")
    out.append(f"## D3 rubric ({config.RUBRIC_VERSION})")
    out.append("")
    for level in ("HIGH", "MEDIUM", "LOW"):
        out.append(f"**{level}:** {rubric[level]['threshold']}")
        out.append("")
    out.append("## Glossary")
    out.append("")
    out.append(_GLOSSARY)
    out.append("")
    out.append("## Matter log (you fill this in)")
    out.append("")
    out.append("| # | Matter (short name) | PDF page(s) | Counterparty | Amount stated | ASC 450-20 | Stage | Criterion |")
    out.append("|---|---|---|---|---|---|---|---|")
    for n in range(1, 9):
        out.append(f"| {n} |  |  |  |  |  |  |  |")
    out.append("")
    out.append(_MATTER_LEGEND)
    out.append("")
    out.append("## Verdict (you fill this in)")
    out.append("")
    out.append("- **Rating:** HIGH / MEDIUM / LOW")
    out.append("- **Deciding criterion:** ")
    out.append("- **Deciding passage (verbatim):** ")
    out.append("- **PDF page:** ")
    out.append("- **Ambiguities — places the rubric did not give a clear answer:** ")
    out.append("- **Labeled by / date:** ")
    out.append("")

    for letter, title, section in (("A", "Item 3 — Legal Proceedings", item3),
                                   ("B", "Commitments and Contingencies note", note)):
        out.append(f"## Source {letter}: {title} (verbatim)")
        out.append("")
        if section is None:
            out.append(f"**Not located automatically.** Find the {title} in the PDF and read it there.")
        else:
            pages = (f"PDF p. {section.first_page}" if section.first_page == section.last_page
                     else f"PDF pp. {section.first_page}–{section.last_page}")
            out.append(f"Heading found: `{section.heading}` — {pages}")
            out.append("")
            out.append("```text")
            out.append(section.text)
            out.append("```")
        out.append("")

    out.append("## Source C: enforcement keyword hits elsewhere in the filing (pointer only)")
    out.append("")
    out.append("Sentences anywhere in the filing containing investigation or enforcement terms. **Not exhaustive and not a list of matters:** "
               "generic Item 1A risk language matches too. Use it to find pages worth reading, then log matters from the PDF itself.")
    out.append("")
    out.append(f"<details><summary>{len(hits)} sentences</summary>")
    out.append("")
    for page, sentence in hits:
        out.append(f"- p. {page} — {sentence}")
    out.append("")
    out.append("</details>")
    out.append("")
    return "\n".join(out)


def build_worksheet(company: str, filings_dir: Path = config.FILINGS_DIR, force: bool = False) -> tuple[Path, dict]:
    meta = PORTFOLIO[company]
    pdf_path = filings_dir / meta["filing"]
    pages = read_pages(pdf_path)
    text = join_pages(pages)

    item3 = find_item3(text)
    note = find_contingencies_note(text)
    current_assets = find_current_assets(pages)
    hits = enforcement_keyword_hits(pages)

    worksheet = render_worksheet(
        company, meta, pdf_path, _sha256(pdf_path), item3, note, current_assets, hits, _load_d3_rubric()
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{company.lower()}.md"
    if out_path.exists() and not force:
        # A worksheet on disk may hold a human's answers; regenerating would erase them.
        raise FileExistsError(f"{out_path} exists and may hold filled-in answers; pass --force to overwrite it")
    out_path.write_text(worksheet, encoding="utf-8")

    summary = {
        "item3": f"pp. {item3.first_page}-{item3.last_page}" if item3 else "NOT LOCATED",
        "contingencies_note": f"pp. {note.first_page}-{note.last_page}" if note else "NOT LOCATED",
        "current_assets": f"p. {current_assets['page']}" if current_assets else "NOT LOCATED",
        "keyword_hits": len(hits),
    }
    return out_path, summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build D3 human-labeling worksheets.")
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
