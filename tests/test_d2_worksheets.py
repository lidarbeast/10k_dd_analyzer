"""Tests for the D2 worksheet extractor.

The extractor must copy filing text, never judge it.  These tests pin what
it copies — auditor's reports, going-concern sentences, notes subsections —
and check it has no path to a model.  The last tests run against the real
filings when they are present locally (the PDFs are git-ignored).
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.evaluation import d2_worksheets as d2
from src.evaluation.d3_worksheets import join_pages

HEADING = "REPORT OF INDEPENDENT REGISTERED PUBLIC ACCOUNTING FIRM"


def _fs_report(tenure_first: bool = True) -> str:
    end = (["/s/ AUDITOR LLP", "We have served as the Company's auditor since 2015.", "Springfield, Ohio", "March 1, 2024"]
           if tenure_first else
           ["/s/ AUDITOR LLP", "Springfield, Ohio", "March 1, 2024", "We have served as the Company's auditor since 2011."])
    return "\n".join([HEADING, "Opinion on the Financial Statements", "In our opinion...", "Going Concern",
                      "These conditions raise substantial doubt about its ability to continue as a going concern.",
                      "Basis for Opinion", "..."] + end + ["77"])


def test_report_runs_from_heading_to_tenure_and_date() -> None:
    text = join_pages([_fs_report(), "CONSOLIDATED BALANCE SHEETS\nTotal assets 10"])
    reports, skipped = d2.find_auditor_reports(text)
    assert skipped == [] and len(reports) == 1
    r = reports[0]
    assert r.covers_financial_statements and not r.covers_icfr and r.complete
    assert r.section.text.splitlines()[-1] == "March 1, 2024"
    assert "BALANCE SHEETS" not in r.section.text
    assert r.signed == "/s/ AUDITOR LLP" and r.report_date == "March 1, 2024"
    assert "Going Concern" in r.headings


def test_report_ending_with_tenure_line_after_date() -> None:
    reports, _ = d2.find_auditor_reports(join_pages([_fs_report(tenure_first=False)]))
    assert reports[0].section.text.splitlines()[-1].startswith("We have served")
    assert reports[0].report_date == "March 1, 2024"


def test_icfr_report_and_reference_only_heading_are_separated() -> None:
    icfr = "\n".join([HEADING, "Opinion on Internal Control Over Financial Reporting", "...",
                      "We have served as the Company's auditor since 2015."])
    reference = f"{HEADING}\nOur auditor has issued a report on internal control, included in Item 8."
    reports, skipped = d2.find_auditor_reports(join_pages([_fs_report(), icfr, reference]))
    assert [(r.covers_financial_statements, r.covers_icfr) for r in reports] == [(True, False), (False, True)]
    assert skipped == [3]


def test_going_concern_terms_skip_ongoing_concern() -> None:
    pages = ["This is an ongoing concern for management.",
             "The statements contemplate continuation as a going concern. Conditions raise substantial doubt."]
    hits = d2.going_concern_sentences(pages)
    assert [p for p, _ in hits] == [2, 2]


def test_notes_subsection_stops_at_next_note_across_a_page_break() -> None:
    notes_page1 = "1. Basis of Presentation\nThe statements are prepared under GAAP.\nMore text on liquidity plans."
    notes_page2 = "ACME CORP AND SUBSIDIARIES\nNOTES TO CONSOLIDATED FINANCIAL STATEMENTS\n(Continued)\nStill basis text.\n2. Summary of Significant Accounting Policies\nCash."
    text = join_pages([_fs_report(), notes_page1, notes_page2])
    reports, _ = d2.find_auditor_reports(text)
    spans = [(text.index(r.section.text), text.index(r.section.text) + len(r.section.text)) for r in reports]
    notes = d2.find_notes_subsections(text, spans[0][0], spans)
    assert len(notes) == 1
    assert "Still basis text." in notes[0].text
    assert "Summary of Significant Accounting Policies" not in notes[0].text


def test_going_concern_heading_inside_report_is_not_a_notes_subsection() -> None:
    text = join_pages([_fs_report()])
    reports, _ = d2.find_auditor_reports(text)
    spans = [(text.index(r.section.text), text.index(r.section.text) + len(r.section.text)) for r in reports]
    assert d2.find_notes_subsections(text, 0, spans) == []


def test_signature_date_from_signature_table() -> None:
    page = "SIGNATURES\nSignature Title Date\nJane Doe CEO February 11, 2026\nJohn Roe CFO February 11, 2026\nDirector February 10, 2026"
    assert d2.signature_date(["cover", page]) == (2, "February 11, 2026")


def test_edgar_link_is_built_from_cik_and_fiscal_year_end() -> None:
    fye = d2._fiscal_year_end({"fiscal_year": "FY2023 (year ended June 30, 2023)"})
    link = d2.edgar_search_link("1639825", fye)
    assert "ciks=0001639825" in link and "startdt=2023-06-30" in link and "forms=10-K" in link


def test_extractor_has_no_model_path() -> None:
    """D2 labels must come from a human; the worksheet builder may not call a model."""
    source = Path(d2.__file__).read_text(encoding="utf-8").lower()
    for forbidden in ("langchain", "genai", "analysis_agent", "anthropic", "openai"):
        assert forbidden not in source, f"worksheet builder references {forbidden}"


@pytest.mark.parametrize("company", sorted(d2.PORTFOLIO))
def test_real_filings_locate_the_financial_statement_report(company: str) -> None:
    pdf = config.FILINGS_DIR / d2.PORTFOLIO[company]["filing"]
    if not pdf.exists():
        pytest.skip(f"{pdf.name} not present locally")
    text = join_pages(d2.read_pages(pdf))
    reports, _ = d2.find_auditor_reports(text)
    fs = [r for r in reports if r.covers_financial_statements]
    assert fs and all(r.complete for r in fs)
    assert d2.edgar_cik(pdf) is not None


def test_held_out_filing_going_concern_paragraphs_are_captured() -> None:
    """The held-out Chapter 11 filing has an auditor's going-concern paragraph
    in each registrant's report; both reports must be captured whole."""
    pdf = config.FILINGS_DIR / "rev-20221231.pdf"
    if not pdf.exists():
        pytest.skip("held-out filing not present locally")
    reports, _ = d2.find_auditor_reports(join_pages(d2.read_pages(pdf)))
    fs = [r for r in reports if r.covers_financial_statements]
    assert len(fs) == 2
    assert all("Going Concern" in r.headings and r.complete for r in fs)


@pytest.mark.parametrize("builder", ["d2", "d3"])
def test_builders_refuse_to_overwrite_a_worksheet(builder, tmp_path, monkeypatch) -> None:
    """A worksheet on disk may hold a human's labels; rebuilding must not erase it silently."""
    from src.evaluation import d3_worksheets as d3
    mod = {"d2": d2, "d3": d3}[builder]
    company = sorted(mod.PORTFOLIO)[0]
    if not (config.FILINGS_DIR / mod.PORTFOLIO[company]["filing"]).exists():
        pytest.skip("filing not present locally")
    monkeypatch.setattr(mod, "OUTPUT_DIR", tmp_path)
    existing = tmp_path / f"{company.lower()}.md"
    existing.write_text("my filled answers", encoding="utf-8")
    with pytest.raises(FileExistsError):
        mod.build_worksheet(company)
    assert existing.read_text(encoding="utf-8") == "my filled answers"
    mod.build_worksheet(company, force=True)
    assert existing.read_text(encoding="utf-8") != "my filled answers"
