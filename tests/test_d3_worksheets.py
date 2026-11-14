"""Tests for the D3 worksheet extractor.

The extractor must copy filing text, never judge it: these tests pin the
section boundaries it copies and check it has no path to a model.  The last
test runs against the real filings when they are present locally (the PDFs
are git-ignored) and skips otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.evaluation import d3_worksheets as ws


def _joined(*pages: str) -> str:
    return ws.join_pages(list(pages))


def test_item3_skips_table_of_contents_and_stops_at_item4() -> None:
    text = _joined(
        "Item 3. Legal Proceedings\nItem 4. Mine Safety Disclosures",  # table of contents
        "Item 3. Legal Proceedings\nWe are party to X v. Co.",
        "continued text\nItem 4. Mine Safety Disclosures\nNot applicable.",
    )
    section = ws.find_item3(text)
    assert section is not None
    assert "X v. Co." in section.text and "continued text" in section.text
    assert "Mine Safety" not in section.text
    assert (section.first_page, section.last_page) == (2, 3)


def test_item3_heading_variants() -> None:
    for heading in (
        "ITEM 3. LEGAL PROCEEDINGS.",
        "Item 3: Legal Proceedings",
        "Item 3. Legal Proceedings",
    ):
        assert ws.find_item3(_joined(f"{heading}\nbody\nItem 4. Mine Safety")) is not None


def test_contingencies_note_stops_at_next_note_number() -> None:
    text = _joined(
        "17. Commitments and Contingencies\nLegal matters.\n2025 was a year. 18 months of runway.",
        "More legal text.\n18. Related Parties\nFlexport.",
    )
    section = ws.find_contingencies_note(text)
    assert section is not None
    assert "More legal text." in section.text
    assert "Related Parties" not in section.text


def test_contingencies_note_em_dash_heading_and_item9_fallback() -> None:
    text = _joined(
        "NOTE 21 — COMMITMENTS AND CONTINGENCIES\nLast note.\nITEM 9. CHANGES IN AND DISAGREEMENTS"
    )
    section = ws.find_contingencies_note(text)
    assert section is not None and "Last note." in section.text and "CHANGES IN" not in section.text


def test_missing_sections_are_reported_not_guessed() -> None:
    text = _joined("Nothing relevant here.")
    assert ws.find_item3(text) is None
    assert ws.find_contingencies_note(text) is None


def test_current_assets_labeled_line() -> None:
    page = "Consolidated Balance Sheets\n(In millions)\nCash 10 9\nTotal current assets 4,594 4,892\nTotal assets 9 9"
    found = ws.find_current_assets(["cover", page])
    assert found == {
        "page": 2,
        "line": "Total current assets 4,594 4,892",
        "units": "(In millions)",
        "unlabeled_subtotal": False,
    }


def test_current_assets_unlabeled_subtotal() -> None:
    page = (
        "Consolidated Balance Sheets\n(In US $ millions, except share amounts)\nAssets\nCurrent assets\n"
        "Cash and cash equivalents 1,545 1,498\nOther current assets 234 209\n8,296 7,254\n"
        "Long-term assets\nTotal assets 15,189 13,924"
    )
    found = ws.find_current_assets([page])
    assert found is not None and found["unlabeled_subtotal"] is True
    assert found["line"].splitlines()[-1] == "8,296 7,254"
    assert found["units"] == "(In US $ millions, except share amounts)"


def test_browser_artefact_lines_removed() -> None:
    raw = "3/8/26, 3:53 AM cvna-20221231\r\nReal text\r\nhttps://www.sec.gov/Archives/x.htm 55/140\r\n"
    assert ws.clean_page(raw).strip() == "Real text"


def test_keyword_hits_are_deduplicated_with_pages() -> None:
    pages = ["The CPSC opened an investigation. Unrelated.", "The CPSC opened an investigation."]
    assert ws.enforcement_keyword_hits(pages) == [(1, "The CPSC opened an investigation.")]


def test_extractor_has_no_model_path() -> None:
    """D3 labels must come from a human; the worksheet builder may not call a model."""
    source = Path(ws.__file__).read_text(encoding="utf-8")
    for forbidden in ("langchain", "genai", "analysis_agent", "anthropic", "openai"):
        assert forbidden not in source.lower(), f"worksheet builder references {forbidden}"


@pytest.mark.parametrize("company", sorted(ws.PORTFOLIO))
def test_real_filings_locate_every_section(company: str) -> None:
    pdf = config.FILINGS_DIR / ws.PORTFOLIO[company]["filing"]
    if not pdf.exists():
        pytest.skip(f"{pdf.name} not present locally")
    pages = ws.read_pages(pdf)
    text = ws.join_pages(pages)
    assert ws.find_item3(text) is not None
    assert ws.find_contingencies_note(text) is not None
    assert ws.find_current_assets(pages) is not None
