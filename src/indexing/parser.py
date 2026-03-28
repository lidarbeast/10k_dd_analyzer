"""
PDF parser for SEC 10-K filings using Docling.

Converts PDF filings into structured JSON and Markdown output.
This module handles pure text extraction — section-type labeling
is performed downstream in the chunker.
"""

import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path so we can import src from anywhere
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableStructureOptions,
    TesseractCliOcrOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption

from src import config

_log = logging.getLogger(__name__)


def parse_pdf(
    pdf_path: Path,
    output_dir: Path | None = None,
    *,
    write_outputs: bool = True,
) -> dict:
    """
    Parse a SEC 10-K PDF using Docling and save structured output.

    Args:
        pdf_path: Path to the PDF filing.
        output_dir: Directory to write JSON and Markdown outputs.
        write_outputs: Whether to persist JSON/Markdown artifacts.

    Returns:
        dict with keys:
            - source_document: stem of the PDF filename (e.g. "shop-20241231")
            - elements: list of dicts, each with {type, text, page, heading_level}
            - markdown: full Markdown text of the document
    """
    if write_outputs:
        if output_dir is None:
            raise ValueError("output_dir is required when write_outputs=True")
        output_dir.mkdir(parents=True, exist_ok=True)
    source_document = pdf_path.stem

    _log.info(f"Parsing {pdf_path.name} with Docling (Tesseract OCR enabled)...")

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options = TableStructureOptions(do_cell_matching=True)
    pipeline_options.ocr_options = TesseractCliOcrOptions()

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    result = converter.convert(str(pdf_path))
    doc = result.document

    _log.info(f"Converted {pdf_path.name}. Exporting formats...")

    # Export Docling document JSON format
    doc_dict = doc.export_to_dict()
    # Add our source_document identifier to the top level for convenience later
    doc_dict["source_document"] = source_document
    
    if write_outputs:
        json_path = output_dir / f"{source_document}_parsed.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(doc_dict, f, ensure_ascii=False)
        _log.info(f"Saved JSON: {json_path}")

        # Export Markdown format
        md_path = output_dir / f"{source_document}.md"
        with md_path.open("w", encoding="utf-8") as f:
            f.write(doc.export_to_markdown())
        _log.info(f"Saved Markdown: {md_path}")

    return doc_dict


# ── CLI entry-point ──────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import time

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

    parser = argparse.ArgumentParser(
        description="Parse SEC 10-K PDF filings into JSON and Markdown using Docling."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        help=(
            "Path to a specific PDF filing to parse. "
            "If omitted, all *.pdf files in data/filings/ are processed."
        ),
    )
    args = parser.parse_args()

    output_dir = config.PARSED_DIR

    if args.input_file:
        pdf_files = [Path(args.input_file)]
    else:
        pdf_files = sorted(config.FILINGS_DIR.glob("*.pdf"))

    if not pdf_files:
        _log.error("No PDF files found.")
        sys.exit(1)

    for pdf_path in pdf_files:
        if not pdf_path.exists():
            _log.error(f"File not found: {pdf_path}")
            continue
        try:
            start = time.time()
            result = parse_pdf(pdf_path, output_dir)
            texts_count = len(result.get("texts", []))
            _log.info(
                f"  ✓ {pdf_path.name} parsed in {time.time() - start:.1f}s: "
                f"{texts_count} text items exported"
            )
        except Exception as e:
            _log.error(f"  ✗ {pdf_path.name}: {e}")

