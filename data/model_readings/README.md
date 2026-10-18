# Model readings — NOT ground truth

Claude's applications of a rubric to the portfolio filings, kept for **comparison with human labels only**.

- Nothing here is a label, and nothing here may be copied into `labels.json` or loaded by the evaluation harness.
- Labels come from XBRL computation, EDGAR full-text search, or a human reading the filing (see CLAUDE.md, "Never do these").
- If a human label and a model reading disagree, a human re-reads the filing. The model reading is never the tie-breaker.
- Finish your own worksheet before opening a reading; seeing the model's rating first anchors yours.

| Path | Dimension | Rubric version |
|---|---|---|
| `d1/carvana.md`, `d1/peloton.md`, `d1/shopify.md` | D1 Financial Health (Z″ from the PDF statements) | v1 |
| `d3/carvana.md`, `d3/peloton.md`, `d3/shopify.md` | D3 Legal & Regulatory | v1 (with the 2026-09-28 clarifications) |
| `d2/carvana.md`, `d2/peloton.md`, `d2/shopify.md` | D2 Going Concern | v1 (with the 2026-10-01 clarifications) |
| `d5/carvana.md`, `d5/peloton.md`, `d5/shopify.md` | D5 Internal Controls | v1 (with the 2026-10-08 guidance) |
